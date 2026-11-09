"""Pure-CPU protocol, strict grid scoring and resumable prediction records."""
import hashlib
import json
import os

if __package__:
    from .result_io import results_lock, snapshot_jsonl
else:
    from result_io import results_lock, snapshot_jsonl

SCORING_POLICY = "exact-grid-cap-failure-independent-grid-v2"
from pathlib import Path

ARMS = ("finetuned", "base")
TASKS = 128
CONTEXT = 18432
MAX_NEW_TOKENS = 8192


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, records):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))


def grid(value):
    if (not isinstance(value, list) or not 1 <= len(value) <= 30
        or not isinstance(value[0], list) or not 1 <= len(value[0]) <= 30):
        raise ValueError("Grid must have 1–30 rows and columns")
    width = len(value[0])
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("Grid must be rectangular")
        if any(type(cell) is not int or not 0 <= cell <= 9 for cell in row):
            raise ValueError("Cells must be integers 0–9")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_outputs(text, count):
    payload = safe_json(text)
    if not isinstance(payload, dict) or set(payload) != {"outputs"}:
        raise ValueError('Response must be exactly {"outputs": [...]}')
    if not isinstance(payload["outputs"], list) or len(payload["outputs"]) != count:
        raise ValueError("Response has the wrong number of grids")
    return [grid(value) for value in payload["outputs"]]


def split_dev(sft, labeled, manifest):
    """Retain the exact user messages; answers go only to a separate scorer file."""
    records, tasks = rows(sft), rows(labeled)
    ids = [record["task_id"] for record in records]
    expected = manifest["task_ids"]["dev"]
    if len(ids) != TASKS or len(set(ids)) != TASKS or set(ids) != set(expected):
        raise ValueError("Development task IDs differ from the fixed 128-task split")
    if len(tasks) != TASKS or {task["task_id"] for task in tasks} != set(ids):
        raise ValueError("Labeled development data differs from SFT data")
    by_id = {task["task_id"]: task for task in tasks}
    prompts, answers = [], []
    for record in records:
        messages = record["messages"]
        if [message["role"] for message in messages] != ["user", "assistant"]:
            raise ValueError("Unexpected message roles")
        task = by_id[record["task_id"]]
        visible = json.loads(messages[0]["content"].split("\nTask:\n", 1)[1])
        if (set(visible) != {"train", "test"} or visible["train"] != task["train"]
            or visible["test"] != [{"input": pair["input"]} for pair in task["test"]]):
            raise ValueError("User prompt differs from visible demonstrations/test inputs")
        expected_outputs = [grid(pair["output"]) for pair in task["test"]]
        if parse_outputs(messages[1]["content"], len(task["test"])) != expected_outputs:
            raise ValueError("SFT answer differs from the development scorer")
        prompts.append({"task_id": record["task_id"], "messages": [messages[0]],
                        "query_count": len(task["test"])})
        answers.append({"task_id": record["task_id"], "outputs": expected_outputs,
                        "source": task["source"]})
    return prompts, answers


def safe_json(text):
    """Bound untrusted model JSON before passing it to the Python parser."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1048576:
        raise ValueError("JSON response exceeds parsing limit")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64: raise ValueError("JSON nesting limit exceeded")
        elif char in "]}": depth -= 1
    try:
        return json.loads(text, object_pairs_hook=unique_object,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Invalid JSON constant")))
    except (RecursionError, OverflowError) as error:
        raise ValueError("JSON parser nesting/resource limit") from error


def append_record(path, value):
    path = Path(path)
    with results_lock(path.parent):
        with path.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def parse_predictions(data, task_ids):
    result = {}
    for line in data.splitlines():
        record = json.loads(line)
        key = (record["task_id"], record["arm"])
        if (record["task_id"] not in task_ids or record["arm"] not in ARMS or key in result
            or record.get("finish_reason") not in ("eos", "token_limit")
            or not isinstance(record.get("text"), str)):
            raise ValueError("Unknown, duplicate or incomplete prediction record")
        result[key] = record
    return result


def load_predictions(path, task_ids, *, repair_tail=False):
    path = Path(path)
    with results_lock(path.parent):
        data, _ = snapshot_jsonl(path, repair_tail=repair_tail)
    return parse_predictions(data, task_ids)


def separate_thinking(raw):
    if not isinstance(raw, str) or raw.count("</think>") != 1:
        raise ValueError("Expected exactly one closing thinking delimiter")
    reasoning, final = raw.split("</think>", 1)
    if "<think>" in reasoning or "<think>" in final:
        raise ValueError("Ambiguous or repeated thinking boundaries")
    if not reasoning.strip(): raise ValueError("No observable thinking before the closing delimiter")
    return reasoning.strip(), final.strip()


def thinking_fields(raw):
    try:
        reasoning, final = separate_thinking(raw)
        return {"raw_text": raw, "reasoning": reasoning, "final_answer": final,
                "text": final, "separation_error": None}
    except ValueError as error:
        return {"raw_text": raw, "reasoning": None, "final_answer": None,
                "text": "", "separation_error": str(error)}


def validate_template(tokenizer, messages):
    on = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False,
                                      add_generation_prompt=True, enable_thinking=True)
    off = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False,
                                       add_generation_prompt=True, enable_thinking=False)
    if on == off or not on.rstrip().endswith("<think>"):
        raise RuntimeError("Tokenizer does not provide a distinct open thinking prefix")
    return {"thinking_template_differs": True, "prefilled_open_think": True,
            "thinking_suffix": on[-150:], "disabled_suffix": off[-150:]}


def token_accounting(completion, marker, eos):
    ids = completion[:-1] if completion and completion[-1] == eos else completion
    positions = [i for i in range(len(ids)-len(marker)+1) if marker and ids[i:i+len(marker)] == marker]
    if len(positions) != 1:
        return {"reasoning_tokens": None, "final_answer_tokens": None,
                "thinking_delimiter_tokens": None, "eos_tokens": len(completion)-len(ids)}
    at = positions[0]
    return {"reasoning_tokens": at, "final_answer_tokens": len(ids)-at-len(marker),
            "thinking_delimiter_tokens": len(marker), "eos_tokens": len(completion)-len(ids)}


def count_thinking_tokens(tokenizer, completion, eos):
    return token_accounting(completion, tokenizer.encode("</think>", add_special_tokens=False), eos)


def validate_thinking_record(record):
    expected = thinking_fields(record.get("raw_text"))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Stored thinking/final separation differs from raw generation")
    tokens = record.get("completion_token_ids")
    marker = record.get("thinking_delimiter_token_ids")
    if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
        or not isinstance(marker, list) or not marker
        or any(type(t) is not int or t < 0 for t in marker)):
        raise ValueError("Missing or invalid token evidence")
    if (type(record.get("generated_tokens")) is not int or record.get("generated_tokens") != len(tokens)
        or type(record.get("token_budget")) is not int
        or not 0 < len(tokens) <= record["token_budget"] <= MAX_NEW_TOKENS
        or type(record.get("prompt_tokens")) is not int or record["prompt_tokens"] < 1
        or record["prompt_tokens"] + len(tokens) > CONTEXT
        or record["token_budget"] != min(MAX_NEW_TOKENS, CONTEXT-record["prompt_tokens"])):
        raise ValueError("Token counts/budget do not match retained token IDs")
    eos = record.get("eos_token_id")
    if type(eos) is not int: raise ValueError("Missing EOS token identity")
    if (record["finish_reason"] == "eos") != bool(tokens and tokens[-1] == eos):
        raise ValueError("EOS stopping reason differs from token evidence")
    counts = token_accounting(tokens, marker, eos)
    if any(record.get(key) != value or type(record.get(key)) is not type(value) for key,value in counts.items()):
        raise ValueError("Thinking token accounting differs from retained token IDs")
    if record.get("accounting_validated") is not (all(value is not None for value in counts.values())):
        raise ValueError("Invalid token-accounting status")
    if not isinstance(record.get("attempt_id"), str) or not record["attempt_id"]:
        raise ValueError("Missing attempt provenance")
    if type(record.get("seconds")) not in (float,int) or not __import__('math').isfinite(record["seconds"]) or record["seconds"] < 0:
        raise ValueError("Invalid generation timing")


def score(answers, predictions, *, thinking=True):
    """One policy for both conditions; structure validity and cap failure are separate."""
    if len({a["task_id"] for a in answers}) != len(answers): raise ValueError("Duplicate scorer task")
    if thinking:
        for record in predictions.values(): validate_thinking_record(record)
    details = []
    correct = {}
    per_arm = {}
    for arm in ARMS:
        processed = solved = invalid = right = processed_grids = capped = 0
        arm_records = []
        for answer in answers:
            record = predictions.get((answer["task_id"], arm))
            if record is None: continue
            arm_records.append(record)
            processed += 1
            expected = answer["outputs"]
            processed_grids += len(expected)
            errors = [None] * len(expected)
            content_matches = [False] * len(expected)
            error = None
            if thinking and record["separation_error"]:
                error = record["separation_error"]
            else:
                try:
                    payload = safe_json(record["text"])
                    if not isinstance(payload,dict) or set(payload) != {"outputs"}:
                        raise ValueError('Response must be exactly {"outputs": [...]}')
                    outputs = payload["outputs"]
                    if not isinstance(outputs,list) or len(outputs) != len(expected):
                        raise ValueError("Response has the wrong number of grids")
                    for i,(actual,target) in enumerate(zip(outputs,expected)):
                        try: content_matches[i] = grid(actual) == target
                        except (ValueError,TypeError) as failure: errors[i] = str(failure)
                except (ValueError,TypeError,KeyError,RecursionError,OverflowError) as failure:
                    error = str(failure)
            format_error = error or next((e for e in errors if e is not None),None)
            invalid += format_error is not None
            cap = record["finish_reason"] == "token_limit"
            capped += cap
            matches = [False]*len(expected) if cap else content_matches
            task_correct = not cap and format_error is None and all(matches)
            solved += task_correct
            right += sum(matches)
            correct[(answer["task_id"],arm)] = task_correct
            details.append({"task_id":answer["task_id"],"arm":arm,"task_correct":task_correct,
                            "grid_correct":matches,"content_grid_correct":content_matches,"format_error":format_error,
                            "grid_format_errors":errors,"token_cap_failure":cap,"finish_reason":record["finish_reason"],
                            "source":answer.get("source")})
        complete = processed == len(answers)
        per_arm[arm] = {"complete":complete,"processed_tasks":processed,"total_tasks":len(answers),
                       "solved_tasks":solved,"task_accuracy":solved/len(answers) if complete else None,
                       "processed_grids":processed_grids,"total_grids":sum(len(a["outputs"]) for a in answers),
                       "correct_grids":right,"grid_accuracy":right/processed_grids if complete and processed_grids else None,
                       "invalid_json_or_grids":invalid,"token_limit_tasks":capped,
                       "thinking_separation_failures":sum(r.get("separation_error") is not None for r in arm_records) if thinking else 0,
                       "completed_generated_tokens":sum(r.get("generated_tokens",0) for r in arm_records),
                       "completed_generation_seconds":sum(r.get("seconds",0) for r in arm_records)}
    paired = [a["task_id"] for a in answers if all((a["task_id"],arm) in correct for arm in ARMS)]
    comparison = {"paired_tasks":len(paired),"both_correct":0,"finetuned_only_correct":0,"base_only_correct":0,"both_wrong":0}
    for task in paired:
        a,b = correct[(task,"finetuned")], correct[(task,"base")]
        comparison["both_correct" if a and b else "finetuned_only_correct" if a else "base_only_correct" if b else "both_wrong"] += 1
    return {"protocol":SCORING_POLICY,"complete":all(a["complete"] for a in per_arm.values()),
            "arms":per_arm,"paired_comparison":comparison}, details
