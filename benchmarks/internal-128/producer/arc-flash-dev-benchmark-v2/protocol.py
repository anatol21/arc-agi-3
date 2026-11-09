"""Pure-CPU protocol, strict grid scoring and resumable prediction records."""
import hashlib
import json
import os
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
    payload = json.loads(text, object_pairs_hook=unique_object)
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


def append_record(path, value):
    """One newline-terminated durable record; interrupted tails are recovered on resume."""
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_predictions(path, task_ids, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        boundary = data.rfind(b"\n") + 1
        if not repair_tail:
            raise ValueError("Incomplete prediction tail; resume generation to recover it")
        # Never silently discard bytes: preserve the interrupted tail beside the log.
        recovery = path.with_suffix(".interrupted-tail")
        with recovery.open("ab") as stream:
            stream.write(data[boundary:] + b"\n")
        data = data[:boundary]
        with path.open("r+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
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


def score(answers, predictions):
    answer_ids = [row["task_id"] for row in answers]
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError("Duplicate scorer task")
    details, per_arm, correct = [], {}, {}
    for arm in ARMS:
        processed = solved = invalid = right_grids = processed_grids = truncated = 0
        for answer in answers:
            task_id, expected = answer["task_id"], answer["outputs"]
            record = predictions.get((task_id, arm))
            if record is None:
                continue
            processed += 1
            processed_grids += len(expected)
            truncated += record["finish_reason"] == "token_limit"
            failure = None
            try:
                outputs = parse_outputs(record["text"], len(expected))
                matches = [actual == target for actual, target in zip(outputs, expected, strict=True)]
            except (ValueError, TypeError, KeyError) as error:
                invalid += 1
                failure = str(error)
                matches = [False] * len(expected)
            task_correct = all(matches)
            solved += task_correct
            right_grids += sum(matches)
            correct[(task_id, arm)] = task_correct
            details.append({"task_id": task_id, "arm": arm, "task_correct": task_correct,
                            "grid_correct": matches, "format_error": failure,
                            "finish_reason": record["finish_reason"], "source": answer.get("source")})
        complete = processed == len(answers)
        per_arm[arm] = {"complete": complete, "processed_tasks": processed,
                        "total_tasks": len(answers), "solved_tasks": solved,
                        "task_accuracy": solved / len(answers) if complete else None,
                        "processed_grids": processed_grids,
                        "total_grids": sum(len(answer["outputs"]) for answer in answers),
                        "correct_grids": right_grids,
                        "grid_accuracy": right_grids / processed_grids if complete else None,
                        "invalid_json_or_grids": invalid, "token_limit_tasks": truncated}
    paired = [task_id for task_id in answer_ids if all((task_id, arm) in correct for arm in ARMS)]
    comparison = {"paired_tasks": len(paired), "both_correct": 0, "finetuned_only_correct": 0,
                  "base_only_correct": 0, "both_wrong": 0}
    for task_id in paired:
        tuned, base = correct[(task_id, "finetuned")], correct[(task_id, "base")]
        label = "both_correct" if tuned and base else "finetuned_only_correct" if tuned else "base_only_correct" if base else "both_wrong"
        comparison[label] += 1
    complete = all(value["complete"] for value in per_arm.values())
    return {"protocol": "strict greedy task accuracy; all query grids exact; one candidate; no repair or TTT",
            "complete": complete, "arms": per_arm, "paired_comparison": comparison}, details
