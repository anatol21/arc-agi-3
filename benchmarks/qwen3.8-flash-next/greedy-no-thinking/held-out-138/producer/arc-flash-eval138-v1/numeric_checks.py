"""CPU-only batch provenance and numerical comparison helpers."""
import hashlib
import json
import math
from pathlib import Path


def finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def check_collated_batch(batch: dict, vocab_size: int) -> dict:
    ids, masks, targets = (batch[name] for name in ("input_ids", "attention_mask", "labels"))
    if len(ids) != 1 or len(masks) != 1 or len(targets) != 1:
        raise ValueError("Expected the original single-example microbatch")
    ids, mask, labels = ids[0], masks[0], targets[0]
    if not ids or len(ids) != len(mask) or len(ids) != len(labels):
        raise ValueError("Collated token-array lengths differ")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError("Collated input token outside the vocabulary")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("Invalid collated attention mask")
    supervised = []
    for index, (token, active, label) in enumerate(zip(ids, mask, labels)):
        if label != -100:
            if not active or label != token:
                raise ValueError("Collation changed answer labels or supervised a padding token")
            if index > 0:
                supervised.append(index)
    if not supervised:
        raise ValueError("No supervised targets remain after causal shifting")
    if 248046 not in [labels[index] for index in supervised]:
        raise ValueError("Assistant end token disappeared during collation")
    return {"sequence_length": len(ids), "nonpadding_tokens": sum(mask),
            "shifted_supervised_tokens": len(supervised),
            "first_supervised_position": supervised[0],
            "last_supervised_position": supervised[-1]}


def classify_failure(cce_loss, first_bad_activation, reference, *, atol=0.03, rtol=0.01):
    if first_bad_activation is not None:
        return "nonfinite_model_activation"
    if not math.isfinite(float(cce_loss)):
        if reference is not None and math.isfinite(float(reference["loss"])):
            return "loss_path_disagreement"
        return "nonfinite_loss_unresolved"
    if reference is not None and not math.isfinite(float(reference["loss"])):
        return "nonfinite_reference_loss"
    if reference is not None and not math.isclose(float(cce_loss), float(reference["loss"]),
                                                 abs_tol=atol, rel_tol=rtol):
        return "loss_path_disagreement"
    return None


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class TokenIndex:
    """Match the real collated arrays to the checksummed, unpadded source row."""
    def __init__(self, records):
        self.records = {}
        for split, task_id, row in records:
            key = (split, token_hash(row["input_ids"]))
            if key in self.records:
                raise ValueError("Ambiguous tokenized task identity")
            self.records[key] = (task_id, row)

    @classmethod
    def from_prepared(cls, config, prepared):
        records = []
        for split, field in (("train", "datasets"), ("dev", "test_datasets")):
            rows = [json.loads(line) for line in Path(config[field][0]["path"]).read_text().splitlines()]
            for task_id, row in zip(prepared["splits"][split]["task_order"], rows, strict=True):
                records.append((split, task_id, row))
        return cls(records)

    def check(self, batch, split, vocab_size=248320):
        info = check_collated_batch(batch, vocab_size)
        mask = batch["attention_mask"][0]
        active = [i for i, value in enumerate(mask) if value]
        if active != list(range(active[0], active[-1] + 1)):
            raise ValueError("Padding may not split a source example")
        row = {name: [batch[name][0][i] for i in active]
               for name in ("input_ids", "attention_mask", "labels")}
        key = (split, token_hash(row["input_ids"]))
        if key not in self.records:
            raise ValueError("Collated example does not match a verified source task")
        task_id, original = self.records[key]
        if row != original:
            raise ValueError(f"Collated labels/masks differ from the complete source answer: {task_id}")
        return {**info, "split": split, "task_id": task_id, "token_sha256": key[1]}
