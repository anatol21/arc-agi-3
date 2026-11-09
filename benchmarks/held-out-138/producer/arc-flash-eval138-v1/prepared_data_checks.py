"""Read-only verification of the existing ARC tokenized data and Arrow cache."""
import json
from pathlib import Path

EXPECTED = {"train": (880, 2772080, 18184), "dev": (128, 342610, 11072)}
CACHE_KEYS = {"train": "1b3f1468e5c826a056f4353898ab34d9", "dev": "8f2b04332877e3208f30d8a6d6e9e8d9"}


def check_tokenized(path: Path, split: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    count, total, longest = EXPECTED[split]
    if len(records) != count:
        raise RuntimeError(f"Wrong {split} task count")
    lengths = []
    for record in records:
        if set(record) != {"input_ids", "attention_mask", "labels"}:
            raise RuntimeError(f"Unexpected tokenized fields in {split}")
        ids, mask, labels = (record[name] for name in ("input_ids", "attention_mask", "labels"))
        if not 0 < len(ids) <= 18432 or len(mask) != len(ids) or len(labels) != len(ids):
            raise RuntimeError(f"Invalid token array lengths in {split}")
        if any(type(value) is not int or not 0 <= value < 248320 for value in ids) or any(value != 1 for value in mask):
            raise RuntimeError(f"Invalid token IDs or attention mask in {split}")
        target = next((i for i, value in enumerate(labels) if value != -100), None)
        if target is None or target == 0 or labels[target:] != ids[target:] or 248046 not in labels[target:]:
            raise RuntimeError(f"Invalid answer-only labels in {split}")
        lengths.append(len(ids))
    if sum(lengths) != total or max(lengths) != longest:
        raise RuntimeError(f"{split} token counts changed since the audit")
    if split == "train" and lengths[0] != longest:
        raise RuntimeError("The longest training task is no longer first")
    return records


def check_cached_records(dataset, expected: list[dict]) -> None:
    dataset.reset_format()
    if len(dataset) != len(expected):
        raise RuntimeError("Prepared Arrow cache has a different row count")
    for index, (cached, original) in enumerate(zip(dataset, expected, strict=True)):
        if any(cached.get(name) != original[name] for name in ("input_ids", "attention_mask", "labels")):
            raise RuntimeError(f"Prepared Arrow cache differs from verified tokens at row {index}")


