# Training-set neighbors of held-out ARC task 1e97544e

16 November 2026. The job 7853809 recipe consumed the prepared training split. The archived data/train.jsonl and sft/train.jsonl match their manifest SHA-256 digests. The ordered task IDs match exactly (880), and the assistant answer on every SFT row matches the corresponding query outputs. The target is in the evaluation manifest. No identical target input/output pair was found in any training task. Demonstrations are in the user prompt; the query answer is the assistant completion. These are statements about this fine-tuning dataset, independent of undocumented pretraining exposure.

Run `python3 research/evidence/direct-answer-audit/training_neighbors.py` from the workspace. The filter is retrospective. Of the 880 tasks, 19 have the same input/output dimensions, at least one change, no changes to visible nonzero input cells, and no zeros in the completed output on every pair. Three of these use colors 1 through K with K increasing by one across consecutive examples, including the supervised query.

| Training task | Palette sizes, demonstrations then query | Query size | Missing query cells | Relevant structure |
|---|---|---|---:|---|
| 73251a56 | 6,7,8 → 9 | 21×21 | 55 | Growing color runs and row-dependent repetition boundaries. Same sequence of palette sizes as the target. |
| c3f564a4 | 5,6,7 → 8 | 16×16 | 39 | Consecutive numbers wrap to 1. Start advances one color per row. |
| 29ec7d0e | 5,6,7,8 → 9 | 18×18 | 51 | Cyclic arithmetic with a row-dependent increment; its second query row is 123456789123456789. |
| Held-out 1e97544e | 6,7,8 → 9 | 23×23 | 73 | Horizontal cyclic sequence; the duplicated value moves one column to the right in each successive row. |

No two-dimensional training output (demonstration or supervised query) matched the target output formula under the tested transformations. The test permits rotations/reflections and arbitrary 3–9-color cycles. In zero-based coordinates the formula selects index (column + int(column < row)) modulo K in the first-row color cycle. It fits all four target outputs. This test does not exhaust possible related generators.

For the saved NF4 base response, all 35 errors are exactly +1 modulo 9, all on or above the main diagonal (7 on, 28 above). Checkpoint-660 has zero errors. Two BF16 sampling conditions each have 9 on and 14 above. Coordinates and values are in audit.json alongside this report. These output comparisons motivate a hypothesis about maintaining alignment at the repeat, but do not identify causal training rows. The query's own three demonstrations illustrate its exact rule. An independent test of training influence would require matched training interventions and fresh examples.
