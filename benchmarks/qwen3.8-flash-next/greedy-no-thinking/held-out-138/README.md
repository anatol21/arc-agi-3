# Held-out ARC validation, 138 tasks

Both the base model and checkpoint-660 finished all 138 tasks (185 query grids per arm). The primary result uses the first completed cluster response per task and model, in the source precedence recorded by the original audit. Each task requires an exact match on every query grid. Incomplete JSON, invalid grids and responses truncated at the token limit count as failures.

| Model | Tasks solved | Task accuracy | Query grids correct | Outputs at token limit |
|---|---:|---:|---:|---:|
| Base | 4 / 138 | 2.90% | 4 / 185 | 4 |
| Checkpoint-660 | 7 / 138 | 5.07% | 8 / 185 | 11 |

| Outcome | Count | Task IDs |
|---|---:|---|
| Both | 4 | `4aab4007`, `af22c60d`, `c663677b`, `ca8f78db` |
| Checkpoint-660 only | 3 | `1e97544e`, `e66aafb8`, `e95e3d8e` |
| Base only | 0 | — |
| Neither | 131 | See the [paired task table](evidence/paired-results.csv) |

## Experimental conditions

The two arms used the native NF4 loading path. The fine-tuned arm enabled the checkpoint-660 adapter and the base arm disabled it. Each task received one direct JSON answer with greedy decoding, thinking disabled, a single beam and an 18,432-token context. The completion budget was the smaller of 8,192 tokens and the space remaining after the full prompt. There was no Python refinement or test-time training. The 138 tasks were reserved from fine-tuning and some were used in earlier API prompt experiments. The [task profiles](../arc-validation/) give the prompt, one example and token counts.

## Combining the allocations

The original audit gives source precedence to the serial debug, serial pilot, three batched utilization runs, full serial run, and dashboard versions 1, 4, 5 and 6, in that order. It collapses duplicate terminal and batch representations within each source. Of the 333 archived terminal observations, 276 are primary and 57 are additional records. The records include carried-forward responses. Repeated or copied responses can represent the same generation. There are 43 task/model pairs with multiple records. Only one task/model pair changes exact-match status across sources.

Checkpoint-660’s first recorded response for `67b4a34d` was incorrect. A later batched response and the full serial response solved it. Giving the full serial run precedence wherever available changes checkpoint-660 to **8/138** and keeps base at **4/138**. The full serial source contains 121 task/model records out of 276, so that sensitivity analysis also combines runs. Serial and batched decoding can produce different greedy outputs. The paired comparison describes this held-out subset under the recorded execution conditions.

## Evidence

- [observations.jsonl](evidence/observations.jsonl) preserves the exact decoded model text for every scored source observation, including malformed outputs. Each line gives the task, model arm, source package, per-pair precedence (`source_order == 0` for the primary record), prompt and output token counts, finish reason, SHA-256 of the UTF-8 response, per-grid matches, format error and task score. `finetuned` denotes checkpoint-660. All 276 primary response digests, source packages, token counts and stop reasons also match the [previously published task profiles](../arc-validation/task-costs.jsonl). Cluster paths and scheduler identifiers are omitted.
- [paired-results.csv](evidence/paired-results.csv) contains one row per task with the source and primary score for each arm and the score under serial precedence.
- [summary.json](evidence/summary.json) contains denominators, solved IDs and both precedence rules. [Producer code](producer/README.md) documents the serial and batched generation paths.

## Reproduce the results

From this directory, rebuild the table and summary from the published evidence using Python 3.10 or later. No GPU or model download is required.

```sh
python3 code/summarize.py --out /tmp/arc-held-out-results
diff evidence/summary.json /tmp/arc-held-out-results/summary.json
diff evidence/paired-results.csv /tmp/arc-held-out-results/paired-results.csv
```

To independently score the actual response text, add the two public evaluation directories. The script checks a digest over the 138 sorted reference rows, calls the original strict scorer, and verifies the recorded task, grid and format outcomes for all 333 observations. The references are from [ARC-AGI](https://github.com/fchollet/ARC-AGI/tree/master/data/evaluation) and [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2/tree/main/data/evaluation). The release distributes generated predictions and score flags. Obtain the underlying task data from the upstream repositories.

```sh
python3 code/summarize.py --out /tmp/arc-held-out-results \
  --references /path/to/ARC-AGI/data/evaluation /path/to/ARC-AGI-2/data/evaluation
```

For an operator with the original workspace and journals, [final_heldout.py](code/final_heldout.py) is the original aggregation script. It verifies source identities and fingerprints, applies the original scorer and records source checksums and repeated responses. [export_observations.py](code/export_observations.py) reads those verified native sources and checks each decoded text digest against the audit before emitting the public evidence. The intermediate audit refers to private archive locations and belongs on the operator’s filesystem.

```sh
python3 code/final_heldout.py /path/to/native-workspace > /tmp/arc-held-out-audit.json
python3 code/export_observations.py /path/to/native-workspace /tmp/arc-held-out-audit.json > /tmp/arc-observations.jsonl
diff evidence/observations.jsonl /tmp/arc-observations.jsonl
```
