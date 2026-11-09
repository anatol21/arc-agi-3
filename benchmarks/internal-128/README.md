# Direct answers on 128 internal validation tasks

The base model and checkpoint-660 each generated one response to every task. The 128 tasks (134 query grids) were drawn from the training-source ARC tasks for internal validation and held out of the 880-task fine-tuning split. The separate 138-task public evaluation is reported in its own experiment.

| Model condition | Exact tasks | Exact query grids | Invalid JSON or grids | Reached token limit |
|---|---:|---:|---:|---:|
| Base, adapter disabled | 19 / 128 (14.84%) | 20 / 134 | 4 | 0 |
| Checkpoint-660, adapter enabled | 33 / 128 (25.78%) | 34 / 134 | 4 | 1 |

Both models solved 16 tasks. Checkpoint-660 alone solved 17, base alone solved 3, and neither solved 92. The task IDs for these categories can be printed with the verification command below.

## Evidence

- [predictions.jsonl](predictions.jsonl) is the original generation journal, with one row per task and model. Each row contains the decoded response verbatim in `text`, the input and output token counts, token allowance, stop reason and generation time. Invalid outputs are preserved.
- [per-task.jsonl](per-task.jsonl) is the original scoring journal. Each row records whether every query grid matched, individual grid scores, format errors and the model condition.
- [summary.json](summary.json) is the archived final report. [instruction.txt](instruction.txt) is the shared instruction extracted from the original user messages. The per-task payload followed the final newline as compact JSON containing demonstrations and query inputs.
- [evidence-sha256.json](evidence-sha256.json) contains hashes for these four public files. The two JSONL files were copied byte-for-byte. The published summary was independently reproduced from the saved responses and the original answer file.

## Generation and scoring

One three-H100 worker loaded the native quantized base and checkpoint-660. Both conditions used the same model, tokenizer and greedy decoder. The worker enabled the adapter for `finetuned` and disabled it for `base`. The protocol used one candidate, one beam, thinking disabled, an 18,432-token context and at most 8,192 new tokens. The complete prompt was retained and its length deducted from the context. Generation alternated conditions by task, reused completed responses across allocations and scored each answer exactly as emitted. There was no Python refinement or test-time training in this run.

The original [stager](producer/arc-flash-dev-benchmark-v2/stage_benchmark.py) separated the user messages from the scoring answers. The [generation entry point](producer/arc-flash-dev-benchmark-v2/run_benchmark.py) loaded and checked the model, set the adapter state, generated a response, and appended it to the journal. The [allocation wrapper](producer/arc-flash-benchmark-budget-v1/launch.py) set a deadline and invoked the same generation entry point for successive allocations. The [scoring entry point](producer/arc-flash-dev-benchmark-v2/score_benchmark.py) called the [original parser and exact-grid scorer](producer/arc-flash-dev-benchmark-v2/protocol.py).

A response must contain only an `outputs` array with the required number of rectangular integer grids, each with values from 0 through 9. A task is correct when all of its query grids match their references exactly. Reaching the token limit does not by itself determine the score.

## Verify locally

From this directory, run the following with Python 3.10 or later. The first command checks file hashes, complete coverage, parsing and aggregates, and prints the three sets of solved task IDs. The second command independently repeats exact-grid scoring if you have the original labeled manifest.

```sh
python3 verify.py
python3 verify.py --labeled-dev /path/to/arc-agi3-data-v1/data/dev.jsonl
```

The labeled file is identified by the archived SHA-256 in [data-manifest.json](producer/arc-flash-dev-benchmark-v2/data-manifest.json). This release was checked with both commands. The fresh score agreed with all 256 archived task/model records, including the individual grid flags and parser exceptions.

## Replay generation on a provisioned cluster

The [producer directory](producer/) contains the generation, staging, scoring and Slurm code. The numeric inference and scoring entry points retain their original bytes. [Source hashes](producer/source-hashes.json) identify each archived and published file. The distributed copy removes site paths and account settings, uses the parent directory of the stager as its default workspace, and retains only the inference functions from two training helpers. Its integrity manifests describe the public copy. The benchmark ID in the published evidence identifies the original staged recipe. Re-staging on another filesystem can produce a different ID because the recipe also recorded local checkpoint and loader paths.

To replay generation, provide the original input data, checkpoint-660, matching base snapshot and tokenizer, the staged v6 training configuration and verification receipts, and the compatible [Axolotl runtime](producer/arc-flash-dev-benchmark-v2/runtime_identity.py). Put both producer subdirectories next to the private `arc-agi3-data-v1`, `arc-flash-run-v6` and `runs/` directories on shared storage. New runs need their own output workspace. From that workspace, with the matching environment active, use:

```sh
python arc-flash-dev-benchmark-v2/stage_benchmark.py
python arc-flash-benchmark-budget-v1/stage_budget.py
mkdir -p logs
export ARC_INFERENCE_ENV=/path/to/the/inference/environment
sbatch --account="$SLURM_ACCOUNT" --partition="$GPU_PARTITION" arc-flash-benchmark-budget-v1/run_full.sbatch
```

Set the account and partition for your site. The wrapper requests three GPUs for up to six hours and checkpoints completed responses between allocations. The historical model artifacts, private manifests with labels, and cluster environment are external inputs. Publication verification used the CPU scorer and existing saved responses.
