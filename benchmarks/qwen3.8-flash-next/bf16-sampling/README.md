# BF16 base model with sampled direct answers

Two sampling strategies each produced one response on the same 138 public validation tasks (185 query grids). A task counted as solved when the decoded JSON contained the correct number of rectangular grids and every query grid matched exactly.

| Strategy | Temperature | Top-p | Top-k | Exact tasks | Exact grids | Format errors | Token-limited responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| `low_nucleus` | 0.2 | 0.95 | 0 | 5/138 (3.62%) | 5/185 | 12 | 4 |
| `instruct_sampling` | 0.7 | 0.8 | 20 | 5/138 (3.62%) | 5/185 | 13 | 6 |

| Paired task outcome | Count | Task IDs |
|---|---:|---|
| Both strategies | 5 | `4aab4007`, `c663677b`, `ca8f78db`, `e66aafb8`, `e95e3d8e` |
| Low nucleus only | 0 | None |
| Instruct sampling only | 0 | None |
| Neither | 133 | See the [paired results](paired-results.csv) |

## Model and inference

The Qwen3.8-Flash-Next base model ran across four H100 GPUs, with the large PLE embedding in host memory. The verified loader recorded 1,449 BF16 parameter tensors and 48 FP32 tensors, zero packed parameters and zero adapter tensors. Both strategies used the same native [direct-answer prompt](../greedy-no-thinking/arc-validation/), with thinking disabled, one beam, one sampled answer and a seed setting of 42. The context limit was 18,432 tokens. Each response had an allowance of up to 8,192 new tokens within that context. Top-k of zero disables top-k filtering.

The pilot completed the first 12 short-prompt tasks per strategy. The continuation reused those observations and evaluated the remaining 126 per strategy. The worker grouped prompts by length, saved completed rows during generation, and resumed unfinished rows from committed token IDs. The sampler seeds each physical call using its batch and saved prefixes. Changing batch membership or interruption timing can change the next draw. The reported results describe one observed draw for each task and strategy.

The [NF4 base model with greedy decoding](../greedy-no-thinking/held-out-138/) solved 4/138 tasks on this cohort. The paired CSV includes that historical baseline. Precision, sampling and execution layout differ between the BF16 sampling runs and the NF4 baseline, so their score difference cannot isolate the effect of quantization.

## Generated evidence and code

- [responses.jsonl](responses.jsonl) contains the decoded answer, text SHA-256, token count, finish reason, score flags and sampling settings for each task and strategy. Each record identifies whether it came from the pilot or continuation.
- [summary.json](summary.json) and [paired-results.csv](paired-results.csv) are rebuilt by [report.py](report.py).
- The [producer snapshots](producer/) include the BF16 loader, sampled decoding loop, batching policy, terminal journal, and native scoring integration. Their hashes were checked against the pilot and continuation package manifests. Repeating generation requires the original weights, tokenizer, staged task inputs and compatible GPU runtime.
- [export_scores.py](export_scores.py) checks the sealed plan and pilot evidence, reads the saved continuation journals, verifies the reference digest and runs the native scorer to export terminal observations.
- [Allocation costs](allocation-costs.json) record the Slurm charges. The 252 continuation responses used 27.84 GPU-hours including loading, interrupted work and failed attempts. The separate 12-task sampling pilot used 0.82 GPU-hours for both strategies together.

## Reproduce the scores

From this directory run:

```sh
python3 report.py --out /tmp/arc-bf16-report
diff summary.json /tmp/arc-bf16-report/summary.json
diff paired-results.csv /tmp/arc-bf16-report/paired-results.csv
```

To rescore the actual text independently, add `--references /path/to/ARC-AGI/data/evaluation /path/to/ARC-AGI-2/data/evaluation`. This checks the 276 saved task and grid outcomes against the separately provisioned public grids. The [benchmarks verifier](../../verify.py) checks the published summaries on CPU.
