# Completed cluster benchmarks

Both experiments used the native quantized base model, greedy direct JSON answers and thinking disabled. Checkpoint-660 enabled the trained adapter. The base condition disabled it. Each task counts as solved only when every query grid matches exactly.

| Evaluation set | Tasks | Base exact | Checkpoint-660 exact | Solved by both |
|---|---:|---:|---:|---:|
| [Internal validation](internal-128/) | 128 | 19 | 33 | 16 |
| [Public held-out validation](held-out-138/) | 138 | 4 | 7 | 4 |

The 128 internal tasks came from the training-source collection and were held out of the 880-task fine-tuning split. The 138 public evaluation tasks cover a different cohort. Their primary result uses the first completion under the original source precedence across serial and batched runs. Preferring the full serial source wherever it has a response changes checkpoint-660 to 8/138. The [paired results](held-out-138/evidence/paired-results.csv) identify the affected task.

Each directory includes the generated responses, per-task decisions, historical generation and scoring code, and commands to check the report on CPU. The [OpenRouter study](../../benchmarks/openrouter-138/) evaluated five prompt methods with two trials per task. Its counts use a different protocol. The BF16 cluster runs and partial Python-refinement experiments are outside these completed NF4 results.
