# ARC benchmark results

## Qwen3.8-Flash-Next · greedy direct answers, thinking disabled

Both experiments used the native quantized base model, greedy direct JSON answers and thinking disabled. Checkpoint-660 enabled the trained adapter. The base condition disabled it. Each task counts as solved only when every query grid matches exactly.

| Evaluation set | Tasks | Base exact | Checkpoint-660 exact | Solved by both |
|---|---:|---:|---:|---:|
| [Internal validation](qwen3.8-flash-next/greedy-no-thinking/internal-128/) | 128 | 19 | 33 | 16 |
| [Public held-out validation](qwen3.8-flash-next/greedy-no-thinking/held-out-138/) | 138 | 4 | 7 | 4 |

The 128 internal tasks came from the training-source collection and were held out of the 880-task fine-tuning split. The 138 public evaluation tasks cover a different cohort. Their primary result uses the first completion under the original source precedence across serial and batched runs. Preferring the full serial source wherever it has a response changes checkpoint-660 to 8/138. The [paired results](qwen3.8-flash-next/greedy-no-thinking/held-out-138/evidence/paired-results.csv) identify the affected task.

Each directory includes the generated responses, per-task decisions, historical generation and scoring code, and commands to check the report on CPU. The [validation task profiles](qwen3.8-flash-next/greedy-no-thinking/arc-validation/) include the original prompt, a worked example and token counts.

## Qwen3.8-Flash-Next · BF16 sampled direct answers

The [BF16 sampling report](qwen3.8-flash-next/bf16-sampling/) contains 276 scored responses on 138 public tasks. Each strategy solved the same five tasks.

## Qwen3.8-Flash-Next · Python refinement

The [Python refinement report](qwen3.8-flash-next/python-refinement/) covers three partial experiments with sandboxed execution on the internal validation split. Each run includes generated programs and per-task scores.

## Qwen3.8-Flash · OpenRouter prompt study

The [OpenRouter study](qwen3.8-flash/openrouter-prompt-study/) evaluated five prompt methods with two trials per task. Its counts use a different protocol.

Run `python3 benchmarks/verify.py` from the repository root to check the benchmark reports against their published evidence.
