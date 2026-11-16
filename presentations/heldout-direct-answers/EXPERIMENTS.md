# Experiment and script index

| Experiment | Evidence | Producer or analysis code |
|---|---|---|
| ARC-only adapter training | [Config, split manifest and completion](../../fine-tuning/qwen3.8-flash-next/checkpoint-660/evidence/) | [Submitted package](../../fine-tuning/qwen3.8-flash-next/checkpoint-660/arc-flash-run-v6/) |
| Tensor-level NF4 loader replay | [Tensor inventory](../../fine-tuning/qwen3.8-flash-next/checkpoint-660/nf4-audit/tensor-inventory.csv) | [Replay](../../fine-tuning/qwen3.8-flash-next/checkpoint-660/nf4-audit/replay.py) |
| NF4 greedy, internal 128 | [Scores and responses](../../benchmarks/qwen3.8-flash-next/greedy-no-thinking/internal-128/) | [Producer](../../benchmarks/qwen3.8-flash-next/greedy-no-thinking/internal-128/producer/) |
| NF4 greedy, held-out 138 | [Paired scores, primary and sensitivity](../../benchmarks/qwen3.8-flash-next/greedy-no-thinking/held-out-138/evidence/) | [Serial and batched producers](../../benchmarks/qwen3.8-flash-next/greedy-no-thinking/held-out-138/producer/) |
| BF16 greedy pilot, 12 tasks | [Per-task scores](../../benchmarks/qwen3.8-flash-next/bf16-greedy/short-pilot/evidence/scores.json) | [Loader, generator and scorer](../../benchmarks/qwen3.8-flash-next/bf16-greedy/short-pilot/producer/arc-bf16-base-debug-v1/) |
| BF16 sampled, two settings on 138 | [Summary and responses](../../benchmarks/qwen3.8-flash-next/bf16-sampling/) | [Producers](../../benchmarks/qwen3.8-flash-next/bf16-sampling/producer/) |
| Five OpenRouter prompts, two trials | [Trial evidence and settings](../../benchmarks/qwen3.8-flash/openrouter-prompt-study/) | [Request and evaluation code](../../apps/arc_practice/server.py), [trial report](../../benchmarks/qwen3.8-flash/openrouter-prompt-study/report.py) |
| Internal Python experiments | [Three refinement conditions and paired pilot](../../benchmarks/qwen3.8-flash-next/python-refinement/) | [Original refinement producers](../../benchmarks/qwen3.8-flash-next/python-refinement/producer/), [paired pilot and forced-code continuation](../../benchmarks/qwen3.8-flash-next/python-refinement/thinking-on/internal-exploration-pilot/producer/) |
| Held-out Python, A/B first pass | [Frozen per-task ledger](../../benchmarks/qwen3.8-flash-next/python-refinement/held-out-138/python-case-audit/per-task-audit.csv) | [Generated worker package and scorer](../../benchmarks/qwen3.8-flash-next/python-refinement/held-out-138/producer/arc-python-heldout-length-v1/) |
| Direct-answer case studies and token costs | [Audit](evidence/report.md), [training neighbors](evidence/1e97544e-training-neighbors.md) | [Response analysis](evidence/analyze.py), [neighbor search](evidence/training_neighbors.py), [token accounting](evidence/token_accounting.py) |
| Python case studies | [Task traces and examples](python-case-audit/) | [Error classification](python-case-audit/build_audit.py), [task maps](python-first-pass/build_maps.py) |

The [weight intervention, synthetic-data comparisons and ARC-AGI-3 plans](proposals-backup/protocols/) describe proposed work. The frozen experiment results above come from earlier runs.
