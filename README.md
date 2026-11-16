# ARC-AGI-3

Research tooling for adaptive agents and world models in ARC-style reasoning.

## Local ARC practice and Ollama benchmark

[`apps/arc_practice/`](apps/arc_practice/) contains a standard-library local web
application for manually solving ARC tasks and benchmarking local Ollama models.
It supports ARC task JSONL files, a 30×30 runtime grid cap, protected evaluation
answers, configurable core instructions, run exports, and an inspector for the
exact local model request, response, parsed prediction, and available reasoning.

The application reads locally provisioned datasets. Keep training data, reference
answers, credentials and private run exports outside version control. The benchmark
directories below contain the reviewed public evidence. See the app’s
[README](apps/arc_practice/README.md) for setup and launch instructions.

## Presentation and training code

[Download the 56-slide presentation](presentations/heldout-direct-answers/heldout-direct-answers.pdf) · [LaTeX sources and slide evidence](presentations/heldout-direct-answers/) · [Checkpoint-660 training code and configuration](fine-tuning/qwen3.8-flash-next/checkpoint-660/) · [Experiment and script index](presentations/heldout-direct-answers/EXPERIMENTS.md)

## Benchmark results

The [benchmark reports](benchmarks/) include generated model responses, per-task scores, producer code and CPU verification commands for both the 128-task internal evaluation and the 138-task public held-out evaluation. The [task profiles](benchmarks/qwen3.8-flash-next/greedy-no-thinking/arc-validation/) give the original prompt, one worked example and measured response lengths.

The [Python refinement runs](benchmarks/qwen3.8-flash-next/python-refinement/) include the internal base/checkpoint-660 comparison and the [held-out base-only comparison with and without optional Python exploration](benchmarks/qwen3.8-flash-next/python-refinement/held-out-138/).

The [OpenRouter prompt study](benchmarks/qwen3.8-flash/openrouter-prompt-study/) reports two trials per task for five methods on the 138 public tasks, including malformed replies and provider errors.

From the repository root, run `python3 benchmarks/verify.py` to reproduce the released summaries from the saved evidence without a GPU or API calls. Each benchmark README also documents how to rescore the generated answers against separately provisioned references.

## Cluster engineering

[Cluster engineering](cluster-engineering/) contains the Cluster Lab dashboard, Slurm scheduler, batching helpers and resumable ARC inference worker. The [worker runbook](cluster-engineering/agent.md) covers local restart tests and launching a budgeted campaign.
