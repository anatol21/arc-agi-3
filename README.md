# ARC-AGI-3

Research tooling for adaptive agents and world models in ARC-style reasoning.

## Local ARC practice and Ollama benchmark

[`apps/arc_practice/`](apps/arc_practice/) contains a standard-library local web
application for manually solving ARC tasks and benchmarking local Ollama models.
It supports ARC task JSONL files, a 30×30 runtime grid cap, protected evaluation
answers, configurable core instructions, run exports, and an inspector for the
exact local model request, response, parsed prediction, and available reasoning.

The application does not include datasets. Keep ARC, GoogleDS, BARC, synthetic
training data, and raw benchmark outputs local. See the app’s
[README](apps/arc_practice/README.md) for setup and launch instructions.

## Held-out ARC validation benchmark

The [held-out validation report](benchmarks/arc-validation/) covers 138 public ARC tasks from the grid benchmark. It includes the original prompt, one worked example, and machine-readable input and generation lengths for the base and checkpoint-660 model arms.

## Cluster scheduling and inference

[Cluster Lab and scheduling guide](scheduling/) documents queue selection, checkpointed workloads and automatic continuation within a GPU-hour budget. Start with the [local dashboard simulator](apps/cluster_dashboard/) or the [standalone Slurm scheduler](scheduling/slurm/). The [inference batching helpers](scheduling/inference/) cover length grouping and memory backoff.

The [resumable ARC inference worker](scheduling/arc_worker/) includes the decoding loop, early response saves, and an offline interruption test. Follow the [worker runbook](agent.md) to adapt it and launch a budgeted campaign.
