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

The [held-out validation report](cluster-engineering/arc-validation/) covers 138 public ARC tasks from the grid benchmark. It includes the original prompt, one worked example, and machine-readable input and generation lengths for the base and checkpoint-660 model arms.

## Cluster engineering

[Cluster engineering](cluster-engineering/) contains the Cluster Lab dashboard, Slurm scheduler, batching helpers and resumable ARC inference worker. The [worker runbook](cluster-engineering/agent.md) covers local restart tests and launching a budgeted campaign.
