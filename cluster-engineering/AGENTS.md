# Working on resumable inference

When creating or changing a worker, follow [the worker runbook](agent.md). It describes stable item IDs, checkpoint ordering, memory backoff, the interruption test, and budgeted Slurm continuation.

Use [the ARC inference example](arc_worker/) to trace a real decoding call. Use synthetic inputs in public tests. Keep the inference prompt, model identity and terminal conditions explicit, and version changes that would affect existing outputs. Run the CPU restart and scheduler tests in the runbook before launching a cluster campaign.
