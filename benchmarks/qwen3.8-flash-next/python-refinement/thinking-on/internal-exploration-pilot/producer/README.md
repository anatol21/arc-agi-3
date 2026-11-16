# Python exploration pilot

The frozen panel and decoding conditions are listed in [PLAN.md](PLAN.md). The two arms use the same frozen base with the adapter disabled. AX and AY use program refinement. BX and BY also offer up to two Python probes before the first solver. Query labels are read by the separate CPU scorer.

Each GPU block runs a fresh hidden-grid check. The model must produce a fenced Python probe, obtain the six digits by running that probe, and write an executable solver that reproduces those digits on a different input. The job records the probe and solver tokens and the returned observation under the block’s control directory. The sampled decoder records forced reasoning closures separately from natural closures. A separate four-token greedy comparison uses the archived two-row, three-GPU base control.

On the cluster the directory is named `arc-python-exploration-pilot-v3` alongside the verified native packages. Run `python arc-python-exploration-pilot-v3/launch.py` from the workspace. The launcher reserves one block at a time and submits a CPU watcher after each GPU job. Each subsequent block requires successful Slurm accounting and an ended worker record. A failed qualification stops the chain. An ended block may contain unfinished tasks at its wall-clock deadline. Read the submission records before acting on an interrupted or ambiguous submission.

To score the committed predictions, load the cluster Miniforge module and run `.venv-arc/bin/python arc-python-exploration-pilot-v3/report.py` in the workspace. The report uses all eight assignments per arm and marks unfinished tasks separately. To obtain actual GPU-hours, sum the four GPU jobs’ Slurm ElapsedRaw values, multiply by four, and divide by 3600. The CPU watchers consume zero GPU-hours.

Repair qualification: the saved response from job 8526529 was replayed with the original tokenizer and isolated executor on two distinct grids. A Python fence containing a top-level print and no solver definition can be an exploration call. The live model still has to write a solver from a fresh observation before panel generation in each block. The four replacement slots are 28 minutes each to account for the original 6 min 10 s on four GPUs.

The current v3 run uses four 26-minute slots. Its panel preflight exercises one CPU-simulated model step and checkpoint restore on the real staged inputs for every block. GPU batch-start records are persisted before dispatch; a saved token prefix confirms generation.
