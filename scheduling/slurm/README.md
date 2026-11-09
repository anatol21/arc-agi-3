# Budgeted Slurm continuation

This is the portable version of the scheduler used for the cluster benchmarks. A campaign consists of named work items, a worker command, a list of eligible production partitions, and an aggregate GPU-hour budget. The worker process controls model loading, inference, batching, and checkpoint frequency. Slurm controls allocation and start time.

## Try it without a cluster

From the repository root, run these commands with Python 3.10 or later on a Unix-like system:

```sh
cd scheduling/slurm
python3 -m unittest discover -s . -p test_scheduler.py -v
python3 worker.py examples/campaign.json
python3 worker.py examples/campaign.json
python3 scheduler.py status examples/campaign.json
```

The synthetic worker commits one item per invocation. Two invocations on a fresh example directory finish two distinct items. The tests use an in-memory Slurm command simulator to exercise queue choice, interruption, duplicate submission recovery, and budget enforcement. Neither the tests nor these example worker invocations request cluster resources.

## Configure a workload

Copy `examples/campaign.json` to a private project directory on storage shared by the login and compute nodes. Place this package on shared storage too. The configuration, worker entry point, Python executable, and scheduler scripts must remain accessible to GPU jobs and CPU watchers. Worker commands run in the configuration file's directory. For example, copy `examples/example_worker.py` beside the new configuration before using the sample command.

| Field | Meaning |
|---|---|
| `items` | Unique work IDs. Include the strategy in an ID when a task has more than one independent condition. |
| `workload_revision` | Identify the exact code, model, dataset, and decoding policy. Use a version or digest and change it when these inputs change. The scheduler hashes the configuration; it cannot detect changes to files referenced by an unchanged configuration. |
| `worker_command` | Executable and arguments as a JSON array. Keep environment setup in a private wrapper where needed. |
| `partitions`, `cpu_partition` | Site-approved production GPU partitions and an eligible CPU partition for settlement. Names in the example are placeholders. |
| `gpus`, `gpu_hours` | GPUs per single-node worker, total budget for **all** allocations in this campaign. |
| `max_minutes`, `reload_seconds` | Per-allocation ceiling and the measured startup time used when comparing partitions. |
| `account`, `qos`, `cpu_threads`, `memory` | Optional site-specific Slurm flags. `account` and `qos` apply to both GPU and CPU jobs; choose values accepted by both or adjust the watcher for your site. `memory` uses Slurm units, for example `32G`. |

All output goes under `.campaigns/<configuration SHA-256>/` beside the private configuration. This directory contains actual job IDs, job names and workload results. Keep it on durable shared storage throughout a campaign. The example's ledger directory is ignored by Git.

Use a private wrapper if model loading requires environment modules or a virtual environment. Set `SCHEDULER_PYTHON` to a Python executable accessible on both kinds of node if `python3` is unavailable in the Slurm environment. Integrate a worker using the two checkpoint functions in `checkpoint.py`:

```python
from checkpoint import read_progress, save_item
```

The worker receives `CAMPAIGN_DIR`, `CAMPAIGN_ID` and `CAMPAIGN_CONFIG`. It reads its own finished items, resumes partial payloads and writes progress with `save_item`. A work unit can be a generated token, record or training step. Counts must be monotonic per item. The payload is application-defined and must be JSON serializable. For inference, persist token IDs so a fresh process can reconstruct a prefix. Mark a validly completed attempt `finished` regardless of its evaluation score. Workers can read `SLURM_JOB_END_TIME` and leave time to flush before the allocation expires. The sample worker demonstrates the contract.

## Submit and monitor

```sh
python3 scheduler.py probe /shared/project/campaign.json
python3 scheduler.py launch /shared/project/campaign.json
python3 scheduler.py status /shared/project/campaign.json
```

`probe` calls `scontrol` and `sbatch --test-only` for the partitions in the configuration. An ETA is a scheduler estimate that may change. `launch` creates the GPU job and a CPU job with an `afterany` dependency on that GPU job. The CPU job settles final usage and submits a successor if eligible work and budget remain. `status` reads the local campaign files. Slurm's `squeue` remains authoritative for pending and running jobs. Configure scheduler accounting and allow compute nodes to invoke the Slurm CLI before using automatic continuation.

### Queue choice

Each eligible partition is probed with the real worker script and resource request. The requested time is the minimum of the live partition maximum, the configured allocation limit, and the unspent budget in whole minutes. Among known start estimates, the scheduler compares the start time plus the cost of repeated loads. It approximates the work horizon from the remaining budget and does not predict queue wait for future waves. An unknown ETA ranks after known estimates. When all estimates are unknown, ties favor longer allocations, then partition name. Reordering the partition list preserves the choice.

### Settlement and recovery

For each completed allocation, the ledger records `ElapsedRaw × allocated GPUs` from `sacct`, then subtracts that charge from the shared cap before calculating the next time limit. A worker that finishes early releases its allocation. Preemption, timeout or node failure can resume from committed work. A clean exit must advance progress if work remains. A failed application, user cancellation or repeated infrastructure interruptions without progress puts the campaign in `needs_attention`.

A file lock serializes scheduler transitions. The ledger records a unique submission intent **before** calling `sbatch`. If the reply is lost, `launch` reconciles the unique job name against both `squeue` and `sacct`. Zero or conflicting matches require investigation, and protect against duplicate submissions. If accounting is delayed after a job ends, run `launch` again once `sacct` has settled. To stop future allocations, create `STOP` in the campaign's ledger directory. Use site tools to cancel an already queued or running job if needed.

The portable scheduler supports ordinary single-node jobs with the GPU count exposed as `gres/gpu` in allocation accounting. It relies on advisory locks and atomic file replacement on the shared filesystem. Site-specific limits and queue policy apply; use development queues only for permitted development work.
