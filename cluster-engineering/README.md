# Cluster engineering

A long evaluation can use a short GPU allocation if its worker saves progress before the time limit. The next worker reloads the model, skips completed responses, and continues from saved prefixes. Queue time and GPU time are measured separately.

| Component | Use |
| --- | --- |
| [Cluster Lab](dashboard/) | Run the dashboard locally, try its simulator, or connect a private cluster profile over SSH. |
| [Slurm scheduler](slurm/) | Choose a production partition, submit a GPU worker and a dependent CPU job, and resume unfinished work within a GPU-hour budget. |
| [ARC inference worker](arc_worker/) | Run the direct-answer worker, interrupt a batch, and inspect which stored tokens it uses on restart. |
| [Inference batching](inference/) | Group comparable inputs, test wider batches, and retain a smaller working width after an out-of-memory failure. |
| [Worker runbook](agent.md) | Implement a resumable worker, test it offline and launch it on Slurm. |
| [ARC validation profiles](arc-validation/) | Inspect prompt lengths, measured response lengths and one worked example from the 138-task evaluation. |

## Start locally

From the repository root, follow the [dashboard setup](dashboard/README.md) to start Cluster Lab in simulator mode, or run the [ARC worker restart test](arc_worker/README.md#run-the-offline-restart-test) using a CPU model. Both work without a cluster allocation.

For a campaign already in progress, keep its checkout and config in place until it finishes. Changing the worker path in the config changes the campaign digest and selects a new journal. Use the new paths for new campaigns.

## How the production runs worked

1. **Define a unit of work.** In the ARC run, one unit was a task ID and a decoding strategy. Each strategy had a separate output directory. A response was complete when it reached EOS or used its token allowance. Exact-grid scoring took place afterward.
2. **Select an allocation.** The controller checked the allowed partitions and asked Slurm for a start estimate using the actual GPU count and candidate wall time. Short allocations can start earlier but pay the model-loading cost more often. Slurm estimates can move after submission.
3. **Keep the model resident.** Four GPUs holding one sharded model acted as one worker. The worker processed multiple batches and alternated between sampling strategies. It saved finished rows immediately and checkpointed unfinished rows. Live generation state lasted for the current call; across allocations the worker reconstructed inputs from saved tokens.
4. **Settle and continue.** A CPU job depended on the GPU job with `afterany`. It inspected the final Slurm state, durable progress, and allocated GPU time before deciding whether to submit another wave. A persisted submission intent made it possible to recover a missing submission receipt without submitting a duplicate job. Deterministic worker failures required inspection.

## What to measure

Use **time from allocation start to the first newly saved response** and **newly completed responses per allocated GPU-hour**. For example, a four-GPU allocation that finishes 12 responses in 791 seconds consumes 0.879 GPU-hours and produces 13.65 responses per GPU-hour. Use the actual elapsed seconds from Slurm accounting, including startup and shutdown.

In a separate three-GPU comparison on the same fixed inputs, staging the weights with 32 CPU workers took 41 seconds. That run reached its first saved response at 12 min 23 s. Direct loading from shared storage reached the first save at 13 min 45 s. Both runs finished 13 responses. These timings include startup; storage contention, model layout and input length can change the result on another system.

The ARC BF16 pilot tested batches of four and eight on short prompts. A later policy probed 12 and 16 separately for comparable input lengths. A successful width on short prompts does not establish the capacity for long inputs. For a workload with different hardware, weights, precision, or context lengths, measure its own safe widths and reload cost.

## Running on another cluster

Start with the local simulator or the Slurm example before connecting a real workload. Supply your own partition names, account, environment, shared workspace and GPU-hour cap. Check your site's policy on development queues and job dependencies. Keep private profiles, SSH sockets, model files, run journals and Slurm logs outside version control.

The dashboard and standalone Slurm controller have separate run formats. Choose one submission path per campaign so that one controller owns the budget and the run journal.
