# Write and run a resumable worker

The scheduler starts a fresh process for each allocation. The worker decides what counts as a finished item, what to save mid-item, and how to reconstruct its next input. The [ARC worker](scheduling/arc_worker/) implements that contract for autoregressive inference. The [minimal worker](scheduling/slurm/examples/example_worker.py) is a shorter starting point for workloads that finish one item at a time.

## Implement a worker

1. **Define the work unit.** Give each independent output a stable, file-safe ID. The ARC example runs one model and decoding policy per campaign and uses a task ID as the item. Keep reference answers out of inference inputs. Identify the exact model, tokenizer, prompt template, data and decoding policy in the campaign. Bump `workload_revision` when changing behavior, including changes in code outside the config.
2. **Accept the scheduler environment.** Read `CAMPAIGN_CONFIG`, `CAMPAIGN_DIR` and `CAMPAIGN_ID`. `worker_command` runs with the config’s parent directory as its working directory. Acquire progress through `read_progress` and publish it with `save_item` from [checkpoint.py](scheduling/slurm/checkpoint.py). The scheduler wrapper already holds the campaign’s single-writer lock. Keep the config, code, interpreter and campaign directory on storage accessible to future allocations.
3. **Restore before loading the expensive resource.** Validate the expected work IDs and checkpoint identity. Skip terminal items. If every item is finished, exit. For a partial ARC row, pass the original prompt IDs followed by the *saved generated token IDs* to a fresh model. Deduct the prefix length from that item’s original generation allowance. Reuse the loaded model for the remaining groups.
4. **Commit progress in order.** Each write may extend the durable prefix. Never overwrite a finished row or alter an earlier prefix. Save a row as soon as it finishes, even when its batch peers are still decoding. At a checkpoint or graceful stop, save new tokens for unfinished rows. The ARC implementation writes its token journal first and then updates the scheduler-facing item. On restart it repairs a missing or lagging item record from the journal. A hard kill can repeat work since the last durable write.
5. **Define completion and failure separately.** For ARC, EOS and exhaustion of the item’s full token allowance are terminal. A malformed or incorrect answer still counts as a finished inference attempt. Score outputs separately. Detect the job deadline, reserve time to flush, and propagate unexpected errors so the controller can stop for inspection. When retrying a device OOM, restore the last compatible batch limit and retain the committed tokens.

## Reproduce the restart locally

From the repository root, run the ARC test with PyTorch and Transformers installed. The test uses a tiny local CPU model and two synthetic prompts.

```sh
python -m unittest discover -s scheduling/arc_worker -v
python -m unittest discover -s scheduling/slurm -v
python -m unittest discover -s scheduling/inference -v
```

For a new worker, test a batch with one terminal and one partial item. Capture the terminal file and partial prefix, interrupt, and invoke the worker again with the same campaign. Assert that the next model input contains only the unfinished item with the exact saved prefix. Check that the terminal file is byte-identical and that a third invocation on the completed campaign performs no inference.

## Run a campaign

1. Start from the [ARC config](scheduling/arc_worker/campaign.example.json) or [generic config](scheduling/slurm/examples/campaign.json). Keep the private copy and its `.campaigns/` directory on persistent shared storage. Set the real item IDs, worker command, data identity, partition names, GPU count, aggregate `gpu_hours` and `max_minutes`. Use absolute paths for the worker and interpreter if the config is beside neither. Set the model cache path consistently across login and compute environments.
2. Exercise the real worker locally on a small manifest where your hardware supports it. A local run of the wrapper performs inference immediately. Use separate configs for the smoke test and measured campaign.
3. From the repository root, substitute your own config path and run:

```sh
config=/shared/project/campaign.private.json
python scheduling/slurm/scheduler.py probe "$config"
python scheduling/slurm/scheduler.py launch "$config"
python scheduling/slurm/scheduler.py status "$config"
```

`probe` uses Slurm’s test-only estimates. `launch` submits the worker and an `afterany` CPU job. The CPU job settles actual allocated GPU time and arranges the next eligible allocation when work and budget remain. Use `status` to inspect the campaign and `squeue` for live queue state. Keep using the same unchanged config to resume. If the controller reports `needs_attention`, inspect the ledger and job log before another launch. A `STOP` file in the campaign directory prevents further launches and the ARC worker checks it during generation.

Keep private configs, datasets, response journals, model files, SSH details, account names and cluster logs out of commits. The [ARC example guide](scheduling/arc_worker/README.md) documents the two loaders and the persisted file formats. The [Slurm guide](scheduling/slurm/README.md) covers settlement, submission recovery and site configuration.
