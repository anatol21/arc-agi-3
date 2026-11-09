# Cluster Lab dashboard

Cluster Lab runs on your laptop. It uses your existing SSH connection to submit Slurm jobs, inspect queue estimates and resume experiments from worker checkpoints. The browser interface lets you review a proposed allocation before submitting, follow metrics and logs, and stop or resume a run. A CPU simulator exercises the same worker and handoff code locally.

## Run the local example

Use Python 3.10 or later on macOS or Linux. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python server.py --demo
```

Open [the dashboard](http://127.0.0.1:8766), select **Local simulator**, and create an experiment using the simulator template. Keep its default 24 items. Review and submit it, then open the run to inspect completed items, checkpoints, worker logs and successive simulated allocations. The simulator works entirely on the laptop. Its queue times and displayed GPU reservations are synthetic.

## Configure a cluster

1. Establish an SSH session from your terminal using your site's normal login procedure. The dashboard uses noninteractive system SSH and supports an existing SSH control connection. Keep credentials and SSH configuration outside the project.
2. In **Cluster settings**, create a profile from [slurm.example.yaml](profiles/slurm.example.yaml). Supply your SSH alias, absolute writable workspace, Python executable, account and production partitions. Set the partition time limits and concurrency limits for your site. The example uses unconfigured placeholders. The dashboard queries Slurm; actual availability and start estimates can change.
3. Create an experiment and choose its command, arguments, environment, task count and GPU request. A draft records the recipe. **Submit** starts a real allocation. The remote cluster needs Python 3.10+ and Slurm. The runtime uses the Python standard library; your workload can use a separate environment.
4. Use the logs, worker receipts and metrics to check progress. **Stop** records a durable stop request before cancelling the tracked allocation. A run can continue after the browser or SSH session closes.

Profile files and the local index live in the ignored `.local/` directory. A remote run has its own directory under the profile workspace's `cluster-ui/`, with a manifest, frozen runtime, worker outputs, checkpoints and Slurm logs. Manifests can contain a user's profile, command-line arguments and paths. Keep these directories private when publishing code.

## How scheduling works

For each allocation, the controller checks the profiles' configured partitions against Slurm and requests start estimates using `sbatch --test-only`. It first considers normal partitions meeting the minimum time limit. A long known wait can make short partitions eligible. Development partitions require an explicit permitted run kind and permission recorded in the profile. Leave those partitions out of a production profile where the site restricts development queues.

Among eligible and affordable partitions, the dashboard ranks known estimated finish times. Its estimate includes queue wait, useful work, startup and the number of expected allocations. If runtime measurements are missing, it prefers the shortest known queue wait. Reordering partition entries does not affect a tie. A start estimate can be unknown. It is never a reservation.

The dashboard reserves the **requested GPU count × requested hours × submitted array tasks** against its optional GPU-hour cap. A short job that exits early can use less than this reservation. Use Slurm accounting for actual elapsed time and allocated GPUs. One worker can use a model spread across several GPUs. Additional independent workers use Slurm array tasks.

Ordinary commands run once. For an adapter configured with `resume_capable: true` and `auto_continue: true`, the controller validates the previous worker receipts, selects unfinished logical shards, and starts a successor allocation after its predecessor ends. A stable worker index preserves the assignment of items to shards. A process lock and a persisted submission intent prevent two workers from submitting the same successor. If Slurm's reply is lost, the runtime first tries to identify the submitted job by its unique name. A hard kill can prevent a worker from publishing its receipt; inspect the run and use Resume after accounting confirms that the previous job has ended.

## Integrate a workload

The runnable [demo adapter](demo_adapter.py) shows how to restore progress, partition work, measure phases and publish a result. [training_example.py](training_example.py) is a finite CPU training example. For a new adapter:

- Read the run manifest at `WM_CONFIG` and write outputs beneath `WM_RUN_DIR`. The immutable run identity is in `WM_IDENTITY`. Worker indices and counts are in `WM_WORKER_INDEX` and `WM_WORKER_COUNT`.
- Use `Adapter.assigned(index)` for stable sharding. Save item IDs and committed progress with `Adapter.checkpoint`. For token generation, retain the committed tokens and the original per-item budgets. On restart, skip already-finished items.
- Between units of work, check `adapter.stopping()`. Publish `partial_checkpointed` with the useful-work counter and completed count, or `complete` once the shard finishes. Adapters can seal additional checkpoint files by relative path. The controller verifies checkpoint identity and contents before continuing.
- Load the model once per allocation. `sdk.prepare_cache` can publish verified reusable model artifacts. `live_generation.generate_chunks` retains the generation cache while the process saves token deltas. A new Slurm allocation reloads the worker and reconstructs its state from persistent files.
- Emit `adapter.metric(...)` to record task progress, loss or tokens per second. `adapter.phase(...)` measures preparation, loading and restore time. The dashboard can report time to the first new saved result and items completed per allocated GPU-hour.

The included ARC template can be configured to launch a compatible adapter. Its commands and data paths are placeholders. The web service binds to loopback, uses a per-session token for changes and executes adapter commands with the cluster account's permissions.

## Test

```sh
.venv/bin/python -m unittest discover -s . -p 'test*.py'
node --check static/app.js
node --check static/workflow.js
```

The Python suite exercises queue choice, detached simulation, checkpoint integrity, submission ambiguity and continuation. It does not require a cluster, GPU or dataset.
