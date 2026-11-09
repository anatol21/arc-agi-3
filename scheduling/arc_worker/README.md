# Resumable ARC inference worker

This example publishes the direct-answer decoding and restart path from the ARC BF16 base campaign. `worker.py` connects it to the repository’s Slurm scheduler. The portable configuration uses a small dense model and two synthetic ARC tasks. The original four-GPU model layout and its loader checks are included separately.

## Run the offline restart test

Use a Unix-like system with Python 3.10+ and compatible PyTorch and Transformers packages. In an existing inference environment, run from the repository root:

```sh
python -m unittest discover -s scheduling/arc_worker -v
```

The restart tests use a locally initialized tiny GPT-2 and controlled logits so that the two responses finish at different times. One test terminates the Python process after the first saved answer and resumes the partial row in a second interpreter. Another repairs a scheduler record from the token journal. The next forward pass includes only the unfinished task. A subsequent invocation after both answers finish can skip model loading. This tests restart behavior independently of ARC accuracy. A separate smoke test runs the scheduler wrapper twice against a locally initialized Qwen3 checkpoint. All tests run offline on CPU.

## Run the example with real weights

The default [campaign](campaign.example.json) uses an unquantized BF16 causal language model loaded by Transformers. Install PyTorch suitable for your device, Transformers with Qwen3 support, and Accelerate for `device_map`. Provision the pinned model and tokenizer on shared storage. With the Hugging Face CLI installed, the following fetches the public weights into its cache:

```sh
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
```

The worker loads local files only. From the repo root, these commands run on the current machine. Use a machine with enough memory and BF16 support. The second invocation reuses the first invocation’s durable state.

```sh
config=scheduling/arc_worker/campaign.example.json
python scheduling/slurm/worker.py "$config"
python scheduling/slurm/worker.py "$config"
python scheduling/slurm/scheduler.py status "$config"
```

The example uses `python3` from `PATH` to start `worker.py`, so activate the inference environment before running the wrapper. For a private campaign, copy the config beside its task file or set `tasks_file` to its full path. Set `worker_command` to the absolute interpreter and the absolute `scheduling/arc_worker/worker.py` path in your checkout. Substitute site partitions and an appropriate budget before submitting to Slurm. Follow the [worker runbook](../../agent.md) for the scheduler commands.

## What is saved

| State | Location inside `.campaigns/<config digest>/` | On restart |
|---|---|---|
| Generated tokens and stop reason | `inference/states/*.json` | Restore the exact prefix and remaining allowance. Skip rows marked `eos` or `token_limit`. |
| Controller progress | `items/<task ID>.json` | Contains a partial or finished flag, generated IDs, and decoded `payload.text` for a terminal row. A lost or lagging progress file is rebuilt from the token journal. |
| Batch history | `inference/attempts/*.json` | Restore working widths and OOM limits for comparable lengths and matching hardware. |

The direct-answer prompt includes demonstration pairs and query inputs. The chat template receives `enable_thinking=False`. The example uses greedy decoding, one output, an 18,432-token context and at most 8,192 new tokens per task. Prefixes count against that original allowance. A result is written at EOS or the limit, independently of correctness. Reference outputs and scoring are outside this worker. See [decode.py](decode.py) for the exact message and decoding settings.

Within a live call the streamer saves finished rows early and writes other progress at the configured interval. Across allocations the worker reloads the model and reconstructs inputs from token IDs. Sampled decoding can take a different trajectory after interruption or regrouping. Keep one decoding policy per config when comparing strategies. The [batching helpers](../inference/) group by current input width and reduce it after an OOM. Batch size can change the numerical output even with greedy decoding.

## Four-GPU BF16 loader

The optional `arc.backend` value `qwen_4gpu_bf16` selects the original Qwen-specific [Axolotl loader](qwen_bf16.py). It requires the matching local checkpoint, tokenizer, compatible Axolotl runtime and four visible GPUs. Set `gpus` to 4, use the model revision in [qwen_layout.json](qwen_layout.json), and supply `arc.axolotl_config` and its `arc.axolotl_config_sha256` privately. The loader checks parameter names, shapes, placement and runtime dtypes, keeps the large PLE embedding in host RAM, and rejects adapters and packed weights. Use a cached Hub model ID at the pinned revision, a `snapshots/<revision>` directory, or a flat local mirror with its `.cache/huggingface/download` metadata. The mirror checks the recorded revision for the config, weight index and every referenced shard. The layout fingerprint checks tensor geometry. Verify the source weights when provisioning them. Supply model weights, evaluation data and site configuration privately.
