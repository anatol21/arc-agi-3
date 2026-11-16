# ARC Flash full run v6

This is the next full ARC-only fine-tuning iteration: one process, three H100s,
six hours, unchanged model revision, data split, optimizer and hyperparameters.
Healthy validation leads directly to normal training, a real first update and
validated checkpoint, then the rest of the fit. There is no separate diagnostic
allocation. Results go to `runs/arc-flash-v6`; earlier runs remain untouched.

Development job **7849447 completed successfully** and isolated the current loss
failure: operands were on GPU 2 while the current CUDA device was GPU 0. Unscoped
calls on the same tensors produced incorrect finite loss or negative infinity;
all four explicitly GPU-2-scoped calls agreed with FP32 references. The isolated
hidden gradient was finite and nonzero, with relative L2 error 0.00348 and cosine
similarity 0.999994 against the independent reference.

v6 scopes the CCE forward call and backward-kernel entry to the embedding's owning
GPU, restoring the previous device afterward. It keeps the corrected v5 target
normalization and all numerical/checkpoint guards. This training launcher has not
run on GPUs yet. The completed diagnostic validates the scoped loss operation
and isolated gradient; full-backbone backward, four-microbatch optimization and
checkpoint creation remain to be established by the guarded full run.

## Transfer from your Mac

Run outside SSH:

```bash
scp "/path/to/research-workspace/artifacts/cluster/arc-flash-run-v6.zip" "/path/to/research-workspace/artifacts/cluster/arc-flash-run-v6.zip.sha256" user@cluster.example.org:/path/to/arc-workspace/
```

## Prepare and submit on the cluster

Use the existing cluster workspace and environment. Do not run the old setup,
reinstall packages, retokenize data or redownload the model.

```bash
cd /path/to/arc-workspace &&
module load devel/miniforge/25.3.1-python-3.12 &&
source .venv-arc/bin/activate &&
sha256sum -c arc-flash-run-v6.zip.sha256 &&
python -m zipfile -e arc-flash-run-v6.zip . &&
python arc-flash-run-v6/stage_full_run.py &&
sbatch arc-flash-run-v6/train_arc.sbatch
```

Run this once, into a new v6 directory; never overwrite a staged/running version.
Every step must succeed before the next runs. A failed CPU check prevents
submission. Keep the returned job number; do not repeat the submission block.

Staging takes the output lock before writing anything, verifies the executed v5
source/readiness record, complete source answer masks and actual Arrow rows, and
reuses the corrected CPU PLE placement. It pins installed versions and CCE's
source commit/code, including both training-step wrappers. v4 staging already
checked all 144 model files against their existing local Hub download digests.
v6 reuses that proof after checking the full inventory, all file sizes/timestamps
and small-file hashes. It does not read the weight shards again. Missing or
changed proof causes an explicit stop. No GPU reservation or network is needed.
Re-running successful staging verifies the existing records without rewriting.
GPU startup checks file sizes/timestamps plus small metadata hashes; it does not
rehash hundreds of GB on allocated GPUs. These checks detect accidental changes;
local Hub metadata is not a separate authenticated manifest from the server.

The new CPU check uses the installed Axolotl/Transformers training-step, count
and loss methods (including the two offloading wrappers) and Qwen CCE Python
forward. Four tiny batches have 1, 2, 3 and 4 supervised targets. From either
inferred flag value, accumulated loss, gradients, an SGD update and per-batch
evaluation must match direct cross entropy. The test uses a tiny CPU model,
a CPU backward adapter and a CPU cross-entropy primitive instead of the CUDA
kernel. It does not load the experiment model or test GPU kernels, distributed
setup, the real backbone or eight-bit optimization. Its result and source hashes
are saved in `compatibility-preflight.json`; startup checks its sealed digest.
Staging also compares the actual completed cluster report
`runs/arc-flash-loss-debug-v1/debug-report.json` to the reviewed report included in
this archive. It verifies all eight additional kernel/runtime source hashes
against that GPU experiment and binds the evidence into readiness. No new GPU
allocation is needed for staging. Success prints `GPU DEVICE EVIDENCE VERIFIED`,
`INSTALLED CPU LOSS CONTRACT PASSED`, then `FULL RUN STAGED`.

`sbatch` snapshots its submission script, but the Python/config files
are read at execution time and must stay unchanged. READY verifies their hashes.

## Integrated guardrails

- Enforce unfiltered raw loss on the actual Trainer arguments. Check each real
  collated batch against its full verified tokenized source, including prompt
  masking, answer labels, padding, task identity and shifted target count.
- Count the actual four-microbatch token window. Move its scalar denominator to
  the CCE loss device inside the loss boundary, preserving autograd and normal
  accumulation. Keep the scalar returned to Trainer on its expected device.
  Set the loss-handling and causal-shift flags explicitly for the verified CCE
  path; reject a second Accelerator accumulation division. Record actual settings
  and the original inferred flags in `effective-guardrails.json`.
- Run the CCE forward and backward kernel under the embedding owner's CUDA
  device context. Record entry/active devices; restore the caller's context on
  return or exception. Preserve all existing loss operands and autograd.
- Stop on the first invalid raw batch loss before backward, rather than waiting
  for all 128 validation batches or a post-update log. Record task/hash, length,
  target count, dtype/device, actual value, and startup activation provenance.
- Compare CCE to detached FP32 projection/cross entropy on bounded token chunks,
  with identical causal shift and denominator, tolerance 0.03 per target + 1%
  relative. Observe the first batch of each validation, first train batches at
  steps 0/1 and every 100 steps, and all nonfinite CCE losses. This observer adds
  about 2.37 GiB for the FP32 classifier copy; it never replaces training loss.
- Check every adapter gradient before and after clipping on every update, and
  every adapter weight plus floating optimizer state after each update. Missing
  gradients fail; legitimate zero gradients are allowed. Packed NF4 weights and
  eight-bit integer moments are not treated as ordinary floating values.
- Assert expected placement, 33,478,656 trainable parameters and adapter-family
  counts; validate all three PLE hashing buffers against deterministic values.
  Record installed kernel callables rather than guessing from warnings.
- Validate the first real checkpoint and every later save: decode finite adapter
  tensors, compare all saved adapters/optimizer/scheduler/CUDA RNG to live state,
  bind that validation to file hashes and the actual step. Keep three accepted
  checkpoints; prune only after accepting the replacement. A partial/corrupt new
  checkpoint falls back to an older accepted save. All GPU RNG streams restore
  at the real framework resume boundary after data skipping.
- Distinguish loading, validation, training, saving, export and failure. A final
  export failure remains recoverable: rerunning the launcher with a validated
  step-660 checkpoint repairs export without an additional training update.
- Request stop/checkpoint at an update boundary on signals or the soft deadline;
  interrupt validation between batches when time expires. Interrupted partial
  accumulation is discarded; only accepted full checkpoints may resume.

Startup module observers remain through the first accepted save, then are removed
to reduce overhead. Raw loss/source/gradient/weight checks stay active. Signals
cannot interrupt an individual long CUDA operation or guarantee a save before
Slurm's hard limit. Cross-node locking and bitwise resumed equivalence are not
demonstrated. No NaN replacement, skipped data, silent kernel fallback or frozen
weight reinitialization is introduced.

## Follow progress

Replace JOBID with the number returned by submission:

```bash
squeue -j JOBID
tail -n 60 logs/arc-flash-v6-JOBID.log
cat runs/arc-flash-v6/run-status.json
```

On failure, keep the full log, `run-status.json`, `numerical-failure.json` if
present, `loss-observations.jsonl`, effective guardrails and kernel records.
The first accepted checkpoint is reported as `CHECKPOINT_VALIDATED`.

## Evidence and limits

v4 stopped in our incorrect flag assertion before a batch. v5 corrected that
contract and reached GPU evaluation, which caught an invalid optimized loss with
a finite reference at zero updates. Development job 7849447 reproduced the
current-device mismatch and verified explicit owner scoping on two tasks, twice
each, plus an isolated hidden gradient. The production forward/backward wrappers
are locally tested for correct device selection, operand preservation, exception
cleanup and CPU autograd compatibility. The exact v6 launcher has not executed
on H100s yet. All 128 development batches, the real longest-task backward,
eight-bit optimizer/update memory fit and resumed continuation still need
cluster evidence. Six hours is a cap, not a completion estimate.

This iteration implements fine-tuning only. ARC generation accuracy, a matched
untuned baseline and per-task TTT are still separate unfinished work; loss is
not ARC solve rate. The data is the fixed ARC-1/ARC-2 subset, not interactive
ARC-AGI-3. Protected evaluation answers remain outside model/training inputs.
