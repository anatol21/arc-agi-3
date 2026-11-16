# Fine-tuning Qwen3.8-Flash-Next to checkpoint 660

Job `7853809` trained 33,478,656 LoRA parameters using Axolotl on three H100s. It completed 660 optimizer updates over three epochs in 3 h 02 min 40 s. The [completion record](evidence/completion.json) identifies the original accounting and trainer evidence.

| Setting | Value used |
|---|---|
| Training data | 880 ARC tasks, with 128 separate internal validation tasks. Zero external synthetic tasks. The split manifest also identifies all 138 excluded public evaluation tasks. |
| Frozen model | Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, NF4 quantization with BF16 compute |
| Adapter | 744 FP32 tensors, rank 16, alpha 32, dropout 0. The config lists every target projection. |
| Objective | Assistant completion token cross-entropy on entire JSON answers. Prompt labels are masked. Thinking disabled. |
| Optimization | Eight-bit AdamW, learning rate 0.00005, cosine schedule, 3% warmup, microbatch size 1, four microbatches per update |

## Executed code and evidence

1. [Data selection and chat construction](data-preparation/experiments/cluster/prepare_data.py) and the [split manifest](evidence/data-manifest.json) give the selection rule and individual task IDs. [Historical tokenization code](data-preparation/experiments/cluster/prepare_run.py) constructs the answer-only labels. Its early placement recipe predates the final training configuration.
2. The [submitted Slurm script](arc-flash-run-v6/train_arc.sbatch) starts [guarded_train.py](arc-flash-run-v6/guarded_train.py). The directory includes the actual loss, placement, checkpoint and export code. The [executed configuration snapshot](evidence/arc-flash.yml) was recovered from cluster output. The [verification record](evidence/verification.json) documents the comparison with the checkpoint settings.
3. The [NF4 audit](nf4-audit/) includes the recorded expert count from training and a later independent loader replay. The replay found 834 quantized linear weights and 96 quantized fused expert tensors. Ordinary BF16 and FP32 parameters were also present.

The historical v6 ZIP has SHA-256 `71ac79065ce8b4cd93c41110dc646d1a7397fde0854009e6cd51a3b228b21a1c`. The [file provenance](source-provenance.json) records original and published hashes. The public copy replaces private paths and scheduler identifiers and updates the redistribution manifest. The original training config hash is recorded separately. The v6 package README describes the state **before** submission.

GPU execution requires the pinned weights, training environment, prepared dataset cache and earlier staging receipts checked by [stage_full_run.py](arc-flash-run-v6/stage_full_run.py). The published files document the executed pipeline. A clean-room training rerun from this release has not been tested.
