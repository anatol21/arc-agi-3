# Tensor-level NF4 audit of ARC training job 7853809

## Historical evidence

Job 7853809 logged 96 MoE expert parameters converted to 4-bit. Its training and debug logs contain an aggregate count. Checkpoint-660 holds the FP32 LoRA adapter. The audit found no packed base-model snapshot or per-tensor quantization-state manifest from that job. [Log and checkpoint evidence](historical-record.json)

## Independent reproduction

Job 8484708 reloaded the original training config and pinned base revision on three H100s, with a separate output directory. It initialized adapters through the training loader and performed zero optimizer steps. For each quantized logical base tensor, the inspector recorded its packed storage size and dtype, GPU, quantization type, block size, logical shape, scale count and nested scale state.

| Replay observation | Logical base tensors | Examples |
|---|---:|---|
| Linear weights with live NF4 state | 834 | Linear-attention projections, self-attention projections, shared-expert projections, vision linear weights |
| Fused routed-expert weights with live NF4 state | 96 | One gate_up_proj and one down_proj in each of 48 decoder layers |
| Ordinary BF16 parameters | 519 | Token embeddings, output head, CPU PLE embedding table |
| Ordinary FP32 parameters | 48 | Routed-expert router weights |

All 930 NF4 names and shapes agreed with the CPU reconstruction. Each had nonempty scales, nested quantization state, and exactly four packed bits per logical element. The packed containers for the linear weights had BF16 dtype; the 96 expert containers had uint8 dtype. Runtime storage dtype alone does not identify the quantization scheme. The 744 trainable FP32 adapter tensors are counted separately from the 1,497 logical base parameters. The replay shows the materialized layout for this configuration; historical byte-for-byte identity cannot be tested from the retained adapter checkpoint.

[Full tensor list](tensor-inventory.csv) · [Live GPU inventory](gpu-inventory.json) · [Replay checks](gpu-replay-report.json) · [Per-family counts](summary.json)

Slurm recorded 492 seconds on three H100s, or 0.410 GPU-hours. [Allocation accounting](status.json)
