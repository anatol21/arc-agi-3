"""Runtime placement check used by run_benchmark.py (excerpt from training source)."""
def check_layout(model) -> dict:
    trainable_devices = set()
    cpu_tables = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            raise RuntimeError(f"Unmaterialized model parameter: {name}")
        is_table = name.endswith("ple.ple_embedding.ngram_embedding.weight")
        if is_table:
            if parameter.device.type != "cpu" or parameter.requires_grad:
                raise RuntimeError("The frozen PLE table must remain on CPU")
            cpu_tables.append(name)
        elif parameter.device.type != "cuda":
            raise RuntimeError(f"Unexpected parameter outside the GPUs: {name}")
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            trainable_devices.add(parameter.device.index)
            trainable_count += parameter.numel()
    if not cpu_tables or trainable_devices != {0, 1, 2}:
        raise RuntimeError("Expected a CPU PLE table and trainable adapters on all three GPUs")
    layers = {0: 0, 1: 0, 2: 0}
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextDecoderLayer":
            device = next(module.parameters()).device
            if device.type != "cuda" or device.index not in layers:
                raise RuntimeError("A text decoder layer is not on an allocated GPU")
            layers[device.index] += 1
    if list(layers.values()) != [16, 16, 16]:
        raise RuntimeError(f"Expected 16 decoder layers per GPU, found {layers}")
    return {"decoder_layers_per_gpu": layers, "cpu_ple_tables": cpu_tables,
            "trainable_parameters": trainable_count, "trainable_devices": sorted(trainable_devices)}
