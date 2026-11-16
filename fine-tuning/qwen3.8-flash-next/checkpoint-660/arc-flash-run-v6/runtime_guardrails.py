"""Guard the real Trainer path, including ordinary accumulation and optimizer updates."""
from __future__ import annotations

import functools
import json
import math
import os
import signal
import time
from pathlib import Path

if __package__:
    from .numeric_checks import TokenIndex, classify_failure, finite_value
    from .run_checks import atomic_json, checkpoint_files, seal_checkpoint
    from .checkpoint_validation import validate_adapters, validate_checkpoint
    from .export_checkpoint import finish_export
    from .loss_normalization import configure_token_normalization
    from .cuda_loss_context import tensor_device_scope, owner_scoped_backward
else:
    from numeric_checks import TokenIndex, classify_failure, finite_value
    from run_checks import atomic_json, checkpoint_files, seal_checkpoint
    from checkpoint_validation import validate_adapters, validate_checkpoint
    from export_checkpoint import finish_export
    from loss_normalization import configure_token_normalization
    from cuda_loss_context import tensor_device_scope, owner_scoped_backward


class TimeBudgetReached(RuntimeError):
    pass


def align_denominator(torch, value, device):
    """Normalize at the inner loss boundary, after Accelerate's input hooks."""
    if value is None:
        raise RuntimeError("The actual token-normalized loss denominator is missing")
    count = value.item() if torch.is_tensor(value) else value
    if isinstance(count, bool) or not math.isfinite(float(count)) or count <= 0 or int(count) != count:
        raise RuntimeError(f"Invalid supervised-token denominator: {count}")
    return value.to(device) if torch.is_tensor(value) else value


def reference_cross_entropy(torch, e, c, labels, denominator, *, chunk_size=16):
    """FP32 projection + CE on supervised positions only; detached, bounded, same shift."""
    with torch.no_grad(), torch.autocast(device_type=e.device.type, enabled=False):
        labels = labels.to(e.device)
        targets = labels[..., 1:].reshape(-1)
        positions = torch.nonzero(targets != -100, as_tuple=False).flatten()
        if not positions.numel():
            raise RuntimeError("No shifted targets at the CCE boundary")
        features = e[..., :-1, :].reshape(-1, e.shape[-1])
        total = 0.0
        # Copy the frozen classifier once per observation, not once per token chunk.
        classifier = c.detach().float()
        for offset in range(0, positions.numel(), chunk_size):
            selected = positions[offset:offset + chunk_size]
            logits = torch.nn.functional.linear(features.index_select(0, selected).detach().float(), classifier)
            value = torch.nn.functional.cross_entropy(logits, targets[selected], reduction="sum")
            total += float(value.item())
            del logits, value
        return {"loss": finite_value(total / float(denominator)),
                "mean_per_target": finite_value(total / positions.numel()),
                "shifted_supervised_tokens": positions.numel(),
                "projection_dtype": "float32", "classifier_copy_GiB": round(classifier.numel() * 4 / 2**30, 3)}


def inspect_parameters(torch, named, *, gradients=False, require_signal=False):
    """Check every adapter, with only one host synchronization per device group."""
    groups = {}
    missing = []
    for name, parameter in named:
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            missing.append(name)
            continue
        groups.setdefault(str(tensor.device), []).append((name, tensor))
    if missing:
        raise RuntimeError(f"Missing adapter gradients ({len(missing)}): {missing[:8]}")
    result = {}
    for device, pairs in groups.items():
        finite = torch.stack([torch.isfinite(value).all() for _, value in pairs])
        if not finite.all().item():
            bad = [name for (name, _), okay in zip(pairs, finite.tolist(), strict=True) if not okay]
            raise RuntimeError(f"Nonfinite adapter {'gradient' if gradients else 'weight'}: {bad[:8]}")
        if gradients and require_signal:
            norms = torch.stack([value.detach().float().norm() for _, value in pairs])
            if not (norms > 0).any().item():
                raise RuntimeError(f"The first accumulated update has entirely zero adapter gradients on {device}")
        result[device] = len(pairs)
    if not result:
        raise RuntimeError("No adapters were checked")
    # Zero gradients are legitimate, including LoRA A at zero-B initialization.
    return result


def inspect_optimizer(torch, optimizer):
    """Eight-bit moments are encoded integers; check their floating scales/metadata."""
    groups = {}
    def visit(value):
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            groups.setdefault(str(value.device), []).append(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("Nonfinite optimizer scalar")
    visit(optimizer.state)
    for device, values in groups.items():
        if not torch.stack(values).all().item():
            raise RuntimeError(f"Nonfinite floating optimizer state on {device}")


class RuntimeGuard:
    def __init__(self, torch, trainer, config, prepared, output, check_layout, place_buffers):
        self.torch, self.trainer = torch, trainer
        self.config, self.prepared, self.output = config, prepared, output
        self.check_layout, self.place_buffers = check_layout, place_buffers
        self.index = TokenIndex.from_prepared(config, prepared)
        self.started = time.time()
        self.stop_at = float(os.environ["ARC_STOP_AT"])
        self.stop_requested = None
        self.context = {}
        self.first_bad_activation = None
        self.reference_seen = set()
        self.handles = []
        self.restore = []
        self.loss_calls = 0
        self.preclip_step = None
        self.last_checkpoint = os.environ.get("ARC_RESUME_CHECKPOINT")
        self.train_finished = False
        self.microbatches = []
        self.expected_window_count = None
        self.expected_classifier_shape = (248320, 2560)
        self.expected_first_length = 18184
        self.named = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]

    def status(self, phase, **extra):
        state = self.trainer.state
        atomic_json(self.output / "run-status.json", {
            "phase": phase, "global_step": state.global_step, "max_steps": state.max_steps,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": round(time.time() - self.started, 1),
            "last_valid_checkpoint": self.last_checkpoint, "context": self.context, **extra})

    def fail(self, message, **extra):
        atomic_json(self.output / "numerical-failure.json", {
            "message": message, "context": self.context, "first_bad_activation": self.first_bad_activation,
            "global_step": self.trainer.state.global_step, **extra})
        raise RuntimeError(message)

    def replace(self, owner, name, method):
        original = getattr(owner, name)
        setattr(owner, name, method)
        self.restore.append(lambda: setattr(owner, name, original))
        return original

    def attach(self):
        from transformers import TrainerCallback
        torch, trainer, guard = self.torch, self.trainer, self
        args = trainer.args
        # This YAML key is ignored by Axolotl 0.20.0; enforce it on the constructed args.
        args.logging_nan_inf_filter = False
        self.context = {"phase": "configuring_loss_contract", "inferred_model_accepts_loss_kwargs":
                        bool(trainer.model_accepts_loss_kwargs),
                        "trainer_microbatches": args.gradient_accumulation_steps,
                        "accelerator_accumulation": trainer.accelerator.gradient_accumulation_steps}
        self.status("configuring")
        normalization = configure_token_normalization(trainer)
        if args.n_gpu != 1 or args.world_size != 1 or args.average_tokens_across_devices:
            raise RuntimeError("Expected one model-parallel process, with no data-parallel loss scaling")
        if args.max_grad_norm <= 0:
            raise RuntimeError("The pre-clipping guard requires the configured gradient clipping path")
        if getattr(trainer, "processing_class", None) is None:
            raise RuntimeError("The real training tokenizer must be present for checkpoint/export validation")
        # Framework rotation runs before on_save. Only prune *after* semantic validation.
        args.save_total_limit = None
        atomic_json(self.output / "effective-guardrails.json", {
            "logging_nan_inf_filter": args.logging_nan_inf_filter, "loss_shifts_labels": True,
            "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "normalization": normalization,
            "cuda_loss_scope": "embedding owner for the CCE forward and backward kernel entry",
            "checkpoint_retention": "three validated checkpoints; prune only after sealing",
            "reference": {"atol_per_target": 0.03, "rtol": 0.01, "projection_dtype": "float32",
                          "observations": "first dev batch per validation; first train batches at steps 0,1 and multiples of 100; all invalid CCE losses"}})

        original_compute = trainer.compute_loss
        def compute(model, inputs, return_outputs=False, num_items_in_batch=None):
            if guard.stop_requested and not model.training:
                raise TimeBudgetReached("Stop requested during validation; preserving the last accepted checkpoint")
            if time.time() >= guard.stop_at:
                raise TimeBudgetReached("Soft deadline reached before the next forward; partial accumulation is not saved")
            split = "train" if model.training else "dev"
            data = {key: inputs[key].detach().cpu().tolist() for key in ("input_ids", "attention_mask", "labels")}
            guard.context = {"split": split, "phase": "checking_batch", "global_step": trainer.state.global_step}
            info = guard.index.check(data, split)
            guard.context.update(info, phase="forward", input_device=str(inputs["input_ids"].device))
            guard.first_bad_activation = None
            count = align_denominator(torch, num_items_in_batch, inputs["labels"].device)
            count_value = int(count.item() if torch.is_tensor(count) else count)
            if split == "train":
                if guard.expected_window_count is None or count_value != guard.expected_window_count:
                    guard.fail("Trainer accumulation-window denominator differs from verified source labels")
                guard.microbatches.append(info)
            elif count_value != info["shifted_supervised_tokens"]:
                guard.fail("Evaluation denominator differs from the actual shifted targets")
            guard.context["num_items_in_batch"] = count_value
            before = guard.loss_calls
            guard.status("training" if model.training else "validating")
            result = original_compute(model, inputs, return_outputs=return_outputs, num_items_in_batch=count)
            if guard.loss_calls != before + 1:
                guard.fail("The guarded CCE path did not run exactly once for this batch")
            loss = result[0] if return_outputs else result
            if loss.numel() != 1:
                guard.fail("Expected one raw scalar loss", shape=list(loss.shape))
            if not torch.isfinite(loss).all().item():
                guard.fail("Nonfinite raw loss before backward", raw_loss=finite_value(loss.item()))
            if model.training and not loss.requires_grad:
                guard.fail("Training loss lost its autograd connection before backward")
            # Preserve autograd if the framework output hook did not align the scalar.
            loss = loss.to(args.device)
            return (loss, result[1]) if return_outputs else loss
        self.replace(trainer, "compute_loss", compute)

        original_count = trainer._get_num_items_in_batch
        def get_count(batches, device):
            value = original_count(batches, device)
            expected = sum(int((batch["labels"][..., 1:] != -100).sum().item()) for batch in batches)
            if value is None or int(value.item() if torch.is_tensor(value) else value) != expected or expected <= 0:
                guard.fail("Constructed token denominator does not count shifted, unmasked labels")
            return value
        self.replace(trainer, "_get_num_items_in_batch", get_count)

        original_samples = trainer.get_batch_samples
        def get_samples(iterator, number, device):
            batches, count = original_samples(iterator, number, device)
            # Model mode may still be eval immediately after startup validation.
            # This method, rather than model.training, identifies the training window.
            if len(batches) != 4:
                guard.fail("The unchanged 880-task recipe must have four complete microbatches per update")
            infos = [guard.index.check({key: batch[key].detach().cpu().tolist()
                       for key in ("input_ids", "attention_mask", "labels")}, "train") for batch in batches]
            guard.expected_window_count = sum(info["shifted_supervised_tokens"] for info in infos)
            if trainer.state.global_step == 0 and infos[0]["nonpadding_tokens"] != guard.expected_first_length:
                guard.fail("The actual first training batch is not the complete longest ARC task")
            if count is None or int(count.item() if torch.is_tensor(count) else count) != guard.expected_window_count:
                guard.fail("Training-window denominator differs from verified answers")
            return batches, count
        self.replace(trainer, "get_batch_samples", get_samples)

        original_clip = trainer._clip_grad_norm
        def clip(model):
            guard.context["phase"] = "before_gradient_clipping"
            guard.gradient_coverage = inspect_parameters(torch, guard.named, gradients=True,
                                                         require_signal=trainer.state.global_step == 0)
            norm = original_clip(model)
            if not math.isfinite(float(norm)):
                guard.fail("Nonfinite pre-clipping gradient norm", grad_norm=finite_value(norm))
            inspect_parameters(torch, guard.named, gradients=True)
            guard.preclip_step = trainer.state.global_step
            return norm
        self.replace(trainer, "_clip_grad_norm", clip)

        original_rng = trainer._load_rng_state
        def load_rng(checkpoint):
            original_rng(checkpoint)
            if checkpoint is not None:
                checkpoint_files(Path(checkpoint))
                states = torch.load(Path(checkpoint) / "arc_cuda_rng_all.pt", map_location="cpu", weights_only=True)
                if len(states) != 3:
                    guard.fail("Resume sidecar lacks the three CUDA RNG streams")
                # Run at the real HF resume boundary, after data skipping and its RNG restore.
                torch.cuda.set_rng_state_all(states)
                print("ALL_GPU_RNG_RESTORED " + str(checkpoint), flush=True)
        self.replace(trainer, "_load_rng_state", load_rng)
        self.install_cce()

        class Callbacks(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
                guard.begin(model, train_dataloader)
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                if guard.preclip_step != state.global_step:
                    guard.fail("Optimizer update reached without the pre-clipping checks")
                inspect_parameters(torch, guard.named, gradients=True)
                expected = sum(item["shifted_supervised_tokens"] for item in guard.microbatches)
                if len(guard.microbatches) != 4 or expected != guard.expected_window_count:
                    guard.fail("Actual four-microbatch update does not match its supervised-token window")
            def on_optimizer_step(self, args, state, control, **kwargs):
                guard.context["phase"] = "after_optimizer_update"
                inspect_parameters(torch, guard.named)
                inspect_optimizer(torch, trainer.optimizer)
                expected = sum(item["shifted_supervised_tokens"] for item in guard.microbatches)
                if len(guard.microbatches) != 4 or expected != guard.expected_window_count:
                    guard.fail("Actual four-microbatch update does not match its supervised-token window")
                if state.global_step < 2:
                    print("REAL_UPDATE_GUARD " + json.dumps({"next_step": state.global_step + 1,
                        "microbatches": len(guard.microbatches), "supervised_tokens": expected,
                        "gradient_coverage": guard.gradient_coverage}), flush=True)
                guard.microbatches = []
                guard.expected_window_count = None
                guard.preclip_step = None
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ("loss", "eval_loss", "train_loss", "grad_norm"):
                    if key in (logs or {}) and not math.isfinite(float(logs[key])):
                        guard.fail("Nonfinite aggregate metric", metric=key, value=finite_value(logs[key]))
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1 or state.global_step >= state.max_steps:
                    control.should_save = True
                if guard.stop_requested or time.time() >= guard.stop_at:
                    control.should_training_stop = True
                    control.should_save = True
                if guard.stop_at - time.time() < 15 * 60 or guard.stop_requested:
                    control.should_evaluate = False
                guard.status("training", peak_allocated_GiB={str(i): round(torch.cuda.max_memory_allocated(i) / 2**30, 3) for i in range(3)})
                return control
            def on_epoch_end(self, args, state, control, **kwargs):
                if guard.stop_at - time.time() < 15 * 60 or guard.stop_requested:
                    control.should_evaluate = False
                return control
            def on_save(self, args, state, control, **kwargs):
                guard.save_checkpoint()
            def on_train_end(self, args, state, control, **kwargs):
                guard.train_finished = True
                guard.status("trained_awaiting_export", epochs_finished=state.global_step >= state.max_steps)
        trainer.add_callback(Callbacks())
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            previous = signal.getsignal(signum)
            def stop(number, frame):
                guard.stop_requested = signal.Signals(number).name
                print("STOP_REQUESTED " + guard.stop_requested, flush=True)
            signal.signal(signum, stop)
            self.restore.append(lambda n=signum, handler=previous: signal.signal(n, handler))

    def begin(self, model, loader):
        torch = self.torch
        buffers = self.place_buffers(model)
        layout = self.check_layout(model)
        if layout["trainable_parameters"] != 33478656:
            self.fail("Unexpected adapter coverage", layout=layout)
        families = {"gdn": 0, "shared": 0, "qsa": 0}
        for name, parameter in self.named:
            if parameter.dtype not in (torch.float32, torch.bfloat16):
                self.fail("Unexpected adapter precision", parameter=name, dtype=str(parameter.dtype))
            family = "gdn" if ".linear_attn." in name else "shared" if ".shared_expert." in name else "qsa"
            families[family] += parameter.numel()
        if families != {"gdn": 20404224, "shared": 7372800, "qsa": 5701632}:
            self.fail("Adapter target families differ from the reviewed recipe", families=families)
        if not getattr(model, "is_gradient_checkpointing", False):
            self.fail("Non-reentrant gradient checkpointing was not enabled")
        if self.trainer.state.max_steps != 660 or loader is None or len(loader.dataset) != 880:
            self.fail("The constructed Trainer no longer matches 880 tasks and 660 planned updates")
        if len(loader.dataset[0]["input_ids"]) != 18184:
            self.fail("The first training task is no longer the complete longest example")
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False
        from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
        import inspect
        backends = {}
        for key in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
            function = getattr(modeling, key, None)
            backends[key] = {"module": getattr(function, "__module__", None),
                             "name": getattr(function, "__qualname__", None), "available": function is not None}
            try:
                path = inspect.getsourcefile(function)
                if path:
                    if __package__:
                        from .run_checks import digest
                    else:
                        from run_checks import digest
                    backends[key].update(source=path, sha256=digest(Path(path)))
            except (TypeError, OSError):
                pass
        atomic_json(self.output / "kernel-backends.json", backends)
        ple = []
        for name, module in model.named_modules():
            if type(module).__name__ == "Qwen4ExpTextNGramEmbedding":
                expected = {
                    "layer_multipliers": _build_layer_multipliers(module.unigram_vocab_size, module.ngram_size, module.ple_layer_index, module.seed),
                    "ngram_heads_vocab_sizes": torch.tensor(module.head_vocab_sizes, dtype=torch.long),
                    "ngram_heads_offsets": torch.tensor(module.head_offsets, dtype=torch.long)}
                for key, value in expected.items():
                    actual = getattr(module, key)
                    if actual.dtype != torch.long or not torch.equal(actual.cpu(), value.cpu()):
                        self.fail("PLE hashing buffer differs from the deterministic configuration", module=name, buffer=key)
                ple.append(name)
        if len(ple) != 1 or len(buffers) != 3:
            self.fail("Unexpected PLE module/buffer count")
        inspect_parameters(torch, self.named)
        self.install_activation_observers(model)
        self.status("validating", layout=layout, ple_buffers_verified=ple,
                    adapter_dtypes=sorted({str(p.dtype) for _, p in self.named}))
        print("FULL_RUN_GUARDRAILS_ACTIVE " + json.dumps(layout), flush=True)

    def install_activation_observers(self, model):
        torch = self.torch
        def observe(name):
            def hook(module, inputs, outputs):
                if self.first_bad_activation is not None:
                    return
                values = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                for value in values:
                    if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all().item():
                        self.first_bad_activation = {"module": name, "class": type(module).__name__,
                            "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                        break
            return hook
        classes = {"Qwen4ExpTextDecoderLayer", "Qwen4ExpTextGatedDeltaNet", "Qwen4ExpTextAttention",
                   "Qwen4ExpTextSparseMoeBlock", "Qwen4ExpTextPLELayer"}
        for name, module in model.named_modules():
            if type(module).__name__ in classes:
                self.handles.append(module.register_forward_hook(observe(name)))

    def install_cce(self):
        import cut_cross_entropy.transformers.utils as cce
        import cut_cross_entropy.cce as core
        torch, original = self.torch, cce.apply_lce
        def observe_backward(e):
            self.context.update(backward_kernel_device=str(e.device),
                                backward_current_device=torch.cuda.current_device() if e.device.type == "cuda" else None)
        self.replace(core, "cce_backward_kernel", owner_scoped_backward(torch, core.cce_backward_kernel, observe_backward))
        @functools.wraps(original)
        def guarded(e, c, labels, opts, bias=None, softcap=None, shift_labels=None, e2=None, c2=None, **kwargs):
            if bias is not None or softcap is not None or shift_labels is not None or e2 is not None or c2 is not None:
                self.fail("CCE operands changed from the reviewed plain, frozen-head causal recipe")
            if opts.reduction != "mean" or opts.train_only:
                self.fail("CCE must observe both validation and token-normalized training")
            if c.requires_grad or tuple(c.shape) != self.expected_classifier_shape or e.shape[-1] != self.expected_classifier_shape[1]:
                self.fail("CCE classifier/hidden layout differs from the frozen Flash output head")
            denominator = align_denominator(torch, kwargs.get("num_items_in_batch"), e.device)
            kwargs["num_items_in_batch"] = denominator
            count = float(denominator.item() if torch.is_tensor(denominator) else denominator)
            self.context.update(loss_device=str(e.device), hidden_dtype=str(e.dtype), classifier_dtype=str(c.dtype),
                                denominator_device=str(getattr(denominator, "device", "python")))
            if c.device != e.device or not torch.isfinite(e).all().item():
                self.fail("Invalid final hidden states or classifier placement before CCE")
            if self.loss_calls == 0 and not torch.isfinite(c).all().item():
                self.fail("Nonfinite frozen output classifier before CCE")
            self.loss_calls += 1
            before_device = torch.cuda.current_device() if e.device.type == "cuda" else None
            with tensor_device_scope(torch, e):
                self.context.update(loss_entry_current_device=before_device,
                                    loss_active_current_device=torch.cuda.current_device() if e.device.type == "cuda" else None)
                loss = original(e, c, labels, opts, bias=bias, softcap=softcap, shift_labels=shift_labels, e2=e2, c2=c2, **kwargs)
            scalar = float(loss.detach().item())
            step, split = self.trainer.state.global_step, self.context["split"]
            key = (split, step)
            reference = None
            if not math.isfinite(scalar) or (key not in self.reference_seen and (split == "dev" or step in (0, 1) or step % 100 == 0)):
                observed_e = e.to(c.dtype) if c.dtype == torch.bfloat16 and e.dtype == torch.float32 else e
                reference = reference_cross_entropy(torch, observed_e, c, labels, count)
                self.reference_seen.add(key)
            atol = 0.03 * self.context["shifted_supervised_tokens"] / count
            failure = classify_failure(scalar, self.first_bad_activation, reference, atol=atol)
            if reference is not None:
                item = {**self.context, "loss": finite_value(scalar), "reference": reference,
                        "absolute_tolerance": atol, "relative_tolerance": 0.01, "diagnosis": failure}
                with (self.output / "loss-observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                print("LOSS_REFERENCE " + json.dumps(item), flush=True)
            if failure:
                self.fail(failure, raw_loss=finite_value(scalar), reference=reference)
            return loss
        self.replace(cce, "apply_lce", guarded)

    def save_checkpoint(self):
        torch, trainer = self.torch, self.trainer
        path = self.output / f"checkpoint-{trainer.state.global_step}"
        self.status("validating_checkpoint")
        inspect_parameters(torch, self.named)
        torch.save(torch.cuda.get_rng_state_all(), path / "arc_cuda_rng_all.pt")
        validation = validate_checkpoint(torch, path, trainer)
        seal_checkpoint(path, validation)
        checkpoint_files(path)
        self.last_checkpoint = str(path)
        print("CHECKPOINT_VALIDATED " + str(path), flush=True)
        # Observers localize the previously failing startup and first real update.
        # Always-on raw-loss, source-label, gradient and weight checks remain active.
        for handle in self.handles:
            handle.remove()
        self.handles = []
        import shutil
        accepted = []
        for candidate in self.output.glob("checkpoint-*"):
            if candidate.is_dir() and candidate.name.removeprefix("checkpoint-").isdigit():
                try:
                    checkpoint_files(candidate)
                    accepted.append(candidate)
                except RuntimeError:
                    continue
        accepted.sort(key=lambda candidate: int(candidate.name.split("-")[-1]))
        for old in accepted[:-3]:
            shutil.rmtree(old)
        self.status("training")

    def exported(self):
        if not self.train_finished or self.last_checkpoint is None:
            self.fail("Training returned without its guarded end and a valid full checkpoint")
        self.status("validating_export")
        validation = validate_adapters(self.torch, self.output, self.trainer.model)
        if self.trainer.state.global_step >= self.trainer.state.max_steps:
            # Also produce the portable tokenizer/base-identity bundle and completion receipt.
            finish_export(self.output, Path(self.last_checkpoint), Path(self.prepared["run_directory"]), self.config)
            return
        atomic_json(self.output / "export-validation.json", validation)
        self.status("complete" if self.trainer.state.global_step >= self.trainer.state.max_steps else "time_budget_checkpointed",
                    export_validated=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for restore in reversed(self.restore):
            restore()


def install(torch, config, prepared, output, check_layout, place_buffers):
    """Attach to the actual Axolotl Trainer instance, then let its ordinary run continue."""
    from transformers import Trainer
    original = Trainer.train
    guards = []
    @functools.wraps(original)
    def train(trainer, *args, **kwargs):
        if guards:
            raise RuntimeError("Unexpected second Trainer.train call")
        guard = RuntimeGuard(torch, trainer, config, prepared, output, check_layout, place_buffers)
        guards.append(guard)
        guard.attach()
        return original(trainer, *args, **kwargs)
    Trainer.train = train
    return guards, lambda: setattr(Trainer, "train", original)
