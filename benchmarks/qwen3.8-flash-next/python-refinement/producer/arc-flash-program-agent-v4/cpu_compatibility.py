"""Exercise installed Trainer + Qwen CCE Python paths on tiny CPU tensors.

It uses a tiny model, a CPU backward adapter and a CPU CE primitive in place of
the CUDA kernel. No experiment model weights, skeleton or CUDA allocation is loaded.
"""
from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from .loss_normalization import configure_token_normalization
    from .run_checks import digest
else:
    from loss_normalization import configure_token_normalization
    from run_checks import digest


def validate_preflight_report(report):
    if (not isinstance(report, dict) or report.get("schema_version") != 1
        or report.get("status") != "passed" or report.get("cuda_kernel_tested") is not False
        or [case.get("inferred_model_accepts_loss_kwargs") for case in report.get("cases", [])] != [False, True]
        or any(case.get("gradients_match") is not True or case.get("optimizer_update_matches") is not True
               or case.get("evaluation_matches") is not True for case in report.get("cases", []))):
        raise RuntimeError("Missing or incomplete installed-framework CPU compatibility result")


def exercise_contract(torch, trainer_class, qwen_module, cce_utils, patch_options, distributed_type):
    """Use real installed class methods; injectable arguments allow pinned-source regressions."""
    class BodyOutput:
        past_key_values = hidden_states = attentions = rope_deltas = router_logits = None
        def __init__(self, hidden):
            self.hidden = hidden
        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.hidden

    class Body(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
        def forward(self, input_ids=None, **kwargs):
            return BodyOutput(self.embedding(input_ids))

    class TinyQwen(torch.nn.Module):
        # Reproduce the false-negative introspection that v4 refused to handle.
        accepts_loss_kwargs = False
        def __init__(self):
            super().__init__()
            self.model = Body(torch.nn.Embedding(11, 4))
            self.lm_head = torch.nn.Linear(4, 11, bias=False)
            self.lm_head.weight.requires_grad_(False)
            self.config = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(vocab_size=11))
        forward = qwen_module.cce_forward_multimodal
        def loss_function(self, **kwargs):
            raise RuntimeError("The unpatched fallback loss must not run")

    class CPUAccelerator:
        gradient_accumulation_steps = num_processes = 1
        parallelism_config = None
        def unwrap_model(self, model):
            return model
        def backward(self, loss, **kwargs):
            loss.backward()

    calls = []
    def cpu_ce(e, c, targets, *, shift, reduction, bias=None, **kwargs):
        if shift is not True or bias is not None:
            raise RuntimeError("Unexpected CCE operands in the CPU compatibility check")
        logits = torch.nn.functional.linear(e[..., :-1, :], c)
        calls.append({"reduction": reduction, "targets": int((targets[..., 1:] != -100).sum())})
        return torch.nn.functional.cross_entropy(logits.reshape(-1, c.shape[0]), targets[..., 1:].reshape(-1),
                                                reduction=reduction)

    batches = []
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    for number in (1, 2, 3, 4):
        labels = ids.clone()
        labels[:, :6 - number] = -100
        # Position zero is never a causal target. Include a sentinel there in
        # this synthetic check so counting unshifted labels cannot also pass.
        if number == 1:
            labels[:, 0] = ids[:, 0]
        batches.append({"input_ids": ids.clone(), "labels": labels, "attention_mask": torch.ones_like(ids)})

    cases = []
    with torch.random.fork_rng(devices=[]), patch.object(cce_utils, "linear_cross_entropy", cpu_ce), \
         patch.object(qwen_module, "_PATCH_OPTS", patch_options):
        for inferred in (False, True):
            torch.manual_seed(7830099)
            model = TinyQwen()
            # Deliberately skip Trainer.__init__: it would select devices and prepare
            # distributed state. The loss/count/backward methods themselves are real.
            trainer = object.__new__(trainer_class)
            trainer.model = model
            trainer.args = SimpleNamespace(gradient_accumulation_steps=4, n_gpu=1, world_size=1,
                average_tokens_across_devices=False, include_tkps=False, orpo_alpha=0,
                sample_packing=False, torch_empty_cache_steps=None, optim="sgd", device=torch.device("cpu"))
            trainer.accelerator = CPUAccelerator()
            trainer.accelerator.distributed_type = distributed_type
            trainer.compute_loss_func = trainer.label_smoother = None
            trainer.model_accepts_loss_kwargs = inferred
            trainer._loss_shifts_labels = False
            trainer.current_gradient_accumulation_steps = 4
            trainer.state = SimpleNamespace(global_step=0)
            trainer.optimizer = torch.optim.SGD([model.model.embedding.weight], lr=0.1)
            trainer._layer_offload_manager = None
            trainer._layer_offload_ctx = contextlib.nullcontext()
            trainer.activation_offload_context = contextlib.nullcontext()
            trainer._prepare_inputs = lambda inputs: inputs
            trainer._prepare_context_parallel_inputs = lambda model, inputs: (contextlib.nullcontext, inputs)
            trainer.compute_loss_context_manager = contextlib.nullcontext
            normalization = configure_token_normalization(trainer)
            count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
            if count is None or int(count) != 10:
                raise RuntimeError(f"Installed Trainer counted the wrong shifted targets: {count}")
            weight = model.model.embedding.weight
            initial = weight.detach().clone()
            expected_loss = sum(torch.nn.functional.cross_entropy(
                model.lm_head(model.model.embedding(batch["input_ids"])[:, :-1]).reshape(-1, 11),
                batch["labels"][:, 1:].reshape(-1), reduction="sum") for batch in batches) / count
            expected_gradient, = torch.autograd.grad(expected_loss, weight)
            trainer.optimizer.zero_grad(set_to_none=True)
            calls.clear()
            observed = []
            model.train()
            for batch in batches:
                observed.append(trainer.training_step(model, dict(batch), num_items_in_batch=count))
            torch.testing.assert_close(sum(observed), expected_loss.detach(), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(weight.grad, expected_gradient, atol=1e-6, rtol=1e-5)
            if len(calls) != 4 or any(call["reduction"] != "sum" for call in calls):
                raise RuntimeError("The real CCE wrapper did not select sum reduction for four microbatches")
            max_error = float((weight.grad - expected_gradient).abs().max())
            trainer.optimizer.step()
            torch.testing.assert_close(weight, initial - 0.1 * expected_gradient, atol=1e-6, rtol=1e-5)
            model.eval()
            for batch in batches:
                eval_count = trainer._get_num_items_in_batch([batch], torch.device("cpu"))
                loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True, num_items_in_batch=eval_count)
                reference = torch.nn.functional.cross_entropy(
                    model.lm_head(model.model.embedding(batch["input_ids"])[:, :-1]).reshape(-1, 11),
                    batch["labels"][:, 1:].reshape(-1), reduction="mean")
                torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-5)
            cases.append({"inferred_model_accepts_loss_kwargs": inferred, "normalization": normalization,
                          "target_counts": [1, 2, 3, 4], "window_target_count": 10,
                          "gradients_match": True, "optimizer_update_matches": True,
                          "evaluation_matches": True, "maximum_gradient_absolute_error": max_error})
    return cases


def run_preflight():
    import torch
    from accelerate.utils import DistributedType
    from axolotl.core.trainers.base import AxolotlTrainer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    import cut_cross_entropy.transformers.qwen4_exp as qwen
    import cut_cross_entropy.transformers.utils as cce
    from cut_cross_entropy.transformers.utils import PatchOptions

    opts = PatchOptions(impl="cce", reduction="mean", filter_eps="auto", accum_e_fp32=False,
                        accum_c_fp32=False, filter_e_grad=True, filter_c_grad=True, train_only=False)
    cases = exercise_contract(torch, AxolotlTrainer, qwen, cce, opts, DistributedType.NO)
    sources = {}
    methods = [base.__dict__["training_step"] for base in AxolotlTrainer.__mro__ if "training_step" in base.__dict__]
    methods += [AxolotlTrainer.compute_loss, AxolotlTrainer._get_num_items_in_batch,
                qwen.cce_forward_multimodal, cce.apply_lce]
    for function in methods:
        source = inspect.getsourcefile(function)
        if source is None:
            raise RuntimeError("Cannot fingerprint the installed training interface")
        sources[str(Path(source))] = digest(Path(source))
    report = {"schema_version": 1, "status": "passed", "cuda_kernel_tested": False,
              "scope": "installed Trainer/Qwen-CCE Python methods; tiny CPU model and backward adapter; CPU CE primitive replaces CUDA kernel; no distributed/device preparation or real backbone execution",
              "native_qwen_accepts_loss_kwargs": Qwen4ExpForConditionalGeneration.accepts_loss_kwargs,
              "trainer_mro": [base.__module__ + "." + base.__qualname__ for base in AxolotlTrainer.__mro__],
              "cases": cases, "source_sha256": sources}
    validate_preflight_report(report)
    return report
