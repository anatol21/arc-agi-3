"""Explicit normalization for the pinned Qwen CCE path, independent of introspection."""
from __future__ import annotations


def configure_token_normalization(trainer):
    """CCE consumes sum-CE / window-token-count; Trainer must not divide again.

    This is only installed by this run's verified Qwen/CCE integration. The actual
    CCE boundary independently asserts a positive denominator and exactly one call.
    """
    args = trainer.args
    before = {"model_accepts_loss_kwargs": bool(trainer.model_accepts_loss_kwargs),
              "loss_shifts_labels": bool(getattr(trainer, "_loss_shifts_labels", False))}
    if args.gradient_accumulation_steps != 4:
        raise RuntimeError(f"Expected four microbatches; constructed Trainer has gradient_accumulation_steps={args.gradient_accumulation_steps!r}")
    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError("Accelerator must use accumulation=1 because Trainer owns the four-microbatch window; "
                           f"found {trainer.accelerator.gradient_accumulation_steps!r}")
    if getattr(trainer, "compute_loss_func", None) is not None or getattr(trainer, "label_smoother", None) is not None:
        raise RuntimeError("The pinned CCE recipe requires its own scalar loss, without another loss function or label smoothing")
    # HF/Axolotl can infer False from the *unpatched* loss function or a class flag.
    # The verified CCE forward actually accepts and consumes this kwarg. Set the
    # effective Trainer contract, rather than relying on that inference.
    trainer.model_accepts_loss_kwargs = True
    trainer._loss_shifts_labels = True
    return {"schema_version": 1, "mode": "sum_shifted_cross_entropy_divided_by_window_target_count",
            "inferred_before_override": before, "model_accepts_loss_kwargs": True, "loss_shifts_labels": True,
            "trainer_microbatches": 4, "accelerator_accumulation": 1}
