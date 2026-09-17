# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:func:`jacobian_for_prompt`): for each output dimension, inject a
one-hot cotangent at *every valid target position at once* and backprop. The
gradient at source position ``p`` is then ``sum_{p' >= p} dh_final[p'] / dh_l[p]``,
the sum over later target positions; we take the mean over source positions
``p``. This is the reduction used in the paper. A per-position estimator
(``dh_final[p] / dh_l[p]`` averaged over ``p``) gives a slightly different
``J_l``; both work as a lens.

Cost: one forward pass and ``ceil(d_model / dim_batch)`` backward passes per
prompt. Shard across machines by running :func:`fit` on disjoint prompt
slices and merging with :meth:`jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16
PositionReduction = Literal["future_sum", "self_hutchinson"]


def _validate_settings(dim_batch, max_seq_len, skip_first, position_reduction):
    if dim_batch < 1 or max_seq_len <= skip_first + 1 or skip_first < 0:
        raise ValueError("require dim_batch >= 1 and max_seq_len > skip_first + 1 >= 1")
    if position_reduction not in ("future_sum", "self_hutchinson"):
        raise ValueError(f"unknown position_reduction: {position_reduction}")


def _position_signs(prompt, seed, d_model, n_valid):
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + int.from_bytes(digest[:8], "big")) % (2**63))
    return torch.randint(2, (d_model, n_valid), generator=generator).float() * 2 - 1


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short to
            leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve None/negative layer indices, bounds-check, enforce source < target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(
            f"target_layer={target_layer} out of range for {n_layers} layers"
        )
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    position_reduction: PositionReduction = "future_sum",
    position_seed: int = 0,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position. See the
    module docstring for the resulting estimator and how it relates to
    a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.
        position_reduction: ``future_sum`` preserves the original estimator;
            ``self_hutchinson`` estimates same-position Jacobian blocks with
            independent Rademacher signs over output dimensions and positions.
        position_seed: CPU sign seed, combined with a stable SHA-256 prompt hash.
        progress_callback: Optional callback after each completed backward batch.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
    """
    _validate_settings(dim_batch, max_seq_len, skip_first, position_reduction)
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())
    signs = (
        _position_signs(prompt, position_seed, d_model, n_valid_positions)
        if position_reduction == "self_hutchinson"
        else None
    )

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[
            target_layer
        ]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            row_signs = (
                signs[dim_start : dim_start + n_dims_this_pass].to(
                    target_activation.device
                )
                if signs is not None
                else None
            )
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every valid target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0 if row_signs is None else row_signs.to(cotangent.dtype)
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, d_model] on whatever device this
                # layer lives on; mean over the valid positions -> dim_batch rows.
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                values = grad[:n_dims_this_pass, positions_on_device, :].float()
                if row_signs is not None:
                    values = values * row_signs.to(grad.device)[..., None]
                rows = values.mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
            del grads
            if progress_callback:
                progress_callback(
                    dict(
                        event="backward",
                        pass_done=pass_idx + 1,
                        pass_total=n_passes,
                        target_layer=target_layer,
                    )
                )
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    position_reduction: PositionReduction = "future_sum",
    position_seed: int = 0,
    checkpoint_metadata: dict | None = None,
    deadline: float | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> JacobianLens:
    """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.
        position_reduction: See :func:`jacobian_for_prompt`; default unchanged.
        position_seed: See :func:`jacobian_for_prompt`.
        checkpoint_metadata: Optional provenance checked on resume (for example
            model revision and backward rule). Missing legacy fields are allowed.
        deadline: Optional ``time.monotonic()`` deadline, checked between prompts.
            A timeout saves the running checkpoint and raises ``TimeoutError``.
        progress_callback: Receives resume, prompt_start, backward, prompt_done,
            and complete events with prompt and backward-batch counters.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    _validate_settings(dim_batch, max_seq_len, skip_first, position_reduction)
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive or None")
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )
    prompt_hashes = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts]

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    fit_seconds = 0.0
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        fit_seconds = state.get("fit_seconds", 0.0)
        state.setdefault("position_reduction", "future_sum")
        state.setdefault("position_seed", 0)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
            ("max_seq_len", max_seq_len),
            ("d_model", d_model),
            ("n_layers", n_layers),
            ("position_reduction", position_reduction),
            ("position_seed", position_seed),
            ("checkpoint_metadata", checkpoint_metadata),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        if next_idx > len(prompts) or (
            "prompt_hashes" in state
            and state["prompt_hashes"][:next_idx] != prompt_hashes[:next_idx]
        ):
            raise ValueError(
                "checkpoint prompt prefix differs from the supplied corpus"
            )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                    "max_seq_len": max_seq_len,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "position_reduction": position_reduction,
                    "position_seed": position_seed,
                    "checkpoint_metadata": checkpoint_metadata,
                    "prompt_hashes": prompt_hashes[:next_idx],
                    "fit_seconds": fit_seconds,
                },
                checkpoint_path,
            )

    def emit(event, **details):
        if progress_callback:
            progress_callback(
                dict(
                    event=event,
                    prompt_done=next_idx,
                    prompt_total=len(prompts),
                    target_layer=target_layer,
                    **details,
                )
            )

    emit("resume")
    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            write_checkpoint()
            raise TimeoutError("fitting time budget reached; checkpoint saved")
        start_time = time.perf_counter()
        emit("prompt_start")
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                position_reduction=position_reduction,
                position_seed=position_seed,
                progress_callback=(
                    lambda event: emit(
                        "backward",
                        pass_done=event["pass_done"],
                        pass_total=event["pass_total"],
                    )
                )
                if progress_callback
                else None,
            )
        except ValueError as exc:
            if "prompt too short:" not in str(exc):
                raise
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            write_checkpoint()
            emit("prompt_done")
            continue
        if not all(torch.isfinite(J).all() for J in per_prompt_J.values()):
            raise FloatingPointError(f"nonfinite Jacobian at prompt {prompt_idx}")

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1
        fit_seconds += time.perf_counter() - start_time

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()
        emit("prompt_done")

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    emit("complete")
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
