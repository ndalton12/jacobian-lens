"""Centered, same-position short-hop Jacobian transport (TJ-Lens)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from jlens.fitting import _atomic_save, fit
from jlens.protocol import LensModel
from jlens.stats import fit_activation_means

TARGET_TO_SOURCES = {
    8: [7, 6, 4, 0],
    14: [13, 12, 10, 6],
    20: [19, 18, 16, 12],
    25: [24, 23, 21, 17],
}


def default_pairs(n_layers):
    """Scale the four target depths; keep literal 1/2/4/8-block hop lengths.

    For E4B (42 blocks) targets are 13, 23, 33, 41. Target 23 is the last
    non-KV-sharing block; later pairs test the shared-KV region. Indices refer
    to block outputs, not percentages or the final RMSNorm output.
    """
    if n_layers < 10:
        raise ValueError("default schedule requires at least 10 layers")
    targets = [round(t * (n_layers - 1) / 25) for t in TARGET_TO_SOURCES]
    return {
        target: [target - hop for hop in (1, 2, 4, 8) if target >= hop]
        for target in targets
    }


def smoke_pairs(model):
    shared = getattr(
        getattr(model.layers[0], "config", None), "num_kv_shared_layers", 0
    )
    if shared:
        boundary = model.n_layers - shared
        target = min(boundary + 1, model.n_layers - 1)
        return {target: sorted({max(0, boundary - 3), target - 1})}
    target = min(8, model.n_layers - 1)
    return {target: sorted({max(0, target - 2), target - 1})}


def validate_pairs(mapping, n_layers):
    if not mapping:
        raise ValueError("target_to_sources must be nonempty")
    for target, sources in mapping.items():
        if not sources:
            raise ValueError(f"target {target} needs at least one source")
        for source in sources:
            if not 0 <= source < target < n_layers:
                raise ValueError(
                    f"require 0 <= source < target < {n_layers}; got {source}, {target}"
                )


class ShortHopAtlas:
    def __init__(
        self,
        jacobians,
        means,
        *,
        n_prompts_by_target,
        d_model,
        n_layers,
        model_id=None,
        position_reduction="self_hutchinson",
        fit_config=None,
    ):
        validate_pairs({j: list(m) for j, m in jacobians.items()}, n_layers)
        if position_reduction not in ("future_sum", "self_hutchinson"):
            raise ValueError("invalid position_reduction")
        self.jacobians = {
            j: {i: k.detach().float().cpu() for i, k in m.items()}
            for j, m in jacobians.items()
        }
        self.means = {i: m.detach().float().cpu() for i, m in means.items()}
        self.n_prompts_by_target = dict(n_prompts_by_target)
        self.d_model, self.n_layers = d_model, n_layers
        self.model_id, self.position_reduction = model_id, position_reduction
        self.fit_config = dict(fit_config or {})
        for source, target in self.pairs:
            k = self.get(source, target)
            if k.shape != (d_model, d_model) or not torch.isfinite(k).all():
                raise ValueError(
                    "every matrix must be finite with shape [d_model, d_model]"
                )
            for layer in (source, target):
                if layer not in self.means or self.means[layer].shape != (d_model,):
                    raise ValueError(f"missing or invalid mean for layer {layer}")
                if not torch.isfinite(self.means[layer]).all():
                    raise ValueError(f"nonfinite mean for layer {layer}")
            if self.n_prompts_by_target.get(target, 0) < 1:
                raise ValueError(f"missing positive prompt count for target {target}")

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [
            (i, j) for j in sorted(self.jacobians) for i in sorted(self.jacobians[j])
        ]

    def get(self, source_layer: int, target_layer: int) -> torch.Tensor:
        return self.jacobians[target_layer][source_layer]

    def transport_centered(self, residual, source_layer, target_layer):
        x = residual.float() - self.means[source_layer].to(residual.device)
        k = self.get(source_layer, target_layer).to(residual.device)
        return self.means[target_layer].to(residual.device) + x @ k.T

    def innovation_vector(self, residual, source_layer, target_layer):
        x = residual.float() - self.means[source_layer].to(residual.device)
        return x @ self.get(source_layer, target_layer).to(residual.device).T - x

    def save(self, path: str, *, dtype=torch.float16):
        matrices = {
            j: {i: k.to(dtype) for i, k in m.items()} for j, m in self.jacobians.items()
        }
        if not all(
            torch.isfinite(k).all() for m in matrices.values() for k in m.values()
        ):
            raise FloatingPointError(
                "atlas overflows storage dtype; use dtype=torch.float32"
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(
            dict(
                format_version=1,
                jacobians=matrices,
                means=self.means,
                n_prompts_by_target=self.n_prompts_by_target,
                d_model=self.d_model,
                n_layers=self.n_layers,
                model_id=self.model_id,
                position_reduction=self.position_reduction,
                fit_config=self.fit_config,
            ),
            str(path),
        )

    @classmethod
    def load(cls, path: str) -> ShortHopAtlas:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.pop("format_version", None) != 1:
            raise ValueError("unsupported ShortHopAtlas format")
        return cls(**state)


def fit_short_hop_atlas(
    model: LensModel,
    prompts: Sequence[str],
    *,
    target_to_sources: Mapping[int, Sequence[int]],
    model_id: str | None = None,
    dim_batch: int = 16,
    max_seq_len: int = 64,
    skip_first: int = 8,
    position_reduction: str = "self_hutchinson",
    position_seed: int = 0,
    checkpoint_dir: str | None = None,
    deadline: float | None = None,
    metadata: dict[str, Any] | None = None,
    means: dict[int, torch.Tensor] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> ShortHopAtlas:
    validate_pairs(target_to_sources, model.n_layers)
    layers = sorted({l for j, src in target_to_sources.items() for l in [j, *src]})
    if means is None:
        means = fit_activation_means(
            model,
            prompts,
            layers=layers,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
        )
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    jacobians, counts = {}, {}
    for target, sources in sorted(target_to_sources.items()):
        lens = fit(
            model,
            prompts,
            source_layers=sources,
            target_layer=target,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            position_reduction=position_reduction,
            position_seed=position_seed,
            checkpoint_path=str(Path(checkpoint_dir) / f"target_{target:02d}.pt")
            if checkpoint_dir
            else None,
            checkpoint_metadata={
                "model_id": model_id,
                "backward_rule": "autograd",
                **(metadata or {}),
            },
            deadline=deadline,
            progress_callback=progress_callback,
        )
        jacobians[target], counts[target] = lens.jacobians, lens.n_prompts
    config = dict(
        target_to_sources={str(k): list(v) for k, v in target_to_sources.items()},
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        position_seed=position_seed,
        position_reduction=position_reduction,
        prompts_sha256=hashlib.sha256(json.dumps(list(prompts)).encode()).hexdigest(),
        backward_rule=(metadata or {}).get("backward_rule", "autograd"),
        **{
            key: value
            for key, value in (metadata or {}).items()
            if key != "backward_rule"
        },
    )
    return ShortHopAtlas(
        jacobians,
        means,
        n_prompts_by_target=counts,
        d_model=model.d_model,
        n_layers=model.n_layers,
        model_id=model_id,
        position_reduction=position_reduction,
        fit_config=config,
    )
