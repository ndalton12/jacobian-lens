"""Prompt-weighted residual means using the fitting position mask."""

from collections.abc import Sequence

import torch

from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@torch.no_grad()
def fit_activation_means(
    model: LensModel,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    max_seq_len: int = 64,
    skip_first: int = 8,
) -> dict[int, torch.Tensor]:
    layers = sorted(set(layers))
    if not layers or layers[0] < 0 or layers[-1] >= model.n_layers:
        raise ValueError("layers must be nonempty and within the residual stack")
    if skip_first < 0 or max_seq_len <= skip_first + 1:
        raise ValueError("invalid max_seq_len or skip_first")
    sums = {layer: torch.zeros(model.d_model, dtype=torch.float32) for layer in layers}
    count = 0
    with ActivationRecorder(model.layers, at=layers) as recorder:
        for prompt in prompts:
            ids = model.encode(prompt, max_length=max_seq_len)
            if ids.shape[1] <= skip_first + 1:
                continue
            mask = valid_position_mask(ids.shape[1], skip_first=skip_first)
            model.forward(ids)
            for layer in layers:
                h = recorder.activations[layer][0]
                sums[layer] += h[mask.to(h.device)].float().mean(0).cpu()
            count += 1
    if not count:
        raise ValueError("no prompts were long enough to fit activation means")
    means = {layer: value / count for layer, value in sums.items()}
    if not all(torch.isfinite(value).all() for value in means.values()):
        raise FloatingPointError("nonfinite activation mean")
    return means
