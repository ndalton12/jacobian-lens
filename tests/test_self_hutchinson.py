import itertools

import pytest
import torch
from torch import nn

from jlens.fitting import fit, jacobian_for_prompt

from .tiny import TinyDecoder


class CausalMix(nn.Module):
    def forward(self, h):
        return h + 0.3 * h.cumsum(dim=1)


def test_exact_diagonal_average_over_all_sign_probes(monkeypatch):
    model = TinyDecoder(n_layers=2, d_model=2)
    model.layers[1] = CausalMix()
    for param in model.parameters():
        param.requires_grad_(False)
    # Four valid positions: skip BOS and exclude final position.
    prompt = "abcde"
    ids = model.encode(prompt)
    h = model.forward(ids).last_hidden_state.detach().requires_grad_()
    full = torch.autograd.functional.jacobian(model.layers[1], h)
    expected = torch.stack([full[0, p, :, 0, p, :] for p in range(1, 5)]).mean(0)
    probes = []
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        monkeypatch.setattr(
            "jlens.fitting._position_signs",
            lambda *_, s=signs: torch.tensor(s).expand(2, -1),
        )
        maps, _, _ = jacobian_for_prompt(
            model,
            prompt,
            [0],
            dim_batch=2,
            max_seq_len=6,
            skip_first=1,
            position_reduction="self_hutchinson",
        )
        probes.append(maps[0])
    torch.testing.assert_close(torch.stack(probes).mean(0), expected, atol=1e-6, rtol=0)
    future, _, _ = jacobian_for_prompt(
        model, prompt, [0], dim_batch=2, max_seq_len=6, skip_first=1
    )
    assert not torch.allclose(future[0], expected)


def test_signs_deterministic_across_dimension_batches():
    model = TinyDecoder(n_layers=2, d_model=5)
    model.layers[1] = CausalMix()
    kwargs = dict(
        source_layers=[0],
        max_seq_len=12,
        skip_first=1,
        position_reduction="self_hutchinson",
        position_seed=9,
    )
    a, _, _ = jacobian_for_prompt(model, "abcdefghij", dim_batch=2, **kwargs)
    torch.manual_seed(999)
    b, _, _ = jacobian_for_prompt(model, "abcdefghij", dim_batch=4, **kwargs)
    torch.testing.assert_close(a[0], b[0])


def test_resume_legacy_and_mode_seed_corpus_rejection(tmp_path):
    model = TinyDecoder()
    path = str(tmp_path / "fit.pt")
    prompts = ["a long fitting prompt " * 3]
    kwargs = dict(source_layers=[0], dim_batch=4, checkpoint_path=path)
    reference = fit(model, prompts, **kwargs)
    state = torch.load(path, weights_only=True)
    state.pop("position_reduction")
    state.pop("position_seed")
    torch.save(state, path)
    resumed = fit(model, prompts, **kwargs)
    torch.testing.assert_close(reference.jacobians[0], resumed.jacobians[0])
    for extra, match in (
        (dict(position_reduction="self_hutchinson"), "position_reduction"),
        (dict(position_seed=1), "position_seed"),
        (dict(max_seq_len=64), "max_seq_len"),
    ):
        with pytest.raises(ValueError, match=match):
            fit(model, prompts, **kwargs, **extra)
    with pytest.raises(ValueError, match="prefix"):
        fit(model, ["a different fitting prompt " * 3], **kwargs)
    extended = fit(model, prompts + ["another valid prompt " * 3], **kwargs)
    assert extended.n_prompts == 2
    with pytest.raises(ValueError, match="position_reduction"):
        fit(model, prompts, position_reduction="invalid")
