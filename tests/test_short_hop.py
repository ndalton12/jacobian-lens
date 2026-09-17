import pytest
import torch

from jlens import ShortHopAtlas, fit_short_hop_atlas

from .tiny import TinyDecoder


def test_exact_maps_orientation_roundtrip_and_resume(tmp_path):
    model = TinyDecoder()
    for param in model.parameters():
        param.requires_grad_(False)
    kwargs = dict(
        target_to_sources={2: [0, 1], 3: [1]},
        dim_batch=3,
        max_seq_len=32,
        skip_first=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        position_seed=17,
    )
    prompts = ["abcdefghij " * 4, "vwxyz " * 4]
    atlas = fit_short_hop_atlas(model, prompts, **kwargs)
    again = fit_short_hop_atlas(model, prompts, **kwargs)
    h = torch.randn(3, model.d_model)
    for source, target in atlas.pairs:
        expected = torch.eye(model.d_model)
        actual_h = h.clone()
        for layer in range(source + 1, target + 1):
            expected = (
                torch.eye(model.d_model) + model.layers[layer].linear.weight
            ) @ expected
            actual_h = model.layers[layer](actual_h)
        torch.testing.assert_close(
            atlas.get(source, target), expected, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(h @ atlas.get(source, target).T, actual_h)
        torch.testing.assert_close(
            atlas.transport_centered(h, source, target), actual_h, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(again.get(source, target), atlas.get(source, target))
    path = str(tmp_path / "atlas.pt")
    atlas.save(path)
    loaded = ShortHopAtlas.load(path)
    assert loaded.pairs == [(0, 2), (1, 2), (1, 3)]
    assert loaded.fit_config == atlas.fit_config
    for i, j in atlas.pairs:
        torch.testing.assert_close(
            loaded.get(i, j), atlas.get(i, j), atol=5e-4, rtol=1e-3
        )
    for layer, mean in atlas.means.items():
        torch.testing.assert_close(loaded.means[layer], mean)


def test_identity_innovation_and_validation():
    model = TinyDecoder(n_layers=2, d_model=4)
    means = {0: torch.randn(4), 1: torch.randn(4)}
    kwargs = dict(n_prompts_by_target={1: 2}, d_model=4, n_layers=2)
    atlas = ShortHopAtlas({1: {0: torch.eye(4)}}, means, **kwargs)
    h = torch.randn(4)
    torch.testing.assert_close(atlas.innovation_vector(h, 0, 1), torch.zeros(4))
    identity = means[1] + h - means[0]
    torch.testing.assert_close(
        model.unembed(atlas.transport_centered(h, 0, 1)) - model.unembed(identity),
        torch.zeros(32),
        atol=1e-6,
        rtol=0,
    )
    with pytest.raises(ValueError, match="source < target"):
        ShortHopAtlas({1: {1: torch.eye(4)}}, means, **kwargs)
    with pytest.raises(ValueError, match="mean"):
        ShortHopAtlas({1: {0: torch.eye(4)}}, {}, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        ShortHopAtlas({1: {0: torch.eye(3)}}, means, **kwargs)
