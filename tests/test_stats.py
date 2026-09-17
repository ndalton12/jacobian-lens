import pytest
import torch

from jlens import ActivationRecorder, fit_activation_means
from jlens.fitting import valid_position_mask

from .tiny import TinyDecoder


def test_means_equal_prompt_weighting_and_skips():
    model = TinyDecoder()
    prompts = ["abcde", "vwxyz " * 8]
    actual = fit_activation_means(
        model, [*prompts, ""], layers=[0, 3], max_seq_len=64, skip_first=2
    )
    collected = {0: [], 3: []}
    for prompt in prompts:
        ids = model.encode(prompt, max_length=64)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[0, 3]) as recorder:
            model.forward(ids)
        mask = valid_position_mask(ids.shape[1], skip_first=2)
        for layer in collected:
            collected[layer].append(recorder.activations[layer][0, mask])
    for layer, values in collected.items():
        expected = torch.stack([v.mean(0) for v in values]).mean(0)
        torch.testing.assert_close(actual[layer], expected)
        assert (
            actual[layer].dtype == torch.float32 and actual[layer].device.type == "cpu"
        )
        assert not torch.allclose(actual[layer], torch.cat(values).mean(0))
    with pytest.raises(ValueError, match="no prompts"):
        fit_activation_means(model, [""], layers=[0])
