"""Opt-in real E4B check. No weights are downloaded in ordinary test runs."""

import os

import pytest
import torch

from experiments.short_hop.common import load_model
from experiments.short_hop.gemma_checks import check_model_paths
from jlens.relp import GEMMA4_RULE_VERSION
from jlens.short_hop import default_pairs, smoke_pairs


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA4_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires RUN_GEMMA4_TESTS=1 and a CUDA GPU; downloads real E4B weights",
)
def test_actual_e4b_text_wrapper_and_r_backward():
    hf, model = load_model("google/gemma-4-E4B-it")
    assert model.n_layers == 42 and model.d_model == 2560
    assert sorted(default_pairs(model.n_layers)) == [13, 23, 33, 41]
    assert smoke_pairs(model) == {25: [21, 24]}
    result = check_model_paths(
        hf,
        model,
        "The scientist compared these results with earlier observations and wrote a report.",
    )
    assert result["backward_rule"] == GEMMA4_RULE_VERSION
    assert result["forward_preserved"] and result["restored"]
