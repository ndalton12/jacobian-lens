"""Optional actual-checkpoint smoke. Never downloads a model in ordinary CI."""

import os

import pytest
import torch

from experiments.short_hop.benchmark_cases import build_arithmetic, build_gallery
from experiments.short_hop.build_eval_cases import build_cases
from experiments.short_hop.common import load_model
from experiments.short_hop.evaluate import evaluate_cases
from jlens import JacobianLens, ShortHopAtlas, fit, fit_short_hop_atlas
from jlens.relp import RULE_VERSION, relp_rules


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA_TESTS") != "1" or not os.environ.get("HF_TOKEN"),
    reason="requires RUN_GEMMA_TESTS=1 and HF_TOKEN",
)
def test_actual_gemma_fit_resume_and_evaluate(tmp_path):
    hf, model = load_model()
    prompts = [
        "The scientist carefully recorded the results of the experiment and compared them with earlier observations. "
        * 4
    ]
    kwargs = dict(
        target_to_sources={2: [1]},
        dim_batch=8,
        max_seq_len=16,
        skip_first=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    atlas = fit_short_hop_atlas(model, prompts, **kwargs)
    again = fit_short_hop_atlas(model, prompts, **kwargs)
    torch.testing.assert_close(atlas.get(1, 2), again.get(1, 2))
    path = str(tmp_path / "atlas.pt")
    atlas.save(path)
    atlas = ShortHopAtlas.load(path)
    fitting = dict(source_layers=[1], dim_batch=8, max_seq_len=16, skip_first=2)
    j_lens = fit(model, prompts, **fitting)
    with relp_rules(model):
        r_lens = fit(model, prompts, **fitting)
        tr_atlas = fit_short_hop_atlas(
            model,
            prompts,
            target_to_sources={2: [1]},
            dim_batch=8,
            max_seq_len=16,
            skip_first=2,
            means=atlas.means,
            metadata={"backward_rule": RULE_VERSION},
        )
    assert isinstance(r_lens, JacobianLens)
    cases = build_cases(model.tokenizer, n_cases=8)[:1]
    rows, _ = evaluate_cases(
        hf, model, atlas, j_lens, r_lens, cases, tmp_path, tr_atlas=tr_atlas
    )
    assert len(rows) == 9
    assert all(torch.isfinite(torch.tensor(row["intermediate_logit"])) for row in rows)
    practice, arithmetic = build_arithmetic(model.tokenizer, 24, preflight_cases=4)
    assert len(practice) == 4 and len(arithmetic) == 24
    gallery = build_gallery(model.tokenizer)
    rows, _ = evaluate_cases(
        hf,
        model,
        atlas,
        j_lens,
        r_lens,
        arithmetic[:1],
        tmp_path / "arithmetic",
        tr_atlas=tr_atlas,
    )
    assert len(rows) == 9
    rows, _ = evaluate_cases(
        hf,
        model,
        atlas,
        j_lens,
        r_lens,
        gallery[:1],
        tmp_path / "gallery",
        tr_atlas=tr_atlas,
    )
    assert len(rows) == 9
