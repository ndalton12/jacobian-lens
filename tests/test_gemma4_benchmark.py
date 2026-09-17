import json
from types import SimpleNamespace

import pytest
import torch

from experiments.short_hop import benchmark
from experiments.short_hop.benchmark_cases import (
    build_arithmetic,
    build_gallery,
    case_ids,
)
from jlens import ShortHopAtlas, from_hf
from jlens.relp import GEMMA4_RULE_VERSION, RULE_VERSION

from .test_arithmetic_benchmark import arithmetic_tokenizer
from .test_gemma4 import tiny_gemma4


def setup_run(tmp_path, monkeypatch):
    tok = arithmetic_tokenizer()
    hf = tiny_gemma4(vocab_size=len(tok))
    model = from_hf(hf, tok)
    practice, cases = build_arithmetic(tok, 24)
    gallery = build_gallery(tok)
    lookup = {
        tuple(case_ids(tok, c)[0].tolist()): c for c in practice + cases + gallery
    }
    generations = []

    def generate(input_ids, **kwargs):
        case = lookup[tuple(input_ids[0].tolist())]
        generations.append(case["id"])
        token = case["answer_token_id"] or tok.eos_token_id
        return torch.cat([input_ids, torch.tensor([[token]])], dim=1)

    monkeypatch.setattr(hf, "generate", generate)
    monkeypatch.setattr(benchmark, "load_model", lambda *args: (hf, model))
    monkeypatch.setattr(
        benchmark,
        "provenance",
        lambda *args: dict(model_revision="tiny-gemma4-revision"),
    )
    import experiments.short_hop.build_fit_corpus as corpus

    monkeypatch.setattr(
        corpus,
        "build_corpus",
        lambda tokenizer, n, *args: [
            dict(
                id=str(i),
                text="red blue green yellow black white orange purple pink brown silver gold",
            )
            for i in range(n)
        ],
    )
    import experiments.short_hop.benchmark_report as reports

    monkeypatch.setattr(reports, "write_gallery", lambda *args: None)
    import experiments.short_hop.plot_results as plots

    monkeypatch.setattr(plots, "plot_results", lambda *args: None)
    args = SimpleNamespace(
        plan=False,
        preflight_only=False,
        fit=True,
        smoke=True,
        reuse_from=None,
        output_dir=str(tmp_path / "new-run"),
        seed=0,
        model="google/gemma-4-E4B-it",
        revision="main",
        n_cases=24,
        max_new_tokens=16,
        n_prompts=32,
        max_seq_len=16,
        skip_first=2,
        dim_batch=4,
        position_reduction="self_hutchinson",
    )
    return hf, model, args, generations


def test_fresh_gemma4_smoke_fit_resumes_incomplete_tr(tmp_path, monkeypatch):
    hf, model, args, generations = setup_run(tmp_path, monkeypatch)
    import jlens.fitting as fitting

    original = fitting.jacobian_for_prompt
    attempts = []
    interrupted = False

    def compute(*pos, **kw):
        nonlocal interrupted
        is_r = "forward" in model.layers[0].mlp.__dict__
        mode = kw["position_reduction"]
        attempts.append((is_r, mode))
        if is_r and mode == "self_hutchinson" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("interrupt TR fit")
        return original(*pos, **kw)

    monkeypatch.setattr(fitting, "jacobian_for_prompt", compute)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(args)
    output = tmp_path / "new-run"
    assert (output / "j_lens.pt").exists() and (output / "r_lens.pt").exists()
    assert (output / "checkpoints" / "tj" / "target_05.pt").exists()
    assert len(generations) == 32
    benchmark.run(args)
    assert len(attempts) == 5  # TJ, J, R, failed TR, resumed TR only.
    assert len(generations) == 32 + 10 + 8
    report = json.loads((output / "summary.json").read_text())
    assert report["verdict"] == "INCONCLUSIVE"  # Smoke is never a science verdict.
    config = json.loads((output / "benchmark_config.json").read_text())
    assert config["backward_rule"] == GEMMA4_RULE_VERSION
    assert config["architecture"]["num_kv_shared_layers"] == 4
    assert config["path_checks"]["forward_preserved"]
    status = json.loads((output / "progress.json").read_text())
    assert status["overall_percent"] == 100
    assert all(
        status["methods"][m]["percent"] == 100
        for m in ("J-Lens", "R-Lens", "TJ-Lens", "TR-Lens")
    )
    benchmark.run(args)
    assert len(attempts) == 5 and len(generations) == 50
    atlas = ShortHopAtlas.load(str(output / "short_hop_atlas.pt"))
    j, r, tr = benchmark.load_bundle(
        output, atlas, model, args.model, "tiny-gemma4-revision"
    )
    assert tr.fit_config["backward_rule"] == GEMMA4_RULE_VERSION
    assert torch.equal(atlas.means[1], tr.means[1])
    assert not torch.allclose(atlas.get(1, 5), tr.get(1, 5))
    from experiments.short_hop.evaluate import evaluate_cases

    _, cases = build_arithmetic(model.tokenizer, 24)
    tr.fit_config["backward_rule"] = RULE_VERSION
    with pytest.raises(ValueError, match="backward rules"):
        evaluate_cases(
            hf, model, atlas, j, r, cases, output / "wrong-rule", tr_atlas=tr
        )


def test_fresh_fit_is_not_started_if_practice_fails(tmp_path, monkeypatch):
    _, _, args, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "preflight",
        lambda *a, **kw: dict(
            passed=False, solved=0, total=32, accuracy=0, threshold=0.8
        ),
    )
    monkeypatch.setattr(
        benchmark, "fit_bundle", lambda *a, **kw: pytest.fail("must not fit")
    )
    benchmark.run(args)
    output = tmp_path / "new-run"
    assert not (output / "checkpoints").exists()
    assert not (output / "qualitative").exists()
    assert "No fitting or qualitative" in (output / "REPORT.md").read_text()
