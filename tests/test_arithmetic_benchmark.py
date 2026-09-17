import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from experiments.short_hop import benchmark
from experiments.short_hop.benchmark_cases import (
    build_arithmetic,
    build_gallery,
    case_ids,
    number_word,
    numeric_answer,
    prepare_case,
    solved_case,
)
from experiments.short_hop.benchmark_report import write_benchmark_report, write_gallery
from experiments.short_hop.common import MODEL_ID, save_baseline
from experiments.short_hop.evaluate import evaluate_cases, readout_metrics
from jlens import fit, fit_short_hop_atlas, from_hf
from jlens.relp import RULE_VERSION, relp_rules

from .test_relp import tiny_gemma
from .test_short_hop_experiment import word_tokenizer


def arithmetic_tokenizer():
    tokenizer = word_tokenizer()
    # Like Gemma, individual digits rather than whole arbitrary multi-digit tokens.
    tokenizer.add_tokens([str(n) for n in range(10)])
    tokenizer.add_tokens(
        [number_word(n) for n in range(21)]
        + [number_word(n) for n in range(30, 100, 10)]
    )
    tokenizer.add_tokens(
        [
            "Calculate",
            "First",
            "add",
            "subtract",
            "multiply",
            "Mars",
            "Venus",
            "language",
            "history",
            "+",
            "-",
            "*",
            "(",
            ")",
        ]
    )
    return tokenizer


def test_arithmetic_families_are_disjoint_and_balanced():
    tok = arithmetic_tokenizer()
    practice, cases = build_arithmetic(tok)
    assert (practice, cases) == build_arithmetic(tok)
    assert len(practice) == 32 and len(cases) == 192
    assert sum(c["split"] == "test" for c in cases) == 128
    assert (practice, cases) != build_arithmetic(tok, seed=1)
    families = defaultdict(list)
    problems = defaultdict(set)
    for case in practice + cases:
        families[case["group_id"]].append(case)
        problems[case["problem_id"]].add(case["split"])
        assert prepare_case(tok, case) == case
        assert set(case["intermediate_token_ids"]).isdisjoint(case["control_token_ids"])
        assert all(i not in tok.all_special_ids for i in case["intermediate_token_ids"])
    assert all(len(splits) == 1 for splits in problems.values())
    for family in families.values():
        assert len(family) == 4
        assert len({r["split"] for r in family}) == 1
        assert len({r["numeric_answer"] for r in family}) == 1
        assert len({r["intermediate"] for r in family}) == 2
        assert {r["wording"] for r in family} == {"symbolic", "verbal"}
        for row in family:
            other = next(r for r in family if r["member"] != row["member"])
            assert row["intermediate"] == other["control_intermediate"]
    with pytest.raises(ValueError, match="divisible"):
        build_arithmetic(tok, 128)


@pytest.mark.parametrize("text", ["20", "twenty", "Twenty.", "20.0", "`20`", " 20\n"])
def test_equivalent_numeric_answers(text):
    assert numeric_answer(text, 20)


@pytest.mark.parametrize(
    "text", ["20 or 30", "The answer is 20", "2", "twenty-one", "20\n30", ".20", "20.5"]
)
def test_numeric_answer_is_not_substring_matching(text):
    assert not numeric_answer(text, 20)


def test_whole_alias_ranking_and_missing_gallery_tokens():
    tok = arithmetic_tokenizer()
    _, cases = build_arithmetic(tok, 24, preflight_cases=4)
    case = cases[0]
    logits = torch.arange(len(tok)).float()
    ids = case["intermediate_token_ids"]
    result = readout_metrics(logits, case, tok)
    assert result["intermediate_rank"] == min(
        result["intermediate_alias_ranks"].values()
    )
    assert result["intermediate_best_token_id"] == max(ids)
    gallery = build_gallery(tok)
    assert len(gallery) == 10
    unavailable = next(c for c in gallery if c["id"] == "paper-three-step-raw-21")
    assert not unavailable["intermediate_token_ids"]
    assert readout_metrics(logits, unavailable, tok)["intermediate_rank"] is None
    typo = next(c for c in gallery if c["id"] == "typo-language-raw-language")
    assert solved_case("any continuation", typo) is None
    raw = next(c for c in gallery if c["id"] == "mars-color-raw-Mars")
    assert solved_case("red. It is rusty.", raw)
    assert not solved_case("reddish", raw)
    assert (
        case_ids(tok, raw).tolist()
        == tok(raw["user_prompt"], return_tensors="pt").input_ids.tolist()
    )


def _model_and_cases():
    tok = arithmetic_tokenizer()
    hf = tiny_gemma(vocab_size=len(tok))
    model = from_hf(hf, tok)
    return hf, model, build_arithmetic(tok, 24, preflight_cases=4)


def test_practice_resume_and_provenance(tmp_path):
    _, model, (practice, _) = _model_and_cases()
    calls = []

    def generate(input_ids, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return torch.cat(
            [input_ids, torch.tensor([[practice[0]["answer_token_id"]]])], dim=1
        )

    hf = SimpleNamespace(generate=generate)
    with pytest.raises(KeyboardInterrupt):
        benchmark.preflight(hf, model, practice, tmp_path, identity={"revision": "a"})
    assert len(list((tmp_path / "cases").glob("*.json"))) == 2
    result = benchmark.preflight(
        hf, model, practice, tmp_path, identity={"revision": "a"}
    )
    assert len(calls) == 5 and result["total"] == 4
    assert (
        benchmark.preflight(hf, model, practice, tmp_path, identity={"revision": "a"})
        == result
    )
    assert len(calls) == 5
    with pytest.raises(ValueError, match="changed"):
        benchmark.preflight(hf, model, practice, tmp_path, identity={"revision": "b"})


def _bundle(tmp_path, model):
    source = tmp_path / "fitted"
    source.mkdir()
    prompts = ["red blue green yellow black white orange purple pink brown silver gold"]
    metadata = dict(model_revision="test-revision", corpus_sha256="test-corpus")
    kwargs = dict(
        target_to_sources={2: [0, 1]},
        model_id=MODEL_ID,
        max_seq_len=12,
        skip_first=1,
        dim_batch=4,
        position_seed=0,
    )
    tj = fit_short_hop_atlas(
        model, prompts, **kwargs, metadata={**metadata, "backward_rule": "autograd"}
    )
    with relp_rules(model):
        tr = fit_short_hop_atlas(
            model,
            prompts,
            **kwargs,
            means=tj.means,
            metadata={**metadata, "backward_rule": RULE_VERSION},
        )
    lens = fit(
        model,
        prompts,
        source_layers=[0, 1],
        target_layer=model.n_layers - 1,
        skip_first=1,
        max_seq_len=12,
        dim_batch=4,
    )
    tj.save(str(source / "short_hop_atlas.pt"))
    tr.save(str(source / "tr_short_hop_atlas.pt"))
    expected = dict(
        **metadata,
        model_id=MODEL_ID,
        target_layer=model.n_layers - 1,
        position_reduction="future_sum",
        position_seed=0,
        skip_first=1,
        max_seq_len=12,
    )
    for name, rule in (("j_lens", "autograd"), ("r_lens", RULE_VERSION)):
        save_baseline(
            lens, source / (name + ".pt"), {**expected, "backward_rule": rule}
        )
    return source, tj, tr, lens


def test_arithmetic_and_raw_evaluation_gallery(tmp_path):
    _, model, (_, cases) = _model_and_cases()
    _, tj, tr, lens = _bundle(tmp_path, model)
    gallery = build_gallery(model.tokenizer)
    lookup = {
        tuple(case_ids(model.tokenizer, c)[0].tolist()): c for c in cases + gallery
    }

    def generate(input_ids, **kwargs):
        case = lookup[tuple(input_ids[0].tolist())]
        token = case["answer_token_id"] or model.tokenizer.eos_token_id
        return torch.cat([input_ids, torch.tensor([[token]])], dim=1)

    hf = SimpleNamespace(generate=generate)
    rows, gen = evaluate_cases(
        hf, model, tj, lens, lens, cases, tmp_path / "arithmetic", tr_atlas=tr
    )
    assert all(g["solved"] for g in gen)
    assert {r["wording"] for r in rows} == {"symbolic", "verbal"}
    qualitative, generated = evaluate_cases(
        hf, model, tj, lens, lens, gallery, tmp_path / "qualitative", tr_atlas=tr
    )
    assert len(qualitative) == 10 * 2 * 9
    # Render a representative example with all actual readouts, without GPU/downloads.
    write_gallery(
        gallery[:1],
        [r for r in qualitative if r["case_id"] == gallery[0]["id"]],
        generated[:1],
        tmp_path / "qualitative",
    )
    assert (tmp_path / "qualitative" / "GALLERY.md").exists()
    assert (tmp_path / "qualitative" / (gallery[0]["id"] + ".png")).exists()
    screening = dict(solved=32, total=32, accuracy=1, threshold=0.8, passed=True)
    write_benchmark_report(tmp_path, screening, cases, rows, gen, fit_n=32)
    report = (tmp_path / "REPORT.md").read_text()
    assert "wins" in report.lower() and "Wording robustness" in report
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["diagnostics"]) == {"symbolic", "verbal", "all", "failed"}
    assert summary["diagnostics"]["failed"]["innovation"]["j_lens"] is None


def test_runner_reuses_fits_and_resumes_main_cases(tmp_path, monkeypatch):
    _, model, _ = _model_and_cases()
    source, _, _, _ = _bundle(tmp_path, model)
    before = benchmark.artifact_hashes(source)
    practice, cases = build_arithmetic(model.tokenizer, 24)
    gallery = build_gallery(model.tokenizer)
    lookup = {
        tuple(case_ids(model.tokenizer, c)[0].tolist()): c
        for c in practice + cases + gallery
    }
    calls = []
    interrupted = False

    def generate(input_ids, **kwargs):
        nonlocal interrupted
        case = lookup[tuple(input_ids[0].tolist())]
        calls.append(case["id"])
        if case["id"] == cases[2]["id"] and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("test main evaluation interruption")
        token = case["answer_token_id"] or model.tokenizer.eos_token_id
        return torch.cat([input_ids, torch.tensor([[token]])], dim=1)

    monkeypatch.setattr(
        benchmark, "load_model", lambda *a: (SimpleNamespace(generate=generate), model)
    )
    monkeypatch.setattr(
        benchmark, "provenance", lambda *a: dict(model_revision="test-revision")
    )
    import experiments.short_hop.benchmark_report as reports

    monkeypatch.setattr(reports, "write_gallery", lambda *a: None)
    import experiments.short_hop.plot_results as plots

    monkeypatch.setattr(plots, "plot_results", lambda *a: None)
    args = SimpleNamespace(
        plan=False,
        preflight_only=False,
        reuse_from=str(source),
        output_dir=str(tmp_path / "new-run"),
        seed=0,
        model=MODEL_ID,
        revision="main",
        n_cases=24,
        max_new_tokens=16,
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(args)
    benchmark.run(args)
    # Gallery duplicates an input for the two tracked numbers; each is an explicit case.
    assert len(calls) == 32 + 10 + 24 + 1
    benchmark.run(args)
    assert len(calls) == 67
    assert benchmark.artifact_hashes(source) == before
    saved = json.loads((tmp_path / "new-run" / "progress.json").read_text())
    assert saved["overall_percent"] == 100
    assert saved["status"] == "complete"


def test_failed_screen_does_not_claim_comparison(tmp_path):
    screening = dict(solved=5, total=32, accuracy=5 / 32, threshold=0.8, passed=False)
    write_benchmark_report(tmp_path, screening, [], [], [], fit_n=32)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict"] == "PREFLIGHT_FAILED"
    assert not summary["arithmetic_evaluated"]
    assert "not run" in (tmp_path / "REPORT.md").read_text()


def test_failed_preflight_runs_gallery_but_not_main(tmp_path, monkeypatch):
    _, model, _ = _model_and_cases()
    source, _, _, _ = _bundle(tmp_path, model)

    def wrong_answer(input_ids, **kwargs):
        return torch.cat(
            [input_ids, torch.tensor([[model.tokenizer.eos_token_id]])], dim=1
        )

    monkeypatch.setattr(
        benchmark,
        "load_model",
        lambda *a: (SimpleNamespace(generate=wrong_answer), model),
    )
    monkeypatch.setattr(
        benchmark, "provenance", lambda *a: dict(model_revision="test-revision")
    )
    evaluated = []

    def evaluate(*args, **kwargs):
        evaluated.append(args[5])
        return [], []

    monkeypatch.setattr(benchmark, "evaluate_cases", evaluate)
    import experiments.short_hop.benchmark_report as reports

    monkeypatch.setattr(reports, "write_gallery", lambda *a: None)
    args = SimpleNamespace(
        plan=False,
        preflight_only=False,
        reuse_from=str(source),
        output_dir=str(tmp_path / "new-run"),
        seed=0,
        model=MODEL_ID,
        revision="main",
        n_cases=24,
        max_new_tokens=16,
    )
    benchmark.run(args)
    assert len(evaluated) == 1
    assert all(c["split"] == "qualitative" for c in evaluated[0])
    assert not (tmp_path / "new-run" / "arithmetic").exists()
    status = json.loads((tmp_path / "new-run" / "benchmark_config.json").read_text())
    assert status["status"] == "preflight_failed"


def test_runpod_benchmark_detaches_and_uses_new_directory(
    tmp_path, monkeypatch, capsys
):
    from scripts import runpod

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                host="example.invalid",
                user="root",
                port=2222,
                remote_dir="/workspace/jacobian-lens",
            )
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runpod.py",
            "--config",
            str(config),
            "--dry-run",
            "benchmark",
            "--",
            "--reuse-from",
            "runs/tjlens-a40",
        ],
    )
    runpod.main()
    command = capsys.readouterr().out
    assert "nohup flock" in command
    assert "experiments.short_hop.benchmark" in command
    assert "--output-dir runs/tjlens-arithmetic" in command
    assert "--reuse-from runs/tjlens-a40" in command


def test_raw_numeric_completion_prefix_boundaries():
    case = dict(
        generation_scoring="answer_prefix",
        numeric_answer=20,
        answer="20",
        answer_aliases=["20", "twenty"],
    )
    assert solved_case("20. Another fact.", case)
    assert solved_case("20.0\n", case)
    assert not solved_case("20.5 is the answer", case)
    assert not solved_case("200", case)
    assert not solved_case("20abc", case)
