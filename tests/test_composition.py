import json
from types import SimpleNamespace

import pytest
import torch

from experiments.short_hop import composition
from experiments.short_hop.benchmark import artifact_hashes, fingerprint
from experiments.short_hop.benchmark_cases import build_arithmetic, case_ids
from experiments.short_hop.common import save_baseline, write_json, write_jsonl
from jlens import JacobianLens, ShortHopAtlas, from_hf
from jlens.relp import GEMMA4_RULE_VERSION

from .test_arithmetic_benchmark import arithmetic_tokenizer
from .test_gemma4 import tiny_gemma4


def test_composition_order_centering_and_actual_target():
    torch.manual_seed(12)
    h, actual, ms, mt = torch.randn(4, 3)
    maps = {name: torch.randn(3, 3) for name in ("k", "kr", "js", "jt", "rs", "rt")}
    result = composition.mapped_states(h, actual, maps, ms, mt)
    torch.testing.assert_close(result["tj_product"], (maps["jt"] @ maps["k"]) @ h)
    torch.testing.assert_close(result["tr_product"], (maps["rt"] @ maps["kr"]) @ h)
    torch.testing.assert_close(
        result["tj_centered"], maps["jt"] @ (mt + maps["k"] @ (h - ms))
    )
    torch.testing.assert_close(
        result["tr_centered"], maps["rt"] @ (mt + maps["kr"] @ (h - ms))
    )
    torch.testing.assert_close(result["j_target"], maps["jt"] @ actual)
    torch.testing.assert_close(result["r_source"], maps["rs"] @ h)
    assert not torch.allclose(result["tj_product"], (maps["k"] @ maps["jt"]) @ h)
    assert not torch.allclose(result["j_target"], result["tj_centered"])
    for name in ("k", "kr"):
        maps[name] = torch.eye(3)
    identity = composition.mapped_states(h, actual, maps, ms, ms)
    torch.testing.assert_close(identity["tj_product"], h @ maps["jt"].T)
    torch.testing.assert_close(identity["tj_centered"], identity["tj_product"])


def setup_source(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    hf = tiny_gemma4(vocab_size=len(tok))
    model = from_hf(hf, tok)
    _, cases = build_arithmetic(tok, 24)
    source = tmp_path / "source"
    source.mkdir()
    revision = "tiny-revision"
    metadata = dict(
        model_id="tiny-gemma4",
        model_revision=revision,
        corpus_sha256="corpus",
        target_layer=7,
        position_reduction="future_sum",
        position_seed=0,
        max_seq_len=16,
        skip_first=2,
    )
    fit_config = {
        k: v
        for k, v in metadata.items()
        if k not in ("target_layer", "position_reduction")
    }
    for name, rule, scale in (("j", "autograd", 1.0), ("r", GEMMA4_RULE_VERSION, 0.8)):
        lens = JacobianLens(
            {i: torch.randn(8, 8) * scale for i in (1, 5)}, n_prompts=32, d_model=8
        )
        save_baseline(
            lens, source / f"{name}_lens.pt", {**metadata, "backward_rule": rule}
        )
        atlas = ShortHopAtlas(
            {5: {1: torch.randn(8, 8) * scale}},
            {1: torch.ones(8), 5: torch.ones(8) * 2},
            n_prompts_by_target={5: 32},
            d_model=8,
            n_layers=8,
            model_id="tiny-gemma4",
            fit_config={**fit_config, "backward_rule": rule},
        )
        atlas.save(
            str(
                source
                / ("short_hop_atlas.pt" if name == "j" else "tr_short_hop_atlas.pt")
            )
        )
    generations = [
        dict(
            case_id=c["id"],
            group_id=c["group_id"],
            split=c["split"],
            task_type=c["task_type"],
            wording=c["wording"],
            expected=c["answer"],
            completion=c["answer"],
            solved=True,
            n_input_tokens=case_ids(tok, c).shape[1],
        )
        for c in cases
    ]
    write_jsonl(source / "arithmetic_cases.jsonl", cases)
    write_jsonl(source / "arithmetic/generation_results.jsonl", generations)
    software = dict(
        model_revision=revision,
        torch=str(torch.__version__),
        transformers="test",
        dtype="bfloat16",
        attention="eager",
        compile=False,
        source_sha256="code",
    )
    config = dict(
        status="complete",
        request=dict(model="tiny-gemma4"),
        software=software,
        datasets=dict(arithmetic_cases=fingerprint(cases)),
        target_to_sources={"5": [1]},
        fitted_artifact_hashes=artifact_hashes(source),
    )
    write_json(source / "benchmark_config.json", config)
    monkeypatch.setattr(composition, "load_model", lambda *args: (hf, model))
    monkeypatch.setattr(composition, "provenance", lambda *args: software)

    def no_generate(*args, **kwargs):
        raise AssertionError("composition must not regenerate answers")

    monkeypatch.setattr(hf, "generate", no_generate)
    args = SimpleNamespace(
        reuse_from=str(source),
        output_dir=str(tmp_path / "composed"),
        source_layer=1,
        target_layer=5,
        seed=0,
        plan=False,
    )
    return args, model, cases, config


def test_composition_replays_model_and_resumes_cases(tmp_path, monkeypatch):
    args, model, cases, config = setup_source(tmp_path, monkeypatch)
    original = composition.score_case
    attempts = []

    def interrupted(*pos, **kw):
        attempts.append(pos[1]["id"])
        if len(attempts) == 3:
            raise KeyboardInterrupt("test interruption")
        return original(*pos, **kw)

    monkeypatch.setattr(composition, "score_case", interrupted)
    with pytest.raises(KeyboardInterrupt):
        composition.run(args)
    output = tmp_path / "composed"
    assert len(list((output / "evaluation_cases").glob("*.json"))) == 2
    assert json.loads((output / "progress.json").read_text())["status"] == "incomplete"
    composition.run(args)
    n = sum(c["split"] == "test" for c in cases)
    assert len(attempts) == n + 1
    composition.run(args)
    assert len(attempts) == n + 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_test"] == summary["n_solved"] == n
    assert set(summary["methods"]) == set(composition.LABELS)
    assert len(summary["comparisons"]["solved"]) == 18
    assert "tj_centered_vs_j_target" in summary["comparisons"]["solved"]
    assert json.loads((output / "progress.json").read_text())["overall_percent"] == 100
    assert artifact_hashes(tmp_path / "source") == config["fitted_artifact_hashes"]
    args.target_layer = 6
    with pytest.raises(ValueError, match="not in the source run"):
        composition.run(args)


def test_composition_source_integrity_and_map_validation(tmp_path, monkeypatch):
    args, model, cases, config = setup_source(tmp_path, monkeypatch)
    maps, means, count = composition.prepare_maps(args.reuse_from, config, model, 1, 5)
    assert len(maps) == 6 and count == 32
    assert len(means) == 2
    tr_path = tmp_path / "source/tr_short_hop_atlas.pt"
    tr = ShortHopAtlas.load(str(tr_path))
    tr.means[1] += 1
    tr.save(str(tr_path))
    with pytest.raises(ValueError, match="means differ"):
        composition.prepare_maps(args.reuse_from, config, model, 1, 5)
    with pytest.raises(ValueError, match="artifacts changed"):
        composition.run(args)
    cases[0]["answer"] = "bad"
    write_jsonl(tmp_path / "source/arithmetic_cases.jsonl", cases)
    with pytest.raises(ValueError, match="definitions changed"):
        composition.load_inputs(args.reuse_from)


def test_runpod_composition_detaches(tmp_path, monkeypatch, capsys):
    from scripts import runpod

    config = tmp_path / "runpod.json"
    write_json(
        config,
        dict(
            host="example.invalid",
            port=2222,
            user="root",
            remote_dir="/workspace/jacobian-lens",
        ),
    )
    monkeypatch.setattr(
        "sys.argv", ["runpod.py", "--config", str(config), "--dry-run", "compose"]
    )
    runpod.main()
    command = capsys.readouterr().out
    assert "nohup flock" in command and "experiments.short_hop.composition" in command
    assert "--output-dir runs/gemma4-e4b-composed25" in command
