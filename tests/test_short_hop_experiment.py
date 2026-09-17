import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.short_hop.build_eval_cases import (
    BRIDGES,
    COLORS,
    ENTITIES,
    VALUES,
    build_cases,
)
from experiments.short_hop.common import chat_ids, load_baseline, save_baseline
from experiments.short_hop.evaluate import evaluate_cases, token_rank
from experiments.short_hop.report import _index, _pick, make_report
from jlens import fit, fit_short_hop_atlas
from scripts.runpod import sync_command, validate

from .tiny import TinyDecoder


def word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["<bos>", "<eos>", "<unk>", "<pad>", *COLORS, *VALUES, *BRIDGES, *ENTITIES]
    vocab = {word: i for i, word in enumerate(words)}
    core = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = "{{ bos_token }}{% for m in messages %}user: {{ m['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return tokenizer


def test_deterministic_cases_split_groups_and_token_context():
    tok = word_tokenizer()
    cases = build_cases(tok)
    assert cases == build_cases(tok)
    assert len(cases) == 128
    assert sum(c["has_distractors"] for c in cases) == 64
    assert {c["task_hops"] for c in cases} == {2, 3}
    for first, second in zip(cases[::2], cases[1::2], strict=True):
        assert first["group_id"] == second["group_id"]
        assert first["split"] == second["split"]
        assert first["intermediate"] == second["control_intermediate"]
        assert (
            first["user_prompt"].split("Question:")[0]
            == second["user_prompt"].split("Question:")[0]
        )
        assert first["intermediate_token_id"] != first["answer_token_id"]
        ids = chat_ids(tok, first["user_prompt"])
        assert (ids == tok.bos_token_id).sum() == 1
    assert token_rank(torch.tensor([3.0, 3.0, 1.0]), 1) == 1


def test_evaluation_full_readouts_and_identity_difference(tmp_path):
    tokenizer = word_tokenizer()
    model = TinyDecoder(n_layers=4, d_model=8, vocab_size=len(tokenizer))
    model.tokenizer = tokenizer
    model.encode = lambda text, max_length=128: (
        tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids
    )
    for param in model.parameters():
        param.requires_grad_(False)
    prompts = ["red blue green yellow black white orange purple pink brown silver gold"]
    atlas = fit_short_hop_atlas(
        model,
        prompts,
        target_to_sources={2: [0, 1], 3: [1]},
        skip_first=1,
        max_seq_len=16,
        dim_batch=4,
    )
    baseline = fit(
        model, prompts, source_layers=[0, 1], skip_first=1, max_seq_len=16, dim_batch=4
    )
    cases = build_cases(tokenizer, n_cases=8)
    case_iter = iter(cases)

    def generate(input_ids, **kwargs):
        case = next(case_iter)
        return torch.cat([input_ids, torch.tensor([[case["answer_token_id"]]])], dim=1)

    rows, generations = evaluate_cases(
        SimpleNamespace(generate=generate),
        model,
        atlas,
        baseline,
        baseline,
        cases,
        tmp_path,
    )
    assert len(rows) == 8 * 3 * 7
    assert all(g["solved"] for g in generations)
    assert {r["readout"] for r in rows} == {
        "innovation",
        "transported",
        "identity",
        "actual_local",
        "source_logit_lens",
        "j_lens",
        "r_lens",
    }
    assert all(np.isfinite(r["intermediate_logit"]) for r in rows)
    report = make_report(rows, generations, tmp_path, fit_n=1)
    assert report["verdict"] == "INCONCLUSIVE"
    assert (tmp_path / "aggregate_metrics.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["completed"]


def synthetic_rows(tj_test_rank):
    rows, generations = [], []
    for index in range(96):
        split = "selection" if index < 32 else "test"
        generations.append(dict(solved=True))
        for name, source, rank in (
            ("innovation", 0, 2 if split == "selection" else tj_test_rank),
            ("innovation", 1, 20 if split == "selection" else 1),
            ("j_lens", 0, 32),
            ("r_lens", 0, 16),
            ("identity", 0, 64),
            ("source_logit_lens", 0, 128),
        ):
            rows.append(
                dict(
                    case_id=str(index),
                    group_id=str(index // 2),
                    split=split,
                    task_type="2_hop_lookup",
                    solved=True,
                    source_layer=source,
                    target_layer=3,
                    hop_length=3 - source,
                    readout=name,
                    intermediate_rank=rank,
                    intermediate_answer_margin=1,
                    intermediate_above_answer=True,
                    intermediate_above_control=True,
                    relative_state_error=0.1,
                    identity_state_error=0.2,
                    update_cosine=0.5,
                )
            )
    return rows, generations


def test_report_selection_is_frozen_and_verdict_uses_heldout(tmp_path):
    rows, generations = synthetic_rows(64)
    assert _pick(_index(rows), "innovation") == "innovation:0:3"
    report = make_report(rows, generations, tmp_path, fit_n=8)
    assert report["verdict"] == "NO_CLEAR_IMPROVEMENT"
    rows, generations = synthetic_rows(2)
    report = make_report(rows, generations, tmp_path, fit_n=8)
    assert report["verdict"] == "PROMISING"
    assert report["comparisons"]["j_lens"]["rank_improvement_factor"] == 16
    report = make_report([], [], tmp_path, fit_n=8)
    assert report["verdict"] == "INCONCLUSIVE"


def test_runpod_sync_arguments():
    config = dict(
        host="203.0.113.1",
        port=2222,
        user="root",
        remote_dir="/workspace/jacobian-lens",
    )
    validate(config)
    for direction in ("push", "pull"):
        args = sync_command(config, direction)
        assert "--no-owner" in args and "--no-group" in args
        assert "--delete" not in args
        assert "ssh -p 2222" in args
    with pytest.raises(ValueError):
        validate({**config, "host": "host; touch /tmp/oops"})
    with pytest.raises(ValueError):
        validate({**config, "remote_dir": "/"})


def test_baseline_provenance_roundtrip_and_rejection(tmp_path):
    from jlens import JacobianLens

    lens = JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3)
    path = tmp_path / "baseline.pt"
    save_baseline(lens, path, dict(model_id="fixture", backward_rule="autograd"))
    loaded = load_baseline(path, expected_metadata=dict(model_id="fixture"))
    torch.testing.assert_close(loaded.jacobians[0], lens.jacobians[0])
    with pytest.raises(ValueError, match="backward_rule"):
        load_baseline(path, expected_metadata=dict(backward_rule="relp"))


def test_runner_four_methods_interruption_and_resume(tmp_path, monkeypatch):
    import experiments.short_hop.run as runner
    import jlens.fitting as fitting
    from jlens import from_hf

    from .test_relp import tiny_gemma

    tokenizer = word_tokenizer()
    hf = tiny_gemma(vocab_size=len(tokenizer))
    model = from_hf(hf, tokenizer)
    monkeypatch.setattr(runner, "TARGET_TO_SOURCES", {2: [0, 1]})
    monkeypatch.setattr(runner, "load_model", lambda *args: (hf, model))
    monkeypatch.setattr(
        runner, "provenance", lambda *args: dict(model_revision="fixture")
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    prompts = [
        "red blue green yellow black white orange purple pink brown silver gold",
        "cat dog bird fish horse bear lion tiger mouse fox wolf snake",
    ]
    monkeypatch.setattr(
        runner, "build_corpus", lambda *args: [dict(text=text) for text in prompts]
    )
    # Plot rendering is exercised separately; this test verifies the orchestration.
    monkeypatch.setattr(runner, "plot_results", lambda *args: None)
    args = SimpleNamespace(
        seed=0,
        model="fixture",
        revision="fixture",
        max_seq_len=12,
        skip_first=1,
        dim_batch=4,
        position_reduction="self_hutchinson",
        output_dir=str(tmp_path),
        minutes=None,
        n_prompts=2,
        n_cases=8,
        hops=[2, 3],
        max_new_tokens=2,
        smoke=False,
        plan=False,
    )
    # Interrupt the second TR prompt, after all J/R/TJ work and one TR prompt
    # are saved. Resume must recompute only that unfinished TR prompt.
    original = fitting.jacobian_for_prompt
    calls = []

    def interrupt_once(*args, **kwargs):
        calls.append((args[1], kwargs["target_layer"]))
        if len(calls) == 8:
            raise KeyboardInterrupt("test interrupted TR prompt")
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "jacobian_for_prompt", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    interrupted = json.loads((tmp_path / "config.json").read_text())
    assert interrupted["status"] == "incomplete"
    checkpoint = torch.load(tmp_path / "checkpoints/tr/target_02.pt", weights_only=True)
    assert checkpoint["n_done"] == 1
    assert "forward" not in model.layers[1].mlp.__dict__
    runner.run(args)
    assert len(calls) == 9
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["status"] == "complete"
    assert config["chosen_n_prompts"] == 2
    assert config["minutes_this_invocation"] is None
    from jlens import ShortHopAtlas

    atlas = ShortHopAtlas.load(str(tmp_path / "short_hop_atlas.pt"))
    tr = ShortHopAtlas.load(str(tmp_path / "tr_short_hop_atlas.pt"))
    assert atlas.pairs == tr.pairs
    assert not torch.allclose(atlas.get(0, 2), tr.get(0, 2))
    for layer in atlas.means:
        torch.testing.assert_close(atlas.means[layer], tr.means[layer], rtol=0, atol=0)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["overall_percent"] == 100
    assert progress["status"] == "complete"
    assert all(
        progress["methods"][method]["percent"] == 100
        for method in ("J-Lens", "R-Lens", "TJ-Lens", "TR-Lens")
    )
    saved = [
        json.loads(line)
        for line in (tmp_path / "pair_results.jsonl").read_text().splitlines()
    ]
    assert len(saved) == 8 * 2 * 9
    assert {row["readout"] for row in saved} >= {"tr_transported", "tr_innovation"}
    assert "TR-Lens result:" in (tmp_path / "REPORT.md").read_text()
    assert json.loads((tmp_path / "evaluation_status.json").read_text())["completed"]
    baseline = load_baseline(
        tmp_path / "r_lens.pt", expected_metadata=dict(model_id="fixture")
    )
    assert baseline.n_prompts == 2
    monkeypatch.setattr(
        runner,
        "load_model",
        lambda *args: pytest.fail("completed resume reloaded model"),
    )
    runner.run(args)
    args.seed = 99
    with pytest.raises(ValueError, match="configuration differs"):
        runner.run(args)
