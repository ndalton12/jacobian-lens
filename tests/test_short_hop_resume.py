import json
from types import SimpleNamespace

import pytest
import torch

from experiments.short_hop.build_eval_cases import build_cases
from experiments.short_hop.evaluate import evaluate_cases
from experiments.short_hop.progress import Progress
from experiments.short_hop.report import make_report
from jlens import fit, fit_short_hop_atlas, from_hf
from jlens.relp import RULE_VERSION, relp_rules

from .test_relp import tiny_gemma
from .test_short_hop_experiment import synthetic_rows, word_tokenizer


def test_case_transactions_resume_without_regeneration(tmp_path):
    tokenizer = word_tokenizer()
    model = from_hf(tiny_gemma(vocab_size=len(tokenizer)), tokenizer)
    prompts = ["red blue green yellow black white orange purple pink brown silver gold"]
    kwargs = dict(target_to_sources={2: [0]}, skip_first=1, max_seq_len=12, dim_batch=4)
    tj = fit_short_hop_atlas(model, prompts, **kwargs)
    with relp_rules(model):
        tr = fit_short_hop_atlas(
            model,
            prompts,
            **kwargs,
            means=tj.means,
            metadata={"backward_rule": RULE_VERSION},
        )
    lens = fit(
        model, prompts, source_layers=[0], skip_first=1, max_seq_len=12, dim_batch=4
    )
    cases = build_cases(tokenizer, n_cases=8)
    calls = []

    def generate(input_ids, **kwargs):
        calls.append(input_ids.tolist())
        if len(calls) == 3:
            raise KeyboardInterrupt("test interruption during generation")
        return torch.cat(
            [input_ids, torch.tensor([[cases[0]["answer_token_id"]]])], dim=1
        )

    hf = SimpleNamespace(generate=generate)
    with pytest.raises(KeyboardInterrupt):
        evaluate_cases(hf, model, tj, lens, lens, cases, tmp_path, tr_atlas=tr)
    assert len(list((tmp_path / "evaluation_cases").glob("*.json"))) == 2
    assert not json.loads((tmp_path / "evaluation_status.json").read_text())[
        "completed"
    ]
    # A half-written temp transaction must never be treated as a completed case.
    (tmp_path / "evaluation_cases" / "abandoned.json.tmp").write_text("{partial")
    events = []
    rows, generations = evaluate_cases(
        hf,
        model,
        tj,
        lens,
        lens,
        cases,
        tmp_path,
        tr_atlas=tr,
        progress_callback=events.append,
    )
    assert len(calls) == 9  # 8 cases plus the interrupted attempt, not 11
    assert events[0]["done"] == 2 and events[-1]["done"] == 8
    assert len(rows) == 8 * 9 and len(generations) == 8
    assert len({(r["case_id"], r["readout"]) for r in rows}) == len(rows)
    rows_again, generations_again = evaluate_cases(
        hf, model, tj, lens, lens, cases, tmp_path, tr_atlas=tr
    )
    assert len(calls) == 9
    assert rows_again == rows and generations_again == generations
    with pytest.raises(ValueError, match="changed"):
        evaluate_cases(
            hf, model, tj, lens, lens, cases, tmp_path, tr_atlas=tr, max_new_tokens=13
        )
    tr.means[0] = tr.means[0] + 1
    with pytest.raises(ValueError, match="identical activation means"):
        evaluate_cases(hf, model, tj, lens, lens, cases, tmp_path, tr_atlas=tr)


def test_dimension_batch_progress_and_method_percent(tmp_path):
    from .tiny import TinyDecoder

    model = TinyDecoder(d_model=5)
    events = []
    fit(
        model,
        ["abcdefghijk"],
        source_layers=[0],
        target_layer=3,
        skip_first=1,
        max_seq_len=12,
        dim_batch=2,
        progress_callback=events.append,
    )
    passes = [event for event in events if event["event"] == "backward"]
    assert [e["pass_done"] for e in passes] == [1, 2, 3]
    assert all(e["pass_total"] == 3 and e["prompt_done"] == 0 for e in passes)
    assert events[-1]["event"] == "complete" and events[-1]["prompt_done"] == 1
    progress = Progress(
        tmp_path,
        [("TR-Lens/target_03", "TR-Lens", 1, 3), ("evaluation", "Evaluation", 2, 1)],
        interval=0,
    )
    receive = progress.callback("TR-Lens")
    for event in events:
        receive(event)
    saved = json.loads((tmp_path / "progress.json").read_text())
    assert saved["methods"]["TR-Lens"]["percent"] == 100
    assert saved["overall_percent"] == 60
    progress.update("evaluation", 2, force=True)
    progress.finish()
    assert (
        json.loads((tmp_path / "progress.json").read_text())["overall_percent"] == 100
    )


def test_tr_report_selects_without_test_leakage(tmp_path):
    rows, generations = synthetic_rows(8)
    tr_rows = []
    for row in rows:
        if row["readout"] == "innovation":
            # TR source 0 wins on selection; source 1 would win on test.
            rank = (
                (1 if row["split"] == "selection" else 2)
                if row["source_layer"] == 0
                else (32 if row["split"] == "selection" else 1)
            )
            tr_rows.append(
                {**row, "readout": "tr_innovation", "intermediate_rank": rank}
            )
    summary = make_report(rows + tr_rows, generations, tmp_path, fit_n=8)
    assert summary["tr_lens"]["selected_innovation"] == "tr_innovation:0:3"
    assert summary["tr_lens"]["comparisons"]["tj_lens"]["rank_improvement_factor"] == 4
    assert summary["tr_lens"]["verdict"] == "PROMISING"
    assert "TR vs TJ-Lens" in (tmp_path / "REPORT.md").read_text()
    incomplete = make_report(
        rows + tr_rows, generations, tmp_path, fit_n=8, completed=False
    )
    assert incomplete["tr_lens"]["verdict"] == "INCONCLUSIVE"
