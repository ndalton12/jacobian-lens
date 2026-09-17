import pytest

from experiments.short_hop.fixed_layer import METHODS, summarize


def fixture():
    generations, rows = [], []
    for i in range(8):
        case = dict(
            case_id=str(i),
            group_id=str(i // 4),
            wording="symbolic" if i % 2 else "verbal",
            solved=i != 7,
            split="test",
            task_type="2_step_arithmetic",
        )
        generations.append(case)
        for method, rank in zip(METHODS, (20, 10, 5, 8), strict=True):
            rows.append(
                dict(
                    **case,
                    source_layer=25,
                    target_layer=33,
                    baseline_target=41,
                    readout=method,
                    intermediate_rank=rank,
                    intermediate_above_control=True,
                )
            )
    return rows, generations


def test_fixed_layer_uses_matched_cases_and_correct_rank_direction():
    rows, generations = fixture()
    # Neither a different layer nor a selection case may influence results.
    rows += [dict(rows[0], source_layer=24, intermediate_rank=1)]
    rows += [dict(rows[0], split="selection", intermediate_rank=1)]
    report, cases = summarize(rows, generations, 25, 33)
    assert report["n_solved"] == 7 and len(cases) == 8
    assert report["r_target_layer"] == 41
    result = report["comparisons"]["solved"]["transported_vs_r_lens"]
    assert result["rank_improvement_factor"] == pytest.approx(2)
    assert result["wins"] == 7 and result["n_groups"] == 2
    assert "meaningful" not in result and report["exploratory"]
    assert report["comparisons"]["all"]["transported_vs_r_lens"]["n_cases"] == 8
    assert report["tr_target_layer"] == 33
    tr = report["comparisons"]["solved"]["tr_transported_vs_transported"]
    assert tr["rank_improvement_factor"] == pytest.approx(5 / 8)
    assert tr["losses"] == 7
    assert len(report["comparisons"]["solved"]) == 6
    assert all(c["tr_transported"] == 8 for c in cases)


def test_fixed_layer_keeps_ties():
    rows, generations = fixture()
    for row in rows:
        row["intermediate_rank"] = 5
    report, _ = summarize(rows, generations, 25, 33)
    for result in report["comparisons"]["solved"].values():
        assert result["ties"] == 7 and result["wins"] == result["losses"] == 0


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "metadata", "rank"])
def test_fixed_layer_rejects_incomplete_or_inconsistent_readouts(corruption):
    rows, generations = fixture()
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "metadata":
        rows[0]["solved"] = False
    else:
        rows[0]["intermediate_rank"] = None
    with pytest.raises(ValueError):
        summarize(rows, generations, 25, 33)
