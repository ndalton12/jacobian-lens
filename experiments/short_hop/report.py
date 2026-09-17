"""Predeclared held-out comparisons with group bootstrap uncertainty."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.short_hop.common import parser, read_jsonl, write_json

MIN_TEST_CASES = 24
MIN_TEST_GROUPS = 12
MIN_FIT_PROMPTS = 8
MIN_LOG2_GAIN = 1.0  # at least a 2x improvement in geometric mean rank
MIN_WIN_RATE = 0.60


def _candidate(row):
    if row["readout"] in ("j_lens", "r_lens", "source_logit_lens"):
        return f"{row['readout']}:{row['source_layer']}"
    return f"{row['readout']}:{row['source_layer']}:{row['target_layer']}"


def _index(rows):
    candidates = defaultdict(dict)
    for row in rows:
        candidates[_candidate(row)][row["case_id"]] = row
    return candidates


def _pick(candidates, readout):
    choices = []
    for key, records in candidates.items():
        values = [
            np.log2(r["intermediate_rank"])
            for r in records.values()
            if r["solved"] and r["split"] == "selection"
        ]
        if key.startswith(readout + ":") and values:
            choices.append((float(np.mean(values)), key))
    return min(choices)[1] if choices else None


def _bootstrap(values, groups, seed, samples=2000):
    # Each group is a pair of prompts with identical facts and different queries.
    unique = sorted(set(groups))
    if not unique:
        return [None, None]
    sums = np.array(
        [
            sum(v for v, g in zip(values, groups, strict=True) if g == key)
            for key in unique
        ]
    )
    counts = np.array([sum(g == key for g in groups) for key in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(samples, len(unique)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare(candidates, selected, baseline, seed=0, task_type=None, subset="solved"):
    if subset not in ("solved", "all", "failed"):
        raise ValueError("subset must be solved, all, or failed")
    if selected is None or baseline is None:
        return None
    if selected not in candidates or baseline not in candidates:
        return None
    left, right = candidates[selected], candidates[baseline]
    ids = sorted(
        i
        for i in left.keys() & right.keys()
        if (
            subset == "all"
            or (
                bool(left[i]["solved"]) == (subset == "solved")
                and bool(right[i]["solved"]) == (subset == "solved")
            )
        )
        and left[i]["split"] == "test"
        and (task_type is None or left[i]["task_type"] == task_type)
    )
    if not ids:
        return None
    ranks_t = np.array([left[i]["intermediate_rank"] for i in ids])
    ranks_b = np.array([right[i]["intermediate_rank"] for i in ids])
    gains = np.log2(ranks_b) - np.log2(ranks_t)
    groups = [left[i]["group_id"] for i in ids]
    ci = _bootstrap(gains, groups, seed)
    gain = float(gains.mean())
    win_rate = float((ranks_t < ranks_b).mean())
    return dict(
        tj_candidate=selected,
        baseline_candidate=baseline,
        n_cases=len(ids),
        n_groups=len(set(groups)),
        mean_log2_rank_gain=gain,
        rank_improvement_factor=2**gain,
        ci95_log2_gain=ci,
        win_rate=win_rate,
        wins=int((ranks_t < ranks_b).sum()),
        ties=int((ranks_t == ranks_b).sum()),
        losses=int((ranks_t > ranks_b).sum()),
        median_tj_rank=float(np.median(ranks_t)),
        median_baseline_rank=float(np.median(ranks_b)),
        meaningful=(gain >= MIN_LOG2_GAIN and ci[0] > 0 and win_rate >= MIN_WIN_RATE),
    )


def aggregate(rows, output_dir):
    groups = defaultdict(list)
    for row in rows:
        for subset in ["all", "solved"] if row["solved"] else ["all"]:
            key = (
                subset,
                row["split"],
                row["task_type"],
                row["source_layer"],
                row["target_layer"],
                row["readout"],
            )
            groups[key].append(row)
    result = []
    for key, values in sorted(groups.items()):
        record = dict(
            zip(
                (
                    "subset",
                    "split",
                    "task_type",
                    "source_layer",
                    "target_layer",
                    "readout",
                ),
                key,
                strict=True,
            )
        )
        record.update(
            n_cases=len(values),
            hop_length=key[4] - key[3],
            median_intermediate_rank=float(
                np.median([v["intermediate_rank"] for v in values])
            ),
            mean_log2_rank=float(
                np.mean([np.log2(v["intermediate_rank"]) for v in values])
            ),
        )
        for metric in (
            "intermediate_answer_margin",
            "intermediate_above_answer",
            "intermediate_above_control",
            "relative_state_error",
            "identity_state_error",
            "update_cosine",
        ):
            record["mean_" + metric] = float(np.mean([v[metric] for v in values]))
        result.append(record)
    if result:
        with (Path(output_dir) / "aggregate_metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    return result


def make_report(rows, generations, output_dir, *, seed=0, fit_n=0, completed=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate(rows, output)
    candidates = _index(rows)
    selected = {
        name: _pick(candidates, name)
        for name in (
            "innovation",
            "transported",
            "j_lens",
            "r_lens",
            "identity",
            "source_logit_lens",
            "tr_innovation",
            "tr_transported",
        )
    }
    comparisons = {
        name: compare(candidates, selected["innovation"], selected[name], seed)
        for name in ("j_lens", "r_lens", "identity", "source_logit_lens")
    }
    matched_source = {}
    if selected["innovation"]:
        source = selected["innovation"].split(":")[1]
        matched_source = {
            name: compare(candidates, selected["innovation"], f"{name}:{source}", seed)
            for name in ("j_lens", "r_lens")
        }
    transported = {
        name: compare(candidates, selected["transported"], selected[name], seed)
        for name in ("j_lens", "r_lens")
    }
    task_types = sorted({r["task_type"] for r in rows})
    by_task = {
        task: {
            name: compare(
                candidates, selected["innovation"], selected[name], seed, task
            )
            for name in ("j_lens", "r_lens")
        }
        for task in task_types
    }
    chosen = candidates.get(selected["innovation"], {})
    test_rows = [r for r in chosen.values() if r["solved"] and r["split"] == "test"]
    specificity = (
        float(np.mean([r["intermediate_above_control"] for r in test_rows]))
        if test_rows
        else None
    )
    state_wins = (
        float(
            np.mean(
                [
                    r["relative_state_error"] < r["identity_state_error"]
                    for r in test_rows
                ]
            )
        )
        if test_rows
        else None
    )
    j, r = comparisons["j_lens"], comparisons["r_lens"]
    enough = bool(
        completed
        and fit_n >= MIN_FIT_PROMPTS
        and j
        and r
        and j["n_cases"] >= MIN_TEST_CASES
        and j["n_groups"] >= MIN_TEST_GROUPS
    )
    if not enough:
        verdict = "INCONCLUSIVE"
        explanation = "Too little completed fitting or solved held-out data to judge meaningful improvement."
    elif j["meaningful"] and r["meaningful"] and specificity > 0.5:
        verdict = "PROMISING"
        explanation = "TJ-Lens shows a meaningful held-out intermediate-rank improvement over both J-Lens and R-Lens in this pilot."
    elif j["meaningful"] and specificity > 0.5:
        verdict = "MIXED"
        explanation = "TJ-Lens improves meaningfully over J-Lens here, but has not established an improvement over R-Lens."
    else:
        verdict = "NO_CLEAR_IMPROVEMENT"
        explanation = "This pilot does not show a meaningful TJ-Lens improvement over J-Lens under the predeclared criteria."
    tr_comparisons = {
        name: compare(candidates, selected["tr_innovation"], selected[other], seed)
        for name, other in (
            ("j_lens", "j_lens"),
            ("r_lens", "r_lens"),
            ("tj_lens", "innovation"),
        )
    }
    tr_test = [
        row
        for row in candidates.get(selected["tr_innovation"], {}).values()
        if row["solved"] and row["split"] == "test"
    ]
    tr_specificity = (
        float(np.mean([row["intermediate_above_control"] for row in tr_test]))
        if tr_test
        else None
    )
    tr_j, tr_r = tr_comparisons["j_lens"], tr_comparisons["r_lens"]
    tr_enough = bool(
        completed
        and fit_n >= MIN_FIT_PROMPTS
        and tr_j
        and tr_r
        and tr_j["n_cases"] >= MIN_TEST_CASES
        and tr_j["n_groups"] >= MIN_TEST_GROUPS
    )
    if not any(row["readout"] == "tr_innovation" for row in rows):
        tr_verdict = "NOT_INCLUDED"
    elif not tr_enough:
        tr_verdict = "INCONCLUSIVE"
    elif tr_j["meaningful"] and tr_r["meaningful"] and tr_specificity > 0.5:
        tr_verdict = "PROMISING"
    elif tr_j["meaningful"] and tr_specificity > 0.5:
        tr_verdict = "MIXED"
    else:
        tr_verdict = "NO_CLEAR_IMPROVEMENT"
    tr_summary = dict(
        verdict=tr_verdict,
        comparisons=tr_comparisons,
        selected_innovation=selected["tr_innovation"],
        selected_transported=selected["tr_transported"],
        query_control_win_fraction=tr_specificity,
        state_error_win_fraction=float(
            np.mean(
                [
                    row["relative_state_error"] < row["identity_state_error"]
                    for row in tr_test
                ]
            )
        )
        if tr_test
        else None,
        transported_state_comparisons={
            name: compare(candidates, selected["tr_transported"], selected[other], seed)
            for name, other in (
                ("j_lens", "j_lens"),
                ("r_lens", "r_lens"),
                ("tj_lens", "transported"),
            )
        },
        by_task={
            task: {
                name: compare(
                    candidates, selected["tr_innovation"], selected[other], seed, task
                )
                for name, other in (
                    ("j_lens", "j_lens"),
                    ("r_lens", "r_lens"),
                    ("tj_lens", "innovation"),
                )
            }
            for task in task_types
        },
    )
    summary = dict(
        verdict=verdict,
        explanation=explanation,
        completed=completed,
        fit_prompts=fit_n,
        selected_on_selection_split=selected,
        comparisons=comparisons,
        tr_lens=tr_summary,
        transported_state_comparisons=transported,
        matched_source_comparisons=matched_source,
        by_task=by_task,
        query_control_win_fraction=specificity,
        state_error_win_fraction=state_wins,
        solved_cases=sum(g["solved"] for g in generations),
        total_cases=len(generations),
        solve_rates_by_task={
            task: dict(
                total=sum(g.get("task_type") == task for g in generations),
                solved=sum(
                    g["solved"] and g.get("task_type") == task for g in generations
                ),
            )
            for task in task_types
        },
        criteria=dict(
            min_test_cases=MIN_TEST_CASES,
            min_test_groups=MIN_TEST_GROUPS,
            min_fit_prompts=MIN_FIT_PROMPTS,
            min_log2_rank_gain=MIN_LOG2_GAIN,
            min_win_rate=MIN_WIN_RATE,
            ci="95% paired group bootstrap; lower bound > 0",
            query_control="label beats paired control intermediate in > 50% of cases",
        ),
    )
    write_json(output / "summary.json", summary)
    lines = [
        f"TJ-Lens result: {verdict}",
        "",
        explanation,
        "",
        f"The base model solved {summary['solved_cases']}/{len(generations)} cases. Each lens used {fit_n} fitting prompts.",
        "The headline uses solved held-out cases. Layer choices were made on a separate selection split.",
        "",
    ]
    for name, label in (("j_lens", "J-Lens"), ("r_lens", "R-Lens")):
        comparison = comparisons[name]
        if comparison:
            lo, hi = comparison["ci95_log2_gain"]
            lines.append(
                f"Compared with {label}: intermediate-rank improvement factor {comparison['rank_improvement_factor']:.2f}x (above 1 is better) "
                f"(95% interval {2**lo:.2f}–{2**hi:.2f}x); TJ-Lens wins on {comparison['win_rate']:.0%} "
                f"of {comparison['n_cases']} solved test cases ({comparison['n_groups']} independent fact groups)."
            )
        else:
            lines.append(
                f"Compared with {label}: insufficient solved selection/test cases."
            )
    if test_rows:
        lines.extend(
            [
                "",
                f"The selected span is {selected['innovation']}. The relevant intermediate beats the paired control label in {specificity:.0%} of test cases.",
                f"Centered transport predicts the target state better than identity in {state_wins:.0%} of test cases.",
            ]
        )
    lines.extend(
        [
            "",
            "Secondary check: the centered transported-state readout (without subtracting identity).",
        ]
    )
    for name, label in (("j_lens", "J-Lens"), ("r_lens", "R-Lens")):
        comparison = transported[name]
        if comparison:
            lines.append(
                f"Compared with {label}: {comparison['rank_improvement_factor']:.2f}x geometric-rank improvement; "
                f"wins on {comparison['win_rate']:.0%} of solved test cases."
            )
        else:
            lines.append(f"Compared with {label}: insufficient data.")
    lines.extend(["", "Results by task (same layer choices, no further tuning):"])
    for task, comparison_set in by_task.items():
        parts = [
            f"{name}: {comparison['rank_improvement_factor']:.2f}x on {comparison['n_cases']} cases"
            for name, comparison in comparison_set.items()
            if comparison
        ]
        lines.append(
            f"{task}: "
            + ("; ".join(parts) if parts else "insufficient solved test cases")
        )
    if tr_verdict != "NOT_INCLUDED":
        lines.extend(
            [
                "",
                f"TR-Lens result: {tr_verdict}",
                "TR uses the same short-hop pairs, means, prompts, sign probes, and evaluation cases as TJ, with R-style backward rules.",
            ]
        )
        for name, label in (
            ("j_lens", "J-Lens"),
            ("r_lens", "R-Lens"),
            ("tj_lens", "TJ-Lens"),
        ):
            comparison = tr_comparisons[name]
            if comparison:
                lo, hi = comparison["ci95_log2_gain"]
                lines.append(
                    f"TR vs {label}: intermediate-rank improvement factor {comparison['rank_improvement_factor']:.2f}x "
                    f"(above 1 is better; 95% interval {2**lo:.2f}–{2**hi:.2f}x); "
                    f"wins on {comparison['win_rate']:.0%} of {comparison['n_cases']} solved test cases."
                )
            else:
                lines.append(f"TR vs {label}: insufficient solved selection/test data.")
        head_to_head = tr_comparisons["tj_lens"]
        if tr_enough and head_to_head:
            lines.append(
                "The R-style corrections meaningfully improve TJ's intermediate ranks in this pilot."
                if head_to_head["meaningful"] and tr_specificity > 0.5
                else "The R-style corrections have not established a meaningful improvement over TJ in this pilot."
            )
        if tr_test:
            lines.append(
                f"TR selected span: {selected['tr_innovation']}; relevance-control wins: {tr_specificity:.0%}; "
                f"target-state prediction beats identity on {tr_summary['state_error_win_fraction']:.0%} of cases."
            )
        lines.append(
            "TR transported-state comparisons and results by task are also saved in summary.json. "
            "R/TR maps use modified backward propagation and are not literal Jacobians; state prediction remains a diagnostic."
        )
    lines.extend(
        [
            "",
            "Meaningful = at least 2x geometric-rank improvement, positive 95% bootstrap lower bound, and wins on at least 60% of cases.",
            "At least 8 fitting prompts, 24 solved test cases, and 12 fact groups are required. These are pilot decision thresholds, not a universal standard.",
            "",
            "Caveats: this measures readout quality, not improved model reasoning or causal proof of computation. "
            + (
                "Arithmetic intermediates are not explicitly supplied; correct answers do not prove that the model used the labelled computation. "
                if task_types == ["2_step_arithmetic"]
                else "The labelled intermediates occur in the supplied facts. Query-switch controls help test relevance, but do not establish causality. "
            )
            + "The headline TJ/TR scores are logit differences; J/R scores are complete readouts. Transported-state comparisons are in summary.json. "
            "One corpus/sign seed cannot establish estimator stability; replicate any positive result with another seed. "
            "R-Lens adapts the published rules to the selected Gemma architecture; it is not a reproduction of the authors' larger-model scores.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(report)
    print(report)
    return summary


def main():
    p = parser(__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fit-prompts", type=int, required=True)
    args = p.parse_args()
    output = Path(args.run_dir)
    manifest_path = output / "evaluation_status.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rows = read_jsonl(output / "pair_results.jsonl")
    generations = read_jsonl(output / "generation_results.jsonl")
    completed = bool(
        manifest.get("completed")
        and manifest.get("rows") == len(rows)
        and manifest.get("expected_cases") == len(generations)
    )
    make_report(
        rows,
        generations,
        output,
        seed=args.seed,
        fit_n=args.fit_prompts,
        completed=completed,
    )


if __name__ == "__main__":
    main()
