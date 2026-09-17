"""Compare cached logit/R/TJ/TR readouts at one fixed source layer; no GPU needed."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.short_hop.common import read_jsonl, write_json
from experiments.short_hop.report import compare

METHODS = {
    "source_logit_lens": "Logit lens",
    "r_lens": "R-Lens",
    "transported": "TJ predicted-state",
    "tr_transported": "TR predicted-state",
}


def summarize(rows, generations, source, target, seed=0):
    expected = {g["case_id"]: g for g in generations if g["split"] == "test"}
    if not expected:
        raise ValueError("no test cases")
    candidates = {name: {} for name in METHODS}
    for row in rows:
        if (
            row["source_layer"] != source
            or row["target_layer"] != target
            or row["readout"] not in METHODS
            or row["split"] != "test"
        ):
            continue
        name, case = row["readout"], row["case_id"]
        if case not in expected or case in candidates[name]:
            raise ValueError("unknown or duplicate case")
        if any(row[k] != expected[case][k] for k in ("solved", "group_id", "wording")):
            raise ValueError("case metadata mismatch")
        rank = row["intermediate_rank"]
        if rank is None or not np.isfinite(rank) or rank < 1:
            raise ValueError("invalid intermediate rank")
        candidates[name][case] = row
    if any(set(records) != set(expected) for records in candidates.values()):
        raise ValueError("missing matched readouts; no cases may be silently dropped")
    if not any(g["solved"] for g in expected.values()):
        raise ValueError("no solved test cases")
    r_targets = {r["baseline_target"] for r in candidates["r_lens"].values()}
    if len(r_targets) != 1 or None in r_targets:
        raise ValueError("inconsistent R-Lens target")
    stats = {}
    for name, records in candidates.items():
        solved = [r for r in records.values() if r["solved"]]
        ranks = np.array([r["intermediate_rank"] for r in solved])
        stats[name] = dict(
            median_rank=float(np.median(ranks)),
            geometric_mean_rank=float(np.exp(np.log(ranks).mean())),
            top5_count=int((ranks <= 5).sum()),
            control_wins=sum(r["intermediate_above_control"] for r in solved),
        )
    comparisons = {}
    for subset in ("solved", "all"):
        comparisons[subset] = {}
        for left, right in (
            ("transported", "source_logit_lens"),
            ("transported", "r_lens"),
            ("r_lens", "source_logit_lens"),
            ("tr_transported", "source_logit_lens"),
            ("tr_transported", "r_lens"),
            ("tr_transported", "transported"),
        ):
            result = compare(candidates, left, right, seed=seed, subset=subset)
            # This is exploratory reuse of a previously inspected test set,
            # not a fresh confirmatory verdict or layer-selection experiment.
            result.pop("meaningful")
            comparisons[subset][f"{left}_vs_{right}"] = result
    per_case = [
        dict(
            case_id=case,
            group_id=expected[case]["group_id"],
            wording=expected[case]["wording"],
            solved=expected[case]["solved"],
            **{
                name: records[case]["intermediate_rank"]
                for name, records in candidates.items()
            },
        )
        for case in sorted(expected)
    ]
    return dict(
        exploratory=True,
        source_layer=source,
        tj_target_layer=target,
        tr_target_layer=target,
        r_target_layer=next(iter(r_targets)),
        n_test=len(expected),
        n_solved=sum(g["solved"] for g in expected.values()),
        seed=seed,
        methods=stats,
        comparisons=comparisons,
    ), per_case


def run(run_dir, output_dir, source=25, target=33, seed=0):
    run_dir, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError("write to a separate directory, outside the original run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    config = json.loads((run_dir / "benchmark_config.json").read_text())
    if config["status"] != "complete":
        raise ValueError("source benchmark is not complete")
    generations = read_jsonl(run_dir / "arithmetic/generation_results.jsonl")
    with (run_dir / "arithmetic/pair_results.jsonl").open() as handle:
        summary, cases = summarize(
            (json.loads(line) for line in handle if line.strip()),
            generations,
            source,
            target,
            seed,
        )
    summary.update(source_run=str(run_dir), source_software=config["software"])
    n = summary["n_solved"]
    lines = [
        f"# Fixed layer {source}: logit lens vs R-Lens vs TJ vs TR",
        "",
        f"Logit lens at {source}; R-Lens {source} → {summary['r_target_layer']}; "
        f"TJ/TR predicted-state {source} → {target}. No subtraction or layer search.",
        "",
        f"Same final input-token position and cached readouts; no new fitting or generation. "
        f"Headline: {n}/{summary['n_test']} correctly answered test questions.",
        "",
        "Exploratory follow-up chosen after inspecting earlier results, not independent confirmation. "
        "Intervals resample problem families together, including their paraphrases.",
        "",
        "| Method | Median intermediate rank ↓ | Geometric mean rank ↓ | Top 5 | Beats control label |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary["methods"].items():
        lines.append(
            f"| {METHODS[name]} | {values['median_rank']:g} | "
            f"{values['geometric_mean_rank']:.1f} | {values['top5_count']}/{n} | "
            f"{values['control_wins']}/{n} |"
        )
    lines += ["", "Rank 1 means the intermediate is the top word; lower is better.", ""]
    for result in summary["comparisons"]["solved"].values():
        left, right = (
            METHODS[result["tj_candidate"]],
            METHODS[result["baseline_candidate"]],
        )
        lo, hi = (2**v for v in result["ci95_log2_gain"])
        factor = result["rank_improvement_factor"]
        lines += [
            f"{left} vs {right}: {factor:.2f}× rank improvement (above 1 is better; "
            f"95% interval {lo:.2f}–{hi:.2f}×). "
            f"Wins / ties / losses: {result['wins']} / {result['ties']} / {result['losses']}.",
            "",
        ]
    lines += [
        "These measure intermediate visibility, not improved answers or proof of causal computation. "
        "All-test comparisons (including failures) are in summary.json; individual ranks are in cases.csv.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    with (output / "cases.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-layer", type=int, default=25)
    parser.add_argument("--target-layer", type=int, default=33)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.source_layer, args.target_layer, args.seed)


if __name__ == "__main__":
    main()
