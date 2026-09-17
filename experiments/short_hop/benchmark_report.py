"""Readable benchmark conclusions and fixed, non-cherry-picked paper examples."""

import json
from collections import defaultdict
from pathlib import Path

from experiments.short_hop.common import write_json
from experiments.short_hop.report import _index, compare, make_report

LABELS = {
    "j_lens": "J-Lens",
    "r_lens": "R-Lens",
    "transported": "TJ: predicted state",
    "innovation": "TJ: change only",
    "tr_transported": "TR: predicted state",
    "tr_innovation": "TR: change only",
}


def _span(candidate):
    if not candidate:
        return "no layer could be selected"
    parts = candidate.split(":")
    return (
        f"layer {parts[1]}"
        if len(parts) == 2
        else f"layers {parts[1]} → {parts[2]} ({'change-only' if 'innovation' in parts[0] else 'predicted-state'} readout)"
    )


def _comparison(label, value):
    if not value:
        return f"{label}: not enough matching cases."
    factor = value["rank_improvement_factor"]
    direction = (
        f"{factor:.2f}× closer to the top"
        if factor >= 1
        else f"{1 / factor:.2f}× farther down the ranking"
    )
    low, high = [2**x for x in value["ci95_log2_gain"]]
    return (
        f"{label}: the intermediate ranked {direction} on geometric average "
        f"(improvement factor {factor:.2f}×; 95% interval {low:.2f}–{high:.2f}×). "
        f"Wins / ties / losses: {value['wins']} / {value['ties']} / {value['losses']} "
        f"over {value['n_cases']} cases in {value['n_groups']} problem families."
    )


def _paired_controls(rows, selected):
    records = _index(rows).get(selected, {})
    grouped = defaultdict(list)
    for row in records.values():
        if row["split"] == "test":
            grouped[row["group_id"], row["wording"]].append(row)
    eligible = [
        rr for rr in grouped.values() if len(rr) == 2 and all(r["solved"] for r in rr)
    ]
    return dict(
        pairs=len(eligible),
        both_correct=sum(
            all(r["intermediate_above_control"] for r in rr) for rr in eligible
        ),
        definition="Both same-answer problems rank their own intermediate above the other problem's intermediate; both answers solved.",
    )


def write_benchmark_report(
    output,
    screening,
    cases,
    rows,
    generations,
    *,
    fit_n,
    seed=0,
    gallery_available=True,
):
    output = Path(output)
    lines = [
        "# Arithmetic lens comparison",
        "",
        f"Practice check: {screening['solved']}/{screening['total']} correct ({screening['accuracy']:.0%}); required at least {screening['threshold']:.0%}.",
        "",
    ]
    config_path = output / "benchmark_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    architecture = config.get("architecture", {})
    if config:
        lines += [
            f"Model: {config['request']['model']}. "
            f"Fitting prompts per method: {fit_n}. "
            f"Backward rule: {config.get('backward_rule', 'recorded in fitted artifacts')}.",
            "",
        ]
    if architecture.get("num_kv_shared_layers") or architecture.get(
        "hidden_size_per_layer_input"
    ):
        lines += [
            "Architecture caveat: this model has per-layer inputs and/or shared attention state. "
            "The maps measure the effect of changing a block's residual output, holding token-derived inputs "
            "and already-computed attention state fixed. Downstream shared-attention paths are differentiated. "
            "A residual vector is not the complete model state; transport fidelity is a diagnostic, not a full simulator.",
            "",
        ]
    if not screening["passed"]:
        lines += [
            "**No numerical verdict: the model failed the practice check.**",
            "The main arithmetic benchmark was not run. No problems were silently replaced or filtered to improve the score.",
            (
                "The fixed paper examples were still evaluated; see [qualitative comparisons](qualitative/GALLERY.md)."
                if gallery_available
                else "No fitting or qualitative lens evaluation was performed: this new model has no fitted maps yet."
            ),
            "Choose a new task/prompt design on practice data before another test. Repeating this command reuses the same practice answers.",
        ]
        write_json(
            output / "summary.json",
            dict(
                verdict="PREFLIGHT_FAILED",
                screening=screening,
                completed=True,
                arithmetic_evaluated=False,
            ),
        )
    else:
        summary = make_report(
            rows, generations, output / "arithmetic", seed=seed, fit_n=fit_n
        )
        selected = summary["selected_on_selection_split"]
        test = [r for r in generations if r["split"] == "test"]
        solved = sum(r["solved"] for r in test)
        lines += [
            f"**TJ: {summary['verdict']}. TR: {summary['tr_lens']['verdict']}.**",
            "",
            f"The model solved {summary['solved_cases']}/{len(generations)} main questions, including {solved}/{len(test)} held-out questions.",
            "The headline compares only correctly answered test questions. Choices of layers/hops used separate selection problems.",
            "Each problem family contains two problems with the same final answer but different intermediates, each in two wordings. These are clustered together for uncertainty estimates.",
            "",
            "## What the numbers mean",
            "",
            "For (2 + 3) × 4, the answer is 20 and the intermediate is 5. Rank 1 means that 5 (or an accepted whole-token spelling such as five) is the lens's highest-scoring word. Lower ranks are better.",
            "An improvement factor of 2× means half the baseline's rank on geometric average; 0.5× means twice as far down the ranking. This is not answer accuracy. A win means a strictly better intermediate rank; ties are shown separately.",
            "Predicted-state readouts inspect the transported state. Change-only readouts subtract the no-change baseline's word scores. They are distinct measurements, not interchangeable probabilities.",
            "",
            "## Held-out comparison",
            "",
            f"TJ selected {_span(selected['innovation'])}; TR selected {_span(selected['tr_innovation'])}.",
            f"Baselines selected J {_span(selected['j_lens'])} and R {_span(selected['r_lens'])}.",
            "",
        ]
        for method, results in (
            ("TJ", summary["comparisons"]),
            ("TR", summary["tr_lens"]["comparisons"]),
        ):
            for baseline, value in results.items():
                if baseline in ("j_lens", "r_lens", "tj_lens"):
                    lines.append(
                        _comparison(
                            f"{method} vs {baseline.replace('_lens', '').upper()}",
                            value,
                        )
                    )
        if summary["verdict"] == "INCONCLUSIVE":
            lines += [
                "",
                "Inconclusive means the minimum sample requirement was not met—not that the methods performed equally. The direction of the numbers above still matters.",
            ]
        lines += ["", "## Predicted-state comparison (without subtraction)", ""]
        for method, comparisons in (
            ("TJ", summary["transported_state_comparisons"]),
            ("TR", summary["tr_lens"]["transported_state_comparisons"]),
        ):
            for baseline, value in comparisons.items():
                lines.append(_comparison(f"{method} vs {baseline}", value))
        diagnostics = {}
        lines += [
            "",
            "## Wording sensitivity and failures",
            "",
            "These checks keep the same globally selected layers; there is no new tuning for a wording or for failed answers.",
        ]
        for subset_name, subset_rows, subset in (
            ("symbolic", [r for r in rows if r["wording"] == "symbolic"], "solved"),
            ("verbal", [r for r in rows if r["wording"] == "verbal"], "solved"),
            ("all", rows, "all"),
            ("failed", rows, "failed"),
        ):
            candidates = _index(subset_rows)
            diagnostics[subset_name] = {}
            lines += ["", f"{subset_name.capitalize()} test questions:"]
            for method in (
                "innovation",
                "tr_innovation",
                "transported",
                "tr_transported",
            ):
                diagnostics[subset_name][method] = {}
                for baseline in ("j_lens", "r_lens"):
                    result = compare(
                        candidates,
                        selected[method],
                        selected[baseline],
                        seed,
                        subset=subset,
                    )
                    diagnostics[subset_name][method][baseline] = result
                    if method in ("innovation", "tr_innovation"):
                        lines.append(
                            _comparison(
                                f"{LABELS[method]} vs {LABELS[baseline]}", result
                            )
                        )
        sensitivity = {}
        lines += [
            "",
            "Wording robustness (descriptive, not an extra significance test):",
        ]
        for method in ("innovation", "tr_innovation", "transported", "tr_transported"):
            values = [
                diagnostics[w][method][b]
                for w in ("symbolic", "verbal")
                for b in ("j_lens", "r_lens")
            ]
            conclusion = (
                "insufficient data"
                if any(v is None for v in values)
                else "better than both baselines in both wordings"
                if all(v["rank_improvement_factor"] > 1 for v in values)
                else "does not improve on both baselines in both wordings"
            )
            sensitivity[method] = conclusion
            lines.append(f"{LABELS[method]}: {conclusion}.")
        controls = {
            method: _paired_controls(rows, selected[method]) for method in LABELS
        }
        lines += [
            "",
            "## Same-answer controls",
            "",
            "Do both problems reveal their own intermediate even though they end at the same answer?",
            "The control is the other problem's intermediate, not necessarily an irrelevant number: it may also occur as an operand in this problem. This checks the labelled intermediate's specificity, not causal necessity.",
        ]
        for method, result in controls.items():
            lines.append(
                f"{LABELS[method]}: both intermediates distinguished in {result['both_correct']}/{result['pairs']} solved pairs (wordings are repeated measurements, not independent pairs)."
            )
        summary.update(
            screening=screening,
            diagnostics=diagnostics,
            wording_robustness=sensitivity,
            same_answer_controls=controls,
            arithmetic_evaluated=True,
            dataset_size=len(cases),
        )
        write_json(output / "summary.json", summary)
        lines += [
            "",
            "Meaningful improvement retains the pilot rule: at least 2× geometric-rank improvement, 95% group-bootstrap interval above 1×, and wins on at least 60% of cases; at least 24 solved test prompts, 12 problem families and 8 fitting prompts.",
            "All aliases and examples were fixed before looking at lens scores. Each concept uses its best-ranked complete single-token alias, identically for every lens; per-alias ranks are retained in pair_results.jsonl. Alias counts can differ across concepts.",
            "Correct arithmetic does not prove the model used the labelled intermediate. Readout quality is not improved model reasoning or causal evidence. This is a second, revised experiment, not a replacement for the original negative result.",
            "",
            "[Paper-example gallery](qualitative/GALLERY.md) · [Detailed metrics](arithmetic/REPORT.md)",
        ]
        from experiments.short_hop.plot_results import plot_results

        plot_results(rows, output / "arithmetic" / "plots")
    text = "\n\n".join(line for line in lines if line) + "\n"
    (output / "REPORT.md").write_text(text)
    print(text)


def _escape(text):
    return str(text).replace("|", "\\|").replace("\n", "\\n")


def write_gallery(cases, rows, generations, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated = {g["case_id"]: g for g in generations}
    lines = [
        "# Fixed paper-example gallery",
        "",
        "Examples were chosen before viewing any lens results. Failures and missing token representations are shown, not removed. These examples do not select benchmark layers or contribute to the arithmetic verdict.",
        "",
        "Original means the upstream raw text, read at its final input token. Chat adaptations are separately labelled and change the input context. Raw completions are checked by expected answer prefix; chat answers by exact answer. The raw typo vignette has no answer-accuracy test.",
        "",
        "These are qualitative transfers to Gemma, not reproductions of the paper's larger-model results. J/R curves use source layer; TJ/TR curves use target layer with separate lines for each fitted hop length. No per-example best layer is selected.",
        "",
    ]
    for case in cases:
        rr = [r for r in rows if r["case_id"] == case["id"]]
        generation = generated[case["id"]]
        pairs = sorted({(r["source_layer"], r["target_layer"]) for r in rr})
        # Structural choice only: middle fitted pair, shared by every method.
        source, target = pairs[len(pairs) // 2]
        status = (
            "not scored"
            if generation["solved"] is None
            else "correct"
            if generation["solved"]
            else "incorrect"
        )
        lines += [
            f"## {case['id']}",
            "",
            f"Source: {case['source']}. {'Chat adaptation' if case['adaptation'] else 'Original raw prompt'}; readout: {case['readout_position']}.",
            "",
            "```text",
            case["user_prompt"],
            "```",
            "",
            f"Model completion: `{_escape(generation['completion'])}`. Expected: `{case['answer']}` — **{status}**.",
            f"Tracked intermediate: **{case['intermediate']}**. Spellings considered: {', '.join(case['intermediate_aliases'])}. Available whole-token variants: {len(case['intermediate_token_ids'])}.",
            "",
        ]
        if not case["intermediate_token_ids"]:
            lines += [
                "**Rank unavailable:** none of the predeclared whole-concept spellings is a single token for this tokenizer. We do not substitute the first digit/token. Top-word comparisons are still shown.",
                "",
            ]
        lines += [
            f"Top words at the fixed representative pair {source} → {target} (J/R use the same source {source}):",
            "",
            "| Readout | Intermediate rank | Top 10 words |",
            "| --- | ---: | --- |",
        ]
        for method, label in LABELS.items():
            row = next(
                r
                for r in rr
                if r["source_layer"] == source
                and r["target_layer"] == target
                and r["readout"] == method
            )
            words = ", ".join(_escape(repr(t["token"])) for t in row["top_tokens"][:10])
            lines.append(
                f"| {label} | {row['intermediate_rank'] or 'unavailable'} | {words} |"
            )
        fig, axes = plt.subplots(
            2, 3, figsize=(13, 7), sharey=True, layout="constrained"
        )
        largest_rank = max(
            (
                r["intermediate_rank"]
                for r in rr
                if r["readout"] in LABELS and r["intermediate_rank"] is not None
            ),
            default=1,
        )
        for ax, (method, label) in zip(axes.flat, LABELS.items(), strict=True):
            selected = [
                r
                for r in rr
                if r["readout"] == method and r["intermediate_rank"] is not None
            ]
            if method in ("j_lens", "r_lens"):
                points = {r["source_layer"]: r["intermediate_rank"] for r in selected}
                ax.plot(sorted(points), [points[x] for x in sorted(points)], marker="o")
                ax.set_xticks(sorted(points))
                ax.set_xlabel("Source layer")
            else:
                for hop in sorted({r["hop_length"] for r in selected}):
                    values = sorted(
                        (r["target_layer"], r["intermediate_rank"])
                        for r in selected
                        if r["hop_length"] == hop
                    )
                    ax.plot(
                        [x for x, _ in values],
                        [y for _, y in values],
                        marker="o",
                        label=f"hop {hop}",
                    )
                ax.set_xlabel("Target layer")
                ax.set_xticks(sorted({r["target_layer"] for r in selected}))
                if selected:
                    ax.legend(fontsize=7)
            ax.set_title(label)
            if selected:
                ax.set_yscale("log")
                ax.set_ylim(0.8, largest_rank * 1.3)
            else:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "No whole single-token alias",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set_ylabel("Intermediate rank (lower is better)")
        fig.suptitle(f"{case['id']} · model answer {status}")
        filename = case["id"] + ".png"
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
        lines += ["", f"![Intermediate ranks]({filename})", ""]
    lines += [
        "All fitted pairs, top-20 readouts and per-alias ranks are saved in pair_results.jsonl. Generation results and exact case definitions are saved alongside the gallery (definitions in ../gallery_cases.jsonl)."
    ]
    (output / "GALLERY.md").write_text("\n".join(lines) + "\n")
