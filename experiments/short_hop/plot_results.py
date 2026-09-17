"""Static diagnostic plots; all plotted layer searches are exploratory."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.short_hop.common import parser, read_jsonl


def plot_results(rows, output_dir, *, readout="innovation", method="TJ-Lens"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if readout == "innovation" and any(
        row["readout"] == "tr_innovation" for row in rows
    ):
        plot_results(
            rows, output / "tr_lens", readout="tr_innovation", method="TR-Lens"
        )
    grouped = defaultdict(list)
    for row in rows:
        if row["solved"] and row["split"] == "test" and row["readout"] == readout:
            grouped[row["target_layer"], row["hop_length"]].append(row)
    if not grouped:
        (output / "NO_PLOTS.txt").write_text(
            "No solved held-out cases; see REPORT.md.\n"
        )
        return
    targets = sorted({t for t, _ in grouped})
    hops = sorted({h for _, h in grouped})
    for metric, transform, label, filename in (
        (
            "intermediate_rank",
            np.log10,
            "log10 median intermediate rank (lower is better)",
            "intermediate_rank_heatmap.png",
        ),
        (
            "intermediate_answer_margin",
            lambda x: x,
            "mean innovation margin (intermediate − answer)",
            "innovation_margin_heatmap.png",
        ),
    ):
        values = np.full((len(targets), len(hops)), np.nan)
        for a, target in enumerate(targets):
            for b, hop in enumerate(hops):
                entries = grouped.get((target, hop), [])
                if entries:
                    scores = [v[metric] for v in entries]
                    values[a, b] = transform(
                        np.median(scores)
                        if metric == "intermediate_rank"
                        else np.mean(scores)
                    )
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        im = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set(
            xticks=range(len(hops)),
            xticklabels=hops,
            xlabel="Layer hop length",
            yticks=range(len(targets)),
            yticklabels=targets,
            ylabel="Target layer",
            title=f"{method} innovation · solved held-out cases (exploratory)",
        )
        fig.colorbar(im, ax=ax, label=label)
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for hop in hops:
        ys = [
            np.median([r["intermediate_rank"] for r in grouped[t, hop]])
            if grouped[t, hop]
            else np.nan
            for t in targets
        ]
        ax.plot(targets, ys, marker="o", label=f"hop {hop}")
    ax.set(
        xlabel="Target layer",
        ylabel="Median intermediate rank (lower is better)",
        yscale="log",
        title=f"{method} innovation · exploratory depth profile",
    )
    ax.legend()
    fig.savefig(output / "rank_by_target_layer.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for metric, label in (
        ("relative_state_error", method),
        ("identity_state_error", "Identity"),
    ):
        ax.plot(
            hops,
            [
                np.mean(
                    [
                        r[metric]
                        for (t, h), rr in grouped.items()
                        if h == hop
                        for r in rr
                    ]
                )
                for hop in hops
            ],
            marker="o",
            label=label,
        )
    ax.set(
        xlabel="Layer hop length",
        ylabel="Mean relative target-state error",
        title="Transport fidelity · solved held-out cases",
    )
    ax.legend()
    fig.savefig(output / "transport_error_by_hop.png", dpi=160)
    plt.close(fig)


def main():
    p = parser(__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    plot_results(read_jsonl(args.results), args.output_dir)


if __name__ == "__main__":
    main()
