"""Fixed-source composed J/R readouts, reusing maps and frozen benchmark cases.

Only forward passes are run. No fitting, generation, layer selection, or
change-only subtraction. Product and centered-state compositions are separate.
"""

import csv
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from experiments.short_hop.benchmark import artifact_hashes, fingerprint, load_bundle
from experiments.short_hop.benchmark_cases import case_ids, prepare_case, solved_case
from experiments.short_hop.common import (
    file_hash,
    load_model,
    parser,
    provenance,
    read_jsonl,
    seed_all,
    write_json,
    write_jsonl,
)
from experiments.short_hop.evaluate import readout_metrics
from experiments.short_hop.progress import Progress
from experiments.short_hop.report import compare
from experiments.short_hop.run import _handle_termination
from jlens import ActivationRecorder, ShortHopAtlas
from jlens.relp import rule_version

LABELS = {
    "j_source": "J at source",
    "r_source": "R at source",
    "j_target": "J at actual target",
    "r_target": "R at actual target",
    "tj_product": "TJ: matrix product",
    "tr_product": "TR: matrix product",
    "tj_centered": "TJ: centered predicted-state",
    "tr_centered": "TR: centered predicted-state",
    "logit_source": "Logit lens at source",
    "logit_target": "Logit lens at actual target",
    "tj_local": "Old TJ: transport then logit lens",
    "tr_local": "Old TR: transport then logit lens",
}
BASELINES = ("j_source", "r_source", "j_target", "r_target")
COMPOSED = ("tj_product", "tr_product", "tj_centered", "tr_centered")
CAVEAT = (
    "Exploratory follow-up on previously inspected test questions, not independent "
    "confirmation. The short-hop maps are same-position Hutchinson estimates; the "
    "J/R maps sum effects across future positions. Gemma shared KV and per-layer "
    "inputs are additional paths. These are compositions of averaged readout maps, "
    "not a clean measurement of E[BA] versus E[B]E[A] for complete Jacobians. "
    "R/TR use modified backward rules, not literal derivatives."
)


def mapped_states(h_source, h_target, maps, mean_source, mean_target):
    """Row vectors: h @ K.T @ J.T = h @ (J @ K).T.

    Apply no normalization/nonlinearity between the maps; unembedding happens
    exactly once after the composed mapping. Centering changes the first hop
    only, matching the earlier predicted-state definition.
    """
    h_source, h_target = h_source.float(), h_target.float()
    tj_raw = h_source @ maps["k"].T
    tr_raw = h_source @ maps["kr"].T
    tj = mean_target + (h_source - mean_source) @ maps["k"].T
    tr = mean_target + (h_source - mean_source) @ maps["kr"].T
    return {
        "j_source": h_source @ maps["js"].T,
        "r_source": h_source @ maps["rs"].T,
        "j_target": h_target @ maps["jt"].T,
        "r_target": h_target @ maps["rt"].T,
        "tj_product": tj_raw @ maps["jt"].T,
        "tr_product": tr_raw @ maps["rt"].T,
        "tj_centered": tj @ maps["jt"].T,
        "tr_centered": tr @ maps["rt"].T,
        "logit_source": h_source,
        "logit_target": h_target,
        "tj_local": tj,
        "tr_local": tr,
    }


def load_inputs(directory):
    directory = Path(directory)
    config = json.loads((directory / "benchmark_config.json").read_text())
    if config.get("status") != "complete":
        raise ValueError("source benchmark must be complete")
    all_cases = read_jsonl(directory / "arithmetic_cases.jsonl")
    if fingerprint(all_cases) != config["datasets"]["arithmetic_cases"]:
        raise ValueError("source case definitions changed")
    generations = read_jsonl(directory / "arithmetic/generation_results.jsonl")
    by_id = {g["case_id"]: g for g in generations}
    if len(by_id) != len(generations) or set(by_id) != {c["id"] for c in all_cases}:
        raise ValueError("missing or duplicate source generations")
    cases = [c for c in all_cases if c["split"] == "test"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("test cases must be nonempty and unique")
    for case in cases:
        g = by_id[case["id"]]
        if any(g[k] != case[k] for k in ("group_id", "split", "wording", "task_type")):
            raise ValueError("source generation metadata mismatch")
        if g["expected"] != case["answer"] or g["solved"] != solved_case(
            g["completion"], case
        ):
            raise ValueError("source answer labels do not match completions")
    return config, cases, by_id


def prepare_maps(directory, config, model, source, target):
    atlas = ShortHopAtlas.load(str(Path(directory) / "short_hop_atlas.pt"))
    revision = config["software"]["model_revision"]
    if (
        atlas.model_id != config["request"]["model"]
        or atlas.fit_config.get("model_revision") != revision
        or atlas.d_model != model.d_model
        or atlas.n_layers != model.n_layers
    ):
        raise ValueError("atlas does not match source model/revision")
    j, r, tr = load_bundle(directory, atlas, model, atlas.model_id, revision)
    if (source, target) not in atlas.pairs or (source, target) not in tr.pairs:
        raise ValueError("requested short-hop pair is not fitted")
    if not {source, target} <= set(j.source_layers) & set(r.source_layers):
        raise ValueError("J/R maps must exist at both source and target")
    if (
        tr.model_id != atlas.model_id
        or tr.d_model != atlas.d_model
        or tr.n_layers != atlas.n_layers
    ):
        raise ValueError("TR architecture mismatch")
    for key in (
        "model_revision",
        "corpus_sha256",
        "prompts_sha256",
        "position_seed",
        "max_seq_len",
        "skip_first",
    ):
        if tr.fit_config.get(key) != atlas.fit_config.get(key):
            raise ValueError(f"TR/TJ mismatch: {key}")
    if (
        atlas.position_reduction != "self_hutchinson"
        or tr.position_reduction != atlas.position_reduction
        or tr.fit_config.get("backward_rule") != rule_version(model)
    ):
        raise ValueError("unexpected short-hop estimator/backward rule")
    counts = {
        atlas.n_prompts_by_target[target],
        tr.n_prompts_by_target[target],
        j.n_prompts,
        r.n_prompts,
    }
    if len(counts) != 1:
        raise ValueError("all maps must use the same fitting count")
    if not all(torch.equal(atlas.means[i], tr.means[i]) for i in (source, target)):
        raise ValueError("TR/TJ means differ")
    maps = dict(
        k=atlas.get(source, target),
        kr=tr.get(source, target),
        js=j.jacobians[source],
        jt=j.jacobians[target],
        rs=r.jacobians[source],
        rt=r.jacobians[target],
    )
    maps = {name: value.to(model.input_device).float() for name, value in maps.items()}
    means = [atlas.means[i].to(model.input_device).float() for i in (source, target)]
    return maps, means, next(iter(counts))


@torch.no_grad()
def score_case(model, case, generation, maps, means, source, target):
    if prepare_case(model.tokenizer, case) != case:
        raise ValueError("tokenizer/aliases changed from the original run")
    ids = case_ids(model.tokenizer, case).to(model.input_device)
    if ids.shape[1] != generation["n_input_tokens"]:
        raise ValueError("prompt tokenization changed")
    with ActivationRecorder(model.layers, at=[source, target]) as recorder:
        model.forward(ids)
        states = mapped_states(
            recorder.activations[source][0, -1],
            recorder.activations[target][0, -1],
            maps,
            *means,
        )
    logits = model.unembed(torch.stack(list(states.values()))).float().cpu()
    rows = []
    for name, scores in zip(states, logits, strict=True):
        rows.append(
            dict(
                case_id=case["id"],
                group_id=case["group_id"],
                split=case["split"],
                task_type=case["task_type"],
                wording=case["wording"],
                solved=generation["solved"],
                source_layer=source,
                target_layer=target,
                readout=name,
                **readout_metrics(scores, case, model.tokenizer),
            )
        )
    return rows


def write_report(rows, cases, generations, output, request):
    index = {name: {} for name in LABELS}
    for row in rows:
        name, case_id = row["readout"], row["case_id"]
        if case_id in index[name]:
            raise ValueError("duplicate result")
        index[name][case_id] = row
    expected = {c["id"] for c in cases}
    if any(set(records) != expected for records in index.values()):
        raise ValueError("incomplete matched results")
    solved = [c["id"] for c in cases if generations[c["id"]]["solved"]]
    if not solved:
        raise ValueError("no solved cases for comparison")
    stats = {}
    for name, records in index.items():
        ranks = np.array([records[i]["intermediate_rank"] for i in solved])
        stats[name] = dict(
            median_rank=float(np.median(ranks)),
            geometric_mean_rank=float(np.exp(np.log(ranks).mean())),
            top5=int((ranks <= 5).sum()),
            control_wins=sum(records[i]["intermediate_above_control"] for i in solved),
        )
    pairs = [(name, baseline) for name in COMPOSED for baseline in BASELINES]
    pairs += [("tr_product", "tj_product"), ("tr_centered", "tj_centered")]
    comparisons = {
        subset: {
            f"{a}_vs_{b}": compare(index, a, b, seed=request["seed"], subset=subset)
            for a, b in pairs
        }
        for subset in ("solved", "all")
    }
    for values in comparisons.values():
        for result in values.values():
            result.pop("meaningful")  # no confirmatory label on a post-hoc follow-up
    summary = dict(
        request=request,
        exploratory=True,
        n_test=len(cases),
        n_solved=len(solved),
        n_groups=len({generations[i]["group_id"] for i in solved}),
        methods=stats,
        comparisons=comparisons,
        caveat=CAVEAT,
    )
    output = Path(output)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "pair_results.jsonl", rows)
    with (output / "cases.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "prompt",
                "intermediate",
                "answer",
                "solved",
                *LABELS,
            ],
        )
        writer.writeheader()
        for case in cases:
            i = case["id"]
            writer.writerow(
                dict(
                    case_id=i,
                    prompt=case["user_prompt"],
                    intermediate=case["intermediate"],
                    answer=case["answer"],
                    solved=generations[i]["solved"],
                    **{
                        name: records[i]["intermediate_rank"]
                        for name, records in index.items()
                    },
                )
            )
    s, t = request["source_layer"], request["target_layer"]
    lines = [
        f"# Composed TJ/TR: {s} → {t} → final",
        "",
        f"Compared on {len(solved)}/{len(cases)} correctly answered existing test questions. "
        "Same final input token, fixed layers, no new fitting or generation.",
        "",
        f"J/R at {s} read the actual source state. J/R at {t} read the actual later state "
        "after the model has done more computation; they are diagnostic baselines, not equal-compute alternatives.",
        "",
        "Matrix-product TJ = unembed(J_target @ K @ h_source); TR = unembed(R_target @ K_R @ h_source). "
        "Centered versions replace K @ h_source with mean_target + K @ (h_source − mean_source). "
        "There is no normalization between maps and no change-only subtraction.",
        "",
        "## Intermediate visibility",
        "",
        "Lower rank is better; rank 1 means the intermediate is the top vocabulary entry.",
        "",
        "| Readout | Median rank ↓ | Geometric mean rank ↓ | Top 5 | Beats control |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stat in stats.items():
        lines.append(
            f"| {LABELS[name]} | {stat['median_rank']:g} | {stat['geometric_mean_rank']:.1f} | "
            f"{stat['top5']}/{len(solved)} | {stat['control_wins']}/{len(solved)} |"
        )
    lines += [
        "",
        "## Comparisons",
        "",
        "Improvement factors above 1 favor the composed readout. "
        "Intervals are 95% problem-family bootstrap intervals, keeping paraphrases together.",
        "",
    ]
    for key, result in comparisons["solved"].items():
        a, b = key.split("_vs_")
        lo, hi = [2**v for v in result["ci95_log2_gain"]]
        lines += [
            f"{LABELS[a]} vs {LABELS[b]}: **{result['rank_improvement_factor']:.2f}×** "
            f"({lo:.2f}–{hi:.2f}×); wins/ties/losses "
            f"{result['wins']}/{result['ties']}/{result['losses']}.",
            "",
        ]
    lines += ["## Plain-language takeaways", ""]
    for name in COMPOSED:
        result = comparisons["solved"][
            f"{name}_vs_{'j_source' if name.startswith('tj') else 'r_source'}"
        ]
        factor = result["rank_improvement_factor"]
        lines += [
            f"{LABELS[name]} puts the intermediate "
            + (
                f"{factor:.2f}× closer to the top"
                if factor >= 1
                else f"{1 / factor:.2f}× farther down"
            )
            + f" on geometric average than its corresponding direct source lens; "
            f"it wins on {result['wins']}/{len(solved)} questions.",
            "",
        ]
    lines += [
        CAVEAT,
        "",
        "This measures readability, not improved answers or causal proof. "
        "All-test comparisons and per-alias scores are saved alongside this report.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def run(args):
    source_dir, output = (
        Path(args.reuse_from).resolve(),
        Path(args.output_dir).resolve(),
    )
    if output == source_dir or source_dir in output.parents:
        raise ValueError("use a separate output directory outside the source run")
    if not 0 <= args.source_layer < args.target_layer:
        raise ValueError("require source < target")
    config, cases, generations = load_inputs(source_dir)
    if args.source_layer not in config["target_to_sources"].get(
        str(args.target_layer), []
    ):
        raise ValueError("requested pair is not in the source run")
    if args.plan:
        print(
            json.dumps(
                dict(
                    cases=len(cases),
                    source=args.source_layer,
                    target=args.target_layer,
                    methods=LABELS,
                    fitting=False,
                    generation=False,
                ),
                indent=2,
            )
        )
        return
    seed_all(args.seed)
    # Token rank comparisons over a small number of vocabulary vectors otherwise
    # oversubscribe CPU threads on large pods. Does not change model GPU kernels.
    torch.set_num_threads(1)
    hashes = artifact_hashes(source_dir)
    if hashes != config["fitted_artifact_hashes"]:
        raise ValueError("fitted source artifacts changed")
    request = dict(
        version=1,
        source_run=str(source_dir),
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        seed=args.seed,
        artifact_hashes=hashes,
        source_config_hash=file_hash(source_dir / "benchmark_config.json"),
        cases_sha256=fingerprint(cases),
        generations_sha256=fingerprint(generations),
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock, _handle_termination():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another composition runner owns this directory"
            ) from exc
        manifest = output / "composition_config.json"
        previous = json.loads(manifest.read_text()) if manifest.exists() else {}
        if previous and previous["request"] != request:
            raise ValueError("composition inputs changed; use a new output directory")
        cache = output / "evaluation_cases"
        if not previous and (
            (cache.exists() and any(cache.iterdir()))
            or (output / "benchmark_config.json").exists()
        ):
            raise ValueError("output directory contains another run or untracked cases")
        progress = Progress(
            output,
            [(name, label, len(cases), 1) for name, label in LABELS.items()]
            + [("report", "Report", 1, 1)],
        )
        state = dict(request=request, status="running")
        try:
            hf, model = load_model(
                config["request"]["model"], config["software"]["model_revision"]
            )
            software = provenance(hf)
            if software["model_revision"] != config["software"]["model_revision"]:
                raise ValueError("loaded model revision differs from source")
            for key in ("torch", "transformers", "dtype", "attention", "compile"):
                if software.get(key) != config["software"].get(key):
                    raise ValueError(f"source runtime differs: {key}")
            if previous and previous.get("software") != software:
                raise ValueError("composition software changed; use a new directory")
            state["software"] = software
            write_json(manifest, state)
            maps, means, fit_n = prepare_maps(
                source_dir, config, model, args.source_layer, args.target_layer
            )
            state["fit_prompts"] = fit_n
            signature = fingerprint(dict(request=request, software=software))
            cache.mkdir(exist_ok=True)
            rows = []
            for i, case in enumerate(cases):
                path = cache / (fingerprint(case["id"]) + ".json")
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (
                        saved["signature"] != signature
                        or saved["case_id"] != case["id"]
                    ):
                        raise ValueError("case checkpoint changed")
                    case_rows = saved["rows"]
                else:
                    case_rows = score_case(
                        model,
                        case,
                        generations[case["id"]],
                        maps,
                        means,
                        args.source_layer,
                        args.target_layer,
                    )
                    write_json(
                        path,
                        dict(signature=signature, case_id=case["id"], rows=case_rows),
                    )
                rows.extend(case_rows)
                for name in LABELS:
                    progress.update(
                        name, i + 1, detail=f"{i + 1}/{len(cases)} cases saved/reused"
                    )
            write_report(rows, cases, generations, output, request)
            state["status"] = "complete"
            write_json(manifest, state)
            progress.update("report", 1, force=True)
            progress.finish(detail="Composed comparison REPORT.md ready")
            print((output / "REPORT.md").read_text())
        except BaseException as exc:
            state.update(status="incomplete", error=str(exc))
            # Preserve an existing manifest if validation failed before accepting
            # its software; do not make incompatible cached cases resumable.
            if "software" in state:
                write_json(manifest, state)
            progress.finish("incomplete", str(exc))
            raise


def main():
    p = parser(__doc__)
    p.add_argument("--reuse-from", default="runs/gemma4-e4b")
    p.add_argument("--output-dir", default="runs/gemma4-e4b-composed25")
    p.add_argument("--source-layer", type=int, default=25)
    p.add_argument("--target-layer", type=int, default=33)
    p.add_argument("--plan", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
