"""Generation-first arithmetic benchmark; fit new lenses or reuse matched artifacts."""

import fcntl
import hashlib
import json
from pathlib import Path

import torch

from experiments.short_hop.benchmark_cases import (
    build_arithmetic,
    build_gallery,
    case_ids,
    solved_case,
)
from experiments.short_hop.common import (
    file_hash,
    fit_args,
    load_baseline,
    load_model,
    model_args,
    parser,
    provenance,
    read_jsonl,
    seed_all,
    write_json,
    write_jsonl,
)
from experiments.short_hop.evaluate import evaluate_cases
from experiments.short_hop.progress import Progress
from experiments.short_hop.run import (
    _handle_termination,
    _jobs,
    _restore_fit_progress,
    fit_bundle,
)
from jlens import ShortHopAtlas
from jlens.relp import rule_version
from jlens.short_hop import default_pairs, smoke_pairs

ARTIFACTS = ("short_hop_atlas.pt", "tr_short_hop_atlas.pt", "j_lens.pt", "r_lens.pt")
PREFLIGHT_THRESHOLD = 0.8


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifact_hashes(directory):
    result = {}
    for name in ARTIFACTS:
        with (Path(directory) / name).open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            result[name] = digest.hexdigest()
    return result


@torch.no_grad()
def preflight(hf, model, cases, output, *, identity, max_new_tokens=16, callback=None):
    """Atomic generation-only transactions; a failed attempt repeats just that case."""
    output = Path(output)
    cache = output / "cases"
    cache.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(
        dict(cases=cases, identity=identity, max_new_tokens=max_new_tokens)
    )
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
        raise ValueError("preflight inputs changed; use a new output directory")
    if not manifest.exists() and any(cache.glob("*.json")):
        raise ValueError("preflight cache lacks provenance")
    write_json(manifest, dict(signature=signature, completed=False))
    rows = []
    for case in cases:
        path = cache / (fingerprint(case["id"]) + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["signature"] != signature or saved["case_id"] != case["id"]:
                raise ValueError("preflight transaction does not match inputs")
        else:
            ids = case_ids(model.tokenizer, case).to(model.input_device)
            generated = hf.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=model.tokenizer.eos_token_id,
            )
            text = model.tokenizer.decode(
                generated[0, ids.shape[1] :], skip_special_tokens=True
            )
            saved = dict(
                signature=signature,
                case_id=case["id"],
                group_id=case["group_id"],
                wording=case["wording"],
                completion=text,
                expected=case["answer"],
                solved=solved_case(text, case),
            )
            write_json(path, saved)
        rows.append(saved)
        if callback:
            callback(
                dict(
                    done=len(rows),
                    detail=f"generation-only practice {len(rows)}/{len(cases)}; cached cases reused",
                )
            )
    solved = sum(r["solved"] for r in rows)
    summary = dict(
        solved=solved,
        total=len(rows),
        accuracy=solved / len(rows),
        threshold=PREFLIGHT_THRESHOLD,
        passed=solved / len(rows) >= PREFLIGHT_THRESHOLD,
        by_wording={
            w: dict(
                solved=sum(r["solved"] for r in rows if r["wording"] == w),
                total=sum(r["wording"] == w for r in rows),
            )
            for w in ("symbolic", "verbal")
        },
    )
    write_jsonl(output / "generations.jsonl", rows)
    write_json(output / "summary.json", summary)
    write_json(manifest, dict(signature=signature, completed=True))
    return summary


def load_bundle(directory, atlas, model, model_id, revision):
    expected = dict(
        model_id=model_id,
        model_revision=revision,
        corpus_sha256=atlas.fit_config.get("corpus_sha256"),
        target_layer=model.n_layers - 1,
        position_reduction="future_sum",
        position_seed=atlas.fit_config.get("position_seed"),
        max_seq_len=atlas.fit_config.get("max_seq_len"),
        skip_first=atlas.fit_config.get("skip_first"),
    )
    return (
        load_baseline(
            Path(directory) / "j_lens.pt",
            expected_metadata={**expected, "backward_rule": "autograd"},
        ),
        load_baseline(
            Path(directory) / "r_lens.pt",
            expected_metadata={**expected, "backward_rule": rule_version(model)},
        ),
        ShortHopAtlas.load(str(Path(directory) / "tr_short_hop_atlas.pt")),
    )


def run(args):
    fit_new = getattr(args, "fit", False)
    smoke = getattr(args, "smoke", False)
    if fit_new and args.reuse_from:
        raise ValueError("--fit and --reuse-from are mutually exclusive")
    if smoke and not fit_new:
        raise ValueError("--smoke requires --fit")
    if fit_new:
        if (
            args.n_prompts < 1
            or args.dim_batch < 1
            or args.max_seq_len <= args.skip_first + 1
        ):
            raise ValueError("invalid fitting count, batch size or sequence mask")
        if args.position_reduction != "self_hutchinson":
            raise ValueError("arithmetic TJ/TR fitting requires self_hutchinson")
        if smoke:
            args.n_prompts = 1
            args.max_seq_len = min(args.max_seq_len, 16)
            args.skip_first = min(args.skip_first, 2)
    if args.plan:
        print(
            json.dumps(
                dict(
                    preflight_cases=32,
                    threshold=PREFLIGHT_THRESHOLD,
                    arithmetic_cases=8 if smoke else args.n_cases,
                    qualitative="4 fixed examples, original + chat adaptations",
                    reuse_from=args.reuse_from,
                    fitting="fresh matched J/R/TJ/TR, after practice passes"
                    if fit_new
                    else "none",
                    target_to_sources="two sources across KV-sharing boundary; E4B 21/24 → 25"
                    if smoke
                    else "depth-scaled 1/2/4/8-block hops; E4B targets 13,23,33,41"
                    if fit_new
                    else "from saved atlas",
                    fit_prompts=args.n_prompts if fit_new else None,
                    dim_batch=args.dim_batch if fit_new else None,
                    smoke=smoke,
                    output=args.output_dir,
                ),
                indent=2,
            )
        )
        return
    if not args.preflight_only and not args.reuse_from and not fit_new:
        raise ValueError(
            "full comparison requires --fit or --reuse-from with all four fitted artifacts"
        )
    output = Path(args.output_dir)
    source = Path(args.reuse_from).resolve() if args.reuse_from else None
    if source == output.resolve():
        raise ValueError(
            "use a new output directory; original fitting run is read-only"
        )
    output.mkdir(parents=True, exist_ok=True)
    request = dict(
        version=1,
        seed=args.seed,
        model=args.model,
        revision=args.revision,
        n_cases=args.n_cases,
        max_new_tokens=args.max_new_tokens,
        reuse_from=str(source) if source else None,
    )
    if fit_new:
        request.update(
            fit=True,
            smoke=smoke,
            n_prompts=args.n_prompts,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
            position_reduction=args.position_reduction,
        )
    with (output / "runner.lock").open("w") as lock, _handle_termination():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner owns this output directory") from exc
        config_path = output / "benchmark_config.json"
        if (output / "config.json").exists():
            raise ValueError(
                "this directory belongs to the original fitting runner; use a new one"
            )
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if config and config["request"] != request:
            raise ValueError(
                "benchmark configuration changed; use a new output directory"
            )
        config["request"] = request
        seed_all(args.seed)
        jobs = [
            ("preflight", "Model preflight", 32, 1),
            ("gallery", "Paper gallery", 10, 4),
        ]
        jobs += [
            (name, name, args.n_cases, 1)
            for name in ("J-Lens", "R-Lens", "TJ-Lens", "TR-Lens")
        ]
        jobs.append(("report", "Report", 1, 8))
        progress = Progress(output, jobs)
        try:
            hashes = artifact_hashes(source) if source else {}
            if config.get("artifact_hashes", hashes) != hashes:
                raise ValueError("fitted artifacts changed; use a new output directory")
            config["artifact_hashes"] = hashes
            atlas = ShortHopAtlas.load(str(source / ARTIFACTS[0])) if source else None
            if atlas and atlas.model_id != args.model:
                raise ValueError("fitted model does not match --model")
            fitted_revision = atlas.fit_config.get("model_revision") if atlas else None
            pinned = fitted_revision or config.get("software", {}).get("model_revision")
            revision = pinned if args.revision == "main" and pinned else args.revision
            hf, model = load_model(args.model, revision)
            if source:
                from experiments.short_hop.gemma_checks import architecture_metadata

                config["architecture"] = architecture_metadata(model)
            software = provenance(hf)
            if fitted_revision and software["model_revision"] != fitted_revision:
                raise ValueError("model revision differs from fitted artifacts")
            if config.get("software", software) != software:
                raise ValueError("software/model changed; use a new output directory")
            config.update(software=software, status="running")
            write_json(config_path, config)
            practice, cases = build_arithmetic(model.tokenizer, args.n_cases, args.seed)
            if smoke:
                cases = cases[:8]
            gallery = build_gallery(model.tokenizer)
            datasets = {
                "practice_cases": practice,
                "arithmetic_cases": cases,
                "gallery_cases": gallery,
            }
            hashes = {name: fingerprint(values) for name, values in datasets.items()}
            if config.get("datasets", hashes) != hashes:
                raise ValueError(
                    "benchmark dataset changed; use a new output directory"
                )
            config["datasets"] = hashes
            write_json(config_path, config)
            for name, values in datasets.items():
                path = output / (name + ".jsonl")
                if path.exists():
                    previous = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    if previous != values:
                        raise ValueError(
                            f"saved {name} was edited; use a new output directory"
                        )
                else:
                    write_jsonl(path, values)
            screening = preflight(
                hf,
                model,
                practice,
                output / "preflight",
                identity=software,
                max_new_tokens=args.max_new_tokens,
                callback=lambda e: progress.update(
                    "preflight", e["done"], detail=e["detail"]
                ),
            )
            if args.preflight_only:
                config["status"] = "preflight_complete"
                write_json(config_path, config)
                progress.finish(
                    "preflight_complete",
                    f"Practice accuracy {screening['accuracy']:.0%}; full comparison not run",
                )
                print(json.dumps(screening, indent=2))
                return
            if fit_new:
                from experiments.short_hop.benchmark_report import (
                    write_benchmark_report,
                )

                if not screening["passed"]:
                    write_benchmark_report(
                        output,
                        screening,
                        cases,
                        [],
                        [],
                        fit_n=0,
                        gallery_available=False,
                    )
                    config["status"] = "preflight_failed"
                    write_json(config_path, config)
                    progress.finish(
                        "preflight_failed",
                        "No fitting or lens evaluation: practice did not pass",
                    )
                    return
                from experiments.short_hop.build_fit_corpus import build_corpus
                from experiments.short_hop.gemma_checks import (
                    architecture_metadata,
                    check_model_paths,
                )

                # Validate the architecture and exact forward path before expensive fits.
                backward_rule = rule_version(model)
                mapping = smoke_pairs(model) if smoke else default_pairs(model.n_layers)
                config.update(
                    target_to_sources=mapping,
                    backward_rule=backward_rule,
                    architecture=architecture_metadata(model),
                )
                write_json(config_path, config)
                # Fitting dominates cost. Keep generation already completed in the
                # same progress snapshot, and restore every per-target checkpoint.
                fit_jobs = _jobs(mapping, args.n_prompts, model.n_layers, len(cases))
                for name, method, total, weight in fit_jobs:
                    if name not in ("evaluation", "report"):
                        progress.jobs[name] = dict(
                            method=method, total=total, done=0, weight=weight
                        )
                for method in ("J-Lens", "R-Lens", "TJ-Lens", "TR-Lens"):
                    progress.jobs[method]["total"] = len(cases)
                _restore_fit_progress(progress, output, mapping, model.n_layers)
                path = output / "fit_prompts.jsonl"
                if not path.exists():
                    if config.get("fit_corpus_sha256"):
                        raise ValueError(
                            "saved fitting corpus is missing; restore it to resume"
                        )
                    write_jsonl(
                        path,
                        build_corpus(
                            model.tokenizer, args.n_prompts, args.max_seq_len, args.seed
                        ),
                    )
                digest = file_hash(path)
                if config.get("fit_corpus_sha256", digest) != digest:
                    raise ValueError("saved fitting corpus changed")
                prompts = [row["text"] for row in read_jsonl(path)]
                if len(prompts) != args.n_prompts:
                    raise ValueError("saved fitting prompt count differs")
                config["fit_corpus_sha256"] = digest
                config["path_checks"] = check_model_paths(
                    hf, model, prompts[0], max_seq_len=args.max_seq_len
                )
                write_json(config_path, config)
                metadata = dict(
                    corpus_sha256=digest,
                    model_revision=software["model_revision"],
                    **architecture_metadata(model),
                )
                fit_bundle(model, prompts, mapping, args, metadata, progress=progress)
                source = output
                atlas = ShortHopAtlas.load(str(output / ARTIFACTS[0]))
                fitted_revision = software["model_revision"]
                config["fitted_artifact_hashes"] = artifact_hashes(output)
                write_json(config_path, config)
            j_lens, r_lens, tr_atlas = load_bundle(
                source, atlas, model, args.model, fitted_revision
            )
            # Fixed gallery always runs, even when practice fails. It never selects layers.
            gallery_rows, gallery_generations = evaluate_cases(
                hf,
                model,
                atlas,
                j_lens,
                r_lens,
                gallery,
                output / "qualitative",
                tr_atlas=tr_atlas,
                max_new_tokens=args.max_new_tokens,
                progress_callback=lambda e: progress.update(
                    "gallery", e["done"], detail=e["detail"]
                ),
            )
            from experiments.short_hop.benchmark_report import (
                write_benchmark_report,
                write_gallery,
            )

            write_gallery(
                gallery, gallery_rows, gallery_generations, output / "qualitative"
            )
            rows, generations = [], []
            if screening["passed"]:

                def callback(event):
                    for name in ("J-Lens", "R-Lens", "TJ-Lens", "TR-Lens"):
                        progress.update(name, event["done"], detail=event["detail"])

                rows, generations = evaluate_cases(
                    hf,
                    model,
                    atlas,
                    j_lens,
                    r_lens,
                    cases,
                    output / "arithmetic",
                    tr_atlas=tr_atlas,
                    max_new_tokens=args.max_new_tokens,
                    progress_callback=callback,
                )
            progress.update(
                "report",
                0,
                detail="Writing plain-language results and qualitative plots",
                force=True,
            )
            write_benchmark_report(
                output,
                screening,
                cases,
                rows,
                generations,
                fit_n=min(atlas.n_prompts_by_target.values()),
                seed=args.seed,
            )
            config["status"] = "complete" if screening["passed"] else "preflight_failed"
            write_json(config_path, config)
            progress.update("report", 1, force=True)
            progress.finish(
                config["status"], "REPORT.md and qualitative/GALLERY.md ready"
            )
        except (Exception, KeyboardInterrupt) as exc:
            config.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
            write_json(config_path, config)
            progress.finish("interrupted", str(exc))
            (output / "REPORT.md").write_text(
                f"Comparison incomplete: {type(exc).__name__}: {exc}\n\n"
                "Completed practice/evaluation cases are saved. Reissue the same command to resume.\n"
            )
            raise


def main():
    p = parser(__doc__)
    model_args(p)
    fit_args(p)
    p.set_defaults(dim_batch=8)
    p.add_argument(
        "--fit",
        action="store_true",
        help="fit fresh J/R/TJ/TR maps after practice passes",
    )
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="with --fit: one 16-token fit prompt, two pairs around KV sharing, eight main cases; never a scientific verdict",
    )
    p.add_argument(
        "--reuse-from", help="read-only directory containing J/R/TJ/TR fitted artifacts"
    )
    p.add_argument("--output-dir", default="runs/tjlens-arithmetic")
    p.add_argument("--n-cases", type=int, default=192)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="generation-only screening; no fitted artifacts required",
    )
    p.add_argument("--plan", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
