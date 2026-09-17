"""Resumable J / R / TJ / TR comparison with per-method and overall progress."""

import fcntl
import json
import logging
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import torch

import jlens
from experiments.short_hop.build_eval_cases import build_cases
from experiments.short_hop.build_fit_corpus import build_corpus
from experiments.short_hop.common import (
    file_hash,
    fit_args,
    load_model,
    model_args,
    parser,
    provenance,
    read_jsonl,
    save_baseline,
    seed_all,
    write_json,
    write_jsonl,
)
from experiments.short_hop.evaluate import evaluate_cases
from experiments.short_hop.plot_results import plot_results
from experiments.short_hop.progress import Progress
from experiments.short_hop.report import make_report
from jlens.fitting import _atomic_save
from jlens.relp import GEMMA4_RULE_VERSION, RULE_VERSION, relp_rules, rule_version
from jlens.short_hop import TARGET_TO_SOURCES, default_pairs, smoke_pairs

logger = logging.getLogger(__name__)


@contextmanager
def _handle_termination():
    # SSH disconnect is handled by nohup. Graceful container/process termination
    # marks the run resumable; a hard kill is recovered from atomic checkpoints.
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _jobs(mapping, n_prompts, n_layers, n_cases):
    sources = sorted({i for values in mapping.values() for i in values})
    jobs = []
    for method in ("TJ-Lens", "TR-Lens"):
        for target, src in sorted(mapping.items()):
            jobs.append(
                (f"{method}/target_{target:02d}", method, n_prompts, target - min(src))
            )
    for method in ("J-Lens", "R-Lens"):
        jobs.append(
            (
                f"{method}/target_{n_layers - 1:02d}",
                method,
                n_prompts,
                n_layers - 1 - min(sources),
            )
        )
    fit_work = sum(total * weight for _, _, total, weight in jobs)
    return [
        ("means", "Preparation", 1, fit_work * 0.01),
        *jobs,
        ("evaluation", "Evaluation", n_cases, fit_work * 0.12 / n_cases),
        ("report", "Report", 1, fit_work * 0.01),
    ]


def _restore_fit_progress(progress, output, mapping, n_layers):
    for method, folder in (("TJ-Lens", "tj"), ("TR-Lens", "tr")):
        for target in mapping:
            path = output / "checkpoints" / folder / f"target_{target:02d}.pt"
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=True)
                progress.jobs[f"{method}/target_{target:02d}"]["done"] = state[
                    "next_idx"
                ]
    for method, name in (("J-Lens", "j_lens"), ("R-Lens", "r_lens")):
        path = output / "checkpoints" / f"{name}.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            progress.jobs[f"{method}/target_{n_layers - 1:02d}"]["done"] = state[
                "next_idx"
            ]
    progress.flush(force=True)


def fit_bundle(model, prompts, mapping, args, metadata, deadline=None, progress=None):
    backward_rule = rule_version(model)
    output = Path(args.output_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    layers = sorted(
        {l for target, sources in mapping.items() for l in [target, *sources]}
    )
    means_path = checkpoint_dir / "activation_means.pt"
    means_settings = dict(
        **metadata,
        layers=layers,
        n_prompts=len(prompts),
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
    )
    if progress:
        progress.update(
            "means",
            0,
            detail="Computing or loading shared activation means",
            force=True,
        )
    if means_path.exists():
        state = torch.load(means_path, map_location="cpu", weights_only=True)
        if state["settings"] != means_settings:
            raise ValueError("activation-mean checkpoint settings differ")
        means = state["means"]
    else:
        means = jlens.fit_activation_means(
            model,
            prompts,
            layers=layers,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
        )
        _atomic_save(dict(means=means, settings=means_settings), str(means_path))
    if progress:
        progress.update("means", 1, detail="Shared means saved", force=True)

    def fit_atlas(method, folder, rule):
        atlas = jlens.fit_short_hop_atlas(
            model,
            prompts,
            target_to_sources=mapping,
            model_id=args.model,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
            position_reduction=args.position_reduction,
            position_seed=args.seed,
            checkpoint_dir=str(checkpoint_dir / folder),
            means=means,
            metadata={**metadata, "backward_rule": rule},
            deadline=deadline,
            progress_callback=progress.callback(method) if progress else None,
        )
        filename = "short_hop_atlas.pt" if folder == "tj" else "tr_short_hop_atlas.pt"
        atlas.save(str(output / filename))
        return atlas

    atlas = fit_atlas("TJ-Lens", "tj", "autograd")
    sources = sorted({i for i, _ in atlas.pairs})
    kwargs = dict(
        source_layers=sources,
        target_layer=model.n_layers - 1,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        position_reduction="future_sum",
        position_seed=args.seed,
        deadline=deadline,
    )
    baseline_metadata = dict(
        **metadata,
        model_id=args.model,
        target_layer=model.n_layers - 1,
        position_reduction="future_sum",
        position_seed=args.seed,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
    )
    j_lens = jlens.fit(
        model,
        prompts,
        checkpoint_path=str(checkpoint_dir / "j_lens.pt"),
        checkpoint_metadata={
            **metadata,
            "backward_rule": "autograd",
            "model_id": args.model,
        },
        progress_callback=progress.callback("J-Lens") if progress else None,
        **kwargs,
    )
    save_baseline(
        j_lens, output / "j_lens.pt", {**baseline_metadata, "backward_rule": "autograd"}
    )

    ids = model.encode(prompts[0], max_length=args.max_seq_len)
    with torch.no_grad():
        reference = model.forward(ids).last_hidden_state.detach().clone()
    with relp_rules(model):
        with torch.no_grad():
            patched = model.forward(ids).last_hidden_state
        if not torch.equal(reference, patched):
            raise RuntimeError(
                "R-Lens patch changed forward outputs; refusing unmatched R/TR comparisons"
            )
        r_lens = jlens.fit(
            model,
            prompts,
            checkpoint_path=str(checkpoint_dir / "r_lens.pt"),
            checkpoint_metadata={
                **metadata,
                "backward_rule": backward_rule,
                "model_id": args.model,
            },
            progress_callback=progress.callback("R-Lens") if progress else None,
            **kwargs,
        )
        save_baseline(
            r_lens,
            output / "r_lens.pt",
            {**baseline_metadata, "backward_rule": backward_rule},
        )
        tr_atlas = fit_atlas("TR-Lens", "tr", backward_rule)
    return atlas, j_lens, r_lens, tr_atlas


def run(args):
    seed_all(args.seed)
    if args.n_prompts is None:
        args.n_prompts = 32
    if (
        (args.minutes is not None and args.minutes <= 0)
        or args.n_cases < 8
        or args.n_prompts < 1
    ):
        raise ValueError("minutes (when supplied), cases and prompts must be positive")
    mapping = {8: [6, 7]} if args.smoke else TARGET_TO_SOURCES
    if args.smoke:
        args.n_prompts, args.n_cases = 1, 8
    request = {
        key: value
        for key, value in vars(args).items()
        if key not in ("minutes", "output_dir", "plan")
    }
    request.update(
        supported_rules=[RULE_VERSION, GEMMA4_RULE_VERSION],
        experiment_version=3,
        methods=["J", "R", "TJ", "TR"],
    )
    if args.plan:
        print(
            json.dumps(
                dict(
                    request=request,
                    target_to_sources=mapping
                    if args.model == "google/gemma-3-1b-it"
                    else "scaled to model depth at load time",
                    baseline_target="final block",
                    baseline_reduction="future_sum",
                    fitting="fixed matched corpus across all four methods",
                    minutes=args.minutes,
                    deadline="none unless --minutes is supplied",
                ),
                indent=2,
            )
        )
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock, _handle_termination():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another runner is already using this output directory"
            ) from exc
        config_path = output / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if config and config["request"] != request:
            raise ValueError("run configuration differs; use a new --output-dir")
        if config.get("status") == "complete":
            print((output / "REPORT.md").read_text())
            return
        config.update(
            request=request, status="starting", minutes_this_invocation=args.minutes
        )
        config.pop("error", None)
        write_json(config_path, config)
        (output / "REPORT.md").write_text(
            "Lens comparison: INCOMPLETE\nRun is starting or resuming. No completed comparison yet.\n"
        )
        progress = Progress(output, [])
        start = time.monotonic()
        previous_elapsed = config.get("elapsed_seconds_total", 0)
        deadline = start + args.minutes * 60 if args.minutes is not None else None
        try:
            hf, model = load_model(args.model, args.revision)
            backward_rule = rule_version(model)
            if backward_rule == GEMMA4_RULE_VERSION:
                mapping = (
                    smoke_pairs(model) if args.smoke else default_pairs(model.n_layers)
                )
            software = provenance(hf)
            if config.get("software") and config["software"] != software:
                raise ValueError(
                    "software/model provenance changed; use a new output directory"
                )
            config["software"] = software
            for name, builder in (
                (
                    "fit_prompts.jsonl",
                    lambda: build_corpus(
                        model.tokenizer, args.n_prompts, args.max_seq_len, args.seed
                    ),
                ),
                (
                    "eval_cases.jsonl",
                    lambda: build_cases(
                        model.tokenizer, args.n_cases, args.seed, tuple(args.hops)
                    ),
                ),
            ):
                path = output / name
                if not path.exists():
                    write_jsonl(path, builder())
                digest = file_hash(path)
                if config.get(name + "_sha256", digest) != digest:
                    raise ValueError(f"{name} changed since this run was configured")
                config[name + "_sha256"] = digest
            prompts = [row["text"] for row in read_jsonl(output / "fit_prompts.jsonl")]
            cases = read_jsonl(output / "eval_cases.jsonl")
            if len(prompts) != args.n_prompts or len(cases) != args.n_cases:
                raise ValueError(
                    "saved dataset sizes differ from the requested experiment"
                )
            metadata = dict(
                corpus_sha256=config["fit_prompts.jsonl_sha256"],
                model_revision=software["model_revision"],
            )
            config.update(
                target_to_sources={str(k): v for k, v in mapping.items()},
                baseline_target=model.n_layers - 1,
                baseline_reduction="future_sum",
                d_model=model.d_model,
                n_layers=model.n_layers,
                chosen_n_prompts=len(prompts),
            )
            progress.jobs = {
                name: dict(method=method, total=total, done=0, weight=weight)
                for name, method, total, weight in _jobs(
                    mapping, len(prompts), model.n_layers, len(cases)
                )
            }
            _restore_fit_progress(progress, output, mapping, model.n_layers)
            config["status"] = "fitting"
            write_json(config_path, config)
            fit_bundle(model, prompts, mapping, args, metadata, deadline, progress)
            atlas = jlens.ShortHopAtlas.load(str(output / "short_hop_atlas.pt"))
            tr_atlas = jlens.ShortHopAtlas.load(str(output / "tr_short_hop_atlas.pt"))
            j_lens = jlens.JacobianLens.load(str(output / "j_lens.pt"))
            r_lens = jlens.JacobianLens.load(str(output / "r_lens.pt"))
            config["status"] = "evaluating"
            write_json(config_path, config)
            rows, generations = evaluate_cases(
                hf,
                model,
                atlas,
                j_lens,
                r_lens,
                cases,
                output,
                tr_atlas=tr_atlas,
                max_new_tokens=args.max_new_tokens,
                deadline=deadline,
                progress_callback=lambda event: progress.update(
                    "evaluation", event["done"], detail=event["detail"]
                ),
            )
            progress.update(
                "report", 0, detail="Writing held-out comparisons and plots", force=True
            )
            summary = make_report(
                rows, generations, output, seed=args.seed, fit_n=len(prompts)
            )
            plot_results(rows, output / "plots")
            config.update(
                status="complete",
                verdict=summary["verdict"],
                tr_verdict=summary["tr_lens"]["verdict"],
                elapsed_seconds=time.monotonic() - start,
                elapsed_seconds_total=previous_elapsed + time.monotonic() - start,
            )
            write_json(config_path, config)
            progress.update(
                "report", 1, detail="REPORT.md and summary.json ready", force=True
            )
            progress.finish()
        except (Exception, KeyboardInterrupt) as exc:
            config.update(
                status="incomplete",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - start,
                elapsed_seconds_total=previous_elapsed + time.monotonic() - start,
            )
            write_json(config_path, config)
            progress.finish("interrupted", str(exc))
            hint = (
                " Reduce --dim-batch in a new run directory."
                if isinstance(exc, torch.cuda.OutOfMemoryError)
                else ""
            )
            report = (
                f"Lens comparison: INCONCLUSIVE\n\nThe run did not finish: {type(exc).__name__}: {exc}.{hint}\n"
                "Fitting checkpoints and completed evaluation cases were retained. Reissue the same command to resume.\n"
            )
            (output / "REPORT.md").write_text(report)
            write_json(
                output / "summary.json",
                dict(verdict="INCONCLUSIVE", completed=False, explanation=report),
            )
            raise


def main():
    p = parser(__doc__)
    model_args(p)
    fit_args(p)
    p.add_argument("--output-dir", default="runs/tjlens-a40")
    p.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="optional per-invocation deadline; default: run to completion",
    )
    p.add_argument(
        "--n-prompts",
        type=int,
        default=32,
        help="same fixed fitting count for J, R, TJ, and TR (default: 32)",
    )
    p.add_argument("--n-cases", type=int, default=128)
    p.add_argument("--hops", type=int, nargs="+", default=[2, 3])
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="one prompt, one short-hop target, all four methods, eight cases",
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help="show the plan without model loading or downloads",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
