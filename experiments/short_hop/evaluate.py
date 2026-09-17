"""Evaluate all readouts at the final input token; retain solved and failed cases."""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.short_hop.benchmark_cases import case_ids, prepare_case, solved_case
from experiments.short_hop.common import (
    answer_token,
    load_baseline,
    load_model,
    model_args,
    parser,
    provenance,
    read_jsonl,
    seed_all,
    write_json,
)
from jlens import ActivationRecorder, ShortHopAtlas
from jlens.relp import rule_version

logger = logging.getLogger(__name__)


def token_rank(logits, token_id):
    return int((logits > logits[token_id]).sum().item()) + 1


def readout_metrics(logits, case, tokenizer, top_k=20):
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite readout logits")

    def best(prefix):
        candidates = case.get(prefix + "_token_ids", [case[prefix + "_token_id"]])
        return max(candidates, key=lambda i: logits[i].item()) if candidates else None

    middle, answer, control = [best(p) for p in ("intermediate", "answer", "control")]

    def score(token):
        return logits[token].item() if token is not None else None

    def margin(first, second):
        return (
            score(first) - score(second)
            if first is not None and second is not None
            else None
        )

    answer_margin, control_margin = margin(middle, answer), margin(middle, control)
    values, ids = logits.topk(min(top_k, logits.numel()))
    return dict(
        intermediate_logit=score(middle),
        answer_logit=score(answer),
        intermediate_rank=token_rank(logits, middle) if middle is not None else None,
        answer_rank=token_rank(logits, answer) if answer is not None else None,
        intermediate_answer_margin=answer_margin,
        control_margin=control_margin,
        intermediate_above_answer=answer_margin > 0
        if answer_margin is not None
        else None,
        intermediate_above_control=control_margin > 0
        if control_margin is not None
        else None,
        intermediate_best_token_id=middle,
        intermediate_alias_ranks={
            str(i): token_rank(logits, i)
            for i in case.get("intermediate_token_ids", [middle])
            if i is not None
        },
        top_tokens=[
            dict(id=i, token=tokenizer.decode([i]), score=v)
            for i, v in zip(ids.tolist(), values.tolist(), strict=True)
        ],
    )


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _solved(text, answer):
    return text.strip().strip(".\"'` ").casefold() == answer.casefold()


def _evaluation_signature(atlases, baselines, cases, max_new_tokens, baseline_target):
    import hashlib

    digest = hashlib.sha256(
        json.dumps(
            dict(
                version=3 if any(c.get("scoring_version") for c in cases) else 2,
                cases=cases,
                max_new_tokens=max_new_tokens,
                baseline_target=baseline_target,
                torch=str(torch.__version__),
            ),
            sort_keys=True,
        ).encode()
    )
    for name, atlas in atlases.items():
        digest.update(
            json.dumps(
                dict(name=name, model_id=atlas.model_id, config=atlas.fit_config),
                sort_keys=True,
            ).encode()
        )
        for layer, mean in sorted(atlas.means.items()):
            digest.update(f"mean:{layer}".encode())
            digest.update(mean.detach().cpu().float().numpy().tobytes())
        for source, target in atlas.pairs:
            digest.update(f"{source}:{target}".encode())
            digest.update(
                atlas.get(source, target).detach().cpu().float().numpy().tobytes()
            )
    for name, lens in baselines.items():
        digest.update(f"{name}:{lens.n_prompts}".encode())
        for layer in lens.source_layers:
            digest.update(str(layer).encode())
            digest.update(
                lens.jacobians[layer].detach().cpu().float().numpy().tobytes()
            )
    return digest.hexdigest()


def _transport_quality(
    predicted,
    identity,
    actual_state,
    source_state,
    mean,
    predicted_logits,
    actual_logits,
):
    denom = (actual_state - mean).norm().clamp_min(1e-8)
    update = predicted - identity
    actual_update = actual_state - source_state
    top_t = set(predicted_logits.topk(min(20, len(predicted_logits))).indices.tolist())
    top_a = set(actual_logits.topk(min(20, len(actual_logits))).indices.tolist())
    return dict(
        relative_state_error=((predicted - actual_state).norm() / denom).item(),
        identity_state_error=((identity - actual_state).norm() / denom).item(),
        update_cosine=_cosine(update, actual_update),
        update_relative_norm=(
            update.norm() / actual_update.norm().clamp_min(1e-8)
        ).item(),
        transported_vs_actual_logit_cosine=_cosine(predicted_logits, actual_logits),
        transported_vs_actual_top20_jaccard=len(top_t & top_a) / len(top_t | top_a),
    )


@torch.no_grad()
def evaluate_cases(
    hf,
    model,
    atlas,
    j_lens,
    r_lens,
    cases,
    output_dir,
    *,
    tr_atlas=None,
    max_new_tokens=12,
    deadline=None,
    baseline_target=None,
    progress_callback=None,
):
    import hashlib

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if atlas.d_model != model.d_model or atlas.n_layers != model.n_layers:
        raise ValueError("atlas architecture does not match model")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("evaluation requires nonempty, unique case IDs")
    atlases = {"TJ": atlas}
    if tr_atlas is not None:
        if (
            tr_atlas.pairs != atlas.pairs
            or tr_atlas.d_model != atlas.d_model
            or tr_atlas.n_layers != atlas.n_layers
            or tr_atlas.model_id != atlas.model_id
            or tr_atlas.position_reduction != atlas.position_reduction
            or tr_atlas.n_prompts_by_target != atlas.n_prompts_by_target
        ):
            raise ValueError(
                "TR and TJ atlases must match in model, pairs, estimator and fitting count"
            )
        for key in (
            "model_revision",
            "corpus_sha256",
            "prompts_sha256",
            "position_seed",
            "max_seq_len",
            "skip_first",
        ):
            if tr_atlas.fit_config.get(key) != atlas.fit_config.get(key):
                raise ValueError(f"TR/TJ provenance mismatch: {key}")
        if tr_atlas.fit_config.get("backward_rule") != rule_version(model):
            raise ValueError("TR atlas does not record the R-Lens backward rules")
        if not all(torch.equal(atlas.means[i], tr_atlas.means[i]) for i in atlas.means):
            raise ValueError("TR and TJ must use identical activation means")
        atlases["TR"] = tr_atlas
    counts = set(atlas.n_prompts_by_target.values()) | {
        j_lens.n_prompts,
        r_lens.n_prompts,
    }
    if len(counts) != 1:
        raise ValueError(
            "all maps and baselines must use the same number of fitting prompts"
        )
    baseline_target = model.n_layers - 1 if baseline_target is None else baseline_target
    sources = sorted({i for i, _ in atlas.pairs})
    if not set(sources) <= set(j_lens.source_layers) or not set(sources) <= set(
        r_lens.source_layers
    ):
        raise ValueError("J-Lens and R-Lens must cover every TJ-Lens source layer")
    lens_objects = {"j_lens": j_lens, "r_lens": r_lens}
    signature = _evaluation_signature(
        atlases, lens_objects, cases, max_new_tokens, baseline_target
    )
    manifest = output / "evaluation_status.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("signature") not in (None, signature):
            raise ValueError(
                "evaluation inputs or artifacts changed; use a new output directory"
            )
    cache = output / "evaluation_cases"
    if cache.exists() and any(cache.glob("*.json")) and not manifest.exists():
        raise ValueError("evaluation cache has no provenance manifest")
    cache.mkdir(exist_ok=True)
    write_json(
        manifest, dict(completed=False, expected_cases=len(cases), signature=signature)
    )
    layers = sorted({l for pair in atlas.pairs for l in pair})
    device = model.input_device
    matrices = {
        name: {(i, j): a.get(i, j).to(device) for i, j in a.pairs}
        for name, a in atlases.items()
    }
    means = {i: m.to(device) for i, m in atlas.means.items()}
    baselines = {
        name: {i: lens.jacobians[i].to(device) for i in sources}
        for name, lens in lens_objects.items()
    }
    cached = {}
    expected_rows = len(atlas.pairs) * (9 if tr_atlas is not None else 7)
    for case in cases:
        path = cache / (hashlib.sha256(case["id"].encode()).hexdigest() + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved.get("signature") != signature
                or saved["generation"]["case_id"] != case["id"]
                or len(saved["rows"]) != expected_rows
            ):
                raise ValueError(f"invalid saved evaluation case: {case['id']}")
            cached[case["id"]] = saved
    completed = len(cached)

    def notify(detail):
        if progress_callback:
            progress_callback(dict(done=completed, total=len(cases), detail=detail))

    notify(f"resuming with {completed}/{len(cases)} cases saved")
    generations, all_rows = [], []
    # Consolidated files are rebuilt from per-case transactions. Only atomic case
    # files are used for resume, so a killed writer cannot duplicate or lose cases.
    with (
        (output / "pair_results.jsonl").open("w") as pair_file,
        (output / "generation_results.jsonl").open("w") as gen_file,
    ):
        for case_number, case in enumerate(cases):
            saved = cached.get(case["id"])
            if saved is None:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        "evaluation time budget reached; completed cases saved"
                    )
                notify(f"case {case_number + 1}/{len(cases)}: generating answer")
                if case.get("scoring_version"):
                    checked = prepare_case(
                        model.tokenizer,
                        case,
                        require_tokens=case["split"] != "qualitative",
                    )
                    if checked != case:
                        raise ValueError(
                            f"case {case['id']} was built with a different tokenizer"
                        )
                for label_key, id_key in (
                    ()
                    if case.get("scoring_version")
                    else (
                        ("intermediate", "intermediate_token_id"),
                        ("answer", "answer_token_id"),
                        ("control_intermediate", "control_token_id"),
                    )
                ):
                    if (
                        answer_token(
                            model.tokenizer, case["user_prompt"], case[label_key]
                        )
                        != case[id_key]
                    ):
                        raise ValueError(
                            f"case {case['id']} was built with a different tokenizer"
                        )
                ids = case_ids(model.tokenizer, case).to(device)
                generated = hf.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
                completion = model.tokenizer.decode(
                    generated[0, ids.shape[1] :], skip_special_tokens=True
                )
                solved = solved_case(completion, case)
                generation = dict(
                    case_id=case["id"],
                    group_id=case["group_id"],
                    split=case["split"],
                    task_type=case["task_type"],
                    solved=solved,
                    expected=case["answer"],
                    completion=completion,
                    n_input_tokens=ids.shape[1],
                    wording=case.get("wording"),
                    problem_id=case.get("problem_id"),
                )
                notify(f"case {case_number + 1}/{len(cases)}: scoring all methods")
                with ActivationRecorder(model.layers, at=layers) as recorder:
                    model.forward(ids)
                    states = {
                        i: recorder.activations[i][0, -1].float().clone()
                        for i in layers
                    }
                named_states = {}
                for source in sources:
                    named_states["source_logit_lens", source, source] = states[source]
                    for name, maps in baselines.items():
                        named_states[name, source, baseline_target] = (
                            states[source] @ maps[source].T
                        )
                for target in sorted(atlas.jacobians):
                    named_states["actual_local", target, target] = states[target]
                for source, target in atlas.pairs:
                    x = states[source] - means[source]
                    named_states["identity", source, target] = means[target] + x
                    for name, maps in matrices.items():
                        key = "transported" if name == "TJ" else "tr_transported"
                        named_states[key, source, target] = (
                            means[target] + x @ maps[source, target].T
                        )
                keys = list(named_states)
                logits = dict(
                    zip(
                        keys,
                        model.unembed(torch.stack(list(named_states.values()))).float(),
                        strict=True,
                    )
                )
                case_rows = []
                for source, target in atlas.pairs:
                    qualities = {}
                    for family in atlases:
                        state_key = (
                            "transported" if family == "TJ" else "tr_transported"
                        )
                        innovation_key = (
                            "innovation" if family == "TJ" else "tr_innovation"
                        )
                        logits[innovation_key, source, target] = (
                            logits[state_key, source, target]
                            - logits["identity", source, target]
                        )
                        qualities[family] = _transport_quality(
                            named_states[state_key, source, target],
                            named_states["identity", source, target],
                            states[target],
                            states[source],
                            means[target],
                            logits[state_key, source, target],
                            logits["actual_local", target, target],
                        )
                    readouts = [
                        ("actual_local", ("actual_local", target, target)),
                        ("source_logit_lens", ("source_logit_lens", source, source)),
                        ("transported", ("transported", source, target)),
                        ("identity", ("identity", source, target)),
                        ("innovation", ("innovation", source, target)),
                        ("j_lens", ("j_lens", source, baseline_target)),
                        ("r_lens", ("r_lens", source, baseline_target)),
                    ]
                    if tr_atlas is not None:
                        readouts.extend(
                            (key, (key, source, target))
                            for key in ("tr_transported", "tr_innovation")
                        )
                    for name, key in readouts:
                        case_rows.append(
                            dict(
                                case_id=case["id"],
                                group_id=case["group_id"],
                                split=case["split"],
                                task_type=case["task_type"],
                                solved=solved,
                                wording=case.get("wording"),
                                problem_id=case.get("problem_id"),
                                source_layer=source,
                                target_layer=target,
                                hop_length=target - source,
                                readout=name,
                                baseline_target=baseline_target
                                if name in baselines
                                else None,
                                **readout_metrics(logits[key], case, model.tokenizer),
                                **qualities["TR" if name.startswith("tr_") else "TJ"],
                            )
                        )
                saved = dict(signature=signature, generation=generation, rows=case_rows)
                path = cache / (
                    hashlib.sha256(case["id"].encode()).hexdigest() + ".json"
                )
                write_json(path, saved)
                completed += 1
                notify(f"case {case_number + 1}/{len(cases)} saved; solved={solved}")
                logger.info(
                    "evaluation: %d/%d cases saved; case=%s solved=%s",
                    completed,
                    len(cases),
                    case["id"],
                    solved,
                )
            generations.append(saved["generation"])
            all_rows.extend(saved["rows"])
            gen_file.write(json.dumps(saved["generation"]) + "\n")
            for row in saved["rows"]:
                pair_file.write(json.dumps(row, allow_nan=False) + "\n")
            pair_file.flush()
            gen_file.flush()
    write_json(
        manifest,
        dict(
            completed=True,
            expected_cases=len(cases),
            evaluated_cases=len(generations),
            rows=len(all_rows),
            signature=signature,
        ),
    )
    return all_rows, generations


def main():
    p = parser(__doc__)
    model_args(p)
    p.add_argument("--atlas", required=True)
    p.add_argument("--tr-atlas", help="TR atlas from the four-method runner")
    p.add_argument("--j-lens", required=True)
    p.add_argument("--r-lens", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=12)
    args = p.parse_args()
    seed_all(args.seed)
    atlas = ShortHopAtlas.load(args.atlas)
    if atlas.model_id and atlas.model_id != args.model:
        raise ValueError("atlas model_id does not match --model")
    fitted_revision = atlas.fit_config.get("model_revision")
    revision = (
        fitted_revision
        if args.revision == "main" and fitted_revision
        else args.revision
    )
    hf, model = load_model(args.model, revision)
    software = provenance(hf)
    if fitted_revision and software["model_revision"] != fitted_revision:
        raise ValueError("atlas and model revisions differ")
    expected = dict(
        model_id=args.model,
        model_revision=fitted_revision,
        corpus_sha256=atlas.fit_config.get("corpus_sha256"),
        target_layer=model.n_layers - 1,
        position_reduction="future_sum",
        position_seed=atlas.fit_config.get("position_seed"),
        max_seq_len=atlas.fit_config.get("max_seq_len"),
        skip_first=atlas.fit_config.get("skip_first"),
    )
    j_lens = load_baseline(
        args.j_lens, expected_metadata={**expected, "backward_rule": "autograd"}
    )
    r_lens = load_baseline(
        args.r_lens,
        expected_metadata={**expected, "backward_rule": rule_version(model)},
    )
    if (
        len(
            set(atlas.n_prompts_by_target.values())
            | {j_lens.n_prompts, r_lens.n_prompts}
        )
        != 1
    ):
        raise ValueError(
            "all maps and baselines must use the same number of fitting prompts"
        )
    rows, generations = evaluate_cases(
        hf,
        model,
        atlas,
        j_lens,
        r_lens,
        read_jsonl(args.cases),
        args.output_dir,
        max_new_tokens=args.max_new_tokens,
        tr_atlas=ShortHopAtlas.load(args.tr_atlas) if args.tr_atlas else None,
    )
    from experiments.short_hop.report import make_report

    make_report(
        rows,
        generations,
        args.output_dir,
        seed=args.seed,
        fit_n=min(atlas.n_prompts_by_target.values()),
    )


if __name__ == "__main__":
    main()
