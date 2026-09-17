"""Fit the sparse TJ-Lens atlas with resumable target checkpoints."""

from pathlib import Path

from experiments.short_hop.common import (
    file_hash,
    fit_args,
    load_model,
    model_args,
    parser,
    provenance,
    read_jsonl,
    seed_all,
    write_json,
)
from jlens import fit_short_hop_atlas
from jlens.short_hop import TARGET_TO_SOURCES


def main():
    p = parser(__doc__)
    model_args(p)
    fit_args(p)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_all(args.seed)
    hf, model = load_model(args.model, args.revision)
    metadata = dict(
        corpus_sha256=file_hash(args.prompts),
        model_revision=provenance(hf)["model_revision"],
    )
    atlas = fit_short_hop_atlas(
        model,
        [r["text"] for r in read_jsonl(args.prompts)],
        target_to_sources={8: [6, 7]} if args.smoke else TARGET_TO_SOURCES,
        model_id=args.model,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        position_reduction=args.position_reduction,
        position_seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        metadata=metadata,
    )
    atlas.save(args.output)
    write_json(
        Path(args.output).with_suffix(".json"),
        dict(**vars(args), **provenance(hf), **metadata),
    )


if __name__ == "__main__":
    main()
