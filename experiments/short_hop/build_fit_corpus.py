"""Build a deterministic, token-length-checked WikiText fitting corpus."""

from experiments.short_hop.common import (
    file_hash,
    load_tokenizer,
    model_args,
    parser,
    seed_all,
    write_json,
    write_jsonl,
)


def build_corpus(tokenizer, n_prompts=32, seq_len=64, seed=0, records=None):
    if n_prompts < 1 or seq_len < 2:
        raise ValueError("n_prompts and seq_len must be positive (seq_len >= 2)")
    if records is None:
        from datasets import load_dataset

        records = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        ).shuffle(seed=seed, buffer_size=2000)
    rows, pieces = [], []
    for record in records:
        text = record["text"].strip()
        if len(text) < 40 or text.startswith("="):
            continue
        pieces.append(text)
        combined = "\n".join(pieces)
        count = len(tokenizer(combined)["input_ids"])
        if count < seq_len:
            continue
        rows.append(dict(id=f"fit-{len(rows):04d}", text=combined, n_tokens=count))
        pieces = []
        if len(rows) == n_prompts:
            return rows
    raise ValueError(f"corpus exhausted after {len(rows)} usable prompts")


def main():
    p = parser(__doc__)
    model_args(p)
    p.add_argument("--output", required=True)
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    args = p.parse_args()
    seed_all(args.seed)
    rows = build_corpus(
        load_tokenizer(args.model, args.revision),
        args.n_prompts,
        args.seq_len,
        args.seed,
    )
    write_jsonl(args.output, rows)
    write_json(
        args.output + ".meta.json",
        dict(
            **vars(args),
            sha256=file_hash(args.output),
            dataset="Salesforce/wikitext",
            subset="wikitext-103-raw-v1",
            split="train",
        ),
    )


if __name__ == "__main__":
    main()
