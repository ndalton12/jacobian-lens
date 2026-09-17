"""Controlled two/three-hop lookups with paired query-switch controls."""

import random

from experiments.short_hop.common import (
    answer_token,
    file_hash,
    load_tokenizer,
    model_args,
    parser,
    seed_all,
    write_json,
    write_jsonl,
)

COLORS = (
    "red blue green yellow black white orange purple pink brown silver gold".split()
)
VALUES = "one two three four five six seven eight nine ten eleven twelve".split()
BRIDGES = "cat dog bird fish horse bear lion tiger mouse fox wolf snake".split()
ENTITIES = "dax wug blicket zorp fep tufa glorp nup koba vash".split()


def _single_pool(tokenizer, pool):
    result = []
    for label in pool:
        try:
            answer_token(tokenizer, "Reply with one word.", label)
            result.append(label)
        except ValueError:
            pass
    if len(result) < 4:
        raise ValueError(
            "fewer than four single-token labels remain in a candidate pool"
        )
    return result


def build_cases(tokenizer, n_cases=128, seed=0, hops=(2, 3)):
    if n_cases < 8 or n_cases % (4 * len(hops)):
        raise ValueError(
            "n_cases must be >= 8 and divisible by 4 * number of hop lengths"
        )
    if not hops or any(h not in (2, 3) for h in hops):
        raise ValueError("supported task hop lengths are 2 and 3")
    rng = random.Random(seed)
    colors, values, bridges = [
        _single_pool(tokenizer, pool) for pool in (COLORS, VALUES, BRIDGES)
    ]
    rows = []
    for group in range(n_cases // 2):
        hops_count = hops[group % len(hops)]
        # Split whole query-switch pairs. Balance hop length across both splits.
        split = "selection" if (group // len(hops)) % 2 == 0 else "test"
        distractors = (group // (2 * len(hops))) % 2 == 1
        count = 4 if distractors else 2
        entities = rng.sample(ENTITIES, count)
        middle, answers, bridge = [
            rng.sample(pool, count) for pool in (colors, values, bridges)
        ]
        facts = []
        for index, entity in enumerate(entities):
            facts.append(f"The {entity} is associated with {middle[index]}.")
            if hops_count == 3:
                facts.extend(
                    [
                        f"Anything associated with {middle[index]} leads to {bridge[index]}.",
                        f"{bridge[index].capitalize()} maps to {answers[index]}.",
                    ]
                )
            else:
                facts.append(
                    f"Anything associated with {middle[index]} maps to {answers[index]}."
                )
        rng.shuffle(facts)
        for query in (0, 1):
            prompt = "Facts:\n" + "\n".join(f"- {fact}" for fact in facts)
            prompt += f"\n\nQuestion: What does the {entities[query]} map to?\nRespond with only the final value."
            other = 1 - query
            rows.append(
                dict(
                    id=f"lookup-{group:04d}-{query}",
                    group_id=f"group-{group:04d}",
                    split=split,
                    task_type=f"{hops_count}_hop_lookup",
                    task_hops=hops_count,
                    has_distractors=distractors,
                    user_prompt=prompt,
                    intermediate=middle[query],
                    answer=answers[query],
                    intermediates=[middle[query]]
                    + ([bridge[query]] if hops_count == 3 else []),
                    control_intermediate=middle[other],
                    intermediate_token_id=answer_token(
                        tokenizer, prompt, middle[query]
                    ),
                    answer_token_id=answer_token(tokenizer, prompt, answers[query]),
                    control_token_id=answer_token(tokenizer, prompt, middle[other]),
                )
            )
    return rows


def main():
    p = parser(__doc__)
    model_args(p)
    p.add_argument("--output", required=True)
    p.add_argument("--n-cases", type=int, default=128)
    p.add_argument("--hops", type=int, nargs="+", default=[2, 3])
    args = p.parse_args()
    seed_all(args.seed)
    rows = build_cases(
        load_tokenizer(args.model, args.revision),
        args.n_cases,
        args.seed,
        tuple(args.hops),
    )
    write_jsonl(args.output, rows)
    write_json(
        args.output + ".meta.json", dict(**vars(args), sha256=file_hash(args.output))
    )


if __name__ == "__main__":
    main()
