"""Frozen arithmetic families and attributed qualitative examples (no model selection)."""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from experiments.short_hop.common import answer_token, chat_ids

PAPER = "https://transformer-circuits.pub/2026/workspace/index.html"
SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_word(number):
    if number < 20:
        return SMALL[number]
    if number == 100:
        return "one hundred"
    tens, units = divmod(number, 10)
    return TENS[tens] + (" " + SMALL[units] if units else "")


def number_aliases(number):
    word = number_word(number)
    return list(dict.fromkeys([str(number), word, word.replace(" ", "-")]))


def numeric_answer(text, number):
    # Entire response, not a substring: "20 or 30" and reasoning are not answers.
    text = text.strip().strip("\"'` ").casefold()
    if text.endswith("."):
        text = text[:-1].rstrip()
    if re.fullmatch(r"[+]?[0-9]+(?:\.0+)?", text):
        return float(text) == number
    return text in number_aliases(number)


def case_ids(tokenizer, case):
    if case.get("input_mode", "chat") == "chat":
        return chat_ids(tokenizer, case["user_prompt"])
    # Original upstream text, including trailing whitespace. No chat wrapper.
    return tokenizer(case["user_prompt"], return_tensors="pt").input_ids


def label_tokens(tokenizer, prompt, aliases, mode="chat"):
    """Predeclared spelling/case/space variants; never partial-word token matches."""
    found = {}
    for alias in aliases:
        for spelling in dict.fromkeys([alias, alias.capitalize()]):
            for text in (spelling, " " + spelling):
                try:
                    if mode == "chat":
                        token = answer_token(tokenizer, prompt, text)
                    else:
                        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                        if len(ids) != 1:
                            continue
                        token = ids[0]
                    if token in tokenizer.all_special_ids:
                        continue
                    if tokenizer.decode([token]).strip().casefold() != alias.casefold():
                        continue
                    found[token] = tokenizer.decode([token])
                except ValueError:
                    continue
    return sorted(found)


def prepare_case(tokenizer, case, *, require_tokens=True):
    case = dict(case)
    mode = case.get("input_mode", "chat")
    for prefix, label in (
        ("intermediate", "intermediate"),
        ("answer", "answer"),
        ("control", "control_intermediate"),
    ):
        aliases = case.get(prefix + "_aliases", [case[label]])
        ids = label_tokens(tokenizer, case["user_prompt"], aliases, mode)
        if require_tokens and not ids:
            raise ValueError(
                f"no complete single-token alias for {label}: {case[label]}"
            )
        case[prefix + "_aliases"] = aliases
        case[prefix + "_token_ids"] = ids
        case[prefix + "_token_id"] = ids[0] if ids else None
    if require_tokens:
        sets = [
            set(case[p + "_token_ids"]) for p in ("intermediate", "answer", "control")
        ]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("intermediate, answer and control token aliases overlap")
    case["scoring_version"] = 1
    return case


def _problems():
    by_answer = defaultdict(list)
    for a in range(1, 10):
        for b in range(1, 10):
            for op in ("+", "-"):
                if op == "+" and a > b:
                    continue  # Do not split a+b and b+a across datasets.
                middle = a + b if op == "+" else a - b
                for c in range(2, 10):
                    answer = middle * c
                    if middle < 2 or answer > 100 or middle in (a, b, c):
                        continue
                    by_answer[answer].append((a, op, b, c, middle, answer))
    return by_answer


def _prompt(problem, wording):
    a, op, b, c, _, _ = problem
    if wording == "symbolic":
        return f"Calculate ({a} {op} {b}) * {c}. Give only the final answer."
    first = f"add {a} and {b}" if op == "+" else f"subtract {b} from {a}"
    return (
        f"First {first}, then multiply the result by {c}. Give only the final answer."
    )


def build_arithmetic(tokenizer, n_cases=192, seed=0, preflight_cases=32):
    """Four cases per group: two same-answer problems, each with two wordings.

    Entire expression/control/paraphrase families stay together. Practice families
    are removed before assigning selection/test, without examining model outputs.
    """
    if n_cases < 24 or n_cases % 12 or preflight_cases < 4 or preflight_cases % 4:
        raise ValueError(
            "n_cases must be >=24 and divisible by 12; preflight cases by 4"
        )
    rng = random.Random(seed)
    pool = _problems()
    answers = sorted(pool)
    rng.shuffle(answers)
    for problems in pool.values():
        rng.shuffle(problems)
    groups = []
    # Round-robin answers prevents a single result from dominating the benchmark.
    while len(groups) < (n_cases + preflight_cases) // 4:
        before = len(groups)
        for answer in answers:
            options = pool[answer]
            while len(options) >= 2:
                first = options.pop()
                # Filter eligibility using the tokenizer only, never lens/model scores.
                matches = [p for p in options if p[4] != first[4]]
                if not matches:
                    continue
                second = matches[0]
                rows = []
                try:
                    for member, problem in enumerate((first, second)):
                        other = (second, first)[member]
                        for wording in ("symbolic", "verbal"):
                            rows.append(
                                prepare_case(
                                    tokenizer,
                                    dict(
                                        id="",
                                        group_id="",
                                        split="",
                                        task_type="2_step_arithmetic",
                                        task_hops=2,
                                        wording=wording,
                                        member=member,
                                        problem_id=f"{problem[0]}{problem[1]}{problem[2]}x{problem[3]}",
                                        user_prompt=_prompt(problem, wording),
                                        intermediate=str(problem[4]),
                                        answer=str(answer),
                                        control_intermediate=str(other[4]),
                                        numeric_answer=answer,
                                        intermediate_aliases=number_aliases(problem[4]),
                                        answer_aliases=number_aliases(answer),
                                        control_aliases=number_aliases(other[4]),
                                        intermediates=[str(problem[4])],
                                    ),
                                )
                            )
                except ValueError:
                    continue
                options.remove(second)
                groups.append(rows)
                break
            if len(groups) == (n_cases + preflight_cases) // 4:
                break
        if len(groups) == before:
            raise ValueError(
                "not enough disjoint, token-compatible arithmetic families"
            )
    rng.shuffle(groups)
    practice, evaluation = [], []
    practice_groups = preflight_cases // 4
    for index, rows in enumerate(groups):
        split = (
            "preflight"
            if index < practice_groups
            else ("selection" if (index - practice_groups) % 3 == 0 else "test")
        )
        group = f"arithmetic-{index:04d}"
        for row in rows:
            row.update(
                id=f"{group}-{row['member']}-{row['wording']}",
                group_id=group,
                split=split,
            )
        (practice if split == "preflight" else evaluation).extend(rows)
    return practice, evaluation


def build_gallery(tokenizer):
    root = Path(__file__).resolve().parents[2] / "data" / "evaluations"
    specs = []
    for filename, name, control in (
        ("lens-eval-multihop.json", "mars-color", "Venus"),
        ("lens-eval-order-ops.json", "parens-add-mult", "4"),
        ("lens-eval-typo.json", "typo-language", "history"),
    ):
        item = next(
            x
            for x in json.loads((root / filename).read_text())["items"]
            if x["name"] == name
        )
        specs.append(
            dict(
                name=name,
                prompt=item["prompt"],
                labels=[item["intermediates"][0]],
                answer=item.get("target", "language"),
                control=control,
                source=f"data/evaluations/{filename}#{name}",
                unscored="target" not in item,
            )
        )
    specs.append(
        dict(
            name="paper-three-step",
            prompt="calc: ( 4 + 17 ) * 2 + 7 =",
            labels=["21", "42"],
            answer="49",
            control="20",
            source=PAPER + " (Figure 17)",
            unscored=False,
        )
    )
    cases = []
    for spec in specs:
        # Exact text is primary; the explicitly labelled chat adaptation is separate.
        for mode in ("raw", "chat"):
            for label in spec["labels"]:
                prompt = spec["prompt"]
                if mode == "chat":
                    prompt = (
                        "Correct only the final misspelled word: " + prompt
                        if spec["unscored"]
                        else "Complete the following with only the answer:\n" + prompt
                    )
                case = dict(
                    id=f"{spec['name']}-{mode}-{label}",
                    group_id=spec["name"],
                    split="qualitative",
                    task_type="paper_example",
                    input_mode=mode,
                    user_prompt=prompt,
                    intermediate=label,
                    answer=spec["answer"],
                    control_intermediate=spec["control"],
                    source=spec["source"],
                    generation_scoring="unscored"
                    if spec["unscored"] and mode == "raw"
                    else "answer_prefix"
                    if mode == "raw"
                    else "exact",
                    readout_position="final input token",
                    adaptation=mode == "chat",
                )
                for prefix, value in (
                    ("intermediate", label),
                    ("answer", spec["answer"]),
                    ("control", spec["control"]),
                ):
                    case[prefix + "_aliases"] = (
                        number_aliases(int(value)) if value.isdigit() else [value]
                    )
                if spec["answer"].isdigit():
                    case["numeric_answer"] = int(spec["answer"])
                cases.append(prepare_case(tokenizer, case, require_tokens=False))
    return cases


def solved_case(text, case):
    scoring = case.get("generation_scoring", "exact")
    if scoring == "unscored":
        return None
    if scoring == "answer_prefix":
        # Raw text completion is allowed to continue after the expected answer.
        text = text.strip().casefold()
        if "numeric_answer" in case:
            numeric = re.match(r"[+]?[0-9]+(?:\.[0-9]+)?(?!\w|\.[0-9])", text)
            if numeric:
                return float(numeric.group()) == case["numeric_answer"]
        return any(
            re.match(re.escape(a.casefold()) + r"(?!\w)", text)
            for a in case.get("answer_aliases", [case["answer"]])
        )
    if "numeric_answer" in case:
        return numeric_answer(text, case["numeric_answer"])
    return text.strip().strip(".\"'` ").casefold() == case["answer"].casefold()
