"""AIME 2024/2025/2026, 0-shot chat, answers in \\boxed{}.

There is no lm-eval AIME task for 2026, but the scoring is lm-eval's own: take
the last \\boxed{} (or the outer $...$ span), strip LaTeX like the harness does
and compare strings. A truncated response counts as incorrect, matching the
Swift 1.5 quantization protocol this bench is meant to compare against.
"""

from . import scoring
from .arguments import add_endpoint_arguments, add_reasoning_arguments, level_template_kwargs
from .endpoint import Endpoint, chat_reply
from .runner import binomial_ci, mean, run, write_summary

# year -> (dataset, split, problem field, answer field)
SOURCES = {
    "24": ("Maxwell-Jia/AIME_2024", "train", "Problem", "Answer"),
    "25": ("math-ai/aime25", "test", "problem", "answer"),
    "26": ("MathArena/aime_2026", "train", "problem", "answer"),
}

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_years(value):
    years = [year.strip() for year in value.split(",") if year.strip()]
    unknown = [year for year in years if year not in SOURCES]
    if unknown:
        raise SystemExit("unknown AIME year(s): %s" % ", ".join(unknown))
    return years


def load_items(years):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("this bench needs the `datasets` package; run through the flake") from error

    items = []
    for year in years:
        path, split, problem_field, answer_field = SOURCES[year]
        dataset = load_dataset(path, split=split)
        for index, row in enumerate(dataset):
            items.append(
                {
                    "index": "%s-%d" % (year, index),
                    "year": year,
                    "problem": row[problem_field],
                    "gold": str(row[answer_field]),
                }
            )
    return items


def main(args):
    items = load_items(parse_years(args.years))
    if args.limit:
        items = items[args.offset : args.offset + args.limit]
    elif args.offset:
        items = items[args.offset :]

    endpoint = Endpoint(args.base_url, args.timeout, args.retries)
    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    template_kwargs = level_template_kwargs(args.level)

    def work(item):
        reply = chat_reply(
            endpoint.chat(
                args.model,
                item["problem"] + PROMPT_SUFFIX,
                generation,
                seed=args.seed,
                template_kwargs=template_kwargs,
            )
        )
        prediction = scoring.aime_extract(reply["text"])
        return {
            "index": item["index"],
            "year": item["year"],
            "gold": item["gold"],
            "output": reply["text"],
            "reasoning": reply["reasoning"],
            "prediction": prediction,
            "correct": 1 if scoring.aime_is_equiv(prediction, item["gold"]) else 0,
            "finish_reason": reply["finish_reason"],
            "completion_tokens": reply["completion_tokens"],
            "reasoning_tokens": reply["reasoning_tokens"],
        }

    rows = run(items, work, args.concurrency, args.out + ".jsonl")
    rows.sort(key=lambda row: row["index"])

    overall = sum(row["correct"] for row in rows) / len(rows)
    by_year = {}
    for row in rows:
        by_year.setdefault(row["year"], []).append(row["correct"])
    per_year = {year: sum(values) / len(values) for year, values in sorted(by_year.items())}

    summary = {
        "model": args.model,
        "bench": "aime",
        "level": args.level,
        "seed": args.seed,
        "years": sorted(by_year),
        "items": len(rows),
        "accuracy": overall,
        "accuracy_ci95": binomial_ci(overall, len(rows)),
        "accuracy_by_year": per_year,
        "truncated": sum(1 for row in rows if row["finish_reason"] == "length"),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        "mean_reasoning_tokens": mean(row["reasoning_tokens"] for row in rows),
        "generation": dict(generation, chat_template_kwargs=template_kwargs),
    }
    write_summary(args.out, summary)
    print(
        "RESULT aime level=%s seed=%d items=%d accuracy=%.4f (%s)"
        % (
            args.level,
            args.seed,
            len(rows),
            overall,
            " ".join("%s=%.3f" % (year, score) for year, score in per_year.items()),
        )
    )


def register(subparsers):
    parser = subparsers.add_parser("aime", help="AIME 2024/2025/2026, 0-shot")
    parser.add_argument("--years", default="24,25,26", help="comma list from 24,25,26")
    add_reasoning_arguments(parser)
    add_endpoint_arguments(parser)
    parser.set_defaults(func=main)
