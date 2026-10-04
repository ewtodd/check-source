"""GSM8K 5-shot, raw completions or chat template.

`gsm8k` mirrors lm-eval's task on `/v1/completions` (the mode names follow AMD's
published MXFP4 evaluation). `gsm8k-chat` places the exact same lm-eval prompt
text in a single user message and lets the target's chat template apply the
reasoning level, which is the protocol used to compare checkpoints.
"""

import random

from . import scoring
from .arguments import (
    add_endpoint_arguments,
    add_reasoning_arguments,
    level_template_kwargs,
    serving_label,
    serving_metadata,
)
from .endpoint import Endpoint, chat_reply, completion_text
from .runner import binomial_ci, mean, run, write_summary

NUM_SHOTS = 5
DEFAULT_FEWSHOT_SEED = 1234
UNTIL = ["Question:", "</s>", "<|im_end|>"]

RAW_GENERATION = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 3072,
    },
    "nothink": {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": 1024,
    },
}


def load_split():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("this bench needs the `datasets` package; run through the flake") from error
    dataset = load_dataset("openai/gsm8k", "main")
    return list(dataset["train"]), list(dataset["test"])


def question_text(question, mode):
    if mode == "thinking":
        return "Question: %s\nAnswer:" % question
    return "Question: %s\nAnswer:<think>\n\n</think>\n\n" % question


def answer_text(answer):
    # lm-eval's default target_delimiter is a single space.
    return " %s" % answer.strip()


def build_prompt(shots, question, mode):
    parts = []
    for shot in shots:
        parts.append(question_text(shot["question"], mode) + answer_text(shot["answer"]))
    parts.append(question_text(question, mode))
    return "\n\n".join(parts)


def _prepare(args):
    """Return the evaluation items and lm-eval's per-document few-shot sets."""
    train, test = load_split()
    items = test[args.offset : args.offset + args.limit] if args.limit else test[args.offset:]
    fewshot = random.Random(args.fewshot_seed)
    for _ in range(args.offset):
        fewshot.sample(train, NUM_SHOTS)
    shots_by_item = [fewshot.sample(train, NUM_SHOTS) for _ in items]
    return items, shots_by_item


def _gsm8k_row(index, gold, text):
    strict = scoring.extract_strict(text)
    flexible = scoring.extract_flexible(text)
    return {
        "gold": gold,
        "output": text,
        "strict": strict,
        "flexible": flexible,
        "strict_ok": strict is not None and gold is not None and strict.lower() == gold.lower(),
        "flexible_ok": flexible is not None and gold is not None and flexible.lower() == gold.lower(),
    }


def run_raw(args):
    items, shots_by_item = _prepare(args)
    endpoint = Endpoint(args.base_url, args.timeout, args.retries)
    generation = dict(RAW_GENERATION[args.mode])
    for key, value in (
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("top_k", args.top_k),
        ("max_tokens", args.max_tokens),
    ):
        if value is not None:
            generation[key] = value

    def work(job):
        index, item, shots = job
        prompt = build_prompt(shots, item["question"], args.mode)
        reply = completion_text(endpoint.complete(args.model, prompt, generation, stop=UNTIL))
        row = _gsm8k_row(index, scoring.gsm8k_gold(item["answer"]), reply["text"])
        row.update(
            {
                "index": index,
                "finish_reason": reply["finish_reason"],
                "completion_tokens": reply["completion_tokens"],
            }
        )
        return row

    jobs = [(args.offset + i, item, shots_by_item[i]) for i, item in enumerate(items)]
    rows = run(jobs, work, args.concurrency, args.out + ".jsonl")
    rows.sort(key=lambda row: row["index"])

    strict_score = sum(row["strict_ok"] for row in rows) / len(rows)
    flex_score = sum(row["flexible_ok"] for row in rows) / len(rows)
    summary = {
        "model": args.model,
        "bench": "gsm8k",
        "protocol": "raw",
        "mode": args.mode,
        "items": len(rows),
        "fewshot_seed": args.fewshot_seed,
        "strict_match": strict_score,
        "flexible_extract": flex_score,
        "strict_match_ci95": binomial_ci(strict_score, len(rows)),
        "flexible_extract_ci95": binomial_ci(flex_score, len(rows)),
        "generation": generation,
        "serving": serving_metadata(args),
    }
    write_summary(args.out, summary)
    print(
        "RESULT gsm8k raw %s items=%d strict=%.4f flexible=%.4f serving=%s"
        % (args.mode, len(rows), strict_score, flex_score, serving_label(args))
    )


def run_chat(args):
    items, shots_by_item = _prepare(args)
    endpoint = Endpoint(args.base_url, args.timeout, args.retries)
    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    template_kwargs = level_template_kwargs(args.level)

    def work(job):
        index, item, shots = job
        prompt = build_prompt(shots, item["question"], "thinking")
        reply = chat_reply(
            endpoint.chat(
                args.model,
                prompt,
                generation,
                seed=args.seed,
                template_kwargs=template_kwargs,
            )
        )
        row = _gsm8k_row(index, scoring.gsm8k_gold(item["answer"]), reply["text"])
        row.update(
            {
                "index": index,
                "level": args.level,
                "seed": args.seed,
                "reasoning": reply["reasoning"],
                "reasoning_chars": len(reply["reasoning"]),
                "finish_reason": reply["finish_reason"],
                "completion_tokens": reply["completion_tokens"],
                "reasoning_tokens": reply["reasoning_tokens"],
            }
        )
        return row

    jobs = [(args.offset + i, item, shots_by_item[i]) for i, item in enumerate(items)]
    rows = run(jobs, work, args.concurrency, args.out + ".jsonl")
    rows.sort(key=lambda row: row["index"])

    strict_score = sum(row["strict_ok"] for row in rows) / len(rows)
    flex_score = sum(row["flexible_ok"] for row in rows) / len(rows)
    summary = {
        "model": args.model,
        "bench": "gsm8k",
        "protocol": "chat",
        "level": args.level,
        "seed": args.seed,
        "items": len(rows),
        "fewshot_seed": args.fewshot_seed,
        "strict_match": strict_score,
        "flexible_extract": flex_score,
        "strict_match_ci95": binomial_ci(strict_score, len(rows)),
        "flexible_extract_ci95": binomial_ci(flex_score, len(rows)),
        "truncated": sum(1 for row in rows if row["finish_reason"] == "length"),
        "parse_fail": sum(1 for row in rows if not row["strict_ok"] and not row["flexible_ok"]),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        "max_completion_tokens": max(
            (row["completion_tokens"] for row in rows if row["completion_tokens"] is not None),
            default=None,
        ),
        "mean_reasoning_tokens": mean(row["reasoning_tokens"] for row in rows),
        "generation": dict(generation, chat_template_kwargs=template_kwargs),
        "serving": serving_metadata(args),
    }
    write_summary(args.out, summary)
    print(
        "RESULT gsm8k chat level=%s seed=%d items=%d strict=%.4f flexible=%.4f serving=%s"
        % (args.level, args.seed, len(rows), strict_score, flex_score, serving_label(args))
    )


def register(subparsers):
    raw = subparsers.add_parser("gsm8k", help="GSM8K 5-shot via /v1/completions")
    raw.add_argument("--mode", choices=("thinking", "nothink"), default="thinking")
    raw.add_argument("--fewshot-seed", type=int, default=DEFAULT_FEWSHOT_SEED)
    raw.add_argument("--temperature", type=float, default=None)
    raw.add_argument("--top-p", type=float, default=None)
    raw.add_argument("--top-k", type=int, default=None)
    raw.add_argument("--max-tokens", type=int, default=None)
    add_endpoint_arguments(raw)
    raw.set_defaults(func=run_raw)

    chat = subparsers.add_parser("gsm8k-chat", help="GSM8K 5-shot via /v1/chat/completions")
    add_reasoning_arguments(chat)
    chat.add_argument("--fewshot-seed", type=int, default=DEFAULT_FEWSHOT_SEED)
    add_endpoint_arguments(chat)
    chat.set_defaults(func=run_chat)
