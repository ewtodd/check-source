"""MATH-500 with the 4-shot Minerva prompt.

The shot text and the two metrics (`exact_match`, `math_verify`) come from
lm-eval's `minerva_math500` task; the four demonstrations are the same fixed
first-n samples the harness uses. The whole prompt goes into one user message
so the target's chat template still controls the thinking level.
"""

from . import scoring
from .arguments import (
    add_endpoint_arguments,
    add_reasoning_arguments,
    level_template_kwargs,
    serving_label,
    serving_metadata,
)
from .endpoint import Endpoint, chat_reply
from .runner import binomial_ci, mean, run, write_summary

MINERVA_SHOTS = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]


def build_prompt(problem):
    parts = []
    for shot in MINERVA_SHOTS:
        parts.append("Problem:\n%s\n\nSolution: %s\n\n" % (shot["problem"], shot["solution"]))
    parts.append("Problem:\n%s\n\nSolution:" % problem)
    return "".join(parts)


def load_items():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("this bench needs the `datasets` package; run through the flake") from error

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "index": index,
            "problem": row["problem"],
            "solution": row["solution"],
            "gold": row["answer"],
            "subject": row.get("subject"),
        }
        for index, row in enumerate(dataset)
    ]


def main(args):
    items = load_items()
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
                build_prompt(item["problem"]),
                generation,
                seed=args.seed,
                template_kwargs=template_kwargs,
            )
        )
        metrics = scoring.minerva_score(item["solution"], item["gold"], reply["text"])
        return {
            "index": item["index"],
            "subject": item["subject"],
            "gold": item["gold"],
            "output": reply["text"],
            "reasoning": reply["reasoning"],
            "exact_match": metrics["exact_match"],
            "math_verify": metrics["math_verify"],
            "finish_reason": reply["finish_reason"],
            "completion_tokens": reply["completion_tokens"],
            "reasoning_tokens": reply["reasoning_tokens"],
        }

    rows = run(items, work, args.concurrency, args.out + ".jsonl")
    rows.sort(key=lambda row: row["index"])

    exact = sum(row["exact_match"] for row in rows) / len(rows)
    verified = sum(row["math_verify"] for row in rows) / len(rows)
    summary = {
        "model": args.model,
        "bench": "math500",
        "level": args.level,
        "seed": args.seed,
        "items": len(rows),
        "exact_match": exact,
        "exact_match_ci95": binomial_ci(exact, len(rows)),
        "math_verify": verified,
        "math_verify_ci95": binomial_ci(verified, len(rows)),
        "truncated": sum(1 for row in rows if row["finish_reason"] == "length"),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        "mean_reasoning_tokens": mean(row["reasoning_tokens"] for row in rows),
        "generation": dict(generation, chat_template_kwargs=template_kwargs),
        "serving": serving_metadata(args),
    }
    write_summary(args.out, summary)
    print(
        "RESULT math500 level=%s seed=%d items=%d exact=%.4f math_verify=%.4f serving=%s"
        % (args.level, args.seed, len(rows), exact, verified, serving_label(args))
    )


def register(subparsers):
    parser = subparsers.add_parser("math500", help="MATH-500, 4-shot Minerva")
    add_reasoning_arguments(parser)
    add_endpoint_arguments(parser)
    parser.set_defaults(func=main)
