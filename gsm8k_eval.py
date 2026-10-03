"""GSM8K 5-shot evaluation against any OpenAI-compatible /v1/completions endpoint.

This mirrors the lm-evaluation-harness `gsm8k` task closely enough to compare
served checkpoints against each other and against published AMD/lm-eval numbers:
the same `doc_to_text`, the same 5-shot sampling from the train split with the
default seed 1234 (a fresh draw per evaluation document, as lm-eval does), the
same "\n\n" delimiters, `until` sequences, and the
strict-match / flexible-extract filter pair. The non-thinking variant mirrors
AMD's `gsm8k_nothink` custom task, which pre-closes an empty <think></think>
block.

Only the standard library plus `datasets` are required -- no torch, no vLLM
client, no lm-eval install. Results are written as a summary JSON and a JSONL
with every prompt, raw completion and extracted answer.

Example:
    python gsm8k_eval.py --base-url http://127.0.0.1:8199 \
        --model my-quant --mode thinking --limit 200 --concurrency 8 \
        --out results/my-quant-thinking.json
"""

import argparse
import concurrent.futures
import json
import math
import random
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_SEED = 1234
NUM_SHOTS = 5
UNTIL = ["Question:", "</s>", "<|im_end|>"]
IGNORE_PATTERNS = [",", r"\$", r"(?s).*#### ", r"\.$"]
STRICT_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

GENERATION = {
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


def clean(text):
    if text is None:
        return None
    for pattern in IGNORE_PATTERNS:
        text = re.sub(pattern, "", text)
    return text.strip()


def extract_strict(text):
    match = STRICT_PATTERN.search(text)
    return clean(match.group(1)) if match else None


def extract_flexible(text):
    last = None
    for match in FLEX_PATTERN.finditer(text):
        last = match.group(0)
    return clean(last) if last else None


def gold_answer(answer):
    match = STRICT_PATTERN.search(answer)
    return clean(match.group(1)) if match else None


def post_json(url, payload, timeout, retries):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception as error:  # noqa: BLE001 - retried above, raised below
            last_error = error
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("request failed after %d tries: %s" % (retries, last_error))


def load_split():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("this script needs the `datasets` package: pip install datasets") from error
    dataset = load_dataset("openai/gsm8k", "main")
    return list(dataset["train"]), list(dataset["test"])


def main():
    parser = argparse.ArgumentParser(
        prog="gsm8k-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8199", help="vLLM server root")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--mode", choices=["thinking", "nothink"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all 1319 test items")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="few-shot sampling seed")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200, help="per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--out", required=True, help="summary JSON path; raw rows go to <out>.jsonl")
    args = parser.parse_args()

    train, test = load_split()
    items = test[args.offset: args.offset + args.limit] if args.limit else test[args.offset:]

    # lm-eval's ContextSampler draws a fresh 5-shot set for every evaluation
    # document, in dataset order, advancing one shared Random(seed) instance
    # (fewshot_random_seed, default 1234) once per document. Replicate that
    # stream exactly so each item sees lm-eval's demonstrations.
    fewshot_rnd = random.Random(args.seed)
    for _ in range(args.offset):
        fewshot_rnd.sample(train, NUM_SHOTS)
    shots_by_item = [fewshot_rnd.sample(train, NUM_SHOTS) for _ in items]
    print("model=%s mode=%s items=%d seed=%d" % (args.model, args.mode, len(items), args.seed))

    generation = GENERATION[args.mode]
    url = args.base_url.rstrip("/") + "/v1/completions"
    lock = threading.Lock()
    rows = []
    done = [0]

    def run_one(indexed_item):
        index, item, shots = indexed_item
        prompt = build_prompt(shots, item["question"], args.mode)
        payload = {"model": args.model, "prompt": prompt, "stop": UNTIL, "n": 1}
        payload.update(generation)
        response = post_json(url, payload, args.timeout, args.retries)
        text = response["choices"][0]["text"]
        gold = gold_answer(item["answer"])
        strict = extract_strict(text)
        flexible = extract_flexible(text)
        row = {
            "index": index,
            "question": item["question"],
            "gold": gold,
            "output": text,
            "strict": strict,
            "flexible": flexible,
            "strict_ok": strict is not None and gold is not None and strict.lower() == gold.lower(),
            "flexible_ok": flexible is not None and gold is not None and flexible.lower() == gold.lower(),
        }
        with lock:
            rows.append(row)
            done[0] += 1
            if done[0] % 25 == 0:
                strict_score = sum(r["strict_ok"] for r in rows) / len(rows)
                flex_score = sum(r["flexible_ok"] for r in rows) / len(rows)
                print("  %d/%d strict=%.4f flexible=%.4f" % (done[0], len(items), strict_score, flex_score))
        return row

    with open(args.out + ".jsonl", "w") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_one, (args.offset + i, item, shots_by_item[i]))
                       for i, item in enumerate(items)]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row) + "\n")
                handle.flush()

    rows.sort(key=lambda row: row["index"])
    strict_score = sum(row["strict_ok"] for row in rows) / len(rows)
    flex_score = sum(row["flexible_ok"] for row in rows) / len(rows)
    strict_err = 1.96 * math.sqrt(strict_score * (1 - strict_score) / len(rows))
    flex_err = 1.96 * math.sqrt(flex_score * (1 - flex_score) / len(rows))
    summary = {
        "model": args.model,
        "mode": args.mode,
        "items": len(rows),
        "shots_seed": args.seed,
        "strict_match": strict_score,
        "strict_match_ci95": strict_err,
        "flexible_extract": flex_score,
        "flexible_extract_ci95": flex_err,
        "generation": generation,
    }
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("RESULT %s %s items=%d strict=%.4f+-%.4f flexible=%.4f+-%.4f"
          % (args.model, args.mode, len(rows), strict_score, strict_err, flex_score, flex_err))


if __name__ == "__main__":
    main()
