"""Argument groups shared by the benches."""

LEVELS = ("none", "medium", "xhigh")


def add_endpoint_arguments(parser):
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server root")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600, help="per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every item in the split")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", required=True, help="summary JSON path; raw rows go to <out>.jsonl")
    # Serving configuration is not introspectable over the API, so it is passed
    # in and recorded with every result. On this stack the activation format and
    # the KV dtype move accuracy independently of the stored weights, so a score
    # without them is not comparable.
    parser.add_argument("--weights", default=None, help="serving weight format, e.g. 'MXFP4 RTN'")
    parser.add_argument(
        "--activations", default=None, help="serving activation format, e.g. 'fp8 per-token (W4A8)'"
    )
    parser.add_argument("--kv-cache", default=None, help="KV-cache dtype, e.g. 'fp16' or 'fp8'")


def serving_metadata(args):
    return {
        "weights": args.weights,
        "activations": args.activations,
        "kv_cache": args.kv_cache,
    }


def serving_label(args):
    parts = [value for value in (args.weights, args.activations, args.kv_cache) if value]
    return " / ".join(parts) if parts else "unlabeled"


def add_reasoning_arguments(parser):
    """Per-request sampling seed plus the reasoning-level knobs of the chat
    template."""
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--seed", type=int, required=True, help="per-request sampling seed")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)


def level_template_kwargs(level):
    """Map a level onto the Qwen3.8 chat template's thinking controls.

    `none` means thinking disabled (no effort value disables it); the other
    levels are the template's own `reasoning_effort` values.
    """
    if level == "none":
        return {"enable_thinking": False}
    return {"reasoning_effort": level}
