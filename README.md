# check-source

Reference benchmarks ("check sources") for OpenAI-compatible LLM endpoints.
A check source is the small, calibrated reference a detector is verified
against; these are the ones a served checkpoint is verified against. The point
is that every checkpoint is measured identically.

Only Python's standard library plus
[`datasets`](https://pypi.org/project/datasets/),
[`sympy`](https://pypi.org/project/sympy/) and
[`math_verify`](https://pypi.org/project/math-verify/) are needed -- no `torch`,
no `lm-eval`, no vLLM client. The server tokenizes and generates; this client
builds prompts, calls the endpoint, extracts answers and scores them.

## Benches

| subcommand | dataset | protocol | metric |
|---|---|---|---|
| `gsm8k` | `openai/gsm8k` test, 5-shot | raw `/v1/completions`, AMD thinking/nothink generation | strict-match + flexible-extract |
| `gsm8k-chat` | same | `/v1/chat/completions`, one user turn, reasoning level | strict-match + flexible-extract |
| `aime` | AIME 2024/2025/2026 | 0-shot chat, `\boxed{}` | integer exact match |
| `math500` | `HuggingFaceH4/MATH-500` | 4-shot Minerva chat | `exact_match` + `math_verify` |
| `knoll` | (planned) nuclear-physics problems | 0-shot chat | numeric with tolerance + `math_verify` |

The chat benches apply the reasoning level through the checkpoint's own chat
template: `none` -> `enable_thinking: false`; `medium` / `xhigh` ->
`reasoning_effort`. Decoding follows Qwen's recommendation
(T=1.0, top_p=0.95, top_k=20) with an explicit per-request `--seed`, so runs are
repeatable and seeds are an arm of the comparison. Truncated responses count as
incorrect.

## Fidelity to lm-evaluation-harness

The GSM8K filters, the AIME string normalizer and the Minerva MATH normalizer
(including the fixed four few-shot samples) are copied from
lm-evaluation-harness 0.4.13, and `math_verify` is pinned to the versions that
harness installs (see `nix/math-verify-overlay.nix`). A checkpoint measured here
therefore lines up with one measured there; the one deliberate divergence is
dropping the five-second `sympy` alarm in the Minerva scorer, because
`signal.alarm` only works on the main thread and the runner is threaded.

## Usage

With Nix (flakes):

```bash
nix run github:ewtodd/check-source -- gsm8k-chat \
  --base-url http://127.0.0.1:8100 \
  --model qwen3.8-27b \
  --level medium \
  --seed 1234 \
  --max-tokens 65536 \
  --limit 200 \
  --concurrency 4 \
  --out results/qwen-medium-s1234.json
```

```bash
nix run . -- aime --years 24,25,26 --level xhigh --seed 0 --max-tokens 32768 \
  --base-url http://127.0.0.1:8100 --model qwen3.8-27b --out results/aime-xhigh.json

nix run . -- math500 --level medium --seed 0 --max-tokens 65536 \
  --base-url http://127.0.0.1:8100 --model qwen3.8-27b --limit 200 --out results/math500.json
```

or from a checkout:

```bash
nix develop
python -m check_source gsm8k-chat --help
```

Without Nix, `pip install datasets sympy math-verify` in a venv and run
`python -m check_source` with `PYTHONPATH=src`.

* `--limit 0` (default) evaluates every item in the split; a limited run prints
  a binomial 95% interval in the summary.
* The server's `--served-model-name` must match `--model`.
* Each run writes `<out>` (summary) and `<out>.jsonl` (one row per item with the
  output, reasoning trace, extracted answer and token counts).

## Calibration

The raw GSM8K bench was calibrated against AMD's published
[`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)
checkpoint and AMD's model-card numbers (GSM8K 5-shot; 94.996% flexible /
95.30% strict thinking, 89.92% / 89.76% non-thinking). Small differences are
expected from sampler seeding and serving environment; the point of the bench is
that every checkpoint is measured identically.

## Knoll bench

`src/check_source/knoll.py` documents the intended item schema for the planned
experimental nuclear-physics / radiation-detection problems. The subcommand
exists but exits with a message until the problem set is added.

## License

MIT
