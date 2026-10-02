# gsm8k-eval

A dependency-light GSM8K 5-shot evaluator for OpenAI-compatible
`/v1/completions` endpoints (vLLM). It is used to compare quantized checkpoints
of the same base model under one identical harness.

Only Python's standard library plus [`datasets`](https://pypi.org/project/datasets/)
are needed -- no `torch`, no `lm-eval`, no vLLM client. The server does the
tokenizing and generation; the client only builds prompts, calls the endpoint,
extracts answers and scores them.

## Why another harness

`lm-evaluation-harness` is the reference implementation, but installing it in a
serving environment is heavy (it depends on `torch`) and its task plumbing
changes between versions. This script reproduces the parts that matter for
comparing quantized checkpoints:

* `doc_to_text`: `Question: {question}\nAnswer:`
* `num_fewshot: 5`, sampled **without replacement** from the train split with
  `random.Random(seed).sample(train, 5)` (default seed `1234`, matching the
  harness default)
* `"\n\n"` few-shot delimiter, target delimiter `" "`
* `until: ["Question:", "</s>", "<|im_end|>"]`
* `strict-match`: first `#### (-?[0-9.,]+)` match
* `flexible-extract`: last match of `(-?[$0-9.,]{2,})|(-?[0-9]+)`
* `regexes_to_ignore`: `","`, `"\\$"`, `"(?s).*#### "`, `"\\.$"`

Generation settings follow AMD's published MXFP4 evaluation:

| mode | temperature | top_p | top_k | presence_penalty | repetition_penalty | max new tokens |
|---|---:|---:|---:|---:|---:|---:|
| `thinking` | 1.0 | 0.95 | 20 | 0.0 | 1.0 | 3072 |
| `nothink` | 0.7 | 0.80 | 20 | 1.5 | 1.0 | 1024 |

`nothink` mirrors AMD's custom `gsm8k_nothink` task: the prompt pre-closes an
empty thinking block (`Answer:<think>\n\n</think>\n\n`).

## Usage

With Nix (flakes):

```bash
nix run github:ewtodd/gsm8k-eval -- \
  --base-url http://127.0.0.1:8199 \
  --model my-quant \
  --mode thinking \
  --limit 200 \
  --concurrency 8 \
  --out results/my-quant-thinking.json
```

or from a checkout:

```bash
nix run . -- --help
nix develop
```

Without Nix:

```bash
python -m venv .venv
.venv/bin/pip install datasets
.venv/bin/python gsm8k_eval.py \
  --base-url http://127.0.0.1:8199 \
  --model my-quant \
  --mode thinking \
  --limit 200 \
  --concurrency 8 \
  --out results/my-quant-thinking.json
```

* `--limit 0` (default) evaluates all 1319 test items; a limited run prints a
  binomial 95% interval in the summary.
* The server's `--served-model-name` must match `--model`.
* Each run writes `<out>` (summary) and `<out>.jsonl` (one row per item with the
  prompt, raw completion and both extracted answers).

Serve with a reasoning/thinking model, for example:

```bash
vllm serve /path/to/checkpoint \
  --served-model-name my-quant \
  --tensor-parallel-size 2 \
  --max-model-len 16384
```

## Calibration

The harness was calibrated by running it against AMD's published
[`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)
checkpoint and comparing with AMD's model-card numbers (GSM8K 5-shot;
94.996% flexible / 95.30% strict thinking, 89.92% / 89.76% non-thinking). Small
differences are expected from sampler seeding and serving environment; the point
of the harness is that every checkpoint is measured identically.

## License

MIT
