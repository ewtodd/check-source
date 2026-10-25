"""Knoll bench, Runner A (knoll-bench-plan.md section 8): experimental
nuclear-physics data-analysis items, one chat turn plus one sandboxed run of
the program the model writes.

Items are not in this repository: they are private (plan section 3.3) and
are read from --items, a JSONL of compiled items (approved.jsonl from the
review stage). Only human_verified items run unless --allow-draft.

The model gets the item's prompt and EPILOGUE, nothing else. Its program runs
once, under the item's pinned dev shell, in bwrap: the filesystem read-only,
/home and /labdata replaced by empty directories (so neither the reference
analysis nor the gold is reachable), no network, and the item's inputs
mounted read-only under work/. Checks are scored from RESULTS.txt
(results_txt_v1, plan section 8.4). A missing program, a failed run or a
missing key fails its checks; nothing is retried.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from .arguments import (
    add_endpoint_arguments,
    add_reasoning_arguments,
    level_template_kwargs,
    serving_label,
    serving_metadata,
)
from .endpoint import Endpoint, chat_reply
from .runner import binomial_ci, mean, run, write_summary

EPILOGUE = """

Solve this by writing one self-contained program. The input files are in the directory work/ beside it. Use Python 3 (PyROOT, numpy and scipy are available) or a ROOT C++ macro whose entry function is `void solution()`. Put the whole program in a single fenced code block (```python or ```cpp). It runs once, non-interactively and without network access, and must write RESULTS.txt in the current directory with one `key = value` line per requested key."""

FENCE = re.compile(r"```([A-Za-z+]*)[^\n]*\n(.*?)```", re.DOTALL)
PYTHON = {"python", "py", "python3"}
CPP = {"cpp", "c++", "cxx", "c", "root"}
RESULT_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)")
EXECUTION_TIMEOUT_S = 900


def extract_program(text):
    """(language, source) of the last python or C++ code block, or None."""
    for language, source in reversed(FENCE.findall(text or "")):
        language = language.lower()
        if language in PYTHON:
            return "python", source
        if language in CPP:
            return "cpp", source
    return None


def parse_results(text):
    """results_txt_v1: `key = value` lines, keys case-insensitive; a key given
    twice with different values is ambiguous and dropped."""
    values, ambiguous = {}, set()
    for line in (text or "").splitlines():
        match = RESULT_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2).rstrip(",;")
        if key in values and values[key] != value:
            ambiguous.add(key)
        values[key] = value
    return {key: value for key, value in values.items() if key not in ambiguous}


def number(value):
    try:
        return float(re.sub(r"[^0-9eE.+-]", "", str(value)))
    except ValueError:
        return None


def score_check(check, got):
    kind = check.get("type")
    if got is None:
        return False
    if kind == "numeric":
        value, expected = number(got), check.get("expected")
        if value is None or not isinstance(expected, (int, float)):
            return False
        tol = check.get("tol") or {}
        return abs(value - expected) <= max(tol.get("abs") or 0, (tol.get("rel") or 0) * abs(expected))
    if kind == "integer":
        value = number(got)
        return value is not None and value == int(value) and int(value) == int(check.get("expected"))
    if kind == "range":
        value = number(got)
        return value is not None and check.get("lo") <= value <= check.get("hi")
    if kind == "boolean":
        return str(got).lower() in ("true", "yes", "1") if check.get("expected") else str(got).lower() in ("false", "no", "0")
    if kind in ("string", "enum"):
        return " ".join(str(got).split()).lower() == " ".join(str(check.get("expected")).split()).lower()
    if kind == "exists":
        return True
    return False


def resolve(origin, repo, repos):
    """An input's host path: ${REPO} is the item's own repository, any other
    ${NAME} comes from --repo NAME=PATH."""
    origin = origin.replace("${REPO}", repo)
    return re.sub(r"\$\{([A-Za-z0-9_-]+)\}", lambda match: repos[match.group(1)], origin)


def flake_ref(base, commit, shell):
    """`github:owner/repo` takes the revision as a path segment; anything else
    (git+file://, git+https://) as ?rev=."""
    if base.startswith(("github:", "gitlab:", "sourcehut:")):
        return "%s/%s#%s" % (base.rstrip("/"), commit, shell)
    return "%s?rev=%s#%s" % (base, commit, shell)


def execute(item, language, source, repos, flakes, timeout):
    """Run the program in the sandbox; (exit status or "timeout", RESULTS.txt
    text or None, combined output tail)."""
    source_info = item.get("source") or {}
    repo_name = source_info.get("repo", "")
    repo = repos.get(repo_name)
    shell = str((item.get("env") or {}).get("nix_develop", "")).rpartition("#")[2] or "default"
    flake = flake_ref(flakes.get(repo_name, "git+file://%s" % repo), source_info.get("commit"), shell)
    with tempfile.TemporaryDirectory(prefix="knoll-") as scratch:
        workdir = pathlib.Path(scratch)
        (workdir / "work").mkdir()
        (workdir / "home").mkdir()
        program = workdir / ("solution.py" if language == "python" else "solution.C")
        program.write_text(source)
        mounts = []
        for entry in item.get("inputs") or []:
            origin = resolve(entry.get("from", ""), repo, repos)
            name = pathlib.Path(entry.get("path", "")).name
            (workdir / "work" / name).touch()
            mounts += ["--ro-bind", os.path.realpath(origin), str(workdir / "work" / name)]
        command = ["python3", program.name] if language == "python" else ["root", "-l", "-b", "-q", program.name]
        sandbox = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--tmpfs", "/home",
            "--tmpfs", "/labdata",
            "--dev", "/dev",
            "--tmpfs", "/dev/shm",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(workdir), str(workdir),
            *mounts,
            "--setenv", "HOME", str(workdir / "home"),
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PYTHONUNBUFFERED", "1",
            "--unshare-net",
            "--die-with-parent",
            "--chdir", str(workdir),
            "--", *command,
        ]  # fmt: skip
        try:
            done = subprocess.run(
                ["nice", "-n", "10", "nix", "develop", flake, "-c", *sandbox],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status, output = str(done.returncode), done.stdout + done.stderr
        except subprocess.TimeoutExpired:
            status, output = "timeout", ""
        results = workdir / "RESULTS.txt"
        return status, results.read_text(errors="replace") if results.exists() else None, output[-4000:]


def load_items(path, allow_draft):
    items = []
    with open(os.path.expanduser(path)) as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            item = entry.get("item", entry)
            state = (item.get("verification") or {}).get("state")
            if state == "human_verified" or allow_draft:
                items.append(item)
    return sorted(items, key=lambda item: item["id"])


def main(args):
    if not shutil.which("bwrap"):
        raise SystemExit("the knoll bench runs model programs in bubblewrap (bwrap); it is not on PATH")
    repos = dict(pair.split("=", 1) for pair in args.repo)
    repos = {name: os.path.expanduser(path) for name, path in repos.items()}
    flakes = dict(pair.split("=", 1) for pair in args.flake)
    items = load_items(args.items, args.allow_draft)
    if not items:
        raise SystemExit("no runnable items in %s (only human_verified items run unless --allow-draft)" % args.items)
    unknown = sorted({(item.get("source") or {}).get("repo") for item in items} - set(repos))
    if unknown:
        raise SystemExit("no local checkout for repo(s) %s; pass --repo NAME=PATH" % ", ".join(map(str, unknown)))
    if args.limit:
        items = items[args.offset : args.offset + args.limit]
    elif args.offset:
        items = items[args.offset :]

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    endpoint = Endpoint(args.base_url, args.timeout, args.retries, api_key=api_key)
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
                item["prompt"] + EPILOGUE,
                generation,
                seed=args.seed,
                template_kwargs=template_kwargs,
            )
        )
        program = extract_program(reply["text"])
        status, results_text, output = (None, None, "")
        if program:
            timeout = int((item.get("env") or {}).get("timeout_s") or EXECUTION_TIMEOUT_S)
            status, results_text, output = execute(item, program[0], program[1], repos, flakes, timeout)
        results = parse_results(results_text)
        required = [check for check in item.get("checks") or [] if check.get("required", True)]
        checks = []
        for check in item.get("checks") or []:
            got = results.get(str(check.get("key", "")).lower())
            checks.append(
                {
                    "id": check.get("id"),
                    "key": check.get("key"),
                    "expected": check.get("expected"),
                    "got": got,
                    "pass": score_check(check, got),
                    "required": check in required,
                }
            )
        passed = [c["pass"] for c in checks if c["required"]]
        return {
            "item_id": item["id"],
            "domain": item.get("domain"),
            "type": item.get("type"),
            "output": reply["text"],
            "reasoning": reply["reasoning"],
            "language": program[0] if program else None,
            "exit_status": status,
            "execution_tail": output,
            "results_txt": results_text,
            "checks": checks,
            "item_score": sum(passed) / len(passed) if passed else 0.0,
            "item_pass": bool(passed) and all(passed),
            "finish_reason": reply["finish_reason"],
            "completion_tokens": reply["completion_tokens"],
            "reasoning_tokens": reply["reasoning_tokens"],
        }

    rows = run(items, work, args.concurrency, args.out + ".jsonl")
    rows.sort(key=lambda row: row["item_id"])

    score = mean(row["item_score"] for row in rows)
    pass_rate = sum(row["item_pass"] for row in rows) / len(rows)
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row["item_score"])
    summary = {
        "model": args.model,
        "bench": "knoll",
        "items_file": os.path.basename(args.items),
        "level": args.level,
        "seed": args.seed,
        "items": len(rows),
        "mean_score": score,
        "pass_rate": pass_rate,
        "pass_rate_ci95": binomial_ci(pass_rate, len(rows)),
        "mean_score_by_domain": {domain: mean(values) for domain, values in sorted(by_domain.items())},
        "no_program": sum(1 for row in rows if row["language"] is None),
        "run_failed": sum(1 for row in rows if row["exit_status"] not in (None, "0")),
        "no_results_txt": sum(1 for row in rows if row["language"] and row["results_txt"] is None),
        "truncated": sum(1 for row in rows if row["finish_reason"] == "length"),
        "mean_completion_tokens": mean(row["completion_tokens"] for row in rows),
        "generation": dict(generation, chat_template_kwargs=template_kwargs),
        "serving": serving_metadata(args),
    }
    write_summary(args.out, summary)
    print(
        "RESULT knoll level=%s seed=%d items=%d mean_score=%.4f pass_rate=%.4f serving=%s"
        % (args.level, args.seed, len(rows), score, pass_rate, serving_label(args))
    )


def register(subparsers):
    parser = subparsers.add_parser("knoll", help="Knoll bench: data-analysis items run in a sandbox (items are private)")
    parser.add_argument("--items", required=True, help="JSONL of compiled items, e.g. approved.jsonl from the review stage")
    parser.add_argument(
        "--repo",
        action="append",
        default=["UM-ANSG=~/UM-ANSG"],
        metavar="NAME=PATH",
        help="where an item's ${REPO} input paths resolve for its source.repo (repeatable): a checkout, or "
        "a directory holding just the input files at their repository paths",
    )
    parser.add_argument(
        "--flake",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="flake for a repo's dev shells, e.g. UM-ANSG=github:ewtodd/UM-ANSG (repeatable); "
        "default git+file:// on the --repo path",
    )
    parser.add_argument("--allow-draft", action="store_true", help="also run items not yet human_verified")
    parser.add_argument("--api-key-env", default=None, help="environment variable holding the endpoint's API key")
    add_reasoning_arguments(parser)
    add_endpoint_arguments(parser)
    parser.set_defaults(func=main)
