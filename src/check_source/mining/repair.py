"""Locator repair: when a drafted check's locator does not select exactly one
number in its reference run's stdout, a local model rewrites it from that
output with every number masked (so it cannot copy a gold value into the
item), and the new locator must select exactly one number in the real output
before it replaces the old one in the item's TOML. The original TOML is kept
beside it as <id>.toml.orig.

Results go to <private>/items/draft/<set>/repairs.jsonl.
"""

import json
import pathlib
import re
import tomllib
import traceback

from . import build, llm
from .draft import SETS

MODEL = "unholy/dsv4-flash"
GENERATION = {"max_tokens": 16000, "temperature": 0.1, "reasoning_effort": "low"}
MASK = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MAX_OUTPUT = 40000
LITERAL = re.compile(r"""locator\s*=\s*('''.*?'''|'[^'\n]*'|"(?:[^"\\\n]|\\.)*")""", re.DOTALL)
OUTCOMES = ["repaired", "unlocatable", "unverified", "model_failed", "no_run", "already_ok"]
VOCAB = frozenset(SETS + OUTCOMES + ["set", "checks", "outcome"])

PROMPT = """A benchmark item's check reads its gold value from the stdout of a reference analysis run through a regex locator, but the locator does not select exactly one number. Write a corrected locator.

What the item asks the solver (for context on which quantity the check holds):
%s

The check: key %s, units %s.
Current locator: %s
Why it failed: %s

The run's stdout follows with every digit replaced by d, so 4 reads d and 30512.3 reads ddddd.d: the columns and magnitudes are visible, the values are not. Write a Python regex that matches exactly one place in the real output and captures the quantity itself (in a table, the value column, not a row number, error or step size) in a group named "value". Use the surrounding text (labels, table headers, the order of blocks) to make the match unique; (?s), lookarounds and non-greedy spans are allowed. If no unique, stable place exists (the value is never printed, or only in repeated unlabeled tables you cannot tell apart), answer with locator null.

Answer with one JSON object: {"locator": "regex: <pattern> group=value", "reason": "..."} or {"locator": null, "reason": "..."}

stdout (numbers masked):
%s
"""


def toml_literal(text):
    return "'''%s'''" % text if "'" in text or "\n" in text else "'%s'" % text


def replace_locator(toml_text, old, new):
    """toml_text with the locator literal whose value is `old` replaced by `new`."""
    for match in LITERAL.finditer(toml_text):
        try:
            value = tomllib.loads("x = " + match.group(1))["x"]
        except tomllib.TOMLDecodeError:
            continue
        if value == old:
            start, end = match.span(1)
            return toml_text[:start] + toml_literal(new) + toml_text[end:]
    return None


def run_stage(private, set_name, model_name=MODEL):
    model = llm.Model(model_name)
    draft_dir = private / "items" / "draft" / set_name
    with open(draft_dir / "index.jsonl") as handle:
        rows = [row for row in map(json.loads, handle) if row.get("state") in ("draft", "needs_human")]
    summary = {"set": set_name, "checks": 0, "outcome": {}}
    log_rows = []
    for row in rows:
        path = draft_dir / ("%s.toml" % row["id"])
        text = path.read_text()
        item = tomllib.loads(text)
        repo = pathlib.Path(row["repo"])
        for check in item.get("checks") or []:
            ref = check.get("gold_ref") or {}
            if ref.get("kind") != "recomputed" or not str(ref.get("locator", "")).startswith("regex: "):
                continue
            workdir, command = ref.get("workdir") or ".", ref.get("command") or ""
            log = None
            for candidate in filter(None, [command, build.compiled(command)]):
                run_dir = private / "build" / "runs" / build.run_key(repo, row["commit"], workdir, candidate)
                status = run_dir / "exit_status"
                if status.exists() and status.read_text().strip() == "0":
                    log = run_dir / "stdout.log"
                    break
            if log is None:
                outcome = "no_run"
            else:
                stdout = log.read_text(errors="replace")
                try:
                    build.locate(ref["locator"], stdout)
                    outcome = "already_ok"
                except build.Unresolved as failure:
                    outcome = repair(model, item, check, ref, str(failure), stdout)
                    if outcome.startswith("regex: "):
                        patched = replace_locator(text, ref["locator"], outcome)
                        if patched is None:
                            outcome = "unverified"
                        else:
                            original = path.with_suffix(".toml.orig")
                            if not original.exists():
                                original.write_text(text)
                            text = patched
                            path.write_text(text)
                            log_rows.append({"id": row["id"], "check": check.get("id"), "locator": outcome})
                            outcome = "repaired"
            summary["checks"] += 1
            summary["outcome"][outcome] = summary["outcome"].get(outcome, 0) + 1
    with open(draft_dir / "repairs.jsonl", "a") as out:
        for entry in log_rows:
            out.write(json.dumps(entry) + "\n")
    return summary


def shape(match):
    """A number with its digits replaced by d: 4 -> d, 30512.3 -> ddddd.d.
    Column and magnitude stay readable; values do not."""
    return re.sub(r"\d", "d", match.group())


def repair(model, item, check, ref, failure, stdout):
    """A verified new locator string, or one of OUTCOMES."""
    masked = MASK.sub(shape, stdout)
    if len(masked) > MAX_OUTPUT:
        masked = masked[: MAX_OUTPUT // 2] + "\n...\n" + masked[-MAX_OUTPUT // 2 :]
    prompt = PROMPT % (
        str(item.get("prompt", ""))[:6000],
        check.get("key"),
        check.get("units", "-"),
        ref["locator"],
        failure,
        masked,
    )
    try:
        answer, _, _ = model.ask_json(prompt, GENERATION)
    except RuntimeError:
        traceback.print_exc()
        return "model_failed"
    if not answer:
        return "model_failed"
    locator = answer.get("locator")
    if not locator:
        return "unlocatable"
    if not str(locator).startswith("regex: "):
        locator = "regex: %s group=value" % locator
    try:
        build.locate(locator, stdout)
    except (build.Unresolved, re.error):
        return "unverified"
    return locator
