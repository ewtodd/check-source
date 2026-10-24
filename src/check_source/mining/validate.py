"""Static machine gates for drafted items (knoll-bench-plan.md section 7.2):
everything that can be checked without running the reference analysis.

Gold resolution, the leakage search for resolved values and the sandbox run
need execution and belong to the builder.
"""

import os
import re

from .link import git

KLASSES = ["coding", "reasoning", "mixed"]
DOMAINS = ["CAL", "SPEC", "PSD", "NUC", "SIM", "STAT", "SW", "DAQ"]
TYPES = ["compute", "interpret", "implement", "script", "debug", "pipeline"]
HORIZONS = ["single_turn", "tool_loop", "agentic"]
SENSITIVITIES = ["public", "internal", "restricted"]
CHECK_TYPES = ["numeric", "integer", "boolean", "string", "enum", "range", "exists", "expression"]
GOLD_KINDS = ["published", "committed_result", "recomputed", "derived", "manual"]
UNRESOLVED = "<resolved at build>"
LOCATOR = re.compile(r"^(row label=.+ column=\S+|regex: .+ group=\w+|hist: \S+ bin=\d+|json: /\S*)$")
GATES = [
    "schema",
    "enums",
    "no_typed_gold",
    "consistency",
    "tolerance",
    "hidden_reference",
    "source_paths",
    "inputs",
    "locators",
]
RUN_REFERENCE = re.compile(r"\broot\s+-[lbq]|\bpython3?\s+\S+\.py\b|\brun the (macro|script)\b", re.IGNORECASE)


def placeholders(text, repo):
    return text.replace("${REPO}", str(repo)).replace("${UM_ANSG}", os.path.expanduser("~/UM-ANSG"))


def check_item(item, repo, commit):
    """{gate: list of problem codes}; an empty list means the gate passed."""
    problems = {gate: [] for gate in GATES}

    for key in ("schema_version", "id", "title", "klass", "domain", "type", "horizon", "sensitivity", "prompt"):
        if key not in item:
            problems["schema"].append("missing_" + key)
    for table in ("source", "output", "scoring", "verification", "generation"):
        if not isinstance(item.get(table), dict):
            problems["schema"].append("missing_" + table)
    checks = item.get("checks")
    if not isinstance(checks, list) or not checks:
        problems["schema"].append("missing_checks")
        checks = []
    if not isinstance(item.get("prompt"), str):
        problems["schema"].append("prompt_not_top_level")

    for key, allowed in (
        ("klass", KLASSES),
        ("domain", DOMAINS),
        ("type", TYPES),
        ("horizon", HORIZONS),
        ("sensitivity", SENSITIVITIES),
    ):
        if key in item and item[key] not in allowed:
            problems["enums"].append("bad_" + key)
    for check in checks:
        if check.get("type") not in CHECK_TYPES:
            problems["enums"].append("bad_check_type")
        if (check.get("gold_ref") or {}).get("kind") not in GOLD_KINDS:
            problems["enums"].append("bad_gold_kind")

    for check in checks:
        kind = (check.get("gold_ref") or {}).get("kind")
        if "expected" in check and kind != "manual" and check["expected"] != UNRESOLVED:
            problems["no_typed_gold"].append("expected_written")

    keys = list((item.get("output") or {}).get("keys") or [])
    check_keys = [check.get("key") for check in checks]
    prompt = item.get("prompt") if isinstance(item.get("prompt"), str) else ""
    for key in keys:
        if key not in check_keys:
            problems["consistency"].append("key_without_check")
        if key not in prompt:
            problems["consistency"].append("key_not_in_prompt")
    for key in check_keys:
        if key not in keys:
            problems["consistency"].append("check_key_not_in_output")
    if len(set(check.get("id") for check in checks)) != len(checks):
        problems["consistency"].append("duplicate_check_id")

    for check in checks:
        if check.get("type") == "numeric":
            tol = check.get("tol") or {}
            if tol.get("abs") is None and tol.get("rel") is None:
                problems["tolerance"].append("no_tolerance")
            if "units" not in check:
                problems["tolerance"].append("no_units")
        if check.get("type") == "range":
            lo, hi = check.get("lo"), check.get("hi")
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo >= hi:
                problems["tolerance"].append("bad_range")
            elif check.get("required") and (hi - lo) <= 0.2 * max(abs(lo), abs(hi)):
                problems["tolerance"].append("narrow_required_range")

    # The solver sees only the prompt and the mounted inputs; an item that
    # names the reference code or says to run it tests nothing.
    references = list((item.get("source") or {}).get("paths") or [])
    references += [(check.get("gold_ref") or {}).get("script") or "" for check in checks]
    for path in filter(None, references):
        if os.path.basename(path) in prompt:
            problems["hidden_reference"].append("reference_named")
    if RUN_REFERENCE.search(prompt):
        problems["hidden_reference"].append("says_run_reference")
    for entry in item.get("inputs") or []:
        if not str(entry.get("path", "")).startswith("work/"):
            problems["hidden_reference"].append("input_not_under_work")

    source = item.get("source") or {}
    for path in source.get("paths") or []:
        if git(repo, "cat-file", "-e", "%s:%s" % (commit, path)) is None:
            problems["source_paths"].append("path_not_at_commit")
    if source.get("commit") and not str(commit).startswith(str(source["commit"])):
        problems["source_paths"].append("commit_mismatch")

    for entry in item.get("inputs") or []:
        origin = entry.get("from") or ""
        if not origin or not os.path.exists(placeholders(origin, repo)):
            problems["inputs"].append("input_missing")

    for check in checks:
        gold = check.get("gold_ref") or {}
        if gold.get("kind") in ("committed_result", "recomputed"):
            if not LOCATOR.match(str(gold.get("locator", ""))):
                problems["locators"].append("bad_locator")
        if gold.get("kind") == "recomputed":
            script = gold.get("script") or ""
            if git(repo, "cat-file", "-e", "%s:%s" % (commit, script)) is None:
                problems["locators"].append("script_not_at_commit")
            if not gold.get("command"):
                problems["locators"].append("no_command")
    return problems
