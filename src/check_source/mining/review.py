"""Human review (knoll-bench-plan.md section 7.3 and Appendix C).

`review` writes <private>/review/<set>.md, one section per drafted item with
its build status, prompt, inputs, checks, gold values, the reference output
lines each gold value was read from, and the checklist; and
<private>/review/<set>-verdicts.toml, one entry per item for the reviewer to
fill in (existing verdicts are kept). Neither is printed: the reviewer opens
them locally.

`approve` reads the verdicts and writes <private>/build/<set>/approved.jsonl:
items marked approve, with gold, verification.state = "human_verified" and
the reviewer's notes. A held item can be approved too; the human is the final
reviewer (plan section 7.3).
"""

import datetime
import json
import pathlib
import re
import subprocess
import tomllib

from . import build
from .draft import SETS

VERDICTS = ["approve", "reject", "rework"]
VOCAB = frozenset(SETS + VERDICTS + ["set", "items", "built", "held", "sheet_written", "verdicts", "approved", "pending"])
CHECKLIST = """- [ ] The gold value is right: the reference output line below shows the number the check holds.
- [ ] The prompt states every convention needed to land within tolerance (units, fit window, background model, weighting, calibration order, empty bins).
- [ ] Key names and units are unambiguous and match the output keys.
- [ ] No answer value or answer-bearing file is mounted or quoted.
- [ ] The task is one you actually perform, at a difficulty worth measuring.
- [ ] Sensitivity and publishable are correct.
- [ ] The tolerance is tight enough to separate a correct analysis from a plausible wrong one, and loose enough for legitimate run-to-run variation."""


def jsonl(path):
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_context(private, row, check, locator):
    """The reference output lines around the number the locator selects."""
    ref = check.get("gold_ref") or {}
    for command in filter(None, [ref.get("command"), build.compiled(ref.get("command") or "")]):
        run_dir = private / "build" / "runs" / build.run_key(
            pathlib.Path(row["repo"]), row["commit"], ref.get("workdir") or ".", command
        )
        log = run_dir / "stdout.log"
        if not log.exists():
            continue
        lines = log.read_text(errors="replace").splitlines()
        match = re.fullmatch(r"regex: (.+) group=(\w+)", locator or "")
        for number, line in enumerate(lines):
            if match and re.search(match.group(1), line):
                window = lines[max(0, number - 2) : number + 3]
                return run_dir, command, "\n".join(window)
        return run_dir, command, "(locator matches nothing in this run's output)"
    return None, ref.get("command"), "(no reference run)"


def run_review(private, set_name):
    draft_dir = private / "items" / "draft" / set_name
    build_dir = private / "build" / set_name
    rows = [row for row in jsonl(draft_dir / "index.jsonl") if row.get("state") in ("draft", "needs_human")]
    built = {entry["item"]["id"]: entry for entry in jsonl(build_dir / "items.jsonl")}
    held = {entry["id"]: entry for entry in jsonl(build_dir / "review.jsonl")}
    out_dir = private / "review"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    verdicts_path = out_dir / ("%s-verdicts.toml" % set_name)
    existing = tomllib.loads(verdicts_path.read_text()).get("items", {}) if verdicts_path.exists() else {}

    sections = [
        "# Knoll bench review: %s" % set_name,
        "",
        "Generated %s. Record a verdict per item in `%s` (approve, reject or rework, plus notes), then run "
        "`check-source knoll-mine approve --set %s`." % (datetime.date.today(), verdicts_path, set_name),
        "",
        "Status: **built** passed every machine gate and has its gold from a sandboxed reference run; "
        "**held** has a reason a human must judge; **unresolved** has no gold yet.",
        "",
    ]
    for row in sorted(rows, key=lambda row: (row["id"] not in built, row["id"])):
        item = tomllib.loads((draft_dir / ("%s.toml" % row["id"])).read_text())
        entry = built.get(row["id"])
        hold = held.get(row["id"])
        if entry:
            status, gold = "built (machine verified)", {c["id"]: c.get("expected") for c in entry["item"]["checks"]}
        elif hold:
            status, gold = "held: %s" % hold["reason"], hold.get("gold") or {}
            if hold.get("gate_problems"):
                status += " (gate problems: %s)" % ", ".join(hold["gate_problems"])
        else:
            status, gold = "unresolved", {}
        sections += ["---", "", "## %s" % row["id"], "", "**%s**  " % item.get("title", ""), "Status: %s  " % status]
        sections += [
            "Project %s, domain %s, type %s, difficulty %s, sensitivity %s"
            % (row["project"], item.get("domain"), item.get("type"), item.get("difficulty"), item.get("sensitivity")),
            "",
        ]
        flags = (item.get("generation") or {}).get("needs_human") or []
        if flags:
            sections += ["Model flags:"] + ["- %s" % flag for flag in flags] + [""]
        sections += ["### Prompt (what the solver sees)", "", "```text", str(item.get("prompt", "")).strip(), "```", ""]
        sections += ["### Inputs", ""]
        hashes = {i["path"]: i["sha256"] for i in (entry or {}).get("inputs", [])}
        for mounted in item.get("inputs") or []:
            sha = hashes.get(mounted.get("path"))
            sections.append(
                "- `%s` <- `%s`%s" % (mounted.get("path"), mounted.get("from"), "  (sha256 %s...)" % sha[:12] if sha else "")
            )
        sections += ["", "### Checks", ""]
        for check in item.get("checks") or []:
            ref = check.get("gold_ref") or {}
            tol = check.get("tol") or {}
            sections.append(
                "- **%s** (`%s`, %s, units %s, tolerance %s): gold = %s"
                % (
                    check.get("id"),
                    check.get("key"),
                    check.get("type"),
                    check.get("units", "-"),
                    ", ".join("%s %s" % kv for kv in tol.items()) or ("range %s..%s" % (check.get("lo"), check.get("hi"))),
                    gold.get(check.get("id"), "not resolved"),
                )
            )
            if ref.get("kind") == "recomputed":
                run_dir, command, context = gold_context(private, row, check, ref.get("locator"))
                sections += [
                    "  - reference: `%s` in `%s`, script `%s`" % (command, ref.get("workdir"), ref.get("script")),
                    "  - locator: `%s`" % ref.get("locator"),
                    "  - run directory (log, plots and outputs): `%s`" % run_dir,
                    "  - output around the gold value:",
                    "",
                    "```text",
                    context,
                    "```",
                ]
            else:
                sections.append("  - gold kind %s: %s" % (ref.get("kind"), ref.get("note", "")))
        sections += ["", "### Checklist", "", CHECKLIST, ""]
    (out_dir / ("%s.md" % set_name)).write_text("\n".join(sections) + "\n")

    lines = [
        "# One entry per item: verdict = approve | reject | rework; notes is free text.",
        "# Reviewer and date are recorded by `check-source knoll-mine approve`.",
        "",
    ]
    for row in sorted(rows, key=lambda row: row["id"]):
        previous = existing.get(row["id"], {})
        lines += [
            '[items."%s"]' % row["id"],
            "verdict = %s" % json.dumps(previous.get("verdict", "")),
            "notes = %s" % json.dumps(previous.get("notes", "")),
            "",
        ]
    verdicts_path.write_text("\n".join(lines))
    return {
        "set": set_name,
        "items": len(rows),
        "built": sum(row["id"] in built for row in rows),
        "held": sum(row["id"] in held for row in rows),
        "sheet_written": True,
    }


def run_approve(private, set_name):
    draft_dir = private / "items" / "draft" / set_name
    build_dir = private / "build" / set_name
    verdicts = tomllib.loads((private / "review" / ("%s-verdicts.toml" % set_name)).read_text()).get("items", {})
    built = {entry["item"]["id"]: entry for entry in jsonl(build_dir / "items.jsonl")}
    held = {entry["id"]: entry for entry in jsonl(build_dir / "review.jsonl")}
    reviewer = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()
    summary = {"set": set_name, "verdicts": {}, "approved": 0, "pending": 0}
    approved = []
    for item_id, verdict in sorted(verdicts.items()):
        choice = verdict.get("verdict", "")
        if choice not in VERDICTS:
            summary["pending"] += 1
            continue
        summary["verdicts"][choice] = summary["verdicts"].get(choice, 0) + 1
        if choice != "approve":
            continue
        if item_id in built:
            item = built[item_id]["item"]
        elif item_id in held and held[item_id].get("gold"):
            item = tomllib.loads((draft_dir / ("%s.toml" % item_id)).read_text())
            for check in item.get("checks") or []:
                if check.get("id") in held[item_id]["gold"]:
                    check["expected"] = held[item_id]["gold"][check["id"]]
        else:
            summary["pending"] += 1
            continue
        item.setdefault("verification", {})["state"] = "human_verified"
        item["verification"]["human"] = {
            "reviewer": reviewer,
            "date": datetime.date.today().isoformat(),
            "method": "review sheet",
            "notes": verdict.get("notes", ""),
        }
        approved.append(item)
        summary["approved"] += 1
    with open(build_dir / "approved.jsonl", "w") as out:
        for item in approved:
            out.write(json.dumps(item, sort_keys=True) + "\n")
    return summary
