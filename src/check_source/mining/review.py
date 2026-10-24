"""Human review (knoll-bench-plan.md section 7.3 and Appendix C).

`review` renders <private>/review/<set>/index.html: one card per drafted item
with its build status, what the reviewer has to decide, the prompt as the
solver sees it, the checks with tolerance, gold value and the reference
output lines the gold was read from, the inputs, and the plots the reference
run produced, converted to PNG (ROOT in the sandbox writes PostScript under
.png names, and some plots are PDF). Verdicts live in
<private>/review/<set>-verdicts.toml; with --serve, a server on 127.0.0.1
writes them from the page's buttons. Nothing is printed: the reviewer opens
the page locally.

`approve` reads the verdicts and writes <private>/build/<set>/approved.jsonl:
items marked approve, with gold, verification.state = "human_verified" and
the reviewer's notes. A held item can be approved too; the human is the final
reviewer (plan section 7.3).
"""

import datetime
import html
import http.server
import json
import pathlib
import re
import shutil
import subprocess
import tomllib
import urllib.parse

from . import build
from .draft import SETS

VERDICTS = ["approve", "reject", "rework"]
VOCAB = frozenset(
    SETS + VERDICTS + ["set", "items", "built", "held", "unresolved", "plots", "verdicts", "approved", "pending"]
)
CHECKLIST = [
    "The gold value is right: the reference output shown holds the number the check expects.",
    "The prompt states every convention needed to land within tolerance (units, fit window, background model, "
    "weighting, calibration order, empty bins).",
    "Key names and units are unambiguous and match the output keys.",
    "No answer value or answer-bearing file is mounted or quoted.",
    "The task is one you actually perform, at a difficulty worth measuring.",
    "Sensitivity and publishable are correct.",
    "The tolerance separates a correct analysis from a plausible wrong one and allows legitimate run-to-run variation.",
]
QUESTIONS = {
    "answer_bearing_input": "An input the reference code itself writes is mounted. Does that file contain the answer, "
    "or only an intermediate the solver should start from?",
    "gold value appears in the prompt": "The prompt states a number within tolerance of the gold (often a fit's starting "
    "value taken from the reference code). Reject, or rework the prompt so it does not give the answer away.",
    "unresolved: locator_no_match": "No unique place in the reference output holds this number. Reject, or write the "
    "locator by hand from the output below.",
    "unresolved: run_failed": "The reference run failed. Reject, or fix the environment and rebuild.",
    "reference exited non-zero after printing its gold": "The reference crashed after printing the gold (PyROOT teardown "
    "in the pinned dev shell). The step that prints the gold completed; approve if the value and its context look right.",
}
STOP = {"ansg", "yap", "78mbr", "the", "and", "for", "from", "with", "keV", "mev", "item", "music", "infra"}
GS_DEVICE = ["-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=png16m", "-r110", "-dFirstPage=1", "-dLastPage=1"]
PAGE_STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f5f5f4; color: #1c1917; }
header { padding: 16px 24px; background: #1c1917; color: #fafaf9; position: sticky; top: 0; z-index: 2; }
header a { color: #fafaf9; margin-right: 10px; font-size: 13px; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
section { background: #fff; border: 1px solid #e7e5e4; border-radius: 8px; padding: 16px 20px; margin: 18px 0; }
h2 { margin: 0 0 4px; font-size: 18px; }
.meta { color: #57534e; font-size: 13px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.built { background: #dcfce7; color: #166534; } .held { background: #fef3c7; color: #92400e; }
.unresolved { background: #fee2e2; color: #991b1b; } .verdict { background: #e0e7ff; color: #3730a3; }
.decide { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 8px 12px; margin: 10px 0; }
pre { background: #fafaf9; border: 1px solid #e7e5e4; padding: 10px; overflow-x: auto; white-space: pre-wrap; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; } td, th { border: 1px solid #e7e5e4; padding: 4px 6px; text-align: left; vertical-align: top; }
.plots img { max-width: 100%; border: 1px solid #e7e5e4; margin: 6px 0; } .thumbs img { width: 240px; margin: 4px; border: 1px solid #e7e5e4; }
form { margin-top: 10px; } textarea { width: 100%; min-height: 50px; font: inherit; }
"""


def jsonl(path):
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def item_runs(private, row, item):
    """[(check, run_dir, command)] for the item's recomputed checks."""
    found = []
    for check in item.get("checks") or []:
        ref = check.get("gold_ref") or {}
        if ref.get("kind") != "recomputed":
            continue
        candidates = []
        for command in filter(None, [ref.get("command"), build.compiled(ref.get("command") or "")]):
            run_dir = private / "build" / "runs" / build.run_key(
                pathlib.Path(row["repo"]), row["commit"], ref.get("workdir") or ".", command
            )
            if (run_dir / "stdout.log").exists():
                status = (run_dir / "exit_status").read_text().strip() if (run_dir / "exit_status").exists() else ""
                candidates.append((status != "0", check, run_dir, command))
        # A clean run first: the interpreted attempt of a macro that needed
        # compiling has a log too.
        found.append(min(candidates, key=lambda c: c[0])[1:] if candidates else (check, None, ref.get("command")))
    return found


def gold_context(run_dir, locator):
    if run_dir is None:
        return "(no reference run)"
    lines = (run_dir / "stdout.log").read_text(errors="replace").splitlines()
    match = re.fullmatch(r"regex: (.+) group=(\w+)", locator or "")
    for number, line in enumerate(lines):
        if match and re.search(match.group(1), line):
            return "\n".join(lines[max(0, number - 3) : number + 4])
    return "(the locator matches nothing in this run's output)"


def real_format(path):
    head = path.read_bytes()[:8]
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"%!PS"):
        return "ps"
    return None


def ghostscript():
    done = subprocess.run(
        ["nix", "build", "nixpkgs#ghostscript", "--no-link", "--print-out-paths"], capture_output=True, text=True
    )
    paths = done.stdout.split()
    return pathlib.Path(paths[-1]) / "bin" / "gs" if paths else None


def run_plots(run_dir, out_dir, gs):
    """Every plot the run produced, as PNG under out_dir; [(name, png path)]."""
    found = []
    if run_dir is None:
        return found
    for root in (run_dir / "src", run_dir / "upper"):
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if not path.is_file() or path.suffix.lower() not in (".png", ".pdf", ".ps", ".eps"):
                continue
            kind = real_format(path)
            if kind is None:
                continue
            name = str(path.relative_to(root))
            target = out_dir / (re.sub(r"[^A-Za-z0-9._/-]", "_", name).rsplit(".", 1)[0] + ".png")
            if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "png":
                    shutil.copyfile(path, target)
                elif gs:
                    extra = ["-dEPSCrop"] if kind == "ps" else []
                    subprocess.run([str(gs), *GS_DEVICE, *extra, "-sOutputFile=%s" % target, str(path)], capture_output=True)
            if target.exists():
                found.append((name, target))
    return found


def relevant(name, item):
    words = re.findall(r"[a-z0-9]{4,}", ("%s %s" % (item.get("id", ""), item.get("title", ""))).lower())
    lowered = name.lower()
    return any(word in lowered for word in words if word not in STOP)


def read_verdicts(path):
    return tomllib.loads(path.read_text()).get("items", {}) if path.exists() else {}


def write_verdicts(path, ids, verdicts):
    lines = [
        "# One entry per item: verdict = approve | reject | rework; notes is free text.",
        "# Reviewer and date are recorded by `check-source knoll-mine approve`.",
        "",
    ]
    for item_id in sorted(ids):
        entry = verdicts.get(item_id, {})
        lines += [
            '[items."%s"]' % item_id,
            "verdict = %s" % json.dumps(entry.get("verdict", "")),
            "notes = %s" % json.dumps(entry.get("notes", "")),
            "",
        ]
    path.write_text("\n".join(lines))


def card(private, row, item, status, kind, decide, gold, entry, verdict, plots_dir, gs, served):
    out = ['<section id="%s">' % html.escape(row["id"])]
    badge = '<span class="badge %s">%s</span>' % (kind, html.escape(status))
    if verdict.get("verdict"):
        badge += ' <span class="badge verdict">verdict: %s</span>' % html.escape(verdict["verdict"])
    out.append("<h2>%s</h2><div>%s</div>" % (html.escape(item.get("title", row["id"])), badge))
    out.append(
        '<div class="meta">%s &middot; project %s &middot; %s / %s &middot; difficulty %s &middot; sensitivity %s</div>'
        % tuple(
            html.escape(str(x))
            for x in (
                row["id"],
                row["project"],
                item.get("domain"),
                item.get("type"),
                item.get("difficulty"),
                item.get("sensitivity"),
            )
        )
    )
    out.append('<div class="decide"><b>What you need to decide:</b><ul>%s</ul></div>' % "".join(
        "<li>%s</li>" % html.escape(text) for text in decide
    ))
    if served:
        out.append('<form method="post" action="/verdict"><input type="hidden" name="id" value="%s">' % html.escape(row["id"]))
        for choice in VERDICTS:
            checked = " checked" if verdict.get("verdict") == choice else ""
            out.append('<label><input type="radio" name="verdict" value="%s"%s> %s</label> ' % (choice, checked, choice))
        out.append(
            '<textarea name="notes" placeholder="notes">%s</textarea><button type="submit">Save verdict</button></form>'
            % html.escape(verdict.get("notes", ""))
        )
    else:
        out.append('<p class="meta">Record the verdict in the verdicts file, or open this page with --serve to use buttons.</p>')
    out.append("<h3>Prompt (what the solver sees)</h3><pre>%s</pre>" % html.escape(str(item.get("prompt", "")).strip()))
    out.append("<h3>Checks</h3><table><tr><th>check</th><th>type / units</th><th>tolerance</th><th>gold</th><th>source</th></tr>")
    runs = item_runs(private, row, item)
    plots = []
    for check, run_dir, command in runs:
        tol = check.get("tol") or {}
        tolerance = ", ".join("%s %s" % kv for kv in tol.items()) or "range %s..%s" % (check.get("lo"), check.get("hi"))
        ref = check.get("gold_ref") or {}
        out.append(
            "<tr><td>%s<br><code>%s</code></td><td>%s / %s</td><td>%s</td><td><b>%s</b></td>"
            "<td><code>%s</code> in <code>%s</code><br>locator <code>%s</code><pre>%s</pre></td></tr>"
            % tuple(
                html.escape(str(x))
                for x in (
                    check.get("id"),
                    check.get("key"),
                    check.get("type"),
                    check.get("units", "-"),
                    tolerance,
                    gold.get(check.get("id"), "not resolved"),
                    command,
                    ref.get("workdir"),
                    ref.get("locator"),
                    gold_context(run_dir, ref.get("locator")),
                )
            )
        )
        plots += run_plots(run_dir, plots_dir / (run_dir.name if run_dir else "none"), gs)
    for check in item.get("checks") or []:
        if (check.get("gold_ref") or {}).get("kind") != "recomputed":
            out.append(
                "<tr><td>%s</td><td>%s</td><td>range %s..%s</td><td>-</td><td>%s</td></tr>"
                % tuple(html.escape(str(x)) for x in (check.get("id"), check.get("type"), check.get("lo"), check.get("hi"),
                                                      (check.get("gold_ref") or {}).get("note", "")))
            )
    out.append("</table>")
    hashes = {mounted["path"]: mounted["sha256"] for mounted in (entry or {}).get("inputs", [])}
    out.append("<h3>Inputs</h3><ul>%s</ul>" % "".join(
        "<li><code>%s</code> &larr; <code>%s</code>%s</li>"
        % (html.escape(str(m.get("path"))), html.escape(str(m.get("from"))),
           " (sha256 %s&hellip;)" % hashes[m.get("path")][:12] if m.get("path") in hashes else "")
        for m in item.get("inputs") or []
    ))
    seen, chosen, others = set(), [], []
    for name, png in plots:
        if png in seen:
            continue
        seen.add(png)
        (chosen if relevant(name, item) else others).append((name, png))
    root = plots_dir.parent
    if chosen:
        out.append('<h3>Plots related to this item</h3><div class="plots">%s</div>' % "".join(
            '<figure><img loading="lazy" src="%s"><figcaption class="meta">%s</figcaption></figure>'
            % (html.escape(str(png.relative_to(root))), html.escape(name)) for name, png in chosen[:12]
        ))
    if others:
        out.append('<details><summary>All other plots from the reference run (%d)</summary><div class="thumbs">%s</div></details>' % (
            len(others),
            "".join('<a href="%s"><img loading="lazy" src="%s" title="%s"></a>'
                    % (html.escape(str(p.relative_to(root))), html.escape(str(p.relative_to(root))), html.escape(n))
                    for n, p in others[:80]),
        ))
    flags = (item.get("generation") or {}).get("needs_human") or []
    if flags:
        out.append("<h3>Model flags</h3><ul>%s</ul>" % "".join("<li>%s</li>" % html.escape(str(f)) for f in flags))
    out.append("<h3>Checklist</h3><ul>%s</ul></section>" % "".join("<li>%s</li>" % html.escape(c) for c in CHECKLIST))
    return "\n".join(out)


def render(private, set_name, served=False):
    draft_dir = private / "items" / "draft" / set_name
    build_dir = private / "build" / set_name
    out_dir = private / "review" / set_name
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows = [row for row in jsonl(draft_dir / "index.jsonl") if row.get("state") in ("draft", "needs_human")]
    built = {entry["item"]["id"]: entry for entry in jsonl(build_dir / "items.jsonl")}
    held = {entry["id"]: entry for entry in jsonl(build_dir / "review.jsonl")}
    verdicts_path = private / "review" / ("%s-verdicts.toml" % set_name)
    verdicts = read_verdicts(verdicts_path)
    write_verdicts(verdicts_path, [row["id"] for row in rows], verdicts)
    gs = ghostscript()
    counts = {"built": 0, "held": 0, "unresolved": 0}
    cards, nav = [], []
    order = sorted(rows, key=lambda row: (row["id"] not in built, row["id"] not in held, row["id"]))
    for row in order:
        item = tomllib.loads((draft_dir / ("%s.toml" % row["id"])).read_text())
        entry, hold = built.get(row["id"]), held.get(row["id"])
        if entry:
            status, kind = "built (machine verified)", "built"
            gold = {c["id"]: c.get("expected") for c in entry["item"]["checks"]}
            decide = ["Go through the checklist below; approve if it all holds."]
        elif hold and not hold["reason"].startswith("unresolved"):
            status, kind, gold = "held: %s" % hold["reason"], "held", hold.get("gold") or {}
            decide = [QUESTIONS.get(code, code) for code in hold.get("gate_problems") or []]
            if hold["reason"] in QUESTIONS:
                decide.append(QUESTIONS[hold["reason"]])
            decide += ["Model flag: %s" % flag for flag in hold.get("model_flags") or []]
        else:
            reason = hold["reason"] if hold else "unresolved"
            status, kind, gold = reason, "unresolved", (hold or {}).get("gold") or {}
            decide = [QUESTIONS.get(reason, reason)]
        counts[kind] += 1
        verdict = verdicts.get(row["id"], {})
        cards.append(card(private, row, item, status, kind, decide or ["Use the checklist."], gold, entry, verdict,
                          out_dir / "plots", gs, served))
        nav.append('<a href="#%s">%s%s</a>' % (html.escape(row["id"]), html.escape(row["id"]),
                                                " (%s)" % verdict["verdict"] if verdict.get("verdict") else ""))
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Knoll review: %s</title><style>%s</style></head><body>"
        "<header><b>Knoll bench review: %s</b> &middot; %d built, %d held, %d unresolved &middot; verdicts in %s"
        "<div>%s</div></header><main>%s</main></body></html>"
        % (html.escape(set_name), PAGE_STYLE, html.escape(set_name), counts["built"], counts["held"],
           counts["unresolved"], html.escape(str(verdicts_path)), " ".join(nav), "\n".join(cards))
    )
    (out_dir / "index.html").write_text(page)
    plots = sum(1 for _ in (out_dir / "plots").rglob("*.png")) if (out_dir / "plots").exists() else 0
    return {"set": set_name, "items": len(rows), **counts, "plots": plots}


def run_review(private, set_name):
    return render(private, set_name)


def serve(private, set_name, port, log):
    """Serve the review page on 127.0.0.1 and save verdicts from its forms.
    Request lines go to `log`, never to the terminal."""
    out_dir = private / "review" / set_name
    verdicts_path = private / "review" / ("%s-verdicts.toml" % set_name)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir), **kwargs)

        def log_message(self, format, *args):
            log.write((format % args) + "\n")
            log.flush()

        def do_POST(self):
            if self.path != "/verdict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            form = urllib.parse.parse_qs(self.rfile.read(length).decode())
            item_id = form.get("id", [""])[0]
            verdicts = read_verdicts(verdicts_path)
            choice = form.get("verdict", [""])[0]
            verdicts[item_id] = {"verdict": choice if choice in VERDICTS else "", "notes": form.get("notes", [""])[0]}
            write_verdicts(verdicts_path, set(verdicts), verdicts)
            render(private, set_name, served=True)
            self.send_response(303)
            self.send_header("Location", "/#%s" % urllib.parse.quote(item_id))
            self.end_headers()

    render(private, set_name, served=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()


def run_approve(private, set_name):
    draft_dir = private / "items" / "draft" / set_name
    build_dir = private / "build" / set_name
    verdicts = read_verdicts(private / "review" / ("%s-verdicts.toml" % set_name))
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
            "method": "review page",
            "notes": verdict.get("notes", ""),
        }
        approved.append(item)
        summary["approved"] += 1
    with open(build_dir / "approved.jsonl", "w") as out:
        for item in approved:
            out.write(json.dumps(item, sort_keys=True) + "\n")
    return summary
