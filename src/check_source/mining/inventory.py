"""Phase 0 inventory (knoll-bench-plan.md section 14, Appendix B pass 1): a
local model describes every analysis code file of a project at a pinned
commit, as the unit list item drafting works from.

Files are read with `git show <commit>:<path>`, so working trees and checked-out
branches are never touched. Two sets:

- um_ansg: YAP-PSD and 78mBr in ~/UM-ANSG, pinned at `main`;
- g0_repos: the MUSIC and shared-infrastructure repositories the link stage
  found, each pinned at its HEAD. Their paths and content stay blind.

Results go to <private>/inventory/<set>.jsonl, one row per file.
"""

import concurrent.futures
import json
import pathlib
import re
import threading
import traceback

from . import llm
from .classify import PROJECTS
from .digest import SOURCES
from .link import git

MODEL = "unholy/dsv4-flash"
GENERATION = {"max_tokens": 16000, "temperature": 0.2, "reasoning_effort": "low"}
UM_ANSG = pathlib.Path("~/UM-ANSG").expanduser()
UM_ANSG_PROJECTS = {"yap_psd": "YAP-PSD", "78mbr": "78mBr"}
G0_PROJECTS = {"music", "infra"}
SETS = ["um_ansg", "g0_repos"]
CODE = re.compile(r"\.(cpp|C|cxx|cc|hpp|hh|h|py|jl|f90|f|sh)$")
SKIP = re.compile(r"(^|/)(third_party|external|vendor|build|__pycache__|\.github|tests?|testing|docs?)(/|$)")
MAX_CHARS = 150000
KINDS = ["analysis_stage", "library", "plotting", "config", "utility", "other"]
VOCAB = frozenset(
    SETS
    + PROJECTS
    + KINDS
    + list(SOURCES)
    + [
        "set",
        "model",
        "repos",
        "files_listed",
        "files_skipped",
        "files_too_big",
        "units",
        "failed",
        "kind",
        "project",
        "wip",
        "units_with_quantities",
        "quantities",
        "dry_run",
        "prompt_tokens",
        "completion_tokens",
    ]
    + sorted(llm.LOCAL_MODELS)
)

PROMPT = """You are indexing a completed physics analysis repository for a benchmark of experimental data-analysis work. Describe the one file below as an analysis unit. Use only what the file and the listed context state; do not interpret physics beyond them. Quote short snippets as evidence.

Answer with one JSON object and nothing else:
{"language": "C++ ROOT macro | Python | ...",
 "kind": "analysis_stage | library | plotting | config | utility | other",
 "description": "one factual line: what it computes",
 "inputs": [{"path": "file or pattern it reads", "what": "...", "evidence": "snippet"}],
 "outputs": [{"path": "file or pattern it writes", "what": "...", "evidence": "snippet"}],
 "command": "how it is run, if the file or README says, else empty",
 "wip": false, "wip_evidence": "snippet marking it unfinished, or empty",
 "quantities": [{"name": "a number it computes (fit parameter, coefficient, half-life, efficiency, figure of merit)", "how": "printed | written to file | plotted only | returned", "where": "stdout | output file and object name | plot", "units": "...", "evidence": "snippet"}],
 "depends_on": ["repository files it includes, imports or calls"],
 "needs_human": ["anything you could not determine"]}

kind is analysis_stage only for a file that turns data into results; shared helpers are library.

Project: %s
File: %s
Other files in the project:
%s

Project README (excerpt):
%s

File content:
%s
"""


def listing(repo, commit, prefix=""):
    names = (git(repo, "ls-tree", "-r", "--name-only", commit, "--", prefix or ".") or "").splitlines()
    return names


def readme(repo, commit, prefix):
    for name in ("README.md", "README", "README.rst"):
        text = git(repo, "show", "%s:%s%s" % (commit, prefix + "/" if prefix else "", name))
        if text:
            return text[:4000]
    return "-"


def units_for(set_name, private):
    """[(project, repo, commit, prefix)] for the set."""
    if set_name == "um_ansg":
        commit = git(UM_ANSG, "rev-parse", "main")
        return [(project, UM_ANSG, commit, prefix) for project, prefix in UM_ANSG_PROJECTS.items()]
    repos = {}
    for source in SOURCES:
        path = private / "link" / ("%s.jsonl" % source)
        if not path.exists():
            continue
        with open(path) as handle:
            for task in map(json.loads, handle):
                if task["repo"] and task["project"] in G0_PROJECTS:
                    repos.setdefault(task["repo"], []).append(task["project"])
    found = []
    for repo, projects in sorted(repos.items()):
        commit = git(repo, "rev-parse", "HEAD")
        if commit:
            found.append((max(set(projects), key=projects.count), pathlib.Path(repo), commit, ""))
    return found


def count(table, key, amount=1):
    table[key] = table.get(key, 0) + amount


def run_stage(private, set_name, model_name=MODEL, dry_run=False, concurrency=4):
    summary = {
        "set": set_name,
        "model": model_name,
        "dry_run": dry_run,
        "repos": 0,
        "files_listed": 0,
        "files_skipped": 0,
        "files_too_big": 0,
        "units": 0,
        "failed": 0,
        "kind": {},
        "project": {},
        "wip": 0,
        "units_with_quantities": 0,
        "quantities": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    jobs = []
    repos = set()
    for project, repo, commit, prefix in units_for(set_name, private):
        repos.add(repo)
        names = listing(repo, commit, prefix)
        code = [name for name in names if CODE.search(name)]
        summary["files_listed"] += len(code)
        about = readme(repo, commit, prefix)
        for name in code:
            if SKIP.search(name):
                summary["files_skipped"] += 1
                continue
            jobs.append((project, repo, commit, name, code, about))
    summary["repos"] = len(repos)
    if dry_run:
        summary["units"] = len(jobs)
        return summary

    model = llm.Model(model_name)
    out_dir = private / "inventory"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    out_path = out_dir / ("%s.jsonl" % set_name)
    lock = threading.Lock()
    rows = []
    # Resume: keep earlier rows that succeeded (or are too big to retry) at the
    # same commit, and redo only the rest.
    if out_path.exists():
        pinned = {(str(job[1]), job[2]) for job in jobs}
        with open(out_path) as handle:
            kept = [
                row
                for row in map(json.loads, handle)
                if row["error"] in (None, "too_big") and (row["repo"], row["commit"]) in pinned
            ]
        done = {(row["repo"], row["commit"], row["path"]) for row in kept}
        jobs = [job for job in jobs if (str(job[1]), job[2], job[3]) not in done]
        rows += kept

    def work(job):
        project, repo, commit, name, code, about = job
        content = git(repo, "show", "%s:%s" % (commit, name)) or ""
        row = {"project": project, "repo": str(repo), "commit": commit, "path": name}
        if len(content) > MAX_CHARS:
            row["error"] = "too_big"
        else:
            others = "\n".join(other for other in code if other != name)[:20000]
            try:
                unit, prompt_tokens, completion_tokens = model.ask_json(
                    PROMPT % (project, name, others, about, content), GENERATION
                )
            except RuntimeError:
                traceback.print_exc()
                unit, prompt_tokens, completion_tokens = None, 0, 0
            row.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if unit is None or unit.get("kind") not in KINDS:
                row["error"] = "bad_answer"
            else:
                row.update(unit=unit, error=None)
        with lock:
            rows.append(row)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, jobs))
    with open(out_path, "w") as out:
        for row in sorted(rows, key=lambda row: (row["repo"], row["path"])):
            out.write(json.dumps(row) + "\n")

    for row in rows:
        summary["prompt_tokens"] += row.get("prompt_tokens", 0)
        summary["completion_tokens"] += row.get("completion_tokens", 0)
        if row["error"] == "too_big":
            summary["files_too_big"] += 1
            continue
        if row["error"]:
            summary["failed"] += 1
            continue
        unit = row["unit"]
        summary["units"] += 1
        count(summary["kind"], unit["kind"])
        count(summary["project"], row["project"])
        summary["wip"] += unit.get("wip") is True
        quantities = unit.get("quantities") or []
        summary["units_with_quantities"] += bool(quantities)
        summary["quantities"] += len(quantities)
    return summary
