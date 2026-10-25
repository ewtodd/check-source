"""Item drafting (knoll-bench-plan.md section 6.1, Appendix B pass 2): a local
model writes one item TOML per job, then the static gates run on it.

Jobs come from two places:

- um_ansg: inventory units of YAP-PSD and 78mBr that are analysis stages, not
  marked work in progress, with at least one computed quantity (G1);
- g0: the best-ranked G0 clusters (MUSIC and shared infrastructure) with a
  repository, read at the repository's HEAD, or at the cited commit for g2
  clusters, whose diff the model also sees. These stay blind.

The model never writes a gold number: checks carry a gold_ref locator that
the builder resolves later. Items go to <private>/items/draft/<set>/<id>.toml,
with an index of jobs, gate results and states beside them.
"""

import concurrent.futures
import json
import os
import pathlib
import re
import threading
import tomllib
import traceback

from . import llm, validate
from .classify import PROJECTS
from .inventory import UM_ANSG, UM_ANSG_PROJECTS
from .link import git

MODEL = "unholy/dsv4-flash"
# High effort can think past 32k tokens and leave no answer, and on large units
# past an hour; a cut-off, empty or timed-out answer is retried once at low
# effort.
GENERATION = {"max_tokens": 100000, "temperature": 0.3, "reasoning_effort": "high"}
FALLBACK = dict(GENERATION, reasoning_effort="low")
TIMEOUT_S = 3 * 3600
SETS = ["um_ansg", "g0"]
G0_TOP = 40
# maitred gives the DeepSeek upstream an hour per request; G0 prompts are kept
# smaller than UM-ANSG's so a draft finishes inside it.
CONTEXT_CHARS = 200000
G0_CONTEXT_CHARS = 120000
STATES = ["draft", "needs_human", "rejected", "skipped", "failed"]
VOCAB = frozenset(
    SETS
    + PROJECTS
    + STATES
    + validate.GATES
    + validate.DOMAINS
    + validate.TYPES
    + validate.GOLD_KINDS
    + sorted(llm.LOCAL_MODELS)
    + [
        "set",
        "model",
        "jobs",
        "state",
        "gate_failed",
        "domain",
        "type",
        "gold_kind",
        "project",
        "checks",
        "prompt_tokens",
        "completion_tokens",
    ]
)

PROMPT = """You are writing one item for the Knoll bench, a private benchmark of experimental nuclear-physics data analysis. It measures whether a model can do the analysis work this group actually does. You are given one analysis unit from a verified, completed analysis, with its reference code. Write one item as TOML, following the rules exactly.

The solver never sees the repository, the reference code or anything in the context below. It gets only your prompt and the files you list in [[inputs]], mounted at their `path` under work/. The builder, not the solver, runs the reference code to obtain the gold value.

Rules:
1. Pick one physically meaningful quantity the reference code computes and prints or writes: a fit result, peak position or width, calibration coefficient, half-life, efficiency, figure of merit, discrimination metric. Not event counts, timings, file sizes or other bookkeeping.
2. Write the prompt as a self-contained analysis specification: what the data are (file, tree, branch names, their meaning and units), what to compute, and every convention needed to land within tolerance (selections and cuts, binning, fit range, fit model including background, weighting, calibration order, units), described in analysis terms. Take the conventions from the reference and dependency code; when the code makes an arbitrary choice, state it.
3. Never name, quote or point to the reference code, its functions, the repository or its scripts, and never tell the solver to run anything that exists in it. The solver must do the analysis itself.
4. Never write a gold number. Every check that has a true value gets expected = "<resolved at build>" and a gold_ref saying where the builder reads it: kind = "recomputed" with script (repository path of the reference code), command (how the builder runs it), workdir (repository-relative), and locator; or kind = "published" with a note naming the publication (it will need human sign-off). Ranges with kind = "manual" are allowed only as loose sanity bounds and never as the headline check.
5. Locators use exactly one of: 'regex: <python regex> group=<name>' over the reference code's stdout (derive it from the code's print statements), 'json: /pointer', 'row label=<label> column=<header>', 'hist: <object name> bin=<n>'.
6. Inputs: only files from the data listing, written as path = "work/<name>" and from = "${REPO}/<repository-relative path>". A file the reference code only adds branches to may be an input; say which branches the solver should use. Never mount result files or anything that contains the answer.
7. The prompt names each output key and its units and asks for `key = value` lines in RESULTS.txt. It must not contain any answer value.
8. Tolerances: numeric checks have tol.rel or tol.abs, tight enough to separate a correct analysis from a plausible wrong one, with a note if wider than 10%%.
9. If the reference result depends on unseeded randomness, if two competent analyses could disagree beyond the tolerance, or if no quantity is scorable, still write the TOML but set skip = true and list the reasons in [generation] needs_human.
10. TOML layout: every top-level key (including prompt) comes before the first [table]. Output only the TOML.

Domains: CAL energy, gain or pixel calibration; SPEC spectroscopy and peak analysis; PSD pulse-shape discrimination and waveforms; NUC nuclear data (half-lives, branching, decay); SIM simulation and transport; STAT statistics and fitting methodology; SW analysis software; DAQ acquisition and signal processing.
Enums: klass coding | reasoning | mixed; type compute | interpret | implement | script | debug | pipeline; horizon single_turn | tool_loop | agentic; sensitivity %s.

Shape (values are placeholders):
schema_version = 1
id = "%s-<short-slug>"
title = "..."
klass = "reasoning"
domain = "NUC"
type = "script"
horizon = "tool_loop"
est_runtime_s = 600
difficulty = 3
sensitivity = "%s"
publishable = false
skip = false
prompt = '''
...
'''

[source]
repo = "%s"
commit = "%s"
paths = ["<reference code path>"]

[[inputs]]
path = "work/<name>"
from = "${REPO}/<repository-relative path>"
tree = "<tree name or empty>"

[env]
nix_develop = "<flake#shell>"
network = false
timeout_s = 900

[output]
contract = "results_txt"
extraction = "results_txt_v1"
keys = ["<key>"]

[[checks]]
id = "<id>"
key = "<key>"
type = "numeric"
expected = "<resolved at build>"
units = "<units>"
required = true
[checks.tol]
rel = 0.01
[checks.gold_ref]
kind = "recomputed"
script = "<path>"
command = "<command>"
workdir = "<dir>"
locator = 'regex: <pattern with (?P<value>...)> group=value'

[scoring]
mode = "partial"
pass_threshold = 1.0

[verification]
state = "draft"

[generation]
method = "%s"
generator_model = "%s"
needs_human = []
notes = ""

Context
-------
%s
"""


DATA_FILE = re.compile(r"\.(root|dat|bin|h5|hdf5|csv|tsv|txt|npz|npy|pkl|parquet|json|evt|mid|fits)$", re.IGNORECASE)


def data_links(repo, prefix):
    """(repository-relative path, real directory) for each data root under
    prefix: symlinks to directories outside the repository (root_files ->
    /labdata/...), and git-ignored directories of the working tree holding
    data-like files (data kept in the checkout, caches). `git archive` carries
    neither, so the sandbox mounts each as an overlay."""
    found = []
    for path in (repo / prefix).rglob("*"):
        if path.is_symlink():
            target = path.resolve()
            if target.is_dir() and not target.is_relative_to(repo):
                found.append((path.relative_to(repo), target))
    listed = git(repo, "status", "--ignored", "--porcelain=v1", "--", prefix or ".") or ""
    for line in listed.splitlines():
        if not line.startswith("!! ") or not line.endswith("/"):
            continue
        directory = repo / line[3:].rstrip("/")
        if directory.is_symlink() or not directory.is_dir():
            continue
        files = (entry for _, entry in zip(range(2000), directory.rglob("*")))
        if any(entry.is_file() and DATA_FILE.search(entry.name) for entry in files):
            found.append((directory.relative_to(repo), directory))
    return found


def data_dirs_listing(repo, prefix):
    """Every file in the project's data directories, addressed through the
    repository, with its size; directories of earlier results are marked."""
    lines = []
    for relative, target in data_links(repo, prefix):
        earlier = re.search(r"cache|plots?|results?|outputs?|fits", str(relative), re.IGNORECASE)
        note = "  [earlier results: never an input]" if earlier else ""
        lines.append("%s/%s" % (relative, note))
        for entry in sorted(target.rglob("*"))[:80]:
            if entry.is_file():
                lines.append("  %s  (%.1f MB)" % (entry.relative_to(target.parent), entry.stat().st_size / 1e6))
    return "\n".join(lines) or "-"


def data_listing(repo, unit_dir, paths):
    """Repository-relative name and size of on-disk files under the
    directories the unit reads. A relative input path is tried against the
    unit's directory, its parent and the repository root, since macros run
    from different places. Paths are never resolved: data directories are
    symlinks (into /labdata) that items must address through the repository.
    """
    lines = []
    seen = set()
    bases = [unit_dir, os.path.dirname(unit_dir), ""]
    for raw in paths:
        stem = raw.split("*")[0].split("$")[0]
        if not stem or os.path.isabs(stem):
            continue
        for base in bases:
            candidate = pathlib.Path(os.path.normpath(os.path.join(repo, base, stem)))
            directory = candidate if candidate.is_dir() else candidate.parent
            if directory.is_dir() and directory != repo:
                break
        else:
            continue
        if directory in seen:
            continue
        seen.add(directory)
        for entry in sorted(directory.iterdir())[:60]:
            if entry.is_file():
                lines.append("%s  (%.1f MB)" % (os.path.relpath(entry, repo), entry.stat().st_size / 1e6))
    return "\n".join(lines) or "-"


UTILS = pathlib.Path("~/Analysis-Utilities").expanduser()
# File stems, matched against #include names, not class names: since
# Analysis-Utilities 26.10.24 WaveformProcessingUtils holds the waveform
# processor variants (PulseIntegralProcessor, PulseHeightProcessor).
UTILS_SOURCES = {"FittingUtils", "WaveformProcessingUtils"}
INCLUDE = re.compile(r'#include\s*"([^"]+)"')


def utils_dependencies(repo, commit, code):
    """Analysis-Utilities headers the code includes, at the revision the
    repository's flake.lock pins; sources too for the fitting code and the
    waveform processors, where analysis conventions live. [(label, text)]."""
    try:
        lock = json.loads(git(repo, "show", "%s:flake.lock" % commit) or "{}")
    except ValueError:
        return []
    rev = next(
        (
            node["locked"]["rev"]
            for node in (lock.get("nodes") or {}).values()
            if (node.get("locked") or {}).get("repo") == "Analysis-Utilities"
        ),
        None,
    )
    if not rev:
        return []
    found = []
    for name in dict.fromkeys(INCLUDE.findall(code)):
        stem = os.path.splitext(os.path.basename(name))[0]
        paths = ["include/%s" % os.path.basename(name)]
        if stem in UTILS_SOURCES:
            paths.append("src/%s.cpp" % stem)
        for path in paths:
            text = git(UTILS, "show", "%s:%s" % (rev, path))
            if text:
                found.append(("Analysis-Utilities@%s:%s" % (rev[:8], path), text[:60000]))
    return found


def um_ansg_jobs(private):
    path = private / "inventory" / "um_ansg.jsonl"
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    files = {row["path"]: row for row in rows if not row["error"]}
    jobs = []
    for row in files.values():
        unit = row["unit"]
        if unit["kind"] != "analysis_stage" or unit.get("wip") or not unit.get("quantities"):
            continue
        repo, commit = pathlib.Path(row["repo"]), row["commit"]
        unit_dir = os.path.dirname(row["path"])
        inputs = [entry.get("path", "") for entry in unit.get("inputs") or []]
        code = git(repo, "show", "%s:%s" % (commit, row["path"])) or ""
        # Short, essential sections first: CONTEXT_CHARS cuts from the end.
        sections = ["unit: %s" % row["path"], "inventory entry:", json.dumps(unit, indent=1)]
        sections += ['the item\'s [env] nix_develop is "UM-ANSG#%s"' % UM_ANSG_PROJECTS[row["project"]]]
        sections += ["", "data files on disk (the unit's input paths):", data_listing(repo, unit_dir, inputs)]
        sections += ["", "all files in the project's data directories:", data_dirs_listing(repo, UM_ANSG_PROJECTS[row["project"]])]
        sections += ["", "flake.nix (excerpt):", (git(repo, "show", "%s:flake.nix" % commit) or "-")[:3000]]
        sections += ["", "reference code (%s):" % row["path"], code]
        for dependency in unit.get("depends_on") or []:
            for name in files:
                if name.endswith(os.path.basename(dependency)) and name != row["path"]:
                    sections += ["", "dependency (%s):" % name, git(repo, "show", "%s:%s" % (commit, name)) or ""]
        for label, text in utils_dependencies(repo, commit, code):
            sections += ["", "dependency (%s):" % label, text]
        jobs.append(
            {
                "set": "um_ansg",
                "project": row["project"],
                "repo": str(repo),
                "commit": commit,
                "unit": row["path"],
                "outputs": [entry.get("path", "") for entry in unit.get("outputs") or []],
                "method": "g1_verified_analysis",
                "context": "\n".join(sections)[:CONTEXT_CHARS],
            }
        )
    return jobs


def g0_jobs(private):
    with open(private / "rank" / "all.jsonl") as handle:
        clusters = [json.loads(line) for line in handle]
    inventory = {}
    path = private / "inventory" / "g0_repos.jsonl"
    if path.exists():
        with open(path) as handle:
            for row in map(json.loads, handle):
                if not row["error"]:
                    inventory[(row["repo"], row["path"])] = row["unit"]
    jobs = []
    for cluster in clusters:
        lead = cluster["lead"]
        if len(jobs) >= G0_TOP:
            break
        if not lead["repo"] or lead["queue"] not in ("g1", "g2") or lead["project"] not in ("music", "infra"):
            continue
        repo = pathlib.Path(lead["repo"])
        commits = [c for c in lead["commits"] if c["on_head"] and c["touches_code"]]
        commit = commits[0]["sha"] if commits else git(repo, "rev-parse", "HEAD")
        sections = ["task observed in the user's sessions: %s" % lead["statement"]]
        sections += ["domain: %s, type: %s, difficulty: %s" % (lead["domain"], lead["type"], lead["difficulty"])]
        files, dependencies, outputs = [], {}, []
        for name in lead["files"][:6]:
            relative = os.path.relpath(name, repo) if os.path.isabs(name) else name
            content = git(repo, "show", "%s:%s" % (commit, relative))
            if not content:
                continue
            files += ["", "file %s:" % relative, content[:50000]]
            dependencies.update(utils_dependencies(repo, commit, content))
            unit = inventory.get((str(repo), relative))
            if unit:
                inputs = [entry.get("path", "") for entry in unit.get("inputs") or []]
                outputs += [entry.get("path", "") for entry in unit.get("outputs") or []]
                sections += ["", "inventory entry for %s:" % relative, json.dumps(unit, indent=1)]
                sections += ["data files on disk for %s:" % relative, data_listing(repo, os.path.dirname(relative), inputs)]
        # Short, essential sections first: CONTEXT_CHARS cuts from the end.
        sections += ["", "all files in the repository's data directories:", data_dirs_listing(repo, "")[:12000]]
        sections += ["", "flake.nix (excerpt):", (git(repo, "show", "%s:flake.nix" % commit) or "-")[:3000]]
        sections += ["", "files at this commit:", (git(repo, "ls-tree", "-r", "--name-only", commit) or "")[:8000]]
        if commits:
            sections += ["", "the commit that delivered it:", (git(repo, "show", "--stat", "-p", commit) or "")[:60000]]
        sections += files
        for label, text in dependencies.items():
            sections += ["", "dependency (%s):" % label, text]
        jobs.append(
            {
                "set": "g0",
                "project": lead["project"],
                "repo": str(repo),
                "commit": commit,
                "unit": cluster["cluster"],
                "outputs": outputs,
                "method": "g2_code_history" if commits else "g1_verified_analysis",
                "context": "\n".join(sections)[:G0_CONTEXT_CHARS],
            }
        )
    return jobs


def strip_fences(text):
    text = text.strip()
    fenced = re.match(r"^```(?:toml)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return fenced.group(1) if fenced else text


def count(table, key, amount=1):
    table[key] = table.get(key, 0) + amount


def run_stage(private, set_name, model_name=MODEL, limit=0, concurrency=4, retry_failed=False, effort="high"):
    model = llm.Model(model_name, timeout=TIMEOUT_S, retries=1)
    first = dict(GENERATION, reasoning_effort=effort)
    jobs = um_ansg_jobs(private) if set_name == "um_ansg" else g0_jobs(private)
    if limit:
        jobs = jobs[:limit]
    out_dir = private / "items" / "draft" / set_name
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    index = out_dir / "index.jsonl"
    rows = []
    if retry_failed and index.exists():
        with open(index) as handle:
            rows = [row for row in map(json.loads, handle) if row["state"] != "failed"]
        finished = {row["unit"] for row in rows}
        jobs = [job for job in jobs if job["unit"] not in finished]
    else:
        for stale in list(out_dir.glob("*.toml")) + [index]:
            stale.unlink(missing_ok=True)
    lock = threading.Lock()

    def work(job):
        project = job["project"]
        sensitivity = "restricted" if project == "music" else "internal"
        repo_name = "UM-ANSG" if job["set"] == "um_ansg" else "<repository>"
        slug = {"yap_psd": "ansg-yap", "78mbr": "ansg-78mbr", "music": "music", "infra": "infra"}[project]
        prompt = PROMPT % (
            sensitivity,
            slug,
            sensitivity,
            repo_name,
            job["commit"],
            job["method"],
            model_name,
            job["context"],
        )
        row = {key: job[key] for key in ("set", "project", "repo", "commit", "unit", "method")}
        row.update(prompt_tokens=0, completion_tokens=0)
        text, error = "", "truncated"
        for generation in (first, FALLBACK):
            try:
                reply = model.ask(prompt, generation)
            except RuntimeError:
                traceback.print_exc()
                error = "http"
                continue
            row["prompt_tokens"] += reply["prompt_tokens"] or 0
            row["completion_tokens"] += reply["completion_tokens"] or 0
            text = strip_fences(reply["text"])
            if reply["finish_reason"] != "length" and text:
                break
            error = "truncated"
        else:
            row.update(state="failed", error=error)
            return row
        # The pinned commit is ours, not the model's to copy.
        text = re.sub(r'(?m)^commit = "[^"]*"', 'commit = "%s"' % job["commit"], text, count=1)
        try:
            item = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            traceback.print_exc()
            row.update(state="failed", error="bad_toml")
            return row
        item_id = re.sub(r"[^a-z0-9-]", "-", str(item.get("id", "item")).lower())[:80]
        with lock:
            while (out_dir / ("%s.toml" % item_id)).exists():
                item_id += "-b"
            (out_dir / ("%s.toml" % item_id)).write_text(text + "\n")
        problems = validate.check_item(item, pathlib.Path(job["repo"]), job["commit"])
        produced = {os.path.basename(path) for path in job["outputs"] if path}
        for entry in item.get("inputs") or []:
            if os.path.basename(entry.get("from", "")) in produced:
                problems["inputs"].append("answer_bearing_input")
        failed = [gate for gate, codes in problems.items() if codes]
        if item.get("skip"):
            state = "skipped"
        elif {"schema", "no_typed_gold", "hidden_reference"} & set(failed):
            state = "rejected"
        elif failed or (item.get("generation") or {}).get("needs_human"):
            state = "needs_human"
        else:
            state = "draft"
        row.update(
            id=item_id,
            state=state,
            error=None,
            problems=problems,
            domain=item.get("domain"),
            type=item.get("type"),
            gold_kinds=sorted({(c.get("gold_ref") or {}).get("kind") for c in item.get("checks") or []} - {None}),
            checks=len(item.get("checks") or []),
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row in pool.map(work, jobs):
            rows.append(row)
    with open(out_dir / "index.jsonl", "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    summary = {
        "set": set_name,
        "model": model_name,
        "jobs": len(rows),
        "state": {},
        "gate_failed": {},
        "domain": {},
        "type": {},
        "gold_kind": {},
        "project": {},
        "checks": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for row in rows:
        count(summary["state"], row["state"])
        summary["prompt_tokens"] += row.get("prompt_tokens", 0)
        summary["completion_tokens"] += row.get("completion_tokens", 0)
        if row["state"] == "failed":
            continue
        count(summary["project"], row["project"])
        summary["checks"] += row["checks"]
        for gate, codes in row["problems"].items():
            if codes:
                count(summary["gate_failed"], gate)
        if row["domain"] in validate.DOMAINS:
            count(summary["domain"], row["domain"])
        if row["type"] in validate.TYPES:
            count(summary["type"], row["type"])
        for kind in row["gold_kinds"]:
            if kind in validate.GOLD_KINDS:
                count(summary["gold_kind"], kind)
    return summary
