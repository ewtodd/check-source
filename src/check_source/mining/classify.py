"""G0 stage 2 (knoll-bench-plan.md section 6.0): a local model reads each
in-scope digest record and lists the tasks the session worked on.

Sessions are selected by rule first, so out-of-scope ones never reach a model:
73mGe and archival work, fleet configuration (/etc/nixos) and son-of-anton
development, and sessions without a user request. Results go to
<private>/classify/<source>.jsonl, one row per session, appended so an
interrupted run resumes where it stopped.
"""

import collections
import concurrent.futures
import json
import threading
import traceback

from . import llm
from .digest import SOURCES

MODEL = "radiance/qwen3.8-27b"
GENERATION = {"max_tokens": 12000, "temperature": 0.2, "reasoning_effort": "low"}
CHUNK_CHARS = 120000
MAX_TASKS = 8
PROJECTS = ["yap_psd", "78mbr", "music", "infra", "73mge", "archival", "other"]
DOMAINS = ["CAL", "SPEC", "PSD", "NUC", "SIM", "STAT", "SW", "DAQ"]
TYPES = ["compute", "interpret", "implement", "script", "debug", "pipeline"]
OUTCOMES = ["done", "partial", "abandoned", "unknown"]
DIFFICULTIES = ["1", "2", "3", "4", "5"]
DROP_SCOPES = {"73mge": "drop_73mge", "yap_qda": "drop_archival", "ansg_archival": "drop_archival"}
DROPS = ["drop_73mge", "drop_archival", "drop_nixos", "drop_soa", "drop_no_requests"]
FAILURES = ["bad_json", "bad_project", "truncated", "http", "crash"]
VOCAB = frozenset(
    list(SOURCES)
    + sorted(llm.LOCAL_MODELS)
    + PROJECTS
    + DOMAINS
    + TYPES
    + OUTCOMES
    + DIFFICULTIES
    + DROPS
    + FAILURES
    + [
        "source",
        "model",
        "considered",
        "selected",
        "dropped",
        "already_done",
        "classified",
        "failed",
        "in_scope",
        "in_scope_project",
        "project",
        "tasks",
        "tasks_dropped",
        "tasks_benchmarkable",
        "task_domain",
        "task_type",
        "task_outcome",
        "task_difficulty",
        "chunked_sessions",
        "prompt_tokens",
        "completion_tokens",
    ]
)

PROMPT = """You are tagging one coding-agent session from an experimental nuclear physicist's work. The goal is a private benchmark (the Knoll bench) of the physics data-analysis work this person actually does. Read the session digest below and answer with one JSON object and nothing else.

Projects:
- yap_psd: pulse-shape discrimination with YAP:Ce scintillators, the production analysis (UM-ANSG YAP-PSD)
- 78mbr: the 78mBr isomer half-life and Monte Carlo efficiency analysis
- music: MUSIC detector work: data processing, pole-zero and trapezoid filtering, simulation (PoleZeroMUSIC, Remix-MUSIC-Sim, musicsim), nuclear-data inputs
- infra: shared analysis software (Analysis-Utilities, analysis-caen, nds-mcp, ROOT and RooFit tooling)
- 73mge: the 73mGe CZT analysis (excluded from the bench)
- archival: ANSG-Archival data, YAP-QDA, or archival YAP data (excluded)
- other: anything else

in_scope is true only when the session does physics analysis, detector or DAQ processing, simulation, nuclear data, statistics and fitting, or analysis-software work for yap_psd, 78mbr, music or infra. System administration, NixOS, LLM and agent tooling, websites, writing unrelated to an analysis, and general programming are out of scope.

Domains: CAL calibration (energy, gain, pixel); SPEC spectroscopy and peak analysis; PSD pulse shapes and waveforms; NUC nuclear data (half-lives, branching, ENSDF); SIM simulation and transport; STAT statistics and fitting; SW analysis software and data plumbing; DAQ acquisition and signal processing.
Task types: compute (a numeric answer from data); interpret (choose or diagnose from given results); implement (write a function or macro to a spec); script (an analysis program from inputs to results); debug (find and fix a fault); pipeline (a multi-stage analysis with intermediate files).

List the substantive tasks the user asked for, at most 8, skipping confirmations such as "y" or "continue". For each task give:
- statement: one or two sentences saying what was asked, in neutral terms
- domain and type: from the lists above
- outcome: done, partial, abandoned or unknown, judged from later requests, commits and the final message
- difficulty: 1 (routine) to 5 (hard, multi-day), judged from the work it took
- benchmarkable: true if a self-contained benchmark task with an objectively checkable answer (a number from data, code that passes tests, a fixed choice) could be built from it; false for writing and emails, prose review, git or release chores, and open-ended discussion
- requests: the [n] numbers of the requests that define the task
- files: up to 5 key file paths involved
- commits: commit shas from the digest that delivered it

Answer format:
{"in_scope": true, "project": "music", "tasks": [{"statement": "...", "domain": "SIM", "type": "implement", "outcome": "done", "difficulty": 3, "benchmarkable": true, "requests": [0, 2], "files": ["..."], "commits": ["abc1234"]}], "notes": "anything a reviewer should know, or empty"}

Session digest%s:
%s
"""


class BadAnswer(Exception):
    """`str(error)` is one of FAILURES."""


class ModelUnavailable(Exception):
    """The preflight request failed, so no session was sent."""


def drop_reason(record):
    if record["scope"] in DROP_SCOPES:
        return DROP_SCOPES[record["scope"]]
    cwd = record["cwd"] or ""
    if record["scope"] == "other" and (cwd.startswith("/etc/nixos") or "/NixOS" in cwd):
        return "drop_nixos"
    if record["scope"] == "other" and "son-of-anton" in cwd:
        return "drop_soa"
    if not record["requests"]:
        return "drop_no_requests"
    return None


def render(record, requests):
    active = (record["active_s"] or 0) / 60
    wall = (record["duration_s"] or 0) / 60
    lines = [
        "title: %s" % (record["title"] or "-"),
        "working directory: %s" % (record["cwd"] or "-"),
        "active minutes: %d, wall-clock minutes: %d" % (active, wall),
        "tool calls: %s; tool errors: %d; subagents: %d"
        % (json.dumps(record["tools"]), record["tool_errors"], record["subagents"]),
        "",
        "requests:",
    ]
    lines += ["[%d] %s" % (index, request["text"]) for index, request in requests]
    lines += ["", "files written:"] + record["files_written"][:60]
    lines += ["", "files read:"] + record["files_read"][:60]
    lines += ["", "shell commands (first 80):"] + [command[:200] for command in record["commands"][:80]]
    lines += ["", "commits:"] + ["%s %s" % (commit["sha"], commit["subject"]) for commit in record["commits"]]
    lines += ["", "final assistant message:", record["final_text"] or "-"]
    return "\n".join(lines)


def chunks(record):
    """Split the numbered requests so each rendered digest fits CHUNK_CHARS."""
    budget = max(CHUNK_CHARS - len(render(record, [])), 20000)
    groups, current, size = [], [], 0
    for index, request in enumerate(record["requests"]):
        length = len(request["text"]) + 8
        if current and size + length > budget:
            groups.append(current)
            current, size = [], 0
        current.append((index, request))
        size += length
    groups.append(current)
    return groups


def parse(text):
    """The answer's JSON object with invalid tasks removed; raises BadAnswer."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise BadAnswer("bad_json")
    try:
        answer = json.loads(text[start : end + 1])
    except ValueError:
        raise BadAnswer("bad_json") from None
    if not isinstance(answer, dict) or answer.get("project") not in PROJECTS:
        raise BadAnswer("bad_project")
    tasks, dropped = [], 0
    for task in answer.get("tasks") or []:
        valid = (
            isinstance(task, dict)
            and isinstance(task.get("statement"), str)
            and task.get("domain") in DOMAINS
            and task.get("type") in TYPES
            and task.get("outcome") in OUTCOMES
            and str(task.get("difficulty")) in DIFFICULTIES
        )
        if not valid:
            dropped += 1
            continue
        tasks.append(
            {
                "statement": task["statement"],
                "domain": task["domain"],
                "type": task["type"],
                "outcome": task["outcome"],
                "difficulty": int(task["difficulty"]),
                "benchmarkable": task.get("benchmarkable") is True,
                "requests": [n for n in task.get("requests") or [] if isinstance(n, int)],
                "files": [f for f in task.get("files") or [] if isinstance(f, str)][:5],
                "commits": [c for c in task.get("commits") or [] if isinstance(c, str)],
            }
        )
    return {
        "in_scope": answer.get("in_scope") is True,
        "project": answer["project"],
        "tasks": tasks[:MAX_TASKS],
        "tasks_dropped": dropped + max(0, len(tasks) - MAX_TASKS),
        "notes": answer.get("notes") if isinstance(answer.get("notes"), str) else "",
    }


def ask(model, prompt):
    """One answer, retried once when it is unparseable or cut off."""
    tokens = [0, 0]
    for attempt in range(2):
        reply = model.ask(prompt if attempt == 0 else prompt + "\nAnswer with the JSON object only.", GENERATION)
        tokens[0] += reply["prompt_tokens"] or 0
        tokens[1] += reply["completion_tokens"] or 0
        try:
            if reply["finish_reason"] == "length":
                raise BadAnswer("truncated")
            return parse(reply["text"]), tokens
        except BadAnswer:
            if attempt:
                raise
    raise AssertionError("unreachable")


def classify(model, record):
    groups = chunks(record)
    answers, tokens = [], [0, 0]
    for number, group in enumerate(groups, 1):
        part = " (part %d of %d, by request)" % (number, len(groups)) if len(groups) > 1 else ""
        answer, used = ask(model, PROMPT % (part, render(record, group)))
        answers.append(answer)
        tokens = [tokens[0] + used[0], tokens[1] + used[1]]
    in_scope = [answer for answer in answers if answer["in_scope"]]
    projects = collections.Counter(answer["project"] for answer in (in_scope or answers))
    return {
        "in_scope": bool(in_scope),
        "project": projects.most_common(1)[0][0],
        "tasks": [task for answer in answers for task in answer["tasks"]],
        "tasks_dropped": sum(answer["tasks_dropped"] for answer in answers),
        "notes": " | ".join(answer["notes"] for answer in answers if answer["notes"]),
        "chunks": len(groups),
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
    }


def count(table, key, amount=1):
    table[key] = table.get(key, 0) + amount


def run_stage(private, source, model_name=MODEL, limit=0, fresh=False, concurrency=3):
    model = llm.Model(model_name)
    out_dir = private / "classify"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    out_path = out_dir / ("%s.jsonl" % source)
    done = set()
    if out_path.exists() and not fresh:
        with open(out_path) as handle:
            done = {row["session"] for row in map(json.loads, handle) if not row.get("error")}
    summary = {"source": source, "model": model_name, "considered": 0, "selected": 0, "dropped": {}, "already_done": 0}
    jobs = []
    with open(private / "digest" / ("%s.jsonl" % source)) as handle:
        for record in map(json.loads, handle):
            summary["considered"] += 1
            reason = drop_reason(record)
            if reason:
                count(summary["dropped"], reason)
            elif record["session"] in done:
                summary["already_done"] += 1
            else:
                jobs.append(record)
    if limit:
        jobs = jobs[:limit]
    summary["selected"] = len(jobs)

    lock = threading.Lock()

    def work(record):
        row = {"session": record["session"], "source": source, "native_id": record["native_id"], "model": model_name}
        try:
            row.update(classify(model, record))
            row["error"] = None
        except BadAnswer as error:
            row["error"] = str(error)
        except RuntimeError:
            traceback.print_exc()
            row["error"] = "http"
        except Exception:  # noqa: BLE001 - one bad session must not stop the run; the log has the traceback
            traceback.print_exc()
            row["error"] = "crash"
        with lock:
            with open(out_path, "a") as out:
                out.write(json.dumps(row) + "\n")
        return row

    try:
        model.ask("Reply with OK.", {"max_tokens": 16, "reasoning_effort": "none"})
    except RuntimeError:
        raise ModelUnavailable() from None
    if fresh and out_path.exists():
        out_path.unlink()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(work, jobs))

    tally = {
        "classified": 0,
        "failed": {},
        "in_scope": 0,
        "in_scope_project": {},
        "project": {},
        "tasks": 0,
        "tasks_dropped": 0,
        "tasks_benchmarkable": 0,
        "task_domain": {},
        "task_type": {},
        "task_outcome": {},
        "task_difficulty": {},
        "chunked_sessions": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for row in rows:
        if row["error"]:
            count(tally["failed"], row["error"])
            continue
        tally["classified"] += 1
        tally["prompt_tokens"] += row["prompt_tokens"]
        tally["completion_tokens"] += row["completion_tokens"]
        tally["chunked_sessions"] += row["chunks"] > 1
        tally["tasks_dropped"] += row["tasks_dropped"]
        count(tally["project"], row["project"])
        if not row["in_scope"]:
            continue
        tally["in_scope"] += 1
        count(tally["in_scope_project"], row["project"])
        for task in row["tasks"]:
            tally["tasks"] += 1
            tally["tasks_benchmarkable"] += task["benchmarkable"]
            count(tally["task_domain"], task["domain"])
            count(tally["task_type"], task["type"])
            count(tally["task_outcome"], task["outcome"])
            count(tally["task_difficulty"], str(task["difficulty"]))
    summary.update(tally)
    return summary
