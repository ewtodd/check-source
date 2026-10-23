"""G0 stage 1 (knoll-bench-plan.md section 6.0): one digest record per agent
session, built by script with no model involved.

Records go to <private>/digest/<source>.jsonl. The summary holds only counts
over `VOCAB`, so it can be shown to the pipeline's operator even when the
sessions themselves are blind.
"""

import json

from . import claude_code, opencode
from .record import SCOPES, TOOLS

SOURCES = {claude_code.SOURCE: claude_code, opencode.SOURCE: opencode}
DURATION_BUCKETS = [(5 * 60, "lt5m"), (30 * 60, "5to30m"), (120 * 60, "30to120m"), (None, "gt120m")]
REQUEST_BUCKETS = [(0, "0"), (1, "1"), (5, "2to5"), (20, "6to20"), (None, "gt20")]
SOURCE_STATS = ["excluded", "databases", "databases_failed", "v2_rows", "legacy_json_sessions"]
VOCAB = frozenset(
    list(SOURCES)
    + SCOPES
    + TOOLS
    + SOURCE_STATS
    + [label for _, label in DURATION_BUCKETS + REQUEST_BUCKETS]
    + [
        "source",
        "sessions",
        "empty",
        "subagents",
        "requests",
        "tool_calls",
        "tool_errors",
        "commits",
        "sessions_with_commits",
        "scope",
        "tools",
        "duration",
        "requests_per_session",
        "unknown",
    ]
)


def bucket(value, buckets):
    if value is None:
        return "unknown"
    for limit, label in buckets:
        if limit is None or value <= limit:
            return label
    return "unknown"


def count(table, key, amount=1):
    table[key] = table.get(key, 0) + amount


def run_stage(private, source, exclude=()):
    out_dir = private / "digest"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    stats = {}
    summary = {
        "source": source,
        "sessions": 0,
        "empty": 0,
        "subagents": 0,
        "requests": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "commits": 0,
        "sessions_with_commits": 0,
        "scope": {},
        "tools": {},
        "duration": {},
        "requests_per_session": {},
    }
    with open(out_dir / ("%s.jsonl" % source), "w") as handle:
        for session in SOURCES[source].sessions(set(exclude), stats):
            record = session.record()
            if record["started"] is None:
                summary["empty"] += 1
                continue
            handle.write(json.dumps(record) + "\n")
            summary["sessions"] += 1
            summary["subagents"] += record["subagents"]
            summary["requests"] += len(record["requests"])
            summary["tool_calls"] += sum(record["tools"].values())
            summary["tool_errors"] += record["tool_errors"]
            summary["commits"] += len(record["commits"])
            summary["sessions_with_commits"] += bool(record["commits"])
            count(summary["scope"], record["scope"])
            for tool, calls in record["tools"].items():
                count(summary["tools"], tool, calls)
            count(summary["duration"], bucket(record["active_s"], DURATION_BUCKETS))
            count(summary["requests_per_session"], bucket(len(record["requests"]), REQUEST_BUCKETS))
    summary.update({key: stats[key] for key in SOURCE_STATS if key in stats})
    return summary
