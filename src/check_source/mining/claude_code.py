"""Claude Code sessions: ~/.claude/projects/<project>/<session>.jsonl, with
subagent transcripts under <project>/<session>/subagents/, folded into their
session."""

import json
import pathlib
import re

from .record import Session, from_iso

ROOT = pathlib.Path("~/.claude/projects").expanduser()
SOURCE = "claude-code"
TOOL_KINDS = {
    "Bash": "bash",
    "Read": "read",
    "Edit": "edit",
    "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Write": "write",
    "Grep": "grep",
    "Glob": "glob",
    "Agent": "task",
    "Task": "task",
    "WebFetch": "web",
    "WebSearch": "web",
}
# User-role text the harness injects rather than the user typing it.
INJECTED = (
    "<task-notification>",
    "<local-command-",
    "<command-name>",
    "<command-message>",
    "This session is being continued",
    "[Request interrupted",
    "Another Claude session sent a message",
    "Caveat:",
    "Base directory for this skill",
)
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def entries(path):
    with open(path) as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
    return ""


def read_file(session, path, main):
    seen = set()
    for entry in entries(path):
        uuid = entry.get("uuid")
        if uuid:
            if uuid in seen:
                continue
            seen.add(uuid)
        kind = entry.get("type")
        if kind == "ai-title" and main:
            session.title = entry.get("aiTitle")
        if kind not in ("user", "assistant"):
            continue
        when = from_iso(entry.get("timestamp"))
        if main and session.cwd is None:
            session.cwd = entry.get("cwd")
        message = entry.get("message") or {}
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        if kind == "user":
            results = [block for block in blocks if block.get("type") == "tool_result"]
            for block in results:
                session.result(text_of(block.get("content")), bool(block.get("is_error")))
            text = REMINDER.sub("", text_of(content)).strip()
            if main and not results and not entry.get("isMeta") and text and not text.startswith(INJECTED):
                session.request(when, text)
            else:
                session.at(when)
            continue
        texts = []
        for block in blocks:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                args = block.get("input") or {}
                tool = TOOL_KINDS.get(block.get("name"), "other")
                session.tool(
                    tool,
                    path=args.get("file_path") or args.get("notebook_path"),
                    written=tool in ("edit", "write"),
                    command=args.get("command") if tool == "bash" else None,
                )
        model = message.get("model")
        if model and model.startswith("<"):
            model = None
        session.assistant(when, message.get("id"), model, "\n".join(texts) if main else None)


def sessions(exclude, stats):
    stats["excluded"] = 0
    for path in sorted(ROOT.glob("*/*.jsonl")):
        if path.stem in exclude:
            stats["excluded"] += 1
            continue
        session = Session(SOURCE, path.stem)
        read_file(session, path, main=True)
        for subagent in sorted((path.parent / path.stem / "subagents").glob("*.jsonl")):
            session.subagents += 1
            read_file(session, subagent, main=False)
        yield session
