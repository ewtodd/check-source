"""opencode sessions, from its SQLite stores (v1.19 schema: `session`, `message`
and `part`, each row's payload a JSON `data` column) and from the JSON files
of the storage format that preceded them (storage/session/<project>/<id>.json,
storage/message/<session>/<id>.json, storage/part/<message>/<id>.json; same
records). Child sessions, which the task tool creates for subagents, are
folded into their top-level session. Databases are opened read-only."""

import json
import os
import pathlib
import sqlite3
import traceback

from .record import Session, from_ms

DATA = pathlib.Path("~/.local/share/opencode").expanduser()
DATABASES = ["opencode.db", "opencode-stable.db"]
SOURCE = "opencode"
TOOL_KINDS = {
    "bash": "bash",
    "read": "read",
    "edit": "edit",
    "multiedit": "edit",
    "write": "write",
    "grep": "grep",
    "glob": "glob",
    "list": "glob",
    "task": "task",
    "apply_patch": "patch",
    "patch": "patch",
    "webfetch": "web",
    "websearch": "web",
}


def descendants(children, root):
    found = []
    stack = list(children.get(root, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def absolute(path, cwd):
    if path and cwd and not os.path.isabs(path):
        return os.path.join(cwd, path)
    return path


def sqlite_messages(con):
    """messages(session_id) -> [(message_id, created_ms, info, parts)] in order."""

    def messages(session_id):
        parts = {}
        for message_id, data in con.execute(
            "select message_id, data from part where session_id = ? order by message_id, id", (session_id,)
        ):
            parts.setdefault(message_id, []).append(json.loads(data))
        return [
            (message_id, created, json.loads(data), parts.get(message_id, []))
            for message_id, created, data in con.execute(
                "select id, time_created, data from message where session_id = ? order by time_created, id",
                (session_id,),
            )
        ]

    return messages


def legacy_messages(storage):
    """The same contract as `sqlite_messages`, over the JSON file store."""

    def load(path):
        with open(path) as handle:
            return json.load(handle)

    def messages(session_id):
        found = []
        for path in (storage / "message" / session_id).glob("*.json"):
            info = load(path)
            parts = [load(part) for part in sorted((storage / "part" / info["id"]).glob("*.json"))]
            found.append((info["id"], (info.get("time") or {}).get("created"), info, parts))
        return sorted(found, key=lambda message: (message[1] or 0, message[0]))

    return messages


def read_session(messages, session_id, directory, title, subagents):
    session = Session(SOURCE, session_id)
    session.cwd = directory
    session.title = title
    session.subagents = len(subagents)
    for current in [session_id] + subagents:
        main = current == session_id
        for message_id, created, info, parts in messages(current):
            when = from_ms((info.get("time") or {}).get("created") or created)
            if info.get("role") == "user":
                text = "\n".join(
                    part.get("text", "")
                    for part in parts
                    if part.get("type") == "text" and not part.get("synthetic") and not part.get("ignored")
                ).strip()
                if main and text:
                    session.request(when, text)
                else:
                    session.at(when)
                continue
            cwd = (info.get("path") or {}).get("cwd") or session.cwd
            texts = []
            for part in parts:
                kind = part.get("type")
                if kind == "text":
                    texts.append(part.get("text", ""))
                elif kind == "patch":
                    for path in part.get("files") or []:
                        session.wrote(absolute(path, cwd))
                elif kind == "tool":
                    state = part.get("state") or {}
                    args = state.get("input") or {}
                    tool = TOOL_KINDS.get(part.get("tool"), "other")
                    session.tool(
                        tool,
                        path=absolute(args.get("filePath"), cwd),
                        written=tool in ("edit", "write"),
                        command=args.get("command") if tool == "bash" else None,
                    )
                    session.result(state.get("output"), state.get("status") == "error")
            model = "/".join(filter(None, [info.get("providerID"), info.get("modelID")])) or None
            session.assistant(when, message_id, model, "\n".join(texts) if main else None)
    return session


def walk(rows, messages, seen, exclude, stats):
    """Yield a Session per top-level row of (id, parent_id, directory, title)."""
    children = {}
    for session_id, parent_id, _, _ in rows:
        if parent_id:
            children.setdefault(parent_id, []).append(session_id)
    for session_id, parent_id, directory, title in rows:
        if parent_id or session_id in seen:
            continue
        seen.add(session_id)
        if session_id in exclude:
            stats["excluded"] += 1
            continue
        yield read_session(messages, session_id, directory, title, descendants(children, session_id))


def sessions(exclude, stats):
    stats.update(databases=0, databases_failed=0, excluded=0, v2_rows=0, legacy_json_sessions=0)
    seen = set()
    for name in DATABASES:
        path = DATA / name
        if not path.exists():
            continue
        try:
            con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
            rows = con.execute("select id, parent_id, directory, title from session").fetchall()
        except sqlite3.Error:
            traceback.print_exc()
            stats["databases_failed"] += 1
            continue
        stats["databases"] += 1
        try:
            stats["v2_rows"] += con.execute("select count(*) from session_message").fetchone()[0]
        except sqlite3.Error:
            pass
        yield from walk(rows, sqlite_messages(con), seen, exclude, stats)
        con.close()
    storage = DATA / "storage"
    rows = []
    for path in (storage / "session").glob("*/*.json"):
        with open(path) as handle:
            info = json.load(handle)
        if info.get("id") and info["id"] not in seen:
            rows.append((info["id"], info.get("parentID"), info.get("directory"), info.get("title")))
    stats["legacy_json_sessions"] = len(rows)
    yield from walk(rows, legacy_messages(storage), seen, exclude, stats)
