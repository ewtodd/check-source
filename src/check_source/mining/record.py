"""One session's digest record, shared by every session source."""

import datetime
import hashlib
import re

MAX_REQUEST = 4000
MAX_COMMAND = 500
MAX_COMMANDS = 300
MAX_FINAL = 2000
IDLE_GAP_S = 30 * 60

# First match wins, so the specific UM-ANSG projects precede the general rules.
# 73mGe, ANSG-Archival and YAP-QDA are out of v1 (plan section 3.2) but still
# get their own scope, so they can be counted and dropped rather than mixed in.
# These only bucket paths and commands; the classify stage decides scope from
# content (production vs archival YAP, for one).
SCOPE_RULES = [
    ("73mge", re.compile(r"73m?ge", re.IGNORECASE)),
    ("yap_qda", re.compile(r"/YAP-QDA(/|$)")),
    ("ansg_archival", re.compile(r"/ANSG-Archival(/|$)")),
    ("yap_psd", re.compile(r"yap", re.IGNORECASE)),
    ("78mbr", re.compile(r"78m?br|bromine", re.IGNORECASE)),
    ("yag", re.compile(r"/YAG-PreProcessing(/|$)")),
    ("music", re.compile(r"music", re.IGNORECASE)),
    ("infra", re.compile(r"/(Analysis-Utilities|analysis-caen|nds-mcp)(/|$)")),
    ("um_ansg", re.compile(r"/UM-ANSG(/|$)")),
]
SCOPES = [name for name, _ in SCOPE_RULES] + ["other"]
# A working directory here says where the session ran, not what it worked on.
GENERIC_SCOPES = {"um_ansg", "other"}
# Without a telling working directory, the leading scope must cover this share
# of the session's scope tally to be its scope; a stray mention is not enough.
MIN_SCOPE_SHARE = 0.25
TOOLS = ["bash", "read", "edit", "write", "grep", "glob", "task", "patch", "web", "other"]
COMMIT = re.compile(r"^\[([^\]\s]+)(?: \(root-commit\))? ([0-9a-f]{7,40})\] (.+)$", re.MULTILINE)


def session_hash(source, native_id):
    return hashlib.sha256(("%s:%s" % (source, native_id)).encode()).hexdigest()[:16]


def scope_of(path):
    for name, pattern in SCOPE_RULES:
        if pattern.search(path):
            return name
    return "other"


def iso(when):
    return when.isoformat() if when else None


def from_iso(text):
    """Parse an ISO timestamp, accepting the `Z` suffix Python 3.10 rejects."""
    if not text:
        return None
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def from_ms(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc) if ms else None


class Session:
    """Accumulates one session's digest record; sources feed it events in
    order and `record()` returns the finished dict."""

    def __init__(self, source, native_id):
        self.source = source
        self.native_id = native_id
        self.title = None
        self.cwd = None
        self.times = []
        self.models = set()
        self.requests = []
        self.assistant_messages = set()
        self.tools = {}
        self.errors = 0
        self.files_read = set()
        self.files_written = set()
        self.commands = []
        self.commits = []
        self.subagents = 0
        self.final_text = None

    def at(self, when):
        if when is not None:
            self.times.append(when)

    def request(self, when, text):
        self.at(when)
        self.requests.append({"time": iso(when), "text": text[:MAX_REQUEST]})

    def assistant(self, when, message_id, model, text):
        """`text` is None for subagent turns, which never become the final text."""
        self.at(when)
        self.assistant_messages.add(message_id)
        if model:
            self.models.add(model)
        if text:
            self.final_text = text[:MAX_FINAL]

    def tool(self, kind, path=None, written=False, command=None):
        self.tools[kind] = self.tools.get(kind, 0) + 1
        if path:
            (self.files_written if written else self.files_read).add(path)
        if command and len(self.commands) < MAX_COMMANDS:
            self.commands.append(command[:MAX_COMMAND])

    def wrote(self, path):
        self.files_written.add(path)

    def result(self, output, error):
        if error:
            self.errors += 1
        if output:
            for branch, sha, subject in COMMIT.findall(output):
                self.commits.append({"branch": branch, "sha": sha, "subject": subject[:200]})

    def scope(self, scopes):
        cwd_scope = scope_of(self.cwd or "")
        if cwd_scope not in GENERIC_SCOPES:
            return cwd_scope
        in_scope = {name: count for name, count in scopes.items() if name != "other"}
        if not in_scope:
            return "other"
        leader = max(in_scope, key=in_scope.get)
        return leader if in_scope[leader] >= MIN_SCOPE_SHARE * sum(scopes.values()) else "other"

    def record(self):
        scopes = {}
        for path in sorted(self.files_read | self.files_written):
            name = scope_of(path)
            scopes[name] = scopes.get(name, 0) + 1
        for command in self.commands:
            name = scope_of(command)
            if name != "other":
                scopes[name] = scopes.get(name, 0) + 1
        times = sorted(self.times)
        start = times[0] if times else None
        end = times[-1] if times else None
        gaps = [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:])]
        return {
            "session": session_hash(self.source, self.native_id),
            "source": self.source,
            "native_id": self.native_id,
            "title": self.title,
            "cwd": self.cwd,
            "started": iso(start),
            "ended": iso(end),
            "duration_s": (end - start).total_seconds() if start else None,
            # Wall time with idle stretches (resumed sessions, overnight) removed.
            "active_s": sum(gap for gap in gaps if gap <= IDLE_GAP_S) if start else None,
            "models": sorted(self.models),
            "requests": self.requests,
            "assistant_messages": len(self.assistant_messages),
            "tools": self.tools,
            "tool_errors": self.errors,
            "files_read": sorted(self.files_read),
            "files_written": sorted(self.files_written),
            "commands": self.commands,
            "commits": self.commits,
            "subagents": self.subagents,
            "final_text": self.final_text,
            "scopes": scopes,
            "scope": self.scope(scopes),
        }
