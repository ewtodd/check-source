"""G0 stage 3 (knoll-bench-plan.md section 6.0): tie each benchmarkable task
to a git repository and verify the commits the classifier attributed to it.

Script only, no model. A task's queue is `g2` when a verified commit on the
repository's current history changes code (a candidate for a code-history
item, plan section 6.2), `g1` when it has a repository but no such commit (an
analysis-unit candidate, section 6.1), and `idea` when no repository is found
on this machine. Commit ids the classifier gave that the repository does not
have are counted, not kept. Results go to <private>/link/<source>.jsonl.
"""

import hashlib
import json
import os
import pathlib
import re
import subprocess

from .digest import SOURCES

QUEUES = ["g1", "g2", "idea"]
TESTS = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.(py|cpp|cc|cxx)$|(^|/)Test[^/]*\.(cpp|cxx|C)$"
)
NOT_CODE = re.compile(r"\.(md|rst|txt|tex|bib|png|pdf|svg|jpg|lock)$|(^|/)(LICENSE|README)[^/]*$", re.IGNORECASE)
VOCAB = frozenset(
    list(SOURCES)
    + QUEUES
    + [
        "source",
        "sessions",
        "tasks",
        "benchmarkable",
        "queue",
        "repos",
        "commits_claimed",
        "commits_verified",
        "commits_missing",
        "commits_on_head",
        "commits_touching_tests",
    ]
)


def git(repo, *args):
    """stdout of `git -C repo args`, or None if git fails."""
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    return done.stdout.strip() if done.returncode == 0 else None


class Repos:
    """git root lookup, cached per directory."""

    def __init__(self):
        self.roots = {}

    def root(self, path):
        """The enclosing work tree of an absolute path; relative paths are
        refused, since they would resolve against this process's directory."""
        if not path or not pathlib.Path(path).is_absolute():
            return None
        directory = pathlib.Path(path)
        chain = [directory, *directory.parents]
        for candidate in chain:
            if candidate in self.roots:
                found = self.roots[candidate]
                break
            if (candidate / ".git").exists():
                found = candidate
                break
        else:
            found = None
        for candidate in chain:
            self.roots.setdefault(candidate, found)
            if candidate == found:
                break
        return found


def commit_info(repo, claimed):
    full = git(repo, "rev-parse", "--verify", "--quiet", "%s^{commit}" % claimed)
    if not full:
        return None
    files = (git(repo, "show", "--name-only", "--format=", full) or "").splitlines()
    on_head = (
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", full, "HEAD"], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    return {
        "sha": full,
        "parent": git(repo, "rev-parse", "--verify", "--quiet", full + "^"),
        "time": git(repo, "show", "-s", "--format=%cI", full),
        "files": files,
        "on_head": on_head,
        "touches_code": any(not NOT_CODE.search(name) for name in files),
        "touches_tests": any(TESTS.search(name) for name in files),
    }


def task_id(session, index):
    return hashlib.sha256(("%s:%d" % (session, index)).encode()).hexdigest()[:16]


def jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def latest(rows):
    """Keep the last successful row per session (classify appends on resume)."""
    kept = {}
    for row in rows:
        if not row.get("error"):
            kept[row["session"]] = row
    return list(kept.values())


def run_stage(private, source):
    digests = {record["session"]: record for record in jsonl(private / "digest" / ("%s.jsonl" % source))}
    rows = latest(jsonl(private / "classify" / ("%s.jsonl" % source)))
    out_dir = private / "link"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    repos = Repos()
    seen_repos = set()
    summary = {
        "source": source,
        "sessions": 0,
        "tasks": 0,
        "benchmarkable": 0,
        "queue": {},
        "repos": 0,
        "commits_claimed": 0,
        "commits_verified": 0,
        "commits_missing": 0,
        "commits_on_head": 0,
        "commits_touching_tests": 0,
    }
    with open(out_dir / ("%s.jsonl" % source), "w") as out:
        for row in rows:
            if not row["in_scope"]:
                continue
            digest = digests[row["session"]]
            summary["sessions"] += 1
            for index, task in enumerate(row["tasks"]):
                summary["tasks"] += 1
                if not task["benchmarkable"]:
                    continue
                summary["benchmarkable"] += 1
                paths = task["files"] or digest["files_written"] or digest["files_read"]
                if digest["cwd"]:
                    paths = [os.path.normpath(os.path.join(digest["cwd"], path)) for path in paths]
                candidates = [repos.root(path) for path in paths] + [repos.root(digest["cwd"])]
                ranked = sorted({c for c in candidates if c}, key=lambda c: -candidates.count(c))
                repo = ranked[0] if ranked else None
                commits = []
                for claimed in dict.fromkeys(task["commits"]):
                    summary["commits_claimed"] += 1
                    info = None
                    for candidate in ranked:
                        info = commit_info(candidate, claimed)
                        if info:
                            break
                    if info is None:
                        summary["commits_missing"] += 1
                        continue
                    summary["commits_verified"] += 1
                    summary["commits_on_head"] += info["on_head"]
                    summary["commits_touching_tests"] += info["touches_tests"]
                    commits.append(info)
                if any(c["on_head"] and c["touches_code"] for c in commits):
                    queue = "g2"
                elif repo:
                    queue = "g1"
                else:
                    queue = "idea"
                summary["queue"][queue] = summary["queue"].get(queue, 0) + 1
                if repo:
                    seen_repos.add(repo)
                times = [digest["requests"][n]["time"] for n in task["requests"] if 0 <= n < len(digest["requests"])]
                out.write(
                    json.dumps(
                        {
                            "task": task_id(row["session"], index),
                            "session": row["session"],
                            "source": source,
                            "project": row["project"],
                            "time": max(times) if times else digest["ended"],
                            **{key: task[key] for key in ("statement", "domain", "type", "outcome", "difficulty")},
                            "files": task["files"],
                            "repo": str(repo) if repo else None,
                            "remote": git(repo, "remote", "get-url", "origin") if repo else None,
                            "commits": commits,
                            "queue": queue,
                        }
                    )
                    + "\n"
                )
    summary["repos"] = len(seen_repos)
    return summary
