"""Item builder (knoll-bench-plan.md sections 5.4, 5.6 and 7.2): resolve each
drafted item's gold by running its reference code, then compile items.jsonl
and a manifest.

Reference code runs in a sandbox, because analysis macros write next to their
inputs and the inputs live in /labdata:

- the repository at the pinned commit is extracted with `git archive` into a
  fresh private directory, so no checkout is touched;
- each data symlink in the working tree (root_files -> /labdata/...) becomes an
  overlayfs mount in the copy, with the data as the read-only lower layer and
  writes going to a private upper directory;
- bwrap mounts the whole filesystem read-only except the copy, the upper
  directories and a private HOME, and unshares the network;
- the environment is the repository's own dev shell at the pinned commit,
  entered from inside the copy so its hooks point at the copy.

Nothing runs unless `allow_run` is set; otherwise the stage reports the runs it
would make. Runs are cached by (repository, commit, workdir, command).
Output: <private>/build/<set>/items.jsonl and manifest.json.
"""

import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import tomllib

from .draft import SETS, data_links
from .link import git
from .validate import UNRESOLVED

SHELLS = {"yap_psd": "YAP-PSD", "78mbr": "78mBr"}
PROJECT_DIRS = {"yap_psd": "YAP-PSD", "78mbr": "78mBr"}
RUN_TIMEOUT_S = 4 * 3600
# Reference runs write processed files into their overlays; stop well before
# the private volume fills.
MIN_FREE_BYTES = 20 * 10**9
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
FAILURES = [
    "no_dev_shell",
    "not_run",
    "run_failed",
    "timeout",
    "locator_no_match",
    "locator_ambiguous",
    "locator_not_number",
    "unsupported_locator",
    "published_pending",
    "input_missing",
]
VOCAB = frozenset(
    SETS
    + FAILURES
    + [
        "set",
        "allow_run",
        "items",
        "built",
        "unresolved",
        "leak_suspected",
        "held_for_review",
        "runs_planned",
        "runs_cached",
        "runs_made",
        "run_status",
        "ok",
        "failed",
        "checks_resolved",
        "inputs_hashed",
    ]
)


SHELL_CACHE = {}


def dev_shell(repo, commit, project, wanted):
    """The dev shell to run a reference in: the UM-ANSG project's own, else
    the item's named shell if the flake has it, else `default`, else the
    first; None if the flake defines none."""
    if project in SHELLS:
        return SHELLS[project]
    key = (str(repo), commit)
    if key not in SHELL_CACHE:
        done = subprocess.run(
            ["nix", "eval", "--json", "git+file://%s?rev=%s#devShells.x86_64-linux" % (repo, commit),
             "--apply", "builtins.attrNames"],
            capture_output=True, text=True, timeout=600,
        )  # fmt: skip
        try:
            SHELL_CACHE[key] = json.loads(done.stdout) if done.returncode == 0 else []
        except ValueError:
            SHELL_CACHE[key] = []
    shells = SHELL_CACHE[key]
    name = str(wanted or "").rpartition("#")[2]
    if name in shells:
        return name
    if "default" in shells:
        return "default"
    return shells[0] if shells else None


def run_key(repo, commit, workdir, command):
    return hashlib.sha256(("%s\0%s\0%s\0%s" % (repo, commit, workdir, command)).encode()).hexdigest()[:16]


def reference_run(private, repo, commit, prefix, shell, workdir, command):
    """Run the reference command in the sandbox; returns (status, stdout path).
    status is the exit code, "timeout", or the cached value of either."""
    run_dir = private / "build" / "runs" / run_key(repo, commit, workdir, command)
    log = run_dir / "stdout.log"
    status_file = run_dir / "exit_status"
    if status_file.exists():
        return status_file.read_text().strip(), log
    if shutil.disk_usage(private).free < MIN_FREE_BYTES:
        return "no_space", log
    src, home = run_dir / "src", run_dir / "home"
    for directory in (src, home, run_dir / "upper", run_dir / "work"):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, capture_output=True, check=True)
    mounts = []
    for number, (relative, target) in enumerate(data_links(repo, prefix)):
        destination = src / relative
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        destination.mkdir(parents=True, exist_ok=True)
        upper, work = run_dir / "upper" / str(number), run_dir / "work" / str(number)
        upper.mkdir(mode=0o700, exist_ok=True)
        work.mkdir(mode=0o700, exist_ok=True)
        mounts += ["--overlay-src", str(target), "--overlay", str(upper), str(work), str(destination)]
    # The analyses use torch (CUDA) and multiprocessing: give the minimal /dev
    # a private /dev/shm and the GPU nodes.
    devices = []
    for node in sorted(pathlib.Path("/dev").glob("nvidia*")) + [pathlib.Path("/dev/dri")]:
        if node.exists():
            devices += ["--dev-bind", str(node), str(node)]
    sandbox = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        *devices,
        "--tmpfs", "/dev/shm",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(src), str(src),
        "--bind", str(home), str(home),
        "--setenv", "HOME", str(home),
        # nix develop points TMPDIR at a directory on the host's /tmp, which
        # the sandbox's fresh /tmp lacks; ACLiC and cling need it.
        "--setenv", "TMPDIR", "/tmp",
        # stdout is a file here, so Python would block-buffer it and lose
        # printed results to a crash at exit (PyROOT with torch does this).
        "--setenv", "PYTHONUNBUFFERED", "1",
        *mounts,
        "--unshare-net",
        "--die-with-parent",
        "--chdir", str(src / workdir),
        "--", "bash", "-c", command,
    ]  # fmt: skip
    flake = "git+file://%s?rev=%s#%s" % (repo, commit, shell)
    with open(log, "w") as out:
        try:
            # Low priority: reference analyses use every core, on a machine
            # someone is working on.
            done = subprocess.run(
                ["nice", "-n", "10", "nix", "develop", flake, "-c", *sandbox],
                cwd=src,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_S,
            )
            status = str(done.returncode)
        except subprocess.TimeoutExpired:
            status = "timeout"
    status_file.write_text(status)
    return status, log


JIT_FAILURE = "Failed to materialize symbols"
INTERPRETED_MACRO = re.compile(r"(\broot\b[^'\"]*?\s)(['\"]?)([\w./-]+\.(?:cpp|C|cxx))\2(?=\s|$)")


def compiled(command):
    """The command with its ROOT macro compiled by ACLiC (`macro.cpp+`), or
    None if it names no interpreted macro."""
    if not INTERPRETED_MACRO.search(command):
        return None
    return INTERPRETED_MACRO.sub(lambda m: "%s'%s+'" % (m.group(1), m.group(3)), command, count=1)


def run_reference(private, repo, commit, prefix, shell, workdir, command):
    """reference_run, retried with the macro compiled when cling cannot JIT it
    (threads and std::async need ACLiC). Returns (status, log, command used)."""
    status, log = reference_run(private, repo, commit, prefix, shell, workdir, command)
    alternative = compiled(command)
    if status not in ("0", "no_space") and alternative and log.exists() and JIT_FAILURE in log.read_text(errors="replace"):
        status, log = reference_run(private, repo, commit, prefix, shell, workdir, alternative)
        return status, log, alternative
    return status, log, command


def locate(locator, stdout):
    """The number a locator selects in the reference run's stdout."""
    match = re.fullmatch(r"regex: (.+) group=(\w+)", locator)
    if not match:
        raise Unresolved("unsupported_locator")
    pattern, group = match.groups()
    found = list(re.finditer(pattern, stdout, re.MULTILINE))
    if not found:
        raise Unresolved("locator_no_match")
    if len(found) > 1:
        raise Unresolved("locator_ambiguous")
    try:
        return float(found[0].group(group))
    except (IndexError, ValueError):
        raise Unresolved("locator_not_number") from None


class Unresolved(Exception):
    """`str(error)` is one of FAILURES."""


def leaked(value, prompt, check):
    """True if the prompt carries a number within the check's tolerance of the
    gold value, written with at least three significant digits."""
    tol = check.get("tol") or {}
    width = max(tol.get("abs") or 0, (tol.get("rel") or 0) * abs(value))
    for text in NUMBER.findall(prompt):
        digits = len(re.sub(r"[^0-9]", "", text.split("e")[0].split("E")[0]).lstrip("0"))
        if digits >= 3 and abs(float(text) - value) <= width:
            return True
    return False


def sha256(path, cache):
    stat = path.stat()
    key = "%s:%d:%d" % (path, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        cache[key] = digest.hexdigest()
    return cache[key]


def count(table, key):
    table[key] = table.get(key, 0) + 1


def run_stage(private, set_name, allow_run=False):
    draft_dir = private / "items" / "draft" / set_name
    with open(draft_dir / "index.jsonl") as handle:
        rows = [row for row in map(json.loads, handle) if row.get("state") in ("draft", "needs_human")]
    out_dir = private / "build" / set_name
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    hash_cache_path = private / "build" / "hashes.json"
    hash_cache = json.loads(hash_cache_path.read_text()) if hash_cache_path.exists() else {}
    summary = {
        "set": set_name,
        "allow_run": allow_run,
        "items": len(rows),
        "built": 0,
        "unresolved": {},
        "leak_suspected": 0,
        "held_for_review": 0,
        "runs_planned": 0,
        "runs_cached": 0,
        "runs_made": 0,
        "run_status": {},
        "checks_resolved": 0,
        "inputs_hashed": 0,
    }
    planned = set()
    built = []
    review = []
    for row in rows:
        text = (draft_dir / ("%s.toml" % row["id"])).read_text()
        item = tomllib.loads(text)
        repo = pathlib.Path(row["repo"])
        prefix = PROJECT_DIRS.get(row["project"], "")
        shell = dev_shell(repo, row["commit"], row["project"], (item.get("env") or {}).get("nix_develop"))
        gold, failure = {}, ("no_dev_shell" if shell is None else None)
        crashed = {}  # check id -> (exit status, value) from a run that crashed after printing
        checks = (item.get("checks") or []) if shell else []
        for check in checks:
            ref = check.get("gold_ref") or {}
            if ref.get("kind") == "published":
                failure = failure or "published_pending"
                continue
            if ref.get("kind") != "recomputed":
                continue
            workdir, command = ref.get("workdir") or ".", ref.get("command") or ""
            key = run_key(repo, row["commit"], workdir, command)
            done = (private / "build" / "runs" / key / "exit_status").exists()
            if key not in planned:
                planned.add(key)
                summary["runs_planned"] += 1
                summary["runs_cached"] += done
            if not done and not allow_run:
                failure = failure or "not_run"
                continue
            if not done:
                summary["runs_made"] += 1
            status, log, used = run_reference(private, repo, row["commit"], prefix, shell, workdir, command)
            ref["command"] = used
            if status != "0":
                count(summary["run_status"], "failed" if status != "timeout" else "timeout")
                failure = failure or ("timeout" if status == "timeout" else "run_failed")
                if status not in ("timeout", "no_space") and log.exists():
                    try:
                        crashed[check["id"]] = (status, locate(ref.get("locator", ""), log.read_text(errors="replace")))
                    except Unresolved:
                        pass
                continue
            count(summary["run_status"], "ok")
            try:
                gold[check["id"]] = locate(ref.get("locator", ""), log.read_text(errors="replace"))
                summary["checks_resolved"] += 1
            except Unresolved as error:
                failure = failure or str(error)
        inputs = []
        for entry in item.get("inputs") or []:
            origin = pathlib.Path(entry.get("from", "").replace("${REPO}", str(repo)))
            if not origin.is_file():
                failure = failure or "input_missing"
                continue
            inputs.append({"path": entry.get("path"), "from": entry.get("from"), "sha256": sha256(origin, hash_cache)})
            summary["inputs_hashed"] += 1
        if failure:
            count(summary["unresolved"], failure)
            # Every gold value found, but some only in a run that exited
            # non-zero: held for a human, never machine-verified.
            checks_with_gold = [c for c in checks if (c.get("gold_ref") or {}).get("kind") == "recomputed"]
            if failure == "run_failed" and crashed and all(c["id"] in gold or c["id"] in crashed for c in checks_with_gold):
                review.append(
                    {
                        "id": item["id"],
                        "reason": "reference exited non-zero after printing its gold",
                        "exit_status": sorted({status for status, _ in crashed.values()}),
                        "gold": {**gold, **{key: value for key, (_, value) in crashed.items()}},
                    }
                )
                summary["held_for_review"] += 1
            else:
                review.append({"id": item["id"], "reason": "unresolved: %s" % failure, "gold": gold})
            continue
        # A gold value the prompt already states is either a leak or a locator
        # that selected the wrong number; either way the item is held back
        # (plan section 7.2), not built.
        if any(leaked(value, item.get("prompt", ""), check) for check in item.get("checks") or []
               for value in [gold.get(check.get("id"))] if value is not None):
            summary["leak_suspected"] += 1
            review.append({"id": item["id"], "reason": "gold value appears in the prompt", "gold": gold})
            continue
        # The draft's own gate problems (the stage writes the pinned commit, so
        # a miscopied one no longer counts) or the model's needs_human list
        # send the resolved item to review, never to items.jsonl.
        item.setdefault("source", {})["commit"] = row["commit"]
        open_problems = sorted(
            {code for codes in (row.get("problems") or {}).values() for code in codes if code != "commit_mismatch"}
        )
        flagged = list((item.get("generation") or {}).get("needs_human") or [])
        if open_problems or flagged:
            review.append(
                {
                    "id": item["id"],
                    "reason": "open gate problems or model flags",
                    "gate_problems": open_problems,
                    "model_flags": flagged,
                    "gold": gold,
                }
            )
            summary["held_for_review"] += 1
            continue
        for check in item.get("checks") or []:
            if check.get("expected") == UNRESOLVED and check["id"] in gold:
                check["expected"] = gold[check["id"]]
        item.setdefault("verification", {})["state"] = "machine_verified"
        canonical = json.dumps(item, sort_keys=True)
        built.append(
            {
                "item": item,
                "item_hash": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
                "inputs": inputs,
                "source_toml_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    hash_cache_path.write_text(json.dumps(hash_cache))
    lines = [json.dumps(entry, sort_keys=True) for entry in sorted(built, key=lambda e: e["item"]["id"])]
    (out_dir / "items.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    manifest = {
        "set": set_name,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schema_version": 1,
        "items": {entry["item"]["id"]: entry["item_hash"] for entry in built},
        "items_jsonl_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out_dir / "review.jsonl").write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in review))
    summary["built"] = len(built)
    return summary
