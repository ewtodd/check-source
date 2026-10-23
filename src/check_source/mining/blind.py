"""Run a pipeline stage blind: its data goes to a private log, the terminal
gets only a vetted summary.

The operator of the mining pipeline (a remote assistant) must not see session
content, so nothing a stage prints, logs or raises reaches stdout. A stage
returns a summary built from counts and a fixed vocabulary; `vet` withholds a
summary holding any other string, and a failure is reported as its exception
type and the pipeline-code lines it passed through, never its message.
"""

import contextlib
import datetime
import json
import os
import pathlib
import re
import traceback

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
HASH = re.compile(r"[0-9a-f]{16}")


def private_dir():
    """The private root, `$KNOLL_PRIVATE` or ~/knoll-private, created 0700.

    A directory inside a git work tree is refused, so private output cannot
    be committed by accident.
    """
    root = pathlib.Path(os.environ.get("KNOLL_PRIVATE", "~/knoll-private")).expanduser().resolve()
    for parent in [root, *root.parents]:
        if (parent / ".git").exists():
            raise SystemExit("KNOLL_PRIVATE is inside a git work tree; refusing")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def vet(value, vocab, where="summary"):
    """None if `value` may be shown, else where the first offending entry is.

    Allowed: numbers, booleans, None, 16-hex-digit hashes, and strings or dict
    keys in `vocab`, nested in lists and dicts. The returned location names
    only keys that already passed.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return None
    if isinstance(value, str):
        if value in vocab or HASH.fullmatch(value):
            return None
        return "%s: string outside the vocabulary" % where
    if isinstance(value, (list, tuple)):
        for item in value:
            problem = vet(item, vocab, where + "[]")
            if problem:
                return problem
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not (isinstance(key, str) and key in vocab):
                return "%s: key outside the vocabulary" % where
            problem = vet(item, vocab, "%s.%s" % (where, key))
            if problem:
                return problem
        return None
    return "%s: unsupported type %s" % (where, type(value).__name__)


def describe(error):
    """`Type at file:line <- file:line`, pipeline frames only, innermost first."""
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if pathlib.Path(frame.filename).resolve().is_relative_to(PACKAGE_DIR)
    ]
    where = " <- ".join(
        "%s:%d" % (pathlib.Path(frame.filename).name, frame.lineno) for frame in reversed(frames)
    )
    return "%s at %s" % (type(error).__name__, where or "a library frame")


def run(stage, func, vocab, **kwargs):
    """Call `func(private_root, **kwargs)` with stdout and stderr sent to a
    private log, then print only the vetted summary or a contentless failure.

    Returns the exit status.
    """
    root = private_dir()
    logs = root / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S-%f")
    log_path = logs / ("%s-%s.log" % (stage, stamp))
    failure = None
    summary = None
    with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            summary = func(root, **kwargs)
        except BaseException as error:  # noqa: BLE001 - Ctrl-C and SystemExit are reported the same way
            traceback.print_exc()
            failure = describe(error)
    print("stage %s, private log %s" % (stage, log_path.name))
    if failure:
        print("FAILED: %s" % failure)
        return 1
    problem = vet(summary, vocab)
    if problem:
        print("summary withheld (%s)" % problem)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0
