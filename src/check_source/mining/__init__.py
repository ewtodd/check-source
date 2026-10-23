"""Knoll bench task mining from agent sessions (knoll-bench-plan.md section
6.0). Every stage runs through `blind.run`."""

from . import blind, digest


def register(subparsers):
    parser = subparsers.add_parser("knoll-mine", help="mine agent sessions for Knoll bench tasks (runs blind)")
    stages = parser.add_subparsers(dest="stage", required=True)
    stage = stages.add_parser("digest", help="stage 1: one digest record per session, no model")
    stage.add_argument("--source", choices=sorted(digest.SOURCES), required=True)
    stage.add_argument(
        "--exclude", action="append", default=[], metavar="SESSION_ID", help="native session id to skip (repeatable)"
    )
    stage.set_defaults(func=run_digest)


def run_digest(args):
    raise SystemExit(
        blind.run(
            "digest-%s" % args.source,
            digest.run_stage,
            digest.VOCAB,
            source=args.source,
            exclude=args.exclude,
        )
    )
