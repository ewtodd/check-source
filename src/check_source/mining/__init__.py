"""Knoll bench task mining from agent sessions (knoll-bench-plan.md section
6.0). Every stage runs through `blind.run`."""

from . import blind, classify, digest, llm


def register(subparsers):
    parser = subparsers.add_parser("knoll-mine", help="mine agent sessions for Knoll bench tasks (runs blind)")
    stages = parser.add_subparsers(dest="stage", required=True)
    stage = stages.add_parser("digest", help="stage 1: one digest record per session, no model")
    stage.add_argument("--source", choices=sorted(digest.SOURCES), required=True)
    stage.add_argument(
        "--exclude", action="append", default=[], metavar="SESSION_ID", help="native session id to skip (repeatable)"
    )
    stage.set_defaults(func=run_digest)
    stage = stages.add_parser("classify", help="stage 2: a local model lists each in-scope session's tasks")
    stage.add_argument("--source", choices=sorted(digest.SOURCES), required=True)
    stage.add_argument("--model", choices=sorted(llm.LOCAL_MODELS), default=classify.MODEL)
    stage.add_argument("--limit", type=int, default=0, help="classify at most this many sessions (0: all)")
    stage.add_argument("--fresh", action="store_true", help="discard earlier results instead of resuming")
    stage.add_argument("--concurrency", type=int, default=3)
    stage.set_defaults(func=run_classify)


def run_classify(args):
    raise SystemExit(
        blind.run(
            "classify-%s" % args.source,
            classify.run_stage,
            classify.VOCAB,
            source=args.source,
            model_name=args.model,
            limit=args.limit,
            fresh=args.fresh,
            concurrency=args.concurrency,
        )
    )


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
