"""Knoll bench task mining from agent sessions (knoll-bench-plan.md section
6.0). Every stage runs through `blind.run`."""

from . import blind, build, classify, digest, draft, inventory, link, llm, rank, repair


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
    stage = stages.add_parser("link", help="stage 3: tie benchmarkable tasks to repositories and verified commits")
    stage.add_argument("--source", choices=sorted(digest.SOURCES), required=True)
    stage.set_defaults(func=run_link)
    stage = stages.add_parser("rank", help="stage 4: merge near-duplicate tasks and rank the clusters")
    stage.add_argument("--scope", choices=sorted(digest.SOURCES) + [rank.ALL], default=rank.ALL)
    stage.set_defaults(func=run_rank)
    stage = stages.add_parser("inventory", help="phase 0: a local model describes each analysis code file")
    stage.add_argument("--set", dest="set_name", choices=inventory.SETS, required=True)
    stage.add_argument("--model", choices=sorted(llm.LOCAL_MODELS), default=inventory.MODEL)
    stage.add_argument("--dry-run", action="store_true", help="count the units without calling a model")
    stage.add_argument("--concurrency", type=int, default=4)
    stage.set_defaults(func=run_inventory)
    stage = stages.add_parser("draft", help="pass 2: a local model drafts one item per unit; static gates run")
    stage.add_argument("--set", dest="set_name", choices=draft.SETS, required=True)
    stage.add_argument("--model", choices=sorted(llm.LOCAL_MODELS), default=draft.MODEL)
    stage.add_argument("--limit", type=int, default=0, help="draft at most this many items (0: all)")
    stage.add_argument("--concurrency", type=int, default=4)
    stage.add_argument("--retry-failed", action="store_true", help="keep finished drafts; redo only failed units")
    stage.add_argument("--effort", choices=["low", "high"], default="high", help="reasoning effort of the first attempt")
    stage.set_defaults(func=run_draft)
    stage = stages.add_parser("build", help="resolve gold by running reference code in a sandbox; compile items")
    stage.add_argument("--set", dest="set_name", choices=draft.SETS, required=True)
    stage.add_argument(
        "--allow-run", action="store_true", help="run reference code (sandboxed); without it, only plan the runs"
    )
    stage.set_defaults(func=run_build)
    stage = stages.add_parser("repair", help="rewrite locators that miss their reference output, verified on it")
    stage.add_argument("--set", dest="set_name", choices=draft.SETS, required=True)
    stage.add_argument("--model", choices=sorted(llm.LOCAL_MODELS), default=repair.MODEL)
    stage.set_defaults(func=run_repair)


def run_repair(args):
    raise SystemExit(
        blind.run("repair-%s" % args.set_name, repair.run_stage, repair.VOCAB, set_name=args.set_name, model_name=args.model)
    )


def run_build(args):
    raise SystemExit(
        blind.run("build-%s" % args.set_name, build.run_stage, build.VOCAB, set_name=args.set_name, allow_run=args.allow_run)
    )


def run_draft(args):
    raise SystemExit(
        blind.run(
            "draft-%s" % args.set_name,
            draft.run_stage,
            draft.VOCAB,
            set_name=args.set_name,
            model_name=args.model,
            limit=args.limit,
            concurrency=args.concurrency,
            retry_failed=args.retry_failed,
            effort=args.effort,
        )
    )


def run_inventory(args):
    raise SystemExit(
        blind.run(
            "inventory-%s" % args.set_name,
            inventory.run_stage,
            inventory.VOCAB,
            set_name=args.set_name,
            model_name=args.model,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    )


def run_link(args):
    raise SystemExit(blind.run("link-%s" % args.source, link.run_stage, link.VOCAB, source=args.source))


def run_rank(args):
    raise SystemExit(blind.run("rank-%s" % args.scope, rank.run_stage, rank.VOCAB, scope=args.scope))


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
