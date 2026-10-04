"""Command line entry point: `check-source <bench> [options]`."""

import argparse

from . import aime, gsm8k, knoll, math500


def build_parser():
    parser = argparse.ArgumentParser(
        prog="check-source",
        description="Reference benchmarks for OpenAI-compatible LLM endpoints",
    )
    subparsers = parser.add_subparsers(dest="bench", required=True)
    gsm8k.register(subparsers)
    aime.register(subparsers)
    math500.register(subparsers)
    knoll.register(subparsers)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
