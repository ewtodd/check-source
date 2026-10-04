"""Knoll bench: experimental nuclear-physics / radiation-detection problems.

Placeholder until the problem set exists. Intended shape of one item:

    id          stable problem identifier
    prompt      the question exactly as it should be shown to the model
    answer      gold answer, a number or a LaTeX expression
    tolerance   optional absolute tolerance for numeric answers
    tags        optional list, e.g. ["spectroscopy", "efficiency", "decay"]

Scoring will follow the other benches' chat protocol (levels none/medium/xhigh,
per-request seed, T=1.0 / top_p=0.95 / top_k=20). The default comparison will be
numeric-with-tolerance when both gold and prediction parse as floats, falling
back to `math_verify` otherwise, because radiation-detection answers are
physical quantities rather than exact symbolic forms.
"""


def register(subparsers):
    parser = subparsers.add_parser("knoll", help="Knoll bench (no problem set yet)")
    parser.set_defaults(func=main)


def main(args):
    raise SystemExit(
        "the knoll bench has no problem set yet; see src/check_source/knoll.py for the intended item format"
    )
