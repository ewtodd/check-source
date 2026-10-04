"""Answer extraction and equivalence, matching lm-evaluation-harness 0.4.13.

The GSM8K filters, the AIME string normalizer and the Minerva MATH normalizer
are copied from that harness so a checkpoint measured here is measured the same
way as one measured with lm-eval. MATH-500 additionally uses `math_verify`,
which is why the flake packages it. The only intentional divergence is the
five-second `sympy` timeout in the Minerva `is_equiv`: signal.alarm only works
on the main thread and the runner is threaded, so that guard is dropped.
"""

import re

# ---------------------------------------------------------------------------
# GSM8K (lm_eval/tasks/gsm8k)
# ---------------------------------------------------------------------------

GSM8K_STRICT_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
GSM8K_FLEX_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
GSM8K_IGNORE_PATTERNS = [",", r"\$", r"(?s).*#### ", r"\.$"]


def clean_gsm8k(text):
    if text is None:
        return None
    for pattern in GSM8K_IGNORE_PATTERNS:
        text = re.sub(pattern, "", text)
    return text.strip()


def extract_strict(text):
    match = GSM8K_STRICT_PATTERN.search(text)
    return clean_gsm8k(match.group(1)) if match else None


def extract_flexible(text):
    last = None
    for match in GSM8K_FLEX_PATTERN.finditer(text):
        last = match.group(0)
    return clean_gsm8k(last) if last else None


def gsm8k_gold(answer):
    match = GSM8K_STRICT_PATTERN.search(answer)
    return clean_gsm8k(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Boxed answers (shared by AIME and Minerva)
# ---------------------------------------------------------------------------

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


# ---------------------------------------------------------------------------
# AIME (lm_eval/tasks/aime/utils.py)
# ---------------------------------------------------------------------------

def aime_extract(response):
    """Last \\boxed{} if present, else the span between the outer $...$."""
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass
    return answer


def aime_is_equiv(prediction, target):
    if prediction is None and target is None:
        return True
    if prediction is None or target is None:
        return False
    try:
        return strip_string(prediction) == strip_string(target)
    except Exception:
        return prediction == target


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)
    return string


# ---------------------------------------------------------------------------
# Minerva MATH (lm_eval/tasks/minerva_math/utils.py)
# ---------------------------------------------------------------------------

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer):
    """Normalize a final answer to a quantitative reasoning question.

    Copied character for character from lm-eval (appendix D of Lewkowycz et al.).
    """
    final_answer = final_answer.split("=")[-1]

    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            final_answer = re.sub(rf"(?<!\\)\b{expr}\b", "", final_answer)

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            segments = re.split(r"(\\[a-zA-Z]+)", final_answer)
            for i in range(0, len(segments), 2):
                segments[i] = segments[i].replace(expr, "")
            final_answer = "".join(segments)
        else:
            final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)(\[[^\]]*\])(\w)", "\\1\\2{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)(?!\[)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", final_answer):
        final_answer = final_answer.replace(",", "")

    return final_answer


def get_unnormalized_answer(text):
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


_MINERVA_CACHE = {}


def minerva_modules():
    """Lazily import the Minerva scorer's heavy dependencies."""
    if not _MINERVA_CACHE:
        try:
            import antlr4  # noqa: F401
            import sympy
            from math_verify import parse, verify
            from sympy.parsing.latex import parse_latex
        except ImportError as error:
            raise SystemExit(
                "the math bench needs sympy, math_verify and antlr4-python3-runtime==4.11; "
                "run check-source through the flake (nix run) so they are on PATH"
            ) from error
        _MINERVA_CACHE.update(
            {"sympy": sympy, "parse": parse, "verify": verify, "parse_latex": parse_latex}
        )
    return _MINERVA_CACHE


def minerva_is_equiv(x1, x2):
    if isinstance(x1, str) and x1 and x1 == x2:
        return True

    modules = minerva_modules()
    sympy = modules["sympy"]
    parse_latex = modules["parse_latex"]

    try:
        parsed_x1 = parse_latex(x1)
        parsed_x2 = parse_latex(x2)
    except Exception:
        return False

    try:
        diff = parsed_x1 - parsed_x2
    except TypeError:
        return False

    try:
        return bool(sympy.simplify(diff) == 0)
    except Exception:
        return False


def minerva_score(gold_solution, gold_answer, candidate):
    """Return lm-eval's two minerva_math500 metrics for one candidate."""
    modules = minerva_modules()
    parse = modules["parse"]
    verify = modules["verify"]

    try:
        unnormalized = get_unnormalized_answer(candidate)
        exact_match = 1 if minerva_is_equiv(normalize_final_answer(unnormalized), gold_answer) else 0
    except Exception:
        exact_match = 0

    # math_verify's timeout uses signal.alarm, which raises in a worker thread;
    # disable it (the runner is threaded) and let the parse itself fail safe.
    try:
        math_verify = (
            1
            if verify(
                gold=parse(gold_solution, parsing_timeout=None),
                target=parse(candidate, parsing_timeout=None),
                timeout_seconds=None,
            )
            else 0
        )
    except Exception:
        math_verify = 0

    return {"exact_match": exact_match, "math_verify": math_verify}
