"""Make the noisy copy of the week: ASR-style word errors (~10%) and misheard names.

Deterministic (seeded), so week_noisy.txt is reproducible:
    uv run python -m tests.eval.recall_benchmark.noise          # rewrite week_noisy.txt
    uv run python -m tests.eval.recall_benchmark.noise --check  # exit 1 if it is out of date

What it does, per line:
  - person names are misheard consistently most of the time (Priya -> Pria, Rahul -> Raul, ...)
  - every other word has a ~10% chance of an error: a homophone or near-sound substitution,
    a dropped word, an inserted filler, or a one-letter misspelling
  - tokens with digits are never edited (numbers are kept so the facts stay answerable;
    the misheard names and wrong words already make retrieval harder)
"""

from __future__ import annotations

import random
import re
import sys

from tests.eval.recall_benchmark import WEEK, WEEK_NOISY, Line, parse_week

SEED = 20261005
WORD_ERROR_P = 0.10
NAME_MISHEAR_P = 0.85

NAMES = {
    "Priyanka": "Priyanca",
    "Priya": "Pria",
    "Rahul": "Raul",
    "Ananya": "Anania",
    "Kavya": "Kavia",
    "Vikram": "Vikrum",
    "Mehta": "Meta",
    "Neha": "Nea",
    "Arjun": "Arjoon",
    "Shobha": "Shoba",
    "Anil": "Aneel",
    "Sam": "Psalm",
    "Lonavala": "Lonavla",
    "Karan": "Karn",
    "Meera": "Mira",
    "Divya": "Divia",
    "Imran": "Imraan",
    "Nisha": "Neesha",
    "Dharampeth": "Dharam path",
    "Sitabuldi": "Sita buldi",
    "Orion": "O'Ryan",
    "DMart": "D mart",
}

SOUNDS_LIKE = {
    "to": "two", "two": "to", "for": "four", "four": "for", "meet": "meat", "their": "there", "there": "their",
    "right": "write", "write": "right", "know": "no", "no": "know", "new": "knew", "our": "are", "are": "our",
    "week": "weak", "weekend": "weak end", "by": "buy", "buy": "by", "here": "hear", "hear": "here", "one": "won",
    "would": "wood", "be": "bee", "see": "sea", "eight": "ate", "pay": "pray", "rent": "rant", "book": "look",
    "cafe": "café", "flat": "flight", "flight": "fight", "code": "coat", "train": "rain", "draft": "graft",
    "session": "season", "sessions": "seasons", "mentor": "meant her", "vegan": "vegans", "allergic": "a allergic",
    "password": "pass word", "wifi": "why fi", "birthday": "birth day", "wedding": "wedding's", "dentist": "dentists",
    "library": "liberty", "account": "a count", "allergy": "allergies", "drive": "dive", "office": "officer",
    "insurance": "assurance", "review": "preview", "promoted": "promote", "moving": "moving in", "doctor": "doctors",
}
FILLERS = ("uh", "um", "like", "the", "a", "so")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*|\S")


def _misspell(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(("drop", "double", "swap"))
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]


def _match_case(src: str, out: str) -> str:
    return out[:1].upper() + out[1:] if src[:1].isupper() else out


def noisy_text(text: str, rng: random.Random) -> str:
    out: list[str] = []
    for word in re.findall(r"\S+", text):
        m = re.match(r"^([\"(]*)(.*?)([.,!?;:\")]*)$", word)
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        base, poss = (core[:-2], core[-2:]) if core.endswith("'s") else (core, "")
        if base in NAMES:
            if rng.random() < NAME_MISHEAR_P:
                base = NAMES[base]
            out.append(lead + base + poss + trail)
            continue
        if any(ch.isdigit() for ch in core) or not core or rng.random() >= WORD_ERROR_P:
            out.append(word)
            continue
        op = rng.choices(("sub", "drop", "insert", "misspell"), weights=(4, 2, 2, 2))[0]
        low = core.lower()
        if op == "sub" and low in SOUNDS_LIKE:
            out.append(lead + _match_case(core, SOUNDS_LIKE[low]) + trail)
        elif op == "drop" and len(text.split()) > 3:
            if trail:  # keep sentence punctuation on the previous word
                if out:
                    out[-1] += trail
                else:
                    out.append(lead + core + trail)
        elif op == "insert":
            first = not out and core != "I" and not core.startswith("I'")
            out.append(rng.choice(FILLERS))
            out.append(lead + (core[:1].lower() + core[1:] if first else core) + trail)
        else:
            out.append(lead + _match_case(core, _misspell(low, rng)) + trail)
    s = " ".join(out)
    return s[:1].upper() + s[1:]


def word_error_rate(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def _added(ln: Line) -> bool:
    """Lines x.40 and later were added for Prompt 5; they draw noise from their own seed, so every earlier
    line keeps exactly the noisy text it had (and earlier runs stay comparable)."""
    return int(ln.id.split(".")[1]) >= 40


def make_noisy(lines: list[Line], seed: int = SEED) -> tuple[list[Line], float]:
    rng = random.Random(seed)
    out, errors, words, seen = [], 0, 0, set()
    for ln in lines:
        r = random.Random(f"{seed}-{ln.id}") if _added(ln) else rng
        text = noisy_text(ln.text, r)
        while text in seen:  # every injected line must be unique (lines are matched by text)
            text = noisy_text(ln.text, r)
        seen.add(text)
        e, n = word_error_rate(ln.text, text)
        errors, words = errors + e, words + n
        out.append(Line(ln.id, ln.when, ln.speaker, ln.keep, text))
    return out, errors / max(words, 1)


def render(lines: list[Line], wer: float) -> str:
    head = [
        "# Kin recall benchmark: the week with ASR-style noise. GENERATED by tests/eval/recall_benchmark/noise.py",
        f"# from week.txt (seed {SEED}); do not edit by hand. Word error rate vs the clean copy: {wer:.1%}.",
        "# Same ids, timestamps, speakers and keep/drop labels as week.txt; only the text differs.",
        "",
    ]
    rows = [f"{ln.id} | {ln.when:%Y-%m-%d %H:%M} | {ln.speaker:<8} | {'keep' if ln.keep else 'drop'} | {ln.text}" for ln in lines]
    return "\n".join(head + rows) + "\n"


def build() -> str:
    lines, wer = make_noisy(parse_week(WEEK.read_text(encoding="utf-8")))
    return render(lines, wer)


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        current = WEEK_NOISY.read_text(encoding="utf-8") if WEEK_NOISY.exists() else ""
        if current != text:
            print("week_noisy.txt is out of date: run python -m tests.eval.recall_benchmark.noise")
            return 1
        return 0
    WEEK_NOISY.write_text(text, encoding="utf-8")
    print(text.splitlines()[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
