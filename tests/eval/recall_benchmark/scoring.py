"""Rule-based checks for benchmark answers, and the metric roll-up. No models here.

An answer is judged on what Kin actually spoke (citations already stripped by the gate).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field

from kin.brain.gate import is_abstention
from tests.eval.recall_benchmark import Question

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fifth": 5, "ninth": 9, "twelfth": 12, "thirteenth": 13,
             "fourteenth": 14, "twentieth": 20}
_NUM_RE = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b")


def normalise(text: str) -> str:
    """Lower-case; "five" -> "5"; "1,850" -> "1850"; "Saffron2026" -> "saffron 2026"; "5pm" -> "5 pm"."""
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = _NUM_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1)]), s)
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _pattern(alt: str) -> re.Pattern:
    a = normalise(alt)
    tail = r"(?![0-9])" if a[-1:].isdigit() else ""  # "5" must not match "50"; "promot" matches "promoted"
    return re.compile(r"(?<![a-z0-9])" + re.escape(a) + tail)


def contains(answer: str, alt: str) -> bool:
    return bool(_pattern(alt).search(normalise(answer)))


def raw_marker(answer: str) -> bool:
    """A citation marker spoken aloud, e.g. "I don't remember that. [#]" (seen in real use)."""
    return bool(re.search(r"\[\s*#|#\s*\d|\[\s*\]", answer))


@dataclass
class Check:
    abstained: bool
    groups_hit: int
    groups_needed: int
    forbidden: list[str]
    correct: bool
    false_answer: bool  # spoke an answer that is wrong (or answered an unanswerable question)
    wrong_abstention: bool  # abstained although the week holds the answer
    marker_leak: bool


def abstained(text: str, outcome: str) -> bool:
    if outcome in ("abstained", "timeout", "error"):
        return True
    return outcome != "answered" and is_abstention(text)


def check(q: Question, text: str, outcome: str) -> Check:
    hits = sum(any(contains(text, a) for a in g) for g in q.expect)
    # "I don't remember Neha's extension [#5]" passes the gate as cited, but it is still an abstention
    abst = abstained(text, outcome) or (is_abstention(text) and hits == 0)
    bad = [f for f in q.forbid if contains(text, f)]
    leak = raw_marker(text)
    if q.abstain:
        correct = abst
        return Check(abst, 0, 0, bad, correct, not abst, False, leak)
    rule_ok = hits >= q.needed and not bad
    if abst:
        correct = q.accept_abstain
        return Check(abst, hits, q.needed, bad, correct, False, not correct, leak)
    return Check(abst, hits, q.needed, bad, rule_ok, not rule_ok, False, leak)


# --- metric roll-up ----------------------------------------------------------------------------------


def pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


@dataclass
class QuestionResult:
    id: str
    cat: str
    q: str
    reference: str
    spoken: str
    outcome: str
    path: str
    citations: str
    blocked_sentences: int
    total_ms: int
    first_audio_ms: int | None
    top_similarity: float | None
    top_score: float | None = None  # the hybrid confidence the gate thresholds (None: runs before hybrid recall)
    explain: str = ""  # the reply's "why this answer" JSON
    retrieved: list[dict] = field(default_factory=list)  # [{id, line, sim, text}]
    support: list[str] = field(default_factory=list)
    support_stored: list[str] = field(default_factory=list)
    support_retrieved: list[str] = field(default_factory=list)
    check: dict = field(default_factory=dict)
    judge: str | None = None
    failure: str | None = None


def failure_reason(q: Question, r: QuestionResult, floor: float) -> str | None:
    """Why a question failed, in pipeline order (write -> retrieve -> answer)."""
    c = r.check
    if c["correct"]:
        return None
    if q.abstain:
        return "false answer (should have abstained)"
    if set(r.support) - set(r.support_stored) and not r.support_retrieved:
        return "dropped at write"
    if not r.support_retrieved:
        return "not retrieved"
    skipped = "recall" not in r.path and "lookup" not in r.path  # the model answered without a tool
    if "lookup" in r.path:
        return "retrieved but wrong answer (routed to lookup)"
    if c["abstained"]:
        if skipped:
            return "wrong abstention (model skipped recall)"
        top = r.top_score if r.top_score is not None else r.top_similarity
        if "fallback: transcript" not in r.path and top is not None and top < floor:
            return "wrong abstention (retrieved, but under the similarity floor)"
        if r.blocked_sentences:
            return "wrong abstention (answer blocked by the citation gate)"
        return "wrong abstention (model said it doesn't remember)"
    if skipped:
        return "retrieved but wrong answer (model skipped recall)"
    return "retrieved but wrong answer"


def summarise(write: list[dict], results: list[QuestionResult], questions: dict[str, Question]) -> dict:
    keep = [w for w in write if w["keep"]]
    drop = [w for w in write if not w["keep"]]
    kept_ok = sum(w["decision"] == "stored" for w in keep)
    drop_ok = sum(w["decision"] != "stored" for w in drop)
    answerable = [r for r in results if not questions[r.id].abstain]
    unanswerable = [r for r in results if questions[r.id].abstain]
    with_support = [r for r in answerable if r.support]
    rec_e2e = [len(r.support_retrieved) / len(r.support) for r in with_support]
    stored_support = [r for r in with_support if r.support_stored]
    rec_stored = [len(set(r.support_retrieved) & set(r.support_stored)) / len(r.support_stored) for r in stored_support]
    any_hit = sum(bool(r.support_retrieved) for r in with_support)
    lat = [r.total_ms for r in results]
    first = [r.first_audio_ms for r in results if r.first_audio_ms is not None]
    paths = {"shortcut": sum("shortcut" in r.path for r in results),
             "transcript_fallback": sum("fallback: transcript" in r.path for r in results),
             "transcript_fallback_correct": sum("fallback: transcript" in r.path and r.check["correct"] for r in results),
             "routed_to_lookup": sum("lookup" in r.path for r in results)}
    short = [r.total_ms for r in results if "shortcut" in r.path]
    tool = [r.total_ms for r in results if "tool: recall" in r.path and "shortcut" not in r.path]
    by_cat: dict[str, dict] = {}
    for r in results:
        b = by_cat.setdefault(r.cat, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r.check["correct"])
    for b in by_cat.values():
        b["accuracy"] = pct(b["correct"], b["n"])
    reasons: dict[str, int] = {}
    for r in results:
        if r.failure:
            reasons[r.failure] = reasons.get(r.failure, 0) + 1
    judged = [r for r in answerable if r.judge is not None]
    return {
        "write": {
            "lines": len(write),
            "keep_drop_accuracy": pct(kept_ok + drop_ok, len(write)),
            "keep_recall": pct(kept_ok, len(keep)),  # keep-lines that were stored
            "drop_precision": pct(drop_ok, len(drop)),  # chatter that was not stored
            "stored": sum(w["decision"] == "stored" for w in write),
            # v3: lines the router was unsure of ("pending"); v4: every non-filler line waits for extraction ("queued")
            "went_to_llm": sum(w["first_decision"] in ("pending", "queued") for w in write),
            "dropped_before_llm": sum(w["first_decision"] in ("dropped", "stored") and w["source"] == "rule"
                                      or w["first_decision"] == "dropped" for w in write),
            "median_ingest_ms": percentile([w["ingest_ms"] for w in write], 50),
        },
        "retrieval": {
            "recall_at_5": round(statistics.mean(rec_e2e), 3) if rec_e2e else None,
            "recall_at_5_of_stored": round(statistics.mean(rec_stored), 3) if rec_stored else None,
            "any_support_in_top5": pct(any_hit, len(with_support)),
        },
        "answers": {
            "questions": len(results),
            "accuracy": pct(sum(r.check["correct"] for r in results), len(results)),
            "answerable_accuracy": pct(sum(r.check["correct"] for r in answerable), len(answerable)),
            "correct_abstention_rate": pct(sum(r.check["abstained"] for r in unanswerable), len(unanswerable)),
            "false_answers": sum(r.check["false_answer"] for r in results),
            "false_answers_on_unanswerable": sum(r.check["false_answer"] for r in unanswerable),
            "false_answers_on_answerable": sum(r.check["false_answer"] for r in answerable),
            "wrong_abstentions": sum(r.check["wrong_abstention"] for r in answerable),
            "raw_marker_spoken": sum(r.check["marker_leak"] for r in results),
            "judge_accuracy": pct(sum(r.judge == "yes" for r in judged), len(judged)) if judged else None,
            "by_category": by_cat,
            "failure_reasons": reasons,
        },
        "latency_ms": {"median": percentile(lat, 50), "p95": percentile(lat, 95), "max": max(lat) if lat else None,
                       "first_audio_median": percentile(first, 50),
                       "shortcut_median": percentile(short, 50), "tool_recall_median": percentile(tool, 50)},
        "paths": paths,
    }


def as_dict(r: QuestionResult) -> dict:
    return asdict(r)
