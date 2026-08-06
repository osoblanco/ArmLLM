"""Verifiers — the referee.

Every task in tasks.jsonl carries a `verify` block, and this file turns that
block into a verdict. It is deliberately short and deliberately readable,
because in agent work **this file is worth more than the agent**: the agent is
a checkpoint that expires, the referee is what lets you measure the next one.

Five verifier types:

  numeric  the answer must contain the target number
  exact    the answer must contain one of the accepted strings
  set      the answer must contain every required item
  abstain  the answer must decline, and must not invent a number

`stale_value` on a verifier is a diagnostic, not a rule: when an answer is
wrong *and* contains the superseded value, we record `stale` rather than plain
`wrong`. Knowing which kind of wrong you are is most of debugging.
"""

from __future__ import annotations

import re
import unicodedata

ABSTAIN_MARKERS = [
    "not published", "not specified", "not stated", "not listed", "not available",
    "cannot be determined", "can't be determined", "could not be determined",
    "unable to determine", "no information", "not found", "does not appear",
    "not provided", "i don't know", "i do not know", "insufficient information",
    "not documented", "no such", "does not exist", "set annually", "by council decision",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[^\w\s:./-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str) -> list[float]:
    """Numbers, tolerating thousands separators. '26,000 AMD' -> 26000.0"""
    out = []
    for raw in re.findall(r"-?\d[\d,\s]*(?:\.\d+)?", str(text or "")):
        cleaned = raw.replace(",", "").replace(" ", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def verify(answer: str, spec: dict) -> dict:
    """-> {'correct': bool, 'reason': str}"""
    kind = spec.get("type")
    answer = str(answer or "")

    if kind == "numeric":
        return _numeric(answer, spec)
    if kind == "exact":
        return _exact(answer, spec)
    if kind == "set":
        return _set(answer, spec)
    if kind == "abstain":
        return _abstain(answer, spec)
    raise ValueError(f"unknown verifier type: {kind!r}")


def _numeric(answer: str, spec: dict) -> dict:
    target = float(spec["value"])
    tol = float(spec.get("tol", 0.5))
    found = extract_numbers(answer)
    if any(abs(v - target) <= tol for v in found):
        return {"correct": True, "reason": "ok"}
    stale = spec.get("stale_value")
    if stale is not None and any(abs(v - float(stale)) <= tol for v in found):
        return {"correct": False, "reason": "stale"}
    if _looks_like_abstention(answer):
        return {"correct": False, "reason": "abstained_on_answerable"}
    if not found:
        return {"correct": False, "reason": "no_number"}
    return {"correct": False, "reason": "wrong_number"}


def _exact(answer: str, spec: dict) -> dict:
    hay = normalize(answer)
    for candidate in spec["accept"]:
        if normalize(candidate) in hay:
            return {"correct": True, "reason": "ok"}
    stale = spec.get("stale_value")
    if stale and normalize(stale) in hay:
        return {"correct": False, "reason": "stale"}
    if _looks_like_abstention(answer):
        return {"correct": False, "reason": "abstained_on_answerable"}
    return {"correct": False, "reason": "wrong_string"}


def _set(answer: str, spec: dict) -> dict:
    hay = normalize(answer)
    missing = [item for item in spec["items"] if normalize(item) not in hay]
    if not missing:
        return {"correct": True, "reason": "ok"}
    return {"correct": False, "reason": f"missing:{len(missing)}/{len(spec['items'])}"}


def _abstain(answer: str, spec: dict) -> dict:
    """Correct = says it cannot be determined, and does not invent a figure.

    Abstention is scored because an agent that never says 'I don't know' has no
    way of being honestly wrong -- it can only be confidently wrong.
    """
    if not _looks_like_abstention(answer):
        return {"correct": False, "reason": "hallucinated"}
    if spec.get("forbid_numbers") and extract_numbers(answer):
        return {"correct": False, "reason": "abstained_but_invented_number"}
    return {"correct": True, "reason": "ok"}


def _looks_like_abstention(answer: str) -> bool:
    hay = normalize(answer)
    return any(normalize(marker) in hay for marker in ABSTAIN_MARKERS)
