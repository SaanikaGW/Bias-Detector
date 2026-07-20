"""
Layer 1 — rule-based detection engine.

Reads detection/patterns.json (the editable source of truth) and scans text.
Design notes:
- Matching runs on the ORIGINAL text (case-insensitive), never a normalized
  copy, so every hit carries exact character offsets for inline highlighting.
- Each pattern may carry `veto` regexes: if one matches the containing
  sentence, the hit is suppressed outright (clear benign use, e.g.
  "competitive salary"). This is a fast, deterministic guard.
- Each pattern may be `ambiguous`: the hit is NOT suppressed but is marked
  for adjudication by the Layer 2 contextual classifier, which decides from
  sentence context (e.g. "aggressive personality" vs "aggressive strategy").
- Layer 1 never calls a model or the network. It is pure, fast, and testable.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

_PATTERNS_PATH = os.path.join(os.path.dirname(__file__), "patterns.json")

SEVERITY_LABELS = {1: "low", 2: "medium", 3: "high"}


@lru_cache(maxsize=1)
def load_rules() -> dict:
    """Load and pre-compile patterns.json once per process."""
    with open(_PATTERNS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for p in data["patterns"]:
        p["_rx"] = re.compile(p["pattern"], re.IGNORECASE)
        p["_veto_rx"] = [re.compile(v, re.IGNORECASE) for v in p.get("veto", [])]
    for s in data["inclusive_signals"]:
        s["_rx"] = re.compile(s["pattern"], re.IGNORECASE)
    return data


def split_sentences(text: str) -> list[dict]:
    """Split into sentences, keeping character offsets for highlight mapping."""
    sentences = []
    start = 0
    for m in re.finditer(r"[.!?\n]+", text):
        seg = text[start:m.end()]
        if seg.strip():
            sentences.append({"text": seg, "start": start, "end": m.end()})
        start = m.end()
    if text[start:].strip():
        sentences.append({"text": text[start:], "start": start, "end": len(text)})
    return sentences


def _containing_sentence(sentences: list[dict], pos: int) -> dict | None:
    for s in sentences:
        if s["start"] <= pos < s["end"]:
            return s
    return sentences[-1] if sentences else None


def detect(text: str) -> dict:
    """
    Run Layer 1 over `text`.

    Returns {"hits": [...], "inclusive_signals": [...]}.
    Each hit: span, start, end, sentence, category, severity (label),
    severity_num, confidence, pattern_id, ambiguous, why, alternatives, gain.
    Hits marked ambiguous=True must be adjudicated by Layer 2 before being
    reported as issues.
    """
    rules = load_rules()
    sentences = split_sentences(text)
    hits: list[dict] = []
    occupied: list[tuple[int, int]] = []  # spans already claimed (dedupe overlaps)

    # Higher-severity, higher-confidence patterns claim spans first.
    ordered = sorted(rules["patterns"],
                     key=lambda p: (p["severity"], p["confidence"]), reverse=True)

    for p in ordered:
        for m in p["_rx"].finditer(text):
            if any(s < m.end() and m.start() < e for s, e in occupied):
                continue  # span already claimed by a stronger pattern
            sent = _containing_sentence(sentences, m.start())
            sent_text = sent["text"] if sent else text
            if any(v.search(sent_text) for v in p["_veto_rx"]):
                continue  # benign use, suppressed by veto guard
            occupied.append((m.start(), m.end()))
            hits.append({
                "span": text[m.start():m.end()],
                "start": m.start(),
                "end": m.end(),
                "sentence": sent_text.strip(),
                "category": p["category"],
                "severity": SEVERITY_LABELS[p["severity"]],
                "severity_num": p["severity"],
                "confidence": p["confidence"],
                "pattern_id": p["id"],
                "ambiguous": p.get("ambiguous", False),
                "why": p.get("why", ""),
                "alternatives": p.get("alternatives", []),
                "gain": p.get("gain", ""),
                "source": "rules",
            })

    signals = []
    for s in rules["inclusive_signals"]:
        m = s["_rx"].search(text)
        if m:
            signals.append({
                "id": s["id"], "label": s["label"], "points": s["points"],
                "span": text[m.start():m.end()], "start": m.start(), "end": m.end(),
            })

    hits.sort(key=lambda h: h["start"])
    return {"hits": hits, "inclusive_signals": signals}


def categories() -> dict:
    """Category metadata (label, weight, research blurb) from patterns.json."""
    return load_rules()["categories"]
