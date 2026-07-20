"""
Hybrid detection pipeline — the single entry point for web app (and later
the Chrome extension, via the same API).

Flow (the LLM is never the decision-maker):
  Layer 1 (rules)      -> candidate hits (+ veto guards, + ambiguity marks)
  Layer 2 (classifier) -> adjudicates ambiguous hits; scans rule-free
                          sentences for subtle bias at a high threshold
  Layer 3 (LLM)        -> explanations, research rationale, rewrites ONLY

SCORING FORMULA (kept transparent so the UI can show the derivation):
  Each confirmed issue contributes:
      penalty = sev_points[severity] * confidence * category_weight
      sev_points: low=4, medium=8, high=14
      category_weight: from patterns.json (0.5–1.0)
  gender_bias_score = min(100, round(sum of penalties))    # higher = worse
  inclusive_language_score = clamp(0..100,
      60 + inclusive signal points (capped at +40)
         - 0.8 * bias penalty)                             # higher = better
  A neutral, signal-free JD therefore scores 0 / 60; adding inclusive
  signals (EEO statement, pay transparency, flexible work…) raises the
  inclusive score toward 100.
"""
from __future__ import annotations

from . import layer1, layer2, layer3

SEV_POINTS = {"low": 4, "medium": 8, "high": 14}
BIAS_LEVELS = [(15, "low"), (40, "medium"), (10**9, "high")]

_MIN_SENTENCE_LEN = 15  # don't classifier-scan trivial fragments


def _bias_level(score: float) -> str:
    for cutoff, label in BIAS_LEVELS:
        if score < cutoff:
            return label
    return "high"


def analyze(text: str, use_llm: bool = True) -> dict:
    cats = layer1.categories()

    # ── Layer 1: rules ──
    l1 = layer1.detect(text)
    clf = layer2.Layer2Classifier.load()

    confirmed: list[dict] = []
    for hit in l1["hits"]:
        if hit["ambiguous"]:
            decided = layer2.adjudicate(hit, clf)
            if decided is not None:
                confirmed.append(decided)
        else:
            confirmed.append(hit)

    # ── Layer 2: scan sentences no rule touched ──
    flagged_ranges = [(h["start"], h["end"]) for h in confirmed]
    for sent in layer1.split_sentences(text):
        if len(sent["text"].strip()) < _MIN_SENTENCE_LEN:
            continue
        if any(s < sent["end"] and sent["start"] < e for s, e in flagged_ranges):
            continue  # sentence already contains a confirmed issue
        issue = layer2.scan_sentence(sent["text"], clf)
        if issue:
            issue["start"], issue["end"] = sent["start"], sent["end"]
            issue["span"] = sent["text"].strip()
            confirmed.append(issue)

    confirmed.sort(key=lambda h: h.get("start", 0))

    # ── Layer 3: enrich (never changes which issues exist) ──
    issues, rewritten_jd, enrichment_source = layer3.enrich(
        text, confirmed) if use_llm else layer3.enrich(text, confirmed)

    # ── Transparent scoring ──
    derivation = []
    penalty_total = 0.0
    for iss in issues:
        weight = cats.get(iss["category"], {}).get("weight", 0.7)
        penalty = SEV_POINTS[iss["severity"]] * iss["confidence"] * weight
        penalty_total += penalty
        derivation.append({
            "phrase": iss["span"],
            "severity": iss["severity"],
            "confidence": iss["confidence"],
            "category_weight": weight,
            "penalty": round(penalty, 1),
        })

    signal_points = min(40, sum(s["points"] for s in l1["inclusive_signals"]))
    gender_bias_score = min(100, round(penalty_total))
    inclusive_score = max(0, min(100, round(60 + signal_points - 0.8 * penalty_total)))

    for iss in issues:
        iss["category_label"] = cats.get(iss["category"], {}).get(
            "label", iss["category"])

    return {
        # ── v2 contract ──
        "issues": issues,
        "scores": {
            "gender_bias_score": gender_bias_score,
            "inclusive_language_score": inclusive_score,
            "bias_level": _bias_level(gender_bias_score),
            "derivation": {
                "issue_penalties": derivation,
                "penalty_total": round(penalty_total, 1),
                "inclusive_signal_points": signal_points,
                "formula": ("bias = min(100, Σ sev_points×confidence×cat_weight); "
                            "inclusive = clamp(60 + signals(≤40) − 0.8×bias_penalty)"),
            },
        },
        "inclusive_signals": l1["inclusive_signals"],
        "rewritten_jd": rewritten_jd,
        "enrichment_source": enrichment_source,
        # ── legacy fields (v1 UI / API consumers) ──
        "bias_score": round(gender_bias_score / 100, 3),
        "bias_level": _bias_level(gender_bias_score),
        "categories": sorted({i["category"] for i in issues}),
        "highlights": [{"span": i["span"], "type": i["category"],
                        "source": i["source"]} for i in issues],
        "suggestions": [f"{i['span']} → {i['rewrite']}" for i in issues
                        if i.get("rewrite")],
    }
