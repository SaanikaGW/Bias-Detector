"""
Faithful reproduction of the v1 detector AS IT ACTUALLY RAN IN PRODUCTION.

The v1 code (old detector.py, preserved in git history) declared a
fine-tuned GPT-2 LoRA as primary and sentence-transformers as a semantic
layer — but torch/transformers/peft/sentence-transformers were never in
requirements.txt, so on Railway both import blocks failed and detection
fell through to exactly two mechanisms, reproduced verbatim here:

1. CSV span matching — substring lookup of every 'Biased_span' from
   bios_check_dataset_full.csv in the normalized text.
2. BIAS_PATTERNS — the v1 regex keyword list, matched on normalized text.

Scoring is the v1 formula: category-count based, not issue based.

This module exists ONLY for the before/after evaluation in run_eval.py.
Do not use it in the product.
"""
from __future__ import annotations

import os
import re
import string

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_PATH = os.path.join(ROOT, "bios_check_dataset_full.csv")

CATEGORY_WEIGHTS = {
    "explicit_gender": 1.0,
    "stereotype": 0.8,
    "masculine_coded_language": 0.8,
    "motherhood_penalty": 0.9,
    "appearance_bias": 0.7,
    "age_bias": 0.7,
    "requirements_bias": 0.6,
}

# v1 regex list, copied verbatim from the old detector.py
BIAS_PATTERNS: dict[str, list[str]] = {
    "explicit_gender": [
        r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b", r"\bhis\b",
        r"\bmale\b", r"\bfemale\b", r"\bgentlemen\b",
        r"female\s+candidate\s+preferred", r"male\s+candidate\s+preferred",
        r"prefer\s+female", r"prefer\s+male",
        r"young\s+man", r"young\s+woman",
        r"businessman", r"businesswoman", r"salesman", r"saleswoman",
        r"manpower", r"mankind", r"\bgirl\b", r"\bboy\b",
        r"chairman", r"spokeswoman", r"spokesman",
    ],
    "stereotype": [
        r"nurturing", r"aggressive\s+(leader|self.starter)", r"dominant\s+leader",
        r"assertive\s+man", r"bossy", r"emotional",
        r"manly", r"womanly", r"motherly", r"fatherly",
    ],
    "masculine_coded_language": [
        r"masculine", r"feminine", r"soft\s+skills",
        r"rock\s+star", r"ninja", r"dominant", r"aggressive",
        r"competitive\s+spirit", r"alpha",
    ],
    "motherhood_penalty": [
        r"no\s+family\s+commitments", r"no\s+caregiving\s+responsibilities",
        r"must\s+be\s+fully\s+dedicated", r"lifestyle\s+fit",
    ],
    "appearance_bias": [
        r"well[\s-]groomed", r"presentable", r"attractive",
        r"good[\s-]looking", r"physically\s+fit", r"clean[\s-]cut",
    ],
    "requirements_bias": [
        r"available\s+24[\/\s]7", r"no\s+exceptions",
        r"must\s+be\s+available\s+at\s+all\s+times",
        r"willing\s+to\s+relocate\s+immediately",
        r"must\s+travel\s+frequently",
        r"no\s+gaps\s+in\s+(employment|work)\s+history",
    ],
    "age_bias": [
        r"fresh\s+graduate\s+only", r"recent\s+graduate\s+preferred",
        r"young\s+(and\s+)?energetic", r"under\s+\d{2}(\s+preferred)?",
        r"\d{2}\s+years\s+or\s+younger", r"digital\s+native",
        r"new\s+grad", r"entry[\-\s]level.*no\s+experience",
    ],
}


def _load_spans() -> dict[str, list[str]]:
    spans: dict[str, list[str]] = {}
    try:
        df = pd.read_csv(_CSV_PATH)
        biased = df[df["Gender_bias_sentiment"] == "biased"]
        for _, row in biased.iterrows():
            cat = str(row.get("Bias_type", "")).strip().lower()
            span = str(row.get("Biased_span", "")).strip()
            if cat and span and span.lower() not in ("nan", ""):
                spans.setdefault(cat, [])
                if span not in spans[cat]:
                    spans[cat].append(span)
    except Exception:
        pass
    return spans


CSV_SPANS = _load_spans()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_jd_legacy(text: str) -> dict:
    """v1 production behavior: CSV spans + regex keywords, category scoring."""
    norm = normalize_text(text)
    seen: set[str] = set()
    highlights: list[dict] = []

    for category, spans in CSV_SPANS.items():
        for span in spans:
            if span.lower() in norm and span not in seen:
                highlights.append({"span": span, "type": category, "source": "csv"})
                seen.add(span)

    for category, patterns in BIAS_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, norm):
                span = m.group(0)
                if span and span not in seen:
                    highlights.append({"span": span, "type": category,
                                       "source": "pattern"})
                    seen.add(span)

    categories_found = list({h["type"] for h in highlights})
    raw_score = min(
        1.0,
        sum(CATEGORY_WEIGHTS.get(c, 0.5) for c in categories_found)
        / len(CATEGORY_WEIGHTS)
    ) if categories_found else 0.0

    bias_level = ("low" if raw_score < 0.25
                  else "medium" if raw_score < 0.6 else "high")
    return {
        "bias_score": round(raw_score, 3),
        "bias_level": bias_level,
        "categories": categories_found,
        "highlights": highlights,
    }
