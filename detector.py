from __future__ import annotations
from typing import List, Dict, Any
import re


# -------------------------------------------------------------------
# Bias patterns (same label space as your CSV)
# -------------------------------------------------------------------

BIAS_PATTERNS: Dict[str, List[str]] = {
    "explicit_gender": [
        r"\bhe\b",
        r"\bshe\b",
        r"\bhis\b",
        r"\bhers\b",
        r"\bman\b",
        r"\bmen\b",
        r"\bwoman\b",
        r"\bwomen\b",
        r"\byoung man\b",
        r"\byoung woman\b",
        r"\bfemale candidate\b",
        r"\bmale candidate\b",
        r"\bprefer (male|female)\b",
        r"\b(prefer|seeking) (a )?woman\b",
        r"\b(prefer|seeking) (a )?man\b",
    ],
    "stereotype": [
        r"\bnurturing female\b",
        r"\bnurturing woman\b",
        r"\bdominant leader\b",
        r"\bnot emotional\b",
        r"\bemotionally stable\b",
        r"\bstrong male presence\b",
        r"\bsoft-spoken woman\b",
        r"\baggressive salesman\b",
        r"\bmotherly\b",
    ],
    "requirements_bias": [
        r"\balways available\b",
        r"\bavailable at all times\b",
        r"\bwork long hours\b",
        r"\bwork late nights\b",
        r"\bno family obligations\b",
        r"\bno family duties\b",
        r"\bno caregiving responsibilities\b",
        r"\b24\/7\b",
        r"\bweekends required\b",
        r"\bmust be on call\b",
    ],
    "age_bias": [
        r"\byoung energetic\b",
        r"\byoung man\b",
        r"\byoung woman\b",
        r"\bfresh graduate only\b",
        r"\brecent graduate\b",
        r"\bdigital native\b",
        r"\bunder 30\b",
        r"\byouthful\b",
    ],
}


def _find_bias_spans(text: str) -> List[Dict[str, Any]]:
    """
    Find biased spans in text using regex patterns.

    Returns:
        [
          {"span": "young man", "type": "explicit_gender"},
          {"span": "work long hours", "type": "requirements_bias"},
          ...
        ]
    """
    text_lower = text.lower()
    results: List[Dict[str, Any]] = []

    for bias_type, patterns in BIAS_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower, flags=re.IGNORECASE):
                span = text[match.start():match.end()]
                results.append(
                    {
                        "span": span,
                        "type": bias_type,
                    }
                )

    # deduplicate (span, type) pairs
    seen = set()
    unique_results = []
    for h in results:
        key = (h["span"].lower(), h["type"])
        if key not in seen:
            seen.add(key)
            unique_results.append(h)

    return unique_results


def _score_to_level(score: float) -> str:
    """Map numeric bias_score to bias_level."""
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "medium"
    else:
        return "high"


def analyze_jd(jd_text: str) -> Dict[str, Any]:
    """
    Analyze a job description string and return:
      - bias_score (0–1)
      - bias_level ("low" | "medium" | "high")
      - categories (list of bias types)
      - highlights (list of {span, type})

    Example output:
    {
      "bias_score": 0.72,
      "bias_level": "high",
      "categories": ["explicit_gender", "requirements_bias"],
      "highlights": [
        {"span": "young man", "type": "explicit_gender"},
        {"span": "work long hours and be always available", "type": "requirements_bias"}
      ]
    }
    """
    highlights = _find_bias_spans(jd_text)

    # categories = unique bias types
    categories = sorted({h["type"] for h in highlights})

    # crude scoring: more spans -> higher score, capped at 1.0
    num_spans = len(highlights)
    # tune weight if you want it more/less sensitive
    bias_score = min(1.0, num_spans * 0.25)  # 0, 0.25, 0.5, 0.75, 1.0
    bias_level = _score_to_level(bias_score)

    return {
        "bias_score": round(bias_score, 2),
        "bias_level": bias_level,
        "categories": categories,
        "highlights": highlights,
    }


if __name__ == "__main__":
    sample = (
        "We are looking for a young man to lead the team and must be willing "
        "to work long hours and be always available."
    )
    out = analyze_jd(sample)
    from pprint import pprint
    pprint(out)
