"""
detector.py — DEPRECATED shim, kept for backward compatibility.

The v1 detector (fine-tuned GPT-2 primary + regex fallback) is replaced by
the v2.0 hybrid pipeline in the `detection/` package. In production the v1
GPT-2 and sentence-transformer layers never actually loaded (torch was not
in requirements.txt), so v1 detection was effectively a keyword list — see
README "What changed in v2.0".

Import `detection.pipeline.analyze` in new code.
"""
from detection.pipeline import analyze


def analyze_jd(text: str) -> dict:
    """v1-compatible signature; returns the v2 result, which is a superset
    of the v1 fields (bias_score, bias_level, categories, highlights)."""
    return analyze(text)
