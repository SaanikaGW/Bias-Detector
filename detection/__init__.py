"""BIOS Career Check v2.0 — hybrid gender-bias detection package.

Layer 1: rule engine over detection/patterns.json (157 curated entries).
Layer 2: trained contextual classifier (TF-IDF n-grams + logistic
         regression, numpy-only runtime) — adjudicates ambiguity, the
         decision-maker for context-dependent terms.
Layer 3: LLM explanations & rewrites, strictly downstream of decisions.

Public entry point: detection.pipeline.analyze(text)
"""
from .pipeline import analyze  # noqa: F401
