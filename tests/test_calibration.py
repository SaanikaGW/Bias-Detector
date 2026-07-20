"""
Calibration acceptance tests (spec Phase 1) + regression checks.

Run:  python -m pytest tests/ -q     (or)  python tests/test_calibration.py

The calibration set is the acceptance bar for the hybrid pipeline. These
exact sentences are deliberately EXCLUDED from Layer 2 training data
(see CALIBRATION_HOLDOUT in ml/train_layer2.py) so this is a real test,
not a memorization check. Layer 3 (LLM) is disabled here — flag/no-flag
decisions must come from Layers 1+2 alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force template fallback: decisions must not depend on the LLM.
os.environ.pop("OPENAI_API_KEY", None)

from detection.pipeline import analyze          # noqa: E402
from detection.layer2 import Layer2Classifier   # noqa: E402

CALIBRATION = [
    ("Aggressive sales strategy", False),
    ("Aggressive personality required", True),
    ("Strong communication skills", False),
    ("Rockstar engineer", True),
    ("Native English speaker", True),
    ("Professional English proficiency", False),
    ("Chairman", True),
    ("Chairperson", False),
    ("Work hard, play hard", True),
    ("Collaborative team environment", False),
]


def test_calibration_pipeline():
    """Full pipeline (L1+L2) must get all 10 calibration cases right."""
    failures = []
    for text, expect_flag in CALIBRATION:
        flagged = len(analyze(text, use_llm=False)["issues"]) > 0
        if flagged != expect_flag:
            failures.append(f"{text!r}: flagged={flagged}, expected={expect_flag}")
    assert not failures, "\n".join(failures)


def test_calibration_layer2_alone():
    """Layer 2 must decide these from context — not inherit them from rules."""
    clf = Layer2Classifier.load()
    failures = []
    for text, expect_flag in CALIBRATION:
        got = clf.p_biased(text) >= 0.5
        if got != expect_flag:
            failures.append(f"{text!r}: p_biased={clf.p_biased(text):.2f}, "
                            f"expected={'flag' if expect_flag else 'clean'}")
    assert not failures, "\n".join(failures)


def test_issue_contract():
    """Every issue carries the full per-issue output contract."""
    required = {"span", "category", "category_label", "severity", "confidence",
                "explanation", "impact", "research_rationale", "rewrite",
                "expected_improvement"}
    result = analyze("We need a rockstar salesman, a digital native with no "
                     "family commitments.", use_llm=False)
    assert result["issues"], "expected issues on a clearly biased JD"
    for issue in result["issues"]:
        missing = required - set(issue)
        assert not missing, f"issue missing fields: {missing}"


def test_scores_derivable():
    """Scores must be reconstructable from the published derivation."""
    result = analyze("Seeking an aggressive personality; native English "
                     "speakers only. We are an equal opportunity employer.",
                     use_llm=False)
    s = result["scores"]
    total = sum(p["penalty"] for p in s["derivation"]["issue_penalties"])
    assert abs(total - s["derivation"]["penalty_total"]) < 0.5
    # per-issue penalties are rounded for display; allow ±1 rounding drift
    assert abs(s["gender_bias_score"] - min(100, round(total))) <= 1
    assert 0 <= s["inclusive_language_score"] <= 100


def test_legacy_fields_present():
    """v1 API consumers must keep working (no regression)."""
    result = analyze("The salesman must be available 24/7.", use_llm=False)
    for key in ("bias_score", "bias_level", "categories", "highlights",
                "suggestions", "rewritten_jd"):
        assert key in result, f"legacy field {key} missing"
    assert 0.0 <= result["bias_score"] <= 1.0
    assert all({"span", "type", "source"} <= set(h)
               for h in result["highlights"])


def test_clean_jd_scores_zero():
    clean = ("We are hiring a software engineer. You will design and ship "
             "backend services in Python. Strong communication skills and "
             "experience with SQL required. We offer flexible working hours "
             "and are an equal opportunity employer.")
    result = analyze(clean, use_llm=False)
    assert result["scores"]["gender_bias_score"] <= 5, result["issues"]
    assert result["scores"]["inclusive_language_score"] >= 70


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} test groups passed.")
