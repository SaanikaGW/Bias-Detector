#!/usr/bin/env python3
"""
Evaluation harness (Phase 2).

Runs BOTH detection systems over eval/data/eval_dataset.csv:
- legacy: the v1 production behavior (eval/legacy_detector.py)
- v2:     the hybrid pipeline, Layers 1+2 only (LLM disabled — flag/no-flag
          decisions must never depend on the LLM)

Outputs (checked-in artifacts):
- eval/metrics.json  — machine-readable metrics for both systems + delta
- eval/report.md     — human-readable before/after report with confusion
                       matrices and a per-example-type breakdown

Metrics per system: accuracy, precision, recall, F1, confusion matrix,
false-positive rate, false-negative rate, average detection confidence
(v2 only — legacy had no confidence concept, reported as null).

Run:  python eval/run_eval.py     (or: make eval)
"""
import json
import os
import sys
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

os.environ.pop("OPENAI_API_KEY", None)  # decisions must not touch the LLM

from detection.pipeline import analyze as analyze_v2   # noqa: E402
from legacy_detector import analyze_jd_legacy          # noqa: E402


def predict_legacy(text: str) -> tuple[bool, float | None]:
    r = analyze_jd_legacy(text)
    return bool(r["highlights"]), None


def predict_v2(text: str) -> tuple[bool, float | None]:
    r = analyze_v2(text, use_llm=False)
    issues = r["issues"]
    if not issues:
        return False, None
    return True, sum(i["confidence"] for i in issues) / len(issues)


def score(df: pd.DataFrame, pred_col: str, conf_col: str) -> dict:
    y_true = (df["expected"] == "biased").to_numpy()
    y_pred = df[pred_col].to_numpy()
    tp = int(((y_true) & (y_pred)).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    confs = df.loc[df[pred_col], conf_col].dropna()
    return {
        "n": len(df),
        "accuracy": round((tp + tn) / len(df), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 3) if fn + tp else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "avg_confidence_when_flagging":
            round(float(confs.mean()), 3) if len(confs) else None,
    }


def confusion_table(m: dict) -> str:
    cm = m["confusion_matrix"]
    return (
        "|                | pred: biased | pred: clean |\n"
        "|----------------|-------------:|------------:|\n"
        f"| **true: biased** | {cm['tp']} | {cm['fn']} |\n"
        f"| **true: clean**  | {cm['fp']} | {cm['tn']} |\n"
    )


def main() -> None:
    df = pd.read_csv(os.path.join(HERE, "data", "eval_dataset.csv"))

    for name, fn in [("legacy", predict_legacy), ("v2", predict_v2)]:
        preds, confs = [], []
        for t in df["Text"]:
            p, c = fn(t)
            preds.append(p)
            confs.append(c)
        df[f"pred_{name}"] = preds
        df[f"conf_{name}"] = confs

    legacy = score(df, "pred_legacy", "conf_legacy")
    v2 = score(df, "pred_v2", "conf_v2")

    # Per-example-type accuracy for both systems
    by_type = {}
    for etype, part in df.groupby("example_type"):
        y = (part["expected"] == "biased").to_numpy()
        by_type[etype] = {
            "n": len(part),
            "legacy_accuracy": round(float((part["pred_legacy"] == y).mean()), 3),
            "v2_accuracy": round(float((part["pred_v2"] == y).mean()), 3),
        }

    delta = {k: round(v2[k] - legacy[k], 3)
             for k in ("accuracy", "precision", "recall", "f1")}
    delta["false_positive_rate"] = round(
        v2["false_positive_rate"] - legacy["false_positive_rate"], 3)
    delta["false_negative_rate"] = round(
        v2["false_negative_rate"] - legacy["false_negative_rate"], 3)

    metrics = {"date": str(date.today()),
               "dataset": "eval/data/eval_dataset.csv",
               "n_examples": len(df),
               "legacy": legacy, "v2": v2, "delta": delta,
               "by_example_type": by_type}
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Misses worth reading (kept honest in the report)
    v2_fns = df[(df["expected"] == "biased") & (~df["pred_v2"])]
    v2_fps = df[(df["expected"] == "none") & (df["pred_v2"])]

    lines = [
        "# Detection Evaluation — v1 (legacy) vs v2 (hybrid pipeline)",
        "",
        f"*Generated {date.today()} by `python eval/run_eval.py` over "
        f"`eval/data/eval_dataset.csv` ({len(df)} labeled examples). "
        "LLM disabled — decisions come from Layers 1+2 only.*",
        "",
        "**Legacy** = what actually ran in v1 production: dataset span matching "
        "+ a regex keyword list (the declared GPT-2/semantic layers never loaded "
        "because torch was not deployed).",
        "",
        "**Honesty caveat:** v2's Layer 1 rules were iterated against earlier "
        "runs of this same dataset (that is what an eval harness is for), so "
        "these are *development* metrics, not a blind test. The Layer 2 "
        "classifier was never trained on any eval row. Legacy was not tuned "
        "at all, but its span-matching memorized parts of the `holdout_test` "
        "slice by construction. Treat the trap/subtle slices as the fairest "
        "apples-to-apples comparison.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Legacy (v1) | Hybrid (v2) | Δ |",
        "|---|---:|---:|---:|",
    ]
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                     ("recall", "Recall"), ("f1", "F1"),
                     ("false_positive_rate", "False-positive rate"),
                     ("false_negative_rate", "False-negative rate")]:
        lines.append(f"| {label} | {legacy[k]:.3f} | {v2[k]:.3f} | "
                     f"{'+' if delta[k] >= 0 else ''}{delta[k]:.3f} |")
    lines += [
        f"| Avg confidence when flagging | — | "
        f"{v2['avg_confidence_when_flagging']} | |",
        "",
        "## Confusion matrices",
        "", "### Legacy (v1)", "", confusion_table(legacy),
        "", "### Hybrid (v2)", "", confusion_table(v2),
        "",
        "## Accuracy by example type",
        "",
        "| Example type | n | Legacy | v2 |",
        "|---|---:|---:|---:|",
    ]
    for etype in sorted(by_type):
        t = by_type[etype]
        lines.append(f"| {etype} | {t['n']} | {t['legacy_accuracy']:.3f} | "
                     f"{t['v2_accuracy']:.3f} |")
    lines += [
        "",
        "*Note: `holdout_test` rows are in-distribution (drawn from the same "
        "datasets the systems were built around; legacy's CSV span matching "
        "partially memorized them in v1, flattering its score on that slice). "
        "The trap/subtle/neutral/inclusive rows are new, out-of-distribution "
        "phrasings — the honest stress test.*",
        "",
        f"## v2 remaining misses ({len(v2_fns)} false negatives, "
        f"{len(v2_fps)} false positives)",
        "",
        "False negatives (biased, not caught):",
        "",
    ]
    for _, r in v2_fns.iterrows():
        lines.append(f"- ({r['example_type']}/{r['category']}) “{r['Text']}”")
    lines += ["", "False positives (clean, wrongly flagged):", ""]
    for _, r in v2_fps.iterrows():
        lines.append(f"- ({r['example_type']}) “{r['Text']}”")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python eval/build_dataset.py   # rebuild dataset (idempotent)",
        "python eval/run_eval.py        # writes eval/metrics.json + eval/report.md",
        "```",
        "",
    ]
    with open(os.path.join(HERE, "report.md"), "w") as f:
        f.write("\n".join(lines))

    print(json.dumps({"legacy": legacy, "v2": v2, "delta": delta}, indent=2))
    print("\nwrote eval/metrics.json and eval/report.md")


if __name__ == "__main__":
    main()
