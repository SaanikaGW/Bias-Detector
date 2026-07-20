#!/usr/bin/env python3
"""
Train the Layer 2 contextual classifier.

WHY THIS DESIGN (documented per the v2.0 spec):
- Model: TF-IDF word n-grams (1-3) + multinomial logistic regression.
  N-grams carry local context, which is exactly what distinguishes
  "aggressive personality required" (flagged) from "aggressive sales
  strategy" (not flagged) — this is a trained contextual classifier over
  sentence context, not a keyword lookup.
- It is cheap, fully deterministic, dependency-light and testable — the
  first implementation option listed in the spec. OpenAI-embedding or
  sentence-transformer backends were considered; both would require either
  a network call per analysis or a ~500MB torch install on the server.
  The exported artifact is plain numpy arrays + a JSON vocabulary, so the
  RUNTIME needs numpy only (no scikit-learn, no pickle-version risk).
- Training data: bios_check_dataset_full.csv (329 real JD sentences) +
  ml/curated_labeled.csv (calibration paraphrases, false-positive traps,
  and the categories the old dataset lacked). The exact calibration
  sentences from the spec are HELD OUT of training on purpose — they are
  the acceptance test, not training data.
  gender_bias_dataset_final_fixed.csv is intentionally excluded: it is
  templated toy text ("girls code" / "boys forget homework"), and training
  on it teaches the model that any mention of women/boys is bias.

Usage:  python ml/train_layer2.py
Outputs:
  detection/layer2_model.npz    (idf, coef, intercept — portable arrays)
  detection/layer2_vocab.json   (vocabulary, classes, config)
  eval/data/{train,val,test}.csv  (stratified splits, seed of Phase 2 eval)
  ml/metrics_layer2.json        (held-out test metrics)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

# Old dataset categories -> v2.0 taxonomy
CATEGORY_MAP = {
    "explicit_gender": "gendered_language",
    "masculine_coded_language": "masculine_coded",
    "stereotype": "stereotype",
    "motherhood_penalty": "caregiver_bias",
    "requirements_bias": "caregiver_bias",
    "appearance_bias": "appearance_bias",
    "age_bias": "age_coded",
    "none": "none",
}

# Exact calibration sentences (spec acceptance set) — excluded from training.
CALIBRATION_HOLDOUT = {
    "aggressive sales strategy", "aggressive personality required",
    "strong communication skills", "rockstar engineer",
    "native english speaker", "professional english proficiency",
    "chairman", "chairperson", "work hard, play hard",
    "collaborative team environment",
}


def load_data() -> pd.DataFrame:
    frames = []
    bios = pd.read_csv(os.path.join(ROOT, "bios_check_dataset_full.csv"))
    bios["label"] = bios["Bias_type"].str.strip().str.lower().map(CATEGORY_MAP)
    frames.append(bios[["Text", "label"]].dropna())

    curated = pd.read_csv(os.path.join(ROOT, "ml", "curated_labeled.csv"))
    frames.append(curated[["Text", "label"]])

    df = pd.concat(frames, ignore_index=True)
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df[df["Text"].str.len() > 3].drop_duplicates(subset=["Text"])
    df = df[~df["Text"].str.lower().str.strip(" .!?").isin(CALIBRATION_HOLDOUT)]
    return df.reset_index(drop=True)


def export_portable(vec: TfidfVectorizer, clf_bin: LogisticRegression,
                    clf_cat: LogisticRegression) -> None:
    """Export as plain arrays + JSON so runtime inference needs numpy only.

    Two heads over the same TF-IDF features:
    - binary head (biased vs none): THE decision-maker. With only two
      classes it produces sharp, usable probabilities from small data.
    - category head: assigns which bias category, given that the binary
      head already decided the sentence is biased.
    """
    vocab = {t: int(i) for t, i in vec.vocabulary_.items()}
    np.savez_compressed(
        os.path.join(ROOT, "detection", "layer2_model.npz"),
        idf=vec.idf_.astype(np.float32),
        bin_coef=clf_bin.coef_.astype(np.float32),
        bin_intercept=clf_bin.intercept_.astype(np.float32),
        cat_coef=clf_cat.coef_.astype(np.float32),
        cat_intercept=clf_cat.intercept_.astype(np.float32),
    )
    with open(os.path.join(ROOT, "detection", "layer2_vocab.json"), "w") as f:
        json.dump({
            "vocabulary": vocab,
            "bin_classes": clf_bin.classes_.tolist(),
            "cat_classes": clf_cat.classes_.tolist(),
            "ngram_range": [1, 3],
            "sublinear_tf": True,
            "token_pattern": r"(?u)\b\w\w+\b",
        }, f)


def verify_parity(vec, clf_bin, texts) -> float:
    """Assert the numpy runtime (detection/layer2.py) reproduces sklearn."""
    sys.path.insert(0, ROOT)
    from detection.layer2 import Layer2Classifier
    Layer2Classifier.load.cache_clear()
    l2 = Layer2Classifier.load()
    biased_col = list(clf_bin.classes_).index("biased")
    ours = np.array([l2.p_biased(t) for t in texts])
    theirs = clf_bin.predict_proba(vec.transform(texts))[:, biased_col]
    return float(np.abs(ours - theirs).max())


def main() -> None:
    df = load_data()
    print(f"Dataset: {len(df)} rows\n{df['label'].value_counts().to_string()}")

    train_df, tmp = train_test_split(df, test_size=0.30, random_state=SEED,
                                     stratify=df["label"])
    val_df, test_df = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                       stratify=tmp["label"])
    os.makedirs(os.path.join(ROOT, "eval", "data"), exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(os.path.join(ROOT, "eval", "data", f"{name}.csv"), index=False)

    vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1,
                          lowercase=True)
    Xtr = vec.fit_transform(train_df["Text"])

    # Binary decision head
    ytr_bin = np.where(train_df["label"] == "none", "none", "biased")
    clf_bin = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced",
                                 random_state=SEED)
    clf_bin.fit(Xtr, ytr_bin)

    # Category head (trained on biased rows only)
    biased_mask = (train_df["label"] != "none").to_numpy()
    clf_cat = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced",
                                 random_state=SEED)
    clf_cat.fit(Xtr[biased_mask], train_df["label"][biased_mask])

    metrics = {}
    for split, part in [("val", val_df), ("test", test_df)]:
        X = vec.transform(part["Text"])
        y_bin = np.where(part["label"] == "none", "none", "biased")
        pred_bin = clf_bin.predict(X)
        bin_f1 = f1_score(y_bin, pred_bin, average="macro")
        print(f"\n== {split} (n={len(part)}) binary macro-F1={bin_f1:.3f}")
        print(classification_report(y_bin, pred_bin, zero_division=0))

        b = (part["label"] != "none").to_numpy()
        cat_acc = float((clf_cat.predict(X[b]) == part["label"][b]).mean())
        print(f"category accuracy on truly-biased rows: {cat_acc:.3f}")
        metrics[split] = {
            "binary_macro_f1": bin_f1, "n": len(part),
            "binary_report": classification_report(y_bin, pred_bin,
                                                   zero_division=0,
                                                   output_dict=True),
            "category_accuracy_on_biased": cat_acc,
        }

    with open(os.path.join(ROOT, "ml", "metrics_layer2.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    export_portable(vec, clf_bin, clf_cat)
    diff = verify_parity(vec, clf_bin, test_df["Text"].tolist()[:50])
    print(f"\nnumpy-runtime vs sklearn parity: max|Δp| = {diff:.2e}")
    assert diff < 1e-4, "portable inference does not match sklearn"
    print("Artifacts written: detection/layer2_model.npz, detection/layer2_vocab.json")


if __name__ == "__main__":
    main()
