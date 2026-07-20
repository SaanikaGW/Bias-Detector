"""
Layer 2 — contextual classifier (runtime inference).

Trained by ml/train_layer2.py and exported as plain numpy arrays + a JSON
vocabulary. This module reimplements sklearn's TF-IDF transform exactly in
numpy, so the server needs NO scikit-learn, no torch, and no network call —
inference is deterministic and costs microseconds. Parity with sklearn's
predict_proba is asserted at training time (max |Δp| < 1e-4).

Two heads over shared TF-IDF (word 1-3 gram) features:
- BINARY head (biased vs none): the decision-maker. N-grams carry local
  context, which is what separates "aggressive personality required"
  (biased) from "aggressive sales strategy" (benign) — a trained
  contextual decision, not a keyword lookup.
- CATEGORY head: names the bias category once the binary head has decided.

Role in the pipeline (the classifier DECIDES, the LLM only explains):
1. Adjudicate ambiguous Layer 1 hits from full-sentence context.
2. Catch subtle bias in sentences no rule matched, at a high confidence
   bar (NEW_ISSUE_THRESHOLD) to keep false positives low.
"""
from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache

import numpy as np

_DIR = os.path.dirname(__file__)

# Ambiguous Layer-1 hit is CLEARED unless the sentence is likely biased.
ADJUDICATE_BIASED_THRESHOLD = 0.50
# A sentence with no rule hit becomes a new issue only above this prob.
NEW_ISSUE_THRESHOLD = 0.80

_TOKEN_RX = re.compile(r"(?u)\b\w\w+\b")


class Layer2Classifier:
    def __init__(self, meta: dict, arrs):
        self.vocab = meta["vocabulary"]
        self.bin_classes = meta["bin_classes"]
        self.cat_classes = meta["cat_classes"]
        self.ngram_range = tuple(meta["ngram_range"])
        self.idf = arrs["idf"]
        self.bin_coef = arrs["bin_coef"]
        self.bin_intercept = arrs["bin_intercept"]
        self.cat_coef = arrs["cat_coef"]
        self.cat_intercept = arrs["cat_intercept"]
        self._biased_idx = self.bin_classes.index("biased")

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "Layer2Classifier":
        with open(os.path.join(_DIR, "layer2_vocab.json"), encoding="utf-8") as f:
            meta = json.load(f)
        arrs = np.load(os.path.join(_DIR, "layer2_model.npz"))
        return cls(meta, arrs)

    # ── exact reimplementation of sklearn's TfidfVectorizer transform ──
    def _vectorize(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RX.findall(text.lower())
        lo, hi = self.ngram_range
        counts: dict[int, int] = {}
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                idx = self.vocab.get(" ".join(tokens[i:i + n]))
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
        x = np.zeros(len(self.idf), dtype=np.float64)
        for idx, c in counts.items():
            x[idx] = (1.0 + math.log(c)) * self.idf[idx]  # sublinear tf * idf
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def p_biased(self, text: str) -> float:
        """P(sentence is biased) from the binary head.

        sklearn's binary LogisticRegression uses the sigmoid of the single
        decision score, oriented toward classes_[1]."""
        x = self._vectorize(text)
        z = float(self.bin_coef[0] @ x + self.bin_intercept[0])
        p_class1 = 1.0 / (1.0 + math.exp(-z))
        return p_class1 if self._biased_idx == 1 else 1.0 - p_class1

    def category(self, text: str) -> tuple[str, float]:
        """Most likely bias category (only meaningful if p_biased is high)."""
        x = self._vectorize(text)
        p = self._softmax(self.cat_coef @ x + self.cat_intercept)
        i = int(np.argmax(p))
        return self.cat_classes[i], float(p[i])

    def classify(self, sentence: str) -> dict:
        pb = self.p_biased(sentence)
        label, cat_conf = self.category(sentence)
        return {"p_biased": pb, "label": label if pb >= 0.5 else "none",
                "category_confidence": cat_conf}


def adjudicate(hit: dict, clf: Layer2Classifier | None = None) -> dict | None:
    """
    Decide an ambiguous Layer 1 hit from its full-sentence context.
    Returns the (possibly updated) hit, or None to clear it.
    """
    clf = clf or Layer2Classifier.load()
    pb = clf.p_biased(hit["sentence"])
    if pb < ADJUDICATE_BIASED_THRESHOLD:
        return None  # context says benign — cleared
    hit = dict(hit)
    # Confidence blends the rule prior with contextual evidence.
    hit["confidence"] = round(min(0.99, 0.5 * hit["confidence"] + 0.5 * pb), 2)
    hit["source"] = "rules+classifier"
    return hit


def scan_sentence(sentence: str, clf: Layer2Classifier | None = None) -> dict | None:
    """Classify a rule-free sentence; return a new issue dict above threshold."""
    clf = clf or Layer2Classifier.load()
    pb = clf.p_biased(sentence)
    if pb < NEW_ISSUE_THRESHOLD:
        return None
    label, _ = clf.category(sentence)
    return {
        "span": sentence.strip(),
        "sentence": sentence.strip(),
        "category": label,
        "severity": "medium",
        "severity_num": 2,
        "confidence": round(pb, 2),
        "pattern_id": None,
        "ambiguous": False,
        "why": "",
        "alternatives": [],
        "gain": "",
        "source": "classifier",
    }
