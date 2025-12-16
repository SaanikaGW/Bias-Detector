from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterable, List, Mapping

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover - dependency optional at runtime
    SentenceTransformer = None
    util = None


# -------------------------------------------------------------------
# Seed data (can be swapped out with CSV-derived spans later)
# -------------------------------------------------------------------

DEFAULT_BIAS_SPANS: Dict[str, List[str]] = {
    "explicit_gender": [
        "he",
        "she",
        "his",
        "hers",
        "man",
        "men",
        "woman",
        "women",
        "young man",
        "young woman",
        "female candidate",
        "male candidate",
        "prefer male",
        "prefer female",
        "seeking a woman",
        "seeking a man",
    ],
    "stereotype": [
        "nurturing female",
        "nurturing woman",
        "dominant leader",
        "not emotional",
        "emotionally stable",
        "strong male presence",
        "soft spoken woman",
        "aggressive salesman",
        "motherly",
    ],
    "requirements_bias": [
        "always available",
        "available at all times",
        "work long hours",
        "work late nights",
        "no family obligations",
        "no family duties",
        "no caregiving responsibilities",
        "24/7",
        "weekends required",
        "must be on call",
    ],
    "age_bias": [
        "young energetic",
        "young man",
        "young woman",
        "fresh graduate",
        "recent graduate",
        "digital native",
        "under 30",
        "youthful",
    ],
}

# Placeholder that can be replaced with spans parsed from the CSV dataset.
BIAS_SPANS_FROM_CSV: Dict[str, List[str]] = DEFAULT_BIAS_SPANS

DEFAULT_EXPLICIT_TERMS = {
    "he",
    "she",
    "his",
    "hers",
    "him",
    "her",
    "mr",
    "mrs",
    "ms",
}


# -------------------------------------------------------------------
# Text normalization + exact matching
# -------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_exact_spans(text: str, phrases: Iterable[str], terms: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Simple surface-form matching for explicit gendered language.
    """
    text_norm = normalize_text(text)
    spans: List[Dict[str, Any]] = []
    seen = set()

    for phrase in phrases:
        phrase_norm = phrase.lower()
        if phrase_norm in text_norm and phrase_norm not in seen:
            spans.append({"span": phrase, "type": "explicit_gender"})
            seen.add(phrase_norm)

    tokens = text_norm.split()
    for token in tokens:
        if token in terms and token not in seen:
            spans.append({"span": token, "type": "explicit_gender"})
            seen.add(token)

    return spans


# -------------------------------------------------------------------
# Semantic matching using embeddings
# -------------------------------------------------------------------

def _embedding_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for semantic matching.")
    return SentenceTransformer("all-MiniLM-L6-v2")


def _span_embeddings() -> Dict[str, Any]:
    """
    Precompute span embeddings per bias type. Called lazily so that tests
    can stub or skip heavy model loads.
    """
    model = _embedding_model()
    return {
        bias_type: model.encode(spans, convert_to_tensor=True)
        for bias_type, spans in BIAS_SPANS_FROM_CSV.items()
    }


def semantic_span_match(text: str, threshold: float = 0.75) -> List[Dict[str, Any]]:
    """
    Find biasy phrases using cosine similarity against seed spans.
    """
    if SentenceTransformer is None or util is None:
        return []

    text_norm = normalize_text(text)
    chunks = re.split(r"[.,;:\n]+", text_norm)
    chunks = [c.strip() for c in chunks if c.strip()]

    results: List[Dict[str, Any]] = []
    model = _embedding_model()
    span_embeds = _span_embeddings()

    for chunk in chunks:
        chunk_emb = model.encode(chunk, convert_to_tensor=True)
        for bias_type, embeds in span_embeds.items():
            scores = util.cos_sim(chunk_emb, embeds)[0]
            max_score = float(scores.max().item())
            if max_score >= threshold:
                results.append(
                    {
                        "span": chunk,
                        "type": bias_type,
                        "score": round(max_score, 3),
                    }
                )

    return results


# -------------------------------------------------------------------
# Aggregation + scoring
# -------------------------------------------------------------------

REASONS: Mapping[str, str] = {
    "explicit_gender": "Explicit gendered terms or pronouns; use neutral phrasing.",
    "stereotype": "Stereotypical trait assignment to a gendered group.",
    "requirements_bias": "Requirements may exclude caregivers or those with limited flexibility.",
    "age_bias": "Age-coded phrasing that can deter experienced or older candidates.",
}

#Suggestions can be replaced by GENAI
SUGGESTIONS: Mapping[str, str] = {
    "explicit_gender": "Use role titles or 'they/them' instead of gendered pronouns.",
    "stereotype": "Describe behaviors/skills directly without tying them to gender.",
    "requirements_bias": "Focus on outcomes and reasonable schedules; avoid 24/7 language.",
    "age_bias": "Emphasize skills and experience over age-coded descriptors.",
}


def _dedupe_spans(spans: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for hit in spans:
        key = (hit["span"].lower(), hit["type"])
        if key in seen:
            continue
        seen.add(key)
        if hit["type"] in REASONS:
            hit = {**hit, "reason": REASONS[hit["type"]]}
        unique.append(hit)
    return unique


def _score_to_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def analyze_jd(jd_text: str) -> Dict[str, Any]:
    """
    Input: raw job description string.
    Output:
    {
        "bias_score": float,
        "bias_level": "low" | "medium" | "high",
        "categories": [ ... ],
        "highlights": [ {"span": str, "type": str, "score": float?, "reason": str?}, ... ],
        "suggestions": [ ... ],
    }
    """
    exact_hits = find_exact_spans(jd_text, DEFAULT_BIAS_SPANS["explicit_gender"], DEFAULT_EXPLICIT_TERMS)
    semantic_hits = semantic_span_match(jd_text)

    highlights = _dedupe_spans([*exact_hits, *semantic_hits])
    categories = sorted({h["type"] for h in highlights})

    # Simple scoring: each unique span contributes weight; semantic strength nudges score.
    num_spans = len(highlights)
    semantic_boost = max((h.get("score", 0) for h in highlights), default=0) * 0.1
    bias_score = min(1.0, num_spans * 0.2 + semantic_boost)
    bias_level = _score_to_level(bias_score)

    suggestions = [SUGGESTIONS[c] for c in categories if c in SUGGESTIONS]

    return {
        "bias_score": round(bias_score, 2),
        "bias_level": bias_level,
        "categories": categories,
        "highlights": highlights,
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    sample = (
        "We are looking for a young man to lead the team. "
        "The candidate must be always available and comfortable working late nights."
    )
    from pprint import pprint

    pprint(analyze_jd(sample))
