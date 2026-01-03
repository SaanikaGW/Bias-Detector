from __future__ import annotations

import json
import os
import re
import string
from typing import Any, Dict, Iterable, List, Mapping, Optional

from flask import Flask, jsonify, request

# Optional dependency (semantic matching)
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    util = None

# Optional dependency (GenAI calls)
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------------------------
# Seed data (swap with CSV-derived spans later)
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

# Placeholder to later replace with spans parsed from CSV
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
# Reasons + suggestions (fallback if GenAI is off)
# -------------------------------------------------------------------

REASONS: Mapping[str, str] = {
    "explicit_gender": "Explicit gendered terms or pronouns; use neutral phrasing.",
    "stereotype": "Stereotypical trait assignment to a gendered group.",
    "requirements_bias": "Requirements may exclude caregivers or those with limited flexibility.",
    "age_bias": "Age-coded phrasing that can deter experienced or older candidates.",
}

SUGGESTIONS_FALLBACK: Mapping[str, str] = {
    "explicit_gender": "Use role titles or 'they/them' instead of gendered pronouns.",
    "stereotype": "Describe behaviors/skills directly without tying them to gender.",
    "requirements_bias": "Focus on outcomes and reasonable schedules; avoid 24/7 language.",
    "age_bias": "Emphasize skills and experience over age-coded descriptors.",
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
    Surface-form matching for explicit gendered language (and any phrase list).
    """
    text_norm = normalize_text(text)
    spans: List[Dict[str, Any]] = []
    seen = set()

    for phrase in phrases:
        phrase_norm = normalize_text(phrase)
        if phrase_norm and phrase_norm in text_norm and (phrase_norm, "explicit_gender") not in seen:
            spans.append({"span": phrase, "type": "explicit_gender"})
            seen.add((phrase_norm, "explicit_gender"))

    tokens = text_norm.split()
    for token in tokens:
        if token in terms and (token, "explicit_gender") not in seen:
            spans.append({"span": token, "type": "explicit_gender"})
            seen.add((token, "explicit_gender"))

    return spans


# -------------------------------------------------------------------
# Semantic matching using embeddings (optional)
# -------------------------------------------------------------------

_EMBED_MODEL: Optional["SentenceTransformer"] = None
_SPAN_EMBEDS: Optional[Dict[str, Any]] = None


def _embedding_model() -> "SentenceTransformer":
    global _EMBED_MODEL
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for semantic matching.")
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _span_embeddings() -> Dict[str, Any]:
    """
    Precompute span embeddings per bias type (lazy).
    """
    global _SPAN_EMBEDS
    if _SPAN_EMBEDS is not None:
        return _SPAN_EMBEDS

    model = _embedding_model()
    _SPAN_EMBEDS = {
        bias_type: model.encode(spans, convert_to_tensor=True)
        for bias_type, spans in BIAS_SPANS_FROM_CSV.items()
        if spans
    }
    return _SPAN_EMBEDS


def semantic_span_match(text: str, threshold: float = 0.75) -> List[Dict[str, Any]]:
    """
    Find biasy phrases using cosine similarity against seed spans.
    If sentence-transformers isn't installed, returns [].
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
                    {"span": chunk, "type": bias_type, "score": round(max_score, 3)}
                )

    return results


# -------------------------------------------------------------------
# Aggregation + scoring
# -------------------------------------------------------------------

def _dedupe_spans(spans: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for hit in spans:
        span = str(hit.get("span", ""))
        btype = str(hit.get("type", ""))
        key = (span.lower(), btype)
        if key in seen:
            continue
        seen.add(key)

        if btype in REASONS:
            hit = {**hit, "reason": REASONS[btype]}
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
    exact_hits = find_exact_spans(
        jd_text,
        DEFAULT_BIAS_SPANS["explicit_gender"],
        DEFAULT_EXPLICIT_TERMS
    )
    semantic_hits = semantic_span_match(jd_text)

    highlights = _dedupe_spans([*exact_hits, *semantic_hits])
    categories = sorted({h["type"] for h in highlights})

    # Simple scoring: each unique span contributes weight; semantic strength nudges score.
    num_spans = len(highlights)
    semantic_boost = max((h.get("score", 0) for h in highlights), default=0) * 0.1
    bias_score = min(1.0, num_spans * 0.2 + semantic_boost)
    bias_level = _score_to_level(bias_score)

    suggestions = [SUGGESTIONS_FALLBACK[c] for c in categories if c in SUGGESTIONS_FALLBACK]

    return {
        "bias_score": round(bias_score, 2),
        "bias_level": bias_level,
        "categories": categories,
        "highlights": highlights,
        "suggestions": suggestions,
    }


# -------------------------------------------------------------------
# GenAI Suggestion Agent (OpenAI) — optional
# -------------------------------------------------------------------

SUGGESTION_SYSTEM_INSTRUCTIONS = """\
Role:
You are a professional job description suggestion agent.

Goal:
Given (1) a job description and (2) a list of potentially biased or exclusionary words/phrases,
suggest exactly ONE inclusive alternative for EACH provided phrase.

Rules (must follow):
- Do NOT perform your own bias detection.
- Only generate suggestions for the provided phrases.
- Provide exactly one suggestion per phrase (no extras).
- Preserve original intent as much as possible while removing bias.
- Output ONLY a bulleted list, each item exactly:
  "biased phrase" -> "inclusive alternative"
- Use straight quotes " and the arrow -> exactly as shown.
- No additional commentary.
"""


def _get_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    return OpenAI()


def suggest_inclusive_alternatives(
    job_description_text: str,
    phrases: List[str],
    model: str = "gpt-5.1-chat-latest",
    temperature: float = 0.2,
) -> str:
    """
    Calls OpenAI to produce `"biased phrase" -> "inclusive alternative"` bullet list.
    """
    client = _get_openai_client()

    # Keep user input as data (JSON) to reduce injection risk
    user_payload = {
        "job_description": job_description_text,
        "phrases": phrases,
        "required_output_format": "\"biased phrase\" -> \"inclusive alternative\" (one per phrase, bullet list only)",
    }

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": SUGGESTION_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    # Extract output text
    out_text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out_text += c.text

    return out_text.strip()


# -------------------------------------------------------------------
# GenAI Rewrite Agent — optional
# -------------------------------------------------------------------

REWRITE_SYSTEM_INSTRUCTIONS = """\
Role:
You are a professional job rewriting agent.

Goal:
Given an original job description and bias findings (spans, types, bias_score, bias_level),
rewrite the job description by replacing each biased span with a more inclusive alternative.

Rules (must follow):
- Do NOT perform your own bias detection.
- Only revise the exact spans provided in the bias findings.
- Preserve meaning, structure, order, and technical requirements.
- Output ONLY the rewritten job description in plain text.
- No headings, no commentary, no bullet lists.
"""

def rewrite_job_description(
    job_description_text: str,
    bias_detector_outputs: List[Dict[str, Any]],
    model: str = "gpt-5.1-chat-latest",
    temperature: float = 0.2,
) -> str:
    client = _get_openai_client()

    user_payload = {
        "original_jd": job_description_text,
        "bias_detector_outputs": bias_detector_outputs,
    }

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": REWRITE_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    out_text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out_text += c.text

    return out_text.strip()


# -------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/analyze")
def api_analyze():
    """
    Body:
      { "job_description": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    jd = data.get("job_description", "")
    if not isinstance(jd, str) or not jd.strip():
        return jsonify({"error": "job_description must be a non-empty string"}), 400

    out = analyze_jd(jd)
    return jsonify(out)


@app.post("/api/suggest")
def api_suggest():
    """
    Body:
      {
        "job_description": "...",
        "phrases": ["young and energetic", "salesman", ...]
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    jd = data.get("job_description", "")
    phrases = data.get("phrases", [])

    if not isinstance(jd, str) or not jd.strip():
        return jsonify({"error": "job_description must be a non-empty string"}), 400
    if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
        return jsonify({"error": "phrases must be a list of strings"}), 400

    try:
        out = suggest_inclusive_alternatives(jd, phrases)
        return jsonify({"suggestions": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/rewrite")
def api_rewrite():
    """
    Body:
      {
        "job_description": "...",
        "bias_detector_outputs": [
          {"span":"young and energetic","type":"age_bias","score":0.92,...},
          ...
        ]
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    jd = data.get("job_description", "")
    outputs = data.get("bias_detector_outputs", [])

    if not isinstance(jd, str) or not jd.strip():
        return jsonify({"error": "job_description must be a non-empty string"}), 400
    if not isinstance(outputs, list):
        return jsonify({"error": "bias_detector_outputs must be a list"}), 400

    try:
        out = rewrite_job_description(jd, outputs)
        return jsonify({"rewritten_job_description": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------
# Local run
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Run locally:
    #   export OPENAI_API_KEY="..."
    #   python detector.py
    #
    # Then test:
    #   curl -X POST http://127.0.0.1:5000/api/analyze -H "Content-Type: application/json" -d '{"job_description":"We want a young man to lead the team..."}'
    #
    app.run(host="127.0.0.1", port=5000, debug=True)
