"""
Layer 3 — LLM explanation & rewrite (strictly downstream).

The LLM NEVER detects, overrides, or invents flags. Layers 1+2 decide what
is flagged; this layer only enriches each confirmed issue with:
- a plain-language explanation of what was flagged and why,
- the research rationale (grounded in the category blurbs from
  patterns.json — the LLM is instructed not to invent citations),
- an inclusive rewrite that preserves the actual job requirement,
- and a full rewritten job description.

Graceful degradation: if OPENAI_API_KEY is missing or the call fails, we
fall back to the template explanations and alternatives that ship with
each rule in patterns.json, so the product keeps working offline — with
slightly less tailored prose, never with fewer detections.
"""
from __future__ import annotations

import json
import os
import re

from .layer1 import categories

MODEL = os.environ.get("BIOS_LLM_MODEL", "gpt-4.1-nano")
TEMPERATURE = 0.1


def _client():
    from openai import OpenAI
    return OpenAI()


def _llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


# ── Template fallback (no API needed) ─────────────────────────────────────────

def _fallback_issue_fields(issue: dict) -> dict:
    cat = categories().get(issue["category"], {})
    alt = issue.get("alternatives") or []
    rewrite = alt[0] if alt else "consider rephrasing in neutral, skills-based terms"
    return {
        "explanation": issue.get("why") or
            f"“{issue['span']}” was flagged as {cat.get('label', issue['category'])}.",
        "research_rationale": cat.get("research", ""),
        "impact": cat.get("impact", ""),
        "rewrite": rewrite,
        "expected_improvement": issue.get("gain") or
            "More candidates who can do the job will actually apply.",
    }


def _is_drop_in(alt: str | None) -> bool:
    """Some alternatives are advice ('remove; state the actual schedule…'),
    not drop-in replacements. Only substitute true drop-ins."""
    if not alt:
        return False
    lowered = alt.lower()
    return (";" not in alt and "(" not in alt
            and not lowered.startswith(("remove", "consider", "state "))
            and len(alt.split()) <= 8)


def _fallback_rewritten_jd(text: str, issues: list[dict]) -> str:
    """Deterministic rewrite: replace each span with its first drop-in
    alternative; advisory-only issues keep their original text (the issue
    card still carries the advice)."""
    out = text
    for issue in sorted(issues, key=lambda i: i.get("start", 0), reverse=True):
        alt = (issue.get("alternatives") or [None])[0]
        if _is_drop_in(alt) and issue.get("start") is not None \
                and issue.get("end") is not None:
            out = out[:issue["start"]] + alt + out[issue["end"]:]
    return out


# ── LLM enrichment ─────────────────────────────────────────────────────────────

_SYSTEM = """You are the explanation layer of BIOS Career Check, a tool that helps recruiters write job descriptions that attract qualified candidates regardless of gender. Detection has ALREADY happened — you must not add, remove, re-judge, or re-detect anything.

For each flagged issue you receive, write:
- explanation: 1-2 sentences, educational and never judgmental. Name what was flagged and why it may discourage applicants. Assume good intent by the writer.
- rewrite: an inclusive replacement for the flagged phrase that PRESERVES the actual job requirement. Never soften a real technical or scheduling requirement — restate it inclusively.
- expected_improvement: one short sentence on what improves.

Ground every research claim ONLY in the research_context you are given. Do not invent studies, statistics, or citations.

Also produce full_rewrite: the complete job description with each flagged phrase replaced by its rewrite. Change nothing else — keep structure, order, tone, and all technical details.

Return strict JSON:
{"issues": [{"id": <id>, "explanation": "...", "rewrite": "...", "expected_improvement": "..."}], "full_rewrite": "..."}"""


def _llm_enrich(text: str, issues: list[dict]) -> dict | None:
    payload = {
        "job_description": text,
        "flagged_issues": [
            {
                "id": i,
                "phrase": iss["span"],
                "sentence": iss["sentence"],
                "category": iss["category"],
                "severity": iss["severity"],
                "why_flagged": iss.get("why", ""),
                "research_context": categories().get(iss["category"], {}).get("research", ""),
                "suggested_directions": iss.get("alternatives", []),
            }
            for i, iss in enumerate(issues)
        ],
    }
    try:
        resp = _client().chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:  # any API/parse failure -> template fallback
        print(f"[layer3] LLM enrichment unavailable ({e}) — using templates.")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def enrich(text: str, issues: list[dict]) -> tuple[list[dict], str, str]:
    """
    Enrich confirmed issues. Returns (issues, rewritten_jd, enrichment_source).
    Never changes WHICH issues exist — only adds explanation fields.
    """
    if not issues:
        return issues, text, "none-needed"

    llm_out = _llm_enrich(text, issues) if _llm_available() else None

    enriched = []
    by_id = {}
    if llm_out:
        for item in llm_out.get("issues", []):
            if isinstance(item, dict) and "id" in item:
                by_id[item["id"]] = item

    for i, issue in enumerate(issues):
        fields = _fallback_issue_fields(issue)
        llm_fields = by_id.get(i, {})
        # LLM text wins where present; templates fill the gaps.
        merged = dict(issue)
        merged["explanation"] = llm_fields.get("explanation") or fields["explanation"]
        merged["research_rationale"] = fields["research_rationale"]
        merged["impact"] = fields["impact"]
        merged["rewrite"] = llm_fields.get("rewrite") or fields["rewrite"]
        merged["expected_improvement"] = (llm_fields.get("expected_improvement")
                                          or fields["expected_improvement"])
        enriched.append(merged)

    if llm_out and llm_out.get("full_rewrite"):
        rewritten = llm_out["full_rewrite"]
        source = "llm"
    else:
        rewritten = _fallback_rewritten_jd(text, enriched)
        source = "templates"
    return enriched, rewritten, source
