import json
from typing import List, Dict, Any, Optional

from openai import OpenAI

# NOTE: never hardcode API keys in code or commit them.
# Use environment variable OPENAI_API_KEY.

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

def _validate_phrases(phrases: List[str]) -> List[str]:
    clean = []
    for p in phrases:
        if not isinstance(p, str):
            continue
        p2 = p.strip()
        if p2:
            clean.append(p2)
    # de-duplicate while preserving order
    seen = set()
    dedup = []
    for p in clean:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup

def suggest_inclusive_alternatives(
    job_description_text: str,
    potentially_biased_phrases: List[str],
    model: str = "gpt-5.1-chat-latest",
    temperature: float = 0.2,
    client: Optional[OpenAI] = None,
) -> str:
    """
    Returns a string containing the bullet list of replacements.

    Raises ValueError if inputs are missing/invalid.
    """
    if not isinstance(job_description_text, str) or not job_description_text.strip():
        raise ValueError("job_description_text must be a non-empty string")

    phrases = _validate_phrases(potentially_biased_phrases)
    if not phrases:
        raise ValueError("potentially_biased_phrases must be a non-empty list of strings")

    client = client or OpenAI()

    # USER CONTENT IS STRICTLY DATA. Do not put it in system instructions.
    user_payload = {
        "job_description": job_description_text,
        "phrases": phrases,
        "required_output_format": "\"biased phrase\" -> \"inclusive alternative\" (one per phrase, bullet list only)"
    }

    # Recommended: Responses API for new projects
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": SUGGESTION_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    # Responses API returns output text across items; easiest is output_text helper if present
    # SDKs differ slightly; this pattern is robust:
    out_text = ""
    for item in resp.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    out_text += c.text

    return out_text.strip()
