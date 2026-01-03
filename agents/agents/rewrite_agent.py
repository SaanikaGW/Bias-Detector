import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

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
    client: Optional[OpenAI] = None,
) -> str:
    if not isinstance(job_description_text, str) or not job_description_text.strip():
        raise ValueError("job_description_text must be a non-empty string")
    if not isinstance(bias_detector_outputs, list) or not bias_detector_outputs:
        raise ValueError("bias_detector_outputs must be a non-empty list")

    client = client or OpenAI()

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
