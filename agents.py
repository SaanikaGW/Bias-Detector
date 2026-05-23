"""
agents.py — GenAI agents for BIOS Check Careers
All agents are strictly DOWNSTREAM of the detector — they never perform
their own bias detection.
"""
import os
import json
import re
from openai import OpenAI

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

MODEL       = "gpt-4.1-nano"
TEMPERATURE = 0.1


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseSuggestAgent:
    system_prompt: str = ""

    def _call(self, user_input: str, max_tokens: int = 1000) -> str:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": user_input},
            ],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_json(self, user_input: str, max_tokens: int = 1500) -> dict:
        """Call the API and parse the JSON response."""
        raw = self._call(user_input, max_tokens)
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)


# ── Suggestion Agent ──────────────────────────────────────────────────────────

class SuggestionAgent(BaseSuggestAgent):
    system_prompt = (
        "You are a professional job description suggestion agent. Your goal is to suggest "
        "inclusive alternatives for each instance of bias based on the provided biased language "
        "in a provided job description. You must strictly follow the following instructions.\n\n"
        "Instructions:\n"
        "1. Input: a job description text and a list of words or phrases that could be biased "
        "or exclusionary.\n"
        "2. For each biased phrase identified, propose a more inclusive alternative that "
        "preserves the original intent but removes the biased wording.\n"
        "3. Provide exactly one suggestion for each detected bias.\n"
        "4. Output Format: List each biased phrase and its suggested inclusive replacement "
        'as a bullet point:\n   - "biased phrase" -> "inclusive alternative"\n\n'
        "The output should ONLY consist of the list of biased phrases and their replacements, "
        "without additional commentary."
    )

    def generate(self, jd_text: str, highlights: list[dict]) -> list[str]:
        if not highlights:
            return []

        spans_str = "\n".join(f'- "{h["span"]}" (type: {h["type"]})' for h in highlights)
        user_input = (
            f"Job Description:\n{jd_text}\n\n"
            f"Detected biased phrases:\n{spans_str}"
        )
        raw = self._call(user_input)
        # Parse bullet list: - "x" -> "y"
        lines      = [l.strip() for l in raw.splitlines() if "->" in l]
        suggestions = []
        for line in lines:
            parts = line.split("->", 1)
            if len(parts) == 2:
                left  = parts[0].strip().strip('-').strip().strip('"')
                right = parts[1].strip().strip('"')
                suggestions.append(f"{left} → {right}")
        return suggestions or [raw]


# ── Rewrite Agent ─────────────────────────────────────────────────────────────

class RewriteAgent(BaseSuggestAgent):
    system_prompt = (
        "You are a professional job rewriting agent. Given an original job description and "
        "a list of detected biases and suggested inclusive alternatives, produce a revised "
        "job description where each biased segment is replaced.\n\n"
        "Instructions:\n"
        "1. Do NOT perform your own bias detection. Use ONLY the biased spans and suggested "
        "alternatives provided in the input.\n"
        "2. For each biased span, use the suggested inclusive alternatives to replace it. "
        "Language can be slightly modified but original meaning and tone must be retained.\n"
        "3. Do not alter any other content. Keep structure, order, and technical details "
        "the same.\n"
        "4. Output Format: Output the complete rewritten job description in plain text only."
    )

    def generate(self, jd_text: str, detection: dict, suggestions: list[str]) -> str:
        bias_outputs = json.dumps({
            "highlights":  detection.get("highlights", []),
            "suggestions": suggestions,
        }, indent=2)
        user_input = (
            "<INPUT>\n"
            f"[ORIGINAL_JD]\n{jd_text}\n\n"
            f"[BIAS_DETECTOR_OUTPUTS]\n{bias_outputs}\n"
            "</INPUT>"
        )
        return self._call(user_input, max_tokens=2000)


# ── PII Stripping Agent ───────────────────────────────────────────────────────

class PIIStripper(BaseSuggestAgent):
    system_prompt = (
        "You are a professional resume anonymization agent. Your task is to remove or "
        "neutralize all personally identifiable information (PII) and demographic signals "
        "from a resume before it is evaluated for job fit. You must do this to ensure the "
        "hiring evaluation is based solely on skills and qualifications.\n\n"
        "Instructions:\n"
        '1. Replace the candidate\'s name everywhere it appears with "Candidate."\n'
        "2. Replace all gendered pronouns (he, she, her, his, him) with \"they/their/them.\"\n"
        "3. Remove or redact: full address, phone number, email address, date of birth, "
        "graduation year if it implies age.\n"
        "4. Do NOT remove or alter: job titles, company names, dates of employment, listed "
        "skills, certifications, education degrees, or any professional accomplishment.\n"
        "5. Do NOT infer or add any information.\n"
        "6. Output ONLY the anonymized resume text. Do not include any commentary, notes, "
        "or explanation."
    )

    def strip(self, resume_text: str) -> str:
        return self._call(resume_text, max_tokens=3000)


# ── Fit Evaluator ─────────────────────────────────────────────────────────────

class FitEvaluator(BaseSuggestAgent):
    system_prompt = (
        "You are a professional, bias-aware hiring evaluation agent. Your sole task is to "
        "evaluate how well a candidate's qualifications match the requirements of a job "
        "description. You must follow these rules strictly and without exception.\n\n"
        "RULES:\n"
        "1. Evaluate ONLY the rewritten job description provided. Ignore any other version.\n"
        "2. Evaluate ONLY the anonymized resume provided. If you detect any demographic "
        "signals (names, pronouns, gender indicators, age indicators), ignore them entirely.\n"
        "3. Score the candidate on skills, years of experience, certifications, and "
        "accomplishments ONLY — as explicitly listed in the job description.\n"
        "4. For every point in your evaluation, cite the specific requirement from the JD "
        "and the specific evidence from the resume. Do not make vague impressions.\n"
        "5. Do NOT use language that correlates with gender stereotypes (e.g., \"assertive,\" "
        "\"empathetic,\" \"aggressive,\" \"nurturing\") as positive or negative signals.\n"
        "6. Do NOT penalize gaps in employment history without explicit evidence that the "
        "role requires continuous employment.\n"
        "7. Output your evaluation in the following JSON format exactly:\n"
        "{\n"
        '  "fit_score": <float 0.0-1.0>,\n'
        '  "fit_level": "<strong|moderate|weak>",\n'
        '  "skill_matches": [{"requirement": "...", "evidence": "...", "match": '
        '"<full|partial|none>"}],\n'
        '  "gaps": [{"requirement": "...", "evidence": "...", "match": "none"}],\n'
        '  "explanation": "<2-3 sentence plain-language summary>",\n'
        '  "bias_signals_suppressed": ["<list any demographic signals detected and ignored>"]\n'
        "}"
    )

    def evaluate(self, rewritten_jd: str, anonymized_resume: str) -> dict:
        user_input = (
            f"[REWRITTEN_JD]\n{rewritten_jd}\n\n"
            f"[ANONYMIZED_RESUME]\n{anonymized_resume}"
        )
        return self._call_json(user_input, max_tokens=2000)

    def evaluate_traditional(self, original_jd: str, original_resume: str) -> dict:
        """Simulate traditional (non-bias-aware) evaluation for comparison view."""
        trad_prompt = (
            "You are a standard hiring evaluation assistant. Evaluate how well this "
            "candidate's resume matches the job description. Score on overall fit.\n\n"
            f"[JOB DESCRIPTION]\n{original_jd}\n\n"
            f"[RESUME]\n{original_resume}\n\n"
            "Output JSON with these fields only: fit_score (float 0-1), fit_level "
            "(strong|moderate|weak), explanation (1-2 sentences)."
        )
        raw      = self._call(trad_prompt, max_tokens=500)
        raw      = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"fit_score": 0.5, "fit_level": "moderate", "explanation": raw}
