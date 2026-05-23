"""
BIOS Check Careers — Flask Backend
"""
import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import analyze_jd
from agents import SuggestionAgent, RewriteAgent, PIIStripper, FitEvaluator

app = Flask(__name__)
_cors_origin = os.environ.get("CORS_ORIGIN", "*")
CORS(app, origins=_cors_origin)

# ── Bias Reducer ─────────────────────────────────────────────────────────────

@app.route("/api/bias-reducer/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text      = data["text"].strip()
    tone      = data.get("tone")
    seniority = data.get("seniority")

    if len(text) < 10:
        return jsonify({"error": "Text too short"}), 400

    # Step 1: run rule-based + semantic detector
    detection = analyze_jd(text)

    # Step 2: call GenAI for suggestions (downstream only — no re-detection)
    suggestion_agent = SuggestionAgent()
    suggestions = suggestion_agent.generate(text, detection["highlights"])

    # Step 3: call GenAI for rewrite
    rewrite_agent = RewriteAgent()
    rewritten_jd  = rewrite_agent.generate(text, detection, suggestions)

    return jsonify({
        "bias_score":   detection["bias_score"],
        "bias_level":   detection["bias_level"],
        "categories":   detection["categories"],
        "highlights":   detection["highlights"],
        "suggestions":  suggestions,
        "rewritten_jd": rewritten_jd,
    })


# ── Hiring AI ─────────────────────────────────────────────────────────────────

@app.route("/api/hiring-ai/strip-pii", methods=["POST"])
def strip_pii():
    data = request.get_json()
    if not data or "resume_text" not in data:
        return jsonify({"error": "Missing 'resume_text' field"}), 400

    stripper   = PIIStripper()
    anonymized = stripper.strip(data["resume_text"])
    return jsonify({"anonymized_resume": anonymized})


@app.route("/api/hiring-ai/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data or "rewritten_jd" not in data or "anonymized_resume" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    evaluator = FitEvaluator()
    result    = evaluator.evaluate(data["rewritten_jd"], data["anonymized_resume"])
    return jsonify(result)


@app.route("/api/hiring-ai/compare", methods=["POST"])
def compare():
    data = request.get_json()
    required = {"original_jd", "rewritten_jd", "original_resume"}
    if not data or not required.issubset(data):
        return jsonify({"error": "Missing required fields"}), 400

    # Bias-aware path
    stripper   = PIIStripper()
    anon       = stripper.strip(data["original_resume"])
    evaluator  = FitEvaluator()
    aware_result = evaluator.evaluate(data["rewritten_jd"], anon)

    # Traditional path (original JD + original resume, no stripping)
    trad_result = evaluator.evaluate_traditional(data["original_jd"], data["original_resume"])

    delta = round(aware_result["fit_score"] - trad_result["fit_score"], 2)
    return jsonify({
        "bias_aware":           aware_result,
        "traditional":          trad_result,
        "score_delta":          f"{'+' if delta >= 0 else ''}{delta}",
        "anonymized_resume":    anon,
    })


# ── Contact ───────────────────────────────────────────────────────────────────

@app.route("/api/contact/submit", methods=["POST"])
def contact():
    data = request.get_json()
    required = {"name", "email", "category", "message"}
    if not data or not required.issubset(data):
        return jsonify({"error": "Missing required fields"}), 400

    # TODO: store to Firebase or send email
    print(f"[Contact] {data['name']} <{data['email']}> — {data['category']}")
    return jsonify({"success": True, "message": "Thank you! We'll be in touch."})


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
