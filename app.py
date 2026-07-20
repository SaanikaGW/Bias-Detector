"""
BIOS Check Careers — Flask Backend
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
from detection.pipeline import analyze as analyze_pipeline
from agents import PIIStripper, FitEvaluator

app = Flask(__name__)
_cors_origin = os.environ.get("CORS_ORIGIN", "*")
CORS(app, origins=_cors_origin)


def _send_contact_email(name: str, sender_email: str, category: str, message: str) -> None:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_APP_PASSWORD")
    recipient = os.getenv("CONTACT_EMAIL", smtp_user)
    if not smtp_user or not smtp_pass:
        return
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg["Subject"] = f"[BIOS Check] {category} — {name}"
    body = f"From: {name} <{sender_email}>\nCategory: {category}\n\n{message}"
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipient, msg.as_string())

# ── Bias Reducer ─────────────────────────────────────────────────────────────

@app.route("/api/bias-reducer/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()

    if len(text) < 10:
        return jsonify({"error": "Text too short"}), 400

    # v2.0 hybrid pipeline: Layer 1 rules -> Layer 2 contextual classifier
    # (the decision-makers) -> Layer 3 LLM explanations/rewrites (downstream
    # only, with offline template fallback). The response contains both the
    # v2 contract (issues, scores, rewritten_jd) and every v1 field
    # (bias_score, bias_level, categories, highlights, suggestions) so
    # existing consumers keep working.
    try:
        result = analyze_pipeline(text)
    except Exception as e:
        print(f"[analyze] pipeline error: {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

    return jsonify(result)


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

    print(f"[Contact] {data['name']} <{data['email']}> — {data['category']}")
    try:
        _send_contact_email(data["name"], data["email"], data["category"], data["message"])
    except Exception as e:
        print(f"[Contact] Email delivery failed: {e}")
    return jsonify({"success": True, "message": "Thank you! We'll be in touch."})


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
