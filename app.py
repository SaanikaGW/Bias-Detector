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
from detection.pipeline import analyze as analyze_pipeline, score_only as score_only_pipeline
import metrics_store
from agents import PIIStripper, FitEvaluator

app = Flask(__name__)
_cors_origin = os.environ.get("CORS_ORIGIN", "*")
CORS(app, origins=_cors_origin)
metrics_store.init_db()


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

    # Honest before/after: re-score the suggested rewrite (Layers 1+2 only,
    # no extra LLM call — see detection/pipeline.score_only) so "percent
    # improved" reflects what actually changed, not an assumed 100% fix.
    rewritten = (result.get("rewritten_jd") or "").strip()
    before_bias = result["scores"]["gender_bias_score"]
    if rewritten and rewritten != text:
        try:
            after = score_only_pipeline(rewritten)
            after_bias = after["gender_bias_score"]
            pct = round(max(0, (before_bias - after_bias) / before_bias * 100), 1) if before_bias > 0 else 0.0
            result["after_scores"] = {
                "gender_bias_score": after_bias,
                "inclusive_language_score": after["inclusive_language_score"],
            }
            result["percent_improved"] = pct
        except Exception as e:
            print(f"[analyze] after-score computation failed: {e}")
            result["percent_improved"] = None
    else:
        result["percent_improved"] = 0.0 if before_bias == 0 else None

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


# ── Usage metrics (global, across all visitors) ──────────────────────────────

@app.route("/api/usage/register", methods=["POST"])
def usage_register():
    data = request.get_json() or {}
    user_id = (data.get("user_id") or "").strip()
    user_type = data.get("user_type")
    company_name = (data.get("company_name") or "").strip() or None
    if not user_id or user_type not in metrics_store.USER_TYPES:
        return jsonify({"error": "Missing or invalid user_id/user_type"}), 400
    try:
        metrics_store.register_user(user_id, user_type, company_name)
    except Exception as e:
        print(f"[usage] register failed: {e}")
        return jsonify({"error": "Could not register"}), 500
    return jsonify({"ok": True})


@app.route("/api/metrics/log", methods=["POST"])
def metrics_log():
    data = request.get_json() or {}
    user_id = (data.get("user_id") or "").strip()
    event_type = data.get("event_type")
    payload = data.get("payload") or {}
    if not user_id or event_type not in metrics_store.EVENT_TYPES:
        return jsonify({"error": "Missing or invalid user_id/event_type"}), 400
    try:
        metrics_store.log_event(user_id, event_type, payload)
    except Exception as e:
        print(f"[metrics] log failed: {e}")
        return jsonify({"error": "Could not log event"}), 500
    return jsonify({"ok": True})


@app.route("/api/metrics/summary")
def metrics_summary_route():
    try:
        return jsonify(metrics_store.summary())
    except Exception as e:
        print(f"[metrics] summary failed: {e}")
        return jsonify({"error": "Could not load summary"}), 500


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
