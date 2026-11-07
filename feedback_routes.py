"""
Flask blueprint for testimonials / feedback.

Public:
  POST /api/feedback/submit        -> saved as "pending"
  GET  /api/feedback/list          -> approved testimonials only

Moderation (requires the ADMIN_KEY env var, sent as the X-Admin-Key header):
  GET    /api/feedback/admin/pending
  POST   /api/feedback/admin/<id>/approve
  POST   /api/feedback/admin/<id>/reject   (back to pending)
  DELETE /api/feedback/admin/<id>
"""
import hmac
import os

from flask import Blueprint, request, jsonify

import feedback_store

feedback_bp = Blueprint("feedback", __name__)
feedback_store.init_db()


def _clean(value, limit):
    value = (value or "").strip()
    return value[:limit] or None


def _is_admin() -> bool:
    key = os.environ.get("ADMIN_KEY", "")
    sent = request.headers.get("X-Admin-Key", "")
    return bool(key) and hmac.compare_digest(key, sent)


@feedback_bp.route("/api/feedback/submit", methods=["POST"])
def submit():
    data = request.get_json() or {}
    user_id = _clean(data.get("user_id"), 80)
    user_type = data.get("user_type")
    message = _clean(data.get("message"), 1000)
    rating = data.get("rating")
    if not user_id or user_type not in feedback_store.USER_TYPES:
        return jsonify({"error": "Missing or invalid user_id/user_type"}), 400
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 5:
        return jsonify({"error": "Rating must be 1 to 5"}), 400
    if not message or len(message) < 10:
        return jsonify({"error": "Please write at least a sentence of feedback"}), 400

    # Name is optional: None means the visitor chose to stay anonymous.
    display_name = _clean(data.get("display_name"), 80) if data.get("show_name") else None
    role = _clean(data.get("role"), 80)
    try:
        feedback_store.add(user_id, user_type, display_name, role, rating, message)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"ok": True})


@feedback_bp.route("/api/feedback/list")
def list_approved():
    items = feedback_store.list_by_status("approved")
    for i in items:
        i.pop("status", None)
    avg = round(sum(i["rating"] for i in items) / len(items), 1) if items else None
    return jsonify({"items": items, "count": len(items), "avg_rating": avg})


@feedback_bp.route("/api/feedback/admin/pending")
def admin_pending():
    if not _is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"items": feedback_store.list_by_status("pending")})


@feedback_bp.route("/api/feedback/admin/<int:fid>/approve", methods=["POST"])
def admin_approve(fid):
    if not _is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"ok": feedback_store.set_status(fid, "approved")})


@feedback_bp.route("/api/feedback/admin/<int:fid>/reject", methods=["POST"])
def admin_reject(fid):
    if not _is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"ok": feedback_store.set_status(fid, "pending")})


@feedback_bp.route("/api/feedback/admin/<int:fid>", methods=["DELETE"])
def admin_delete(fid):
    if not _is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"ok": feedback_store.delete(fid)})
