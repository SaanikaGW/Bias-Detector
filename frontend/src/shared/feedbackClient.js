/**
 * shared/feedbackClient.js — testimonials / feedback API client.
 * Best-effort helpers; the UI handles errors it cares about (submit/list).
 */

const STATE_KEY = "bios_feedback_state"; // "submitted" | "dismissed" | null

const base = (api) => (api || "").replace(/\/+$/, "");

export function getFeedbackState() {
  try { return localStorage.getItem(STATE_KEY); } catch { return null; }
}

export function setFeedbackState(value) {
  try { localStorage.setItem(STATE_KEY, value); } catch { /* no-op */ }
}

async function parse(res) {
  let body = {};
  try { body = await res.json(); } catch { /* no-op */ }
  if (!res.ok) throw new Error(body.error || `Request failed (HTTP ${res.status}).`);
  return body;
}

export async function submitFeedback(api, payload) {
  const res = await fetch(`${base(api)}/api/feedback/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parse(res);
}

export async function fetchTestimonials(api) {
  const res = await fetch(`${base(api)}/api/feedback/list`);
  return parse(res);
}

const adminHeaders = (key) => ({ "X-Admin-Key": key });

export async function adminPending(api, key) {
  return parse(await fetch(`${base(api)}/api/feedback/admin/pending`, { headers: adminHeaders(key) }));
}

export async function adminApprove(api, key, id) {
  return parse(await fetch(`${base(api)}/api/feedback/admin/${id}/approve`, { method: "POST", headers: adminHeaders(key) }));
}

export async function adminDelete(api, key, id) {
  return parse(await fetch(`${base(api)}/api/feedback/admin/${id}`, { method: "DELETE", headers: adminHeaders(key) }));
}
