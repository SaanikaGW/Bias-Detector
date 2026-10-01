/**
 * shared/usageMetrics.js — global (site-wide) usage metrics client.
 *
 * Complements fhiHistory.js (which tracks a visitor's OWN history in their
 * own browser). This module talks to the backend's small SQLite-backed
 * metrics store so the "Impact" page can show real, non-fabricated totals
 * across every visitor: how many job descriptions have been analyzed, by
 * what percent they improved, the Fair Hiring Index average, how much
 * difference the bias-aware Hiring AI made, and how many companies vs.
 * individuals have used the tool.
 *
 * Every call here is best-effort: a failed metrics call must never break
 * the tool the visitor is actually using, so every function swallows its
 * own errors.
 */

const USER_ID_KEY = "bios_user_id";
const USER_TYPE_KEY = "bios_user_type"; // "company" | "individual" | null
const COMPANY_NAME_KEY = "bios_company_name";

/** A stable anonymous id for this browser, created once and reused. */
export function getUserId() {
  let id = null;
  try { id = localStorage.getItem(USER_ID_KEY); } catch { /* no-op */ }
  if (!id) {
    id = (crypto?.randomUUID?.() || `u-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    try { localStorage.setItem(USER_ID_KEY, id); } catch { /* no-op */ }
  }
  return id;
}

/** Whether this browser has already answered the company/individual prompt. */
export function hasUsageType() {
  try { return !!localStorage.getItem(USER_TYPE_KEY); } catch { return false; }
}

export function getUsageType() {
  try {
    return {
      userType: localStorage.getItem(USER_TYPE_KEY) || null,
      companyName: localStorage.getItem(COMPANY_NAME_KEY) || null,
    };
  } catch {
    return { userType: null, companyName: null };
  }
}

/** Register this browser as a company or individual (asked once, see App.jsx). */
export async function registerUsageType(apiBase, userType, companyName) {
  const base = (apiBase || "").replace(/\/+$/, "");
  const userId = getUserId();
  try {
    await fetch(`${base}/api/usage/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, user_type: userType, company_name: companyName || null }),
    });
    localStorage.setItem(USER_TYPE_KEY, userType);
    if (companyName) localStorage.setItem(COMPANY_NAME_KEY, companyName);
    else localStorage.removeItem(COMPANY_NAME_KEY);
  } catch {
    // Non-fatal — the event-logging endpoint will still count this user_id
    // as an anonymous individual if registration never went through.
  }
}

/**
 * Log one usage event toward the global totals. Fire-and-forget: never
 * throws, never blocks the UI it's called from.
 *   eventType: "jd_analyzed" | "fhi_submitted" | "hiring_ai_compared"
 */
export function logUsageEvent(apiBase, eventType, payload = {}) {
  const base = (apiBase || "").replace(/\/+$/, "");
  const userId = getUserId();
  try {
    fetch(`${base}/api/metrics/log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, event_type: eventType, payload }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* no-op — metrics are a nice-to-have, never block the real feature */
  }
}

/** Fetch the global aggregate summary for the Impact page. */
export async function fetchUsageSummary(apiBase) {
  const base = (apiBase || "").replace(/\/+$/, "");
  const res = await fetch(`${base}/api/metrics/summary`);
  if (!res.ok) throw new Error(`Could not load impact stats (HTTP ${res.status}).`);
  return res.json();
}
