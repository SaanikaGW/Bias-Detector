// AUTO-SYNCED from /shared/biosClient.js — edit that file, then run: make ext
/**
 * shared/biosClient.js — the SINGLE shared client module for BIOS Career Check.
 *
 * Imported by BOTH:
 *   - the web app   (frontend/src/App.jsx)
 *   - the extension (extension/shared/biosClient.js is an auto-synced copy;
 *     run `make ext` after editing this file — see Makefile)
 *
 * Detection itself lives server-side (detection/ Python package). Both
 * clients call the SAME endpoint, so there is exactly one detection pipeline;
 * this module is the one place that knows how to call it, interpret errors,
 * and turn results into a report.
 */

/**
 * Analyze text against the BIOS detection API.
 * Returns the v2 result object, or throws a normalized Error with:
 *   err.kind = "offline" | "server" | "invalid"
 */
export async function analyzeText(apiBase, text, { timeoutMs = 30000 } = {}) {
  const base = (apiBase || "").replace(/\/+$/, "");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(`${base}/api/bias-reducer/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: ctrl.signal,
    });
  } catch (e) {
    const err = new Error(
      "Can't reach the BIOS analysis service. Check your connection " +
      "and the API address, then try again.");
    err.kind = "offline";
    err.cause = e;
    throw err;
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const err = new Error(data?.error || `Analysis failed (HTTP ${res.status}).`);
    err.kind = res.status >= 500 ? "server" : "invalid";
    throw err;
  }
  return data;
}

/** Severity → display color (shared between web app and extension). */
export const SEV_COLORS = { low: "#10B981", medium: "#F59E0B", high: "#F43F5E" };

/** Category → display color. */
export const CAT_COLORS = {
  gendered_language:       "#F43F5E",
  masculine_coded:         "#F59E0B",
  feminine_coded:          "#EC4899",
  stereotype:              "#F97316",
  caregiver_bias:          "#8B5CF6",
  age_coded:               "#EAB308",
  appearance_bias:         "#14B8A6",
  exclusionary:            "#38BDF8",
  qualification_inflation: "#A78BFA",
};
export const catColor = (cat) => CAT_COLORS[cat] || "#0EA5E9";

/** Build the downloadable Markdown report from an analysis result. */
export function buildReport(text, result) {
  const s = result.scores || {};
  const lines = [
    "# BIOS Career Check — Bias Analysis Report",
    `Generated: ${new Date().toISOString().slice(0, 10)}`,
    "",
    `**Gender Bias Score:** ${s.gender_bias_score ?? Math.round((result.bias_score || 0) * 100)}/100 (${result.bias_level} bias — lower is better)`,
    `**Inclusive Language Score:** ${s.inclusive_language_score ?? "—"}/100 (higher is better)`,
    "",
    `## Issues found (${(result.issues || []).length})`,
    "",
  ];
  (result.issues || []).forEach((i, n) => {
    lines.push(`### ${n + 1}. “${i.span}”`);
    lines.push(`- Category: ${i.category_label || i.category} | Severity: ${i.severity} | Confidence: ${Math.round(i.confidence * 100)}%`);
    if (i.explanation) lines.push(`- Why it was flagged: ${i.explanation}`);
    if (i.impact) lines.push(`- Why it may discourage applicants: ${i.impact}`);
    if (i.research_rationale) lines.push(`- Research context: ${i.research_rationale}`);
    if (i.rewrite) lines.push(`- Suggested rewrite: ${i.rewrite}`);
    if (i.expected_improvement) lines.push(`- Expected improvement: ${i.expected_improvement}`);
    lines.push("");
  });
  if ((result.inclusive_signals || []).length) {
    lines.push("## Inclusive signals already present");
    result.inclusive_signals.forEach(sig => lines.push(`- ${sig.label} (“${sig.span}”)`));
    lines.push("");
  }
  if (s.derivation) {
    lines.push("## How the scores were derived");
    lines.push(s.derivation.formula || "");
    lines.push("");
  }
  lines.push("## Original job description", "", text, "",
             "## Improved job description", "", result.rewritten_jd || "");
  return lines.join("\n");
}

/** Trigger a browser download of the Markdown report. */
export function downloadReport(text, result, filename = "bias-analysis-report.md") {
  const blob = new Blob([buildReport(text, result)], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
