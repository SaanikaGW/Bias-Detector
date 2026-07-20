/**
 * Result rendering shared by the popup and the side panel.
 * All dynamic strings are inserted via textContent (never innerHTML) so page
 * or API content can't inject markup into the extension UI.
 */
import { SEV_COLORS, catColor } from "../shared/biosClient.js";

export function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

export function scoreBadge(label, value, invert = false) {
  const good = invert ? value >= 75 : value < 25;
  const mid = invert ? value >= 45 : value < 60;
  const color = good ? "#10B981" : mid ? "#F59E0B" : "#F43F5E";
  const wrap = el("div", "score");
  const num = el("div", "score-num", String(value));
  num.style.color = color;
  wrap.append(num, el("div", "score-label", label));
  return wrap;
}

export function issueCard(issue, { compact = false } = {}) {
  const card = el("div", "issue");
  card.style.borderLeftColor = catColor(issue.category);

  const head = el("div", "issue-head");
  head.append(el("span", "issue-span", `“${issue.span}”`));
  const sev = el("span", "sev", issue.severity.toUpperCase());
  sev.style.color = SEV_COLORS[issue.severity];
  const conf = el("span", "conf", `${Math.round(issue.confidence * 100)}%`);
  conf.title = "Detection confidence";
  head.append(sev, conf);
  card.append(head);

  const cat = el("span", "cat", issue.category_label || issue.category);
  cat.style.color = catColor(issue.category);
  cat.style.borderColor = catColor(issue.category);
  card.append(cat);

  if (!compact) {
    if (issue.explanation) card.append(el("p", "expl", issue.explanation));
    if (issue.rewrite) {
      const rw = el("div", "rewrite");
      rw.append(el("strong", null, "Try instead: "),
                document.createTextNode(issue.rewrite));
      card.append(rw);
    }
    if (issue.research_rationale || issue.impact) {
      const details = el("details", "research");
      details.append(el("summary", null, "Why this matters (research)"));
      if (issue.impact) details.append(el("p", null, issue.impact));
      if (issue.research_rationale) details.append(el("p", null, issue.research_rationale));
      card.append(details);
    }
  }
  return card;
}

/** Original text with flagged phrases wrapped in <mark>, offsets from the API. */
export function highlightedText(text, issues) {
  const box = el("div", "hl-box");
  const spans = (issues || [])
    .filter(i => Number.isInteger(i.start) && Number.isInteger(i.end) && i.end > i.start)
    .sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const iss of spans) {
    if (iss.start < cursor) continue;
    if (iss.start > cursor) box.append(document.createTextNode(text.slice(cursor, iss.start)));
    const mark = el("mark", "hl", text.slice(iss.start, iss.end));
    mark.title = `${iss.category_label || iss.category} — ${iss.severity} severity`;
    box.append(mark);
    cursor = iss.end;
  }
  box.append(document.createTextNode(text.slice(cursor)));
  return box;
}

export function errorBox(err, onRetry) {
  const box = el("div", "error");
  box.append(el("div", "error-title",
    err.kind === "offline" ? "Can't reach the analysis service" : "Analysis failed"));
  box.append(el("p", null, err.message));
  if (err.kind === "offline") {
    box.append(el("p", "hint",
      "Check the API address in Settings below, and that you're online. " +
      "Your text stays on this device until analysis succeeds."));
  }
  if (onRetry) {
    const btn = el("button", "btn", "Retry");
    btn.addEventListener("click", onRetry);
    box.append(btn);
  }
  return box;
}

const DEFAULT_API = "http://localhost:5001";

export async function getApiBase() {
  const { apiBase } = await chrome.storage.sync.get("apiBase");
  return apiBase || DEFAULT_API;
}

export async function setApiBase(v) {
  await chrome.storage.sync.set({ apiBase: (v || DEFAULT_API).trim() });
}
