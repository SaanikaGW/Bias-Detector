/**
 * Side panel: full analysis experience — manual paste, page/selection
 * analysis, both scores, inline highlights, issue cards with research,
 * copy improved version, download report.
 * Uses the SAME shared client module as the web app (synced from /shared).
 */
import { analyzeText, buildReport } from "../shared/biosClient.js";
import { extractFromActiveTab, extractSelection } from "../lib/extract.js";
import { el, scoreBadge, issueCard, highlightedText, errorBox,
         getApiBase, setApiBase } from "../lib/render.js";

const $ = (id) => document.getElementById(id);
const statusBox = $("status");
const resultBox = $("result");

let current = { text: "", result: null };

function setBusy(msg) {
  statusBox.replaceChildren(el("div", "spinner"), el("p", "muted", msg));
}
function clearStatus() { statusBox.replaceChildren(); }

async function run(getText) {
  try {
    setBusy("Reading…");
    const { text } = await getText();
    setBusy("Analyzing for gender-coded language…");
    const result = await analyzeText(await getApiBase(), text);
    current = { text, result };
    clearStatus();
    render();
  } catch (err) {
    clearStatus();
    resultBox.replaceChildren(errorBox(err, () => run(getText)));
  }
}

function render() {
  const { text, result } = current;
  const s = result.scores || {};
  resultBox.replaceChildren();

  const scores = el("div", "scores");
  scores.append(
    scoreBadge("Gender Bias (lower is better)", s.gender_bias_score ?? 0),
    scoreBadge("Inclusive Language (higher is better)",
               s.inclusive_language_score ?? 0, true),
  );
  resultBox.append(scores);

  if ((result.inclusive_signals || []).length) {
    const sig = el("div", "signals");
    result.inclusive_signals.forEach(x => sig.append(el("span", "signal", `✓ ${x.label}`)));
    resultBox.append(el("p", "muted", "Inclusive signals already present:"), sig);
  }

  // Tabs: Issues / Highlighted / Improved
  const tabs = el("div", "tabs");
  const panes = {
    issues: el("div"),
    highlighted: el("div"),
    improved: el("div"),
  };
  const tabDefs = [
    ["issues", `⚑ Issues (${(result.issues || []).length})`],
    ["highlighted", "🔍 Highlighted"],
    ["improved", "✏️ Improved"],
  ];
  let active = "issues";
  function selectTab(name) {
    active = name;
    tabs.querySelectorAll(".tab").forEach(b =>
      b.setAttribute("aria-selected", b.dataset.tab === name));
    Object.entries(panes).forEach(([k, p]) =>
      p.style.display = k === name ? "" : "none");
  }
  tabDefs.forEach(([name, label]) => {
    const b = el("button", "tab", label);
    b.dataset.tab = name;
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => selectTab(name));
    tabs.append(b);
  });
  resultBox.append(tabs, panes.issues, panes.highlighted, panes.improved);

  const issues = result.issues || [];
  if (issues.length) issues.forEach(i => panes.issues.append(issueCard(i)));
  else panes.issues.append(el("p", "muted", "No gender-coded language found. Nice work — post away."));

  panes.highlighted.append(highlightedText(text, issues));

  const improved = el("div", "hl-box", result.rewritten_jd || "(no rewrite available)");
  panes.improved.append(improved);

  const actions = el("div", "row");
  actions.style.marginTop = "10px";
  const copyBtn = el("button", "btn", "⧉ Copy Improved Version");
  copyBtn.disabled = !result.rewritten_jd;
  copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(result.rewritten_jd || "");
    copyBtn.textContent = "✓ Copied";
    setTimeout(() => (copyBtn.textContent = "⧉ Copy Improved Version"), 1800);
  });
  const dlBtn = el("button", "btn ghost", "⭳ Download Report");
  dlBtn.addEventListener("click", () => {
    const blob = new Blob([buildReport(text, result)], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bias-analysis-report.md";
    a.click();
    URL.revokeObjectURL(url);
  });
  actions.append(copyBtn, dlBtn);
  resultBox.append(actions);

  selectTab(active);
}

// ── wire up ───────────────────────────────────────────────────────────────────
$("analyze-page").addEventListener("click", () => run(extractFromActiveTab));
$("analyze-selection").addEventListener("click", () => run(extractSelection));
$("analyze-paste").addEventListener("click", () => {
  const text = $("manual").value.trim();
  if (text.length < 30) {
    resultBox.replaceChildren(
      errorBox({ message: "Paste at least a sentence or two first." }));
    return;
  }
  run(async () => ({ source: "paste", text }));
});

$("save-api").addEventListener("click", async () => {
  await setApiBase($("api-base").value);
  $("api-saved").textContent = "Saved ✓";
  setTimeout(() => ($("api-saved").textContent = ""), 1800);
});

(async function init() {
  $("api-base").value = await getApiBase();
  // Pending text from the context menu, or last popup result.
  const { pendingText, lastText, lastResult } =
    await chrome.storage.session.get(["pendingText", "lastText", "lastResult"]);
  if (pendingText) {
    await chrome.storage.session.remove(["pendingText", "pendingSource"]);
    $("manual").value = pendingText;
    run(async () => ({ source: "selection", text: pendingText }));
  } else if (lastText && lastResult) {
    current = { text: lastText, result: lastResult };
    render();
  }
})();
