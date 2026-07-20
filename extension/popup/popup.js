/**
 * Popup: quick analyze + compact summary. Full results live in the side
 * panel; both use the shared client module (extension/shared/biosClient.js,
 * synced from /shared — the same code the web app imports).
 */
import { analyzeText } from "../shared/biosClient.js";
import { extractFromActiveTab, extractSelection } from "../lib/extract.js";
import { el, scoreBadge, issueCard, errorBox, getApiBase } from "../lib/render.js";

const $ = (id) => document.getElementById(id);
const statusBox = $("status");
const resultBox = $("result");

function setBusy(msg) {
  statusBox.replaceChildren(el("div", "spinner"), el("p", "muted", msg));
  resultBox.replaceChildren();
}

function clearStatus() { statusBox.replaceChildren(); }

async function run(getText) {
  try {
    setBusy("Reading the page…");
    const { text, source } = await getText();
    setBusy(`Analyzing ${source === "google-docs" ? "your Google Doc" :
             source === "linkedin" ? "this LinkedIn posting" : "the text"}…`);
    const result = await analyzeText(await getApiBase(), text);
    clearStatus();
    renderCompact(text, result);
    // Hand the full result to the side panel for the detailed view.
    await chrome.storage.session.set({ lastText: text, lastResult: result });
  } catch (err) {
    clearStatus();
    resultBox.replaceChildren(errorBox(err, () => run(getText)));
  }
}

function renderCompact(text, result) {
  const s = result.scores || {};
  const scores = el("div", "scores");
  scores.append(
    scoreBadge("Gender Bias", s.gender_bias_score ?? 0),
    scoreBadge("Inclusive Language", s.inclusive_language_score ?? 0, true),
  );
  resultBox.replaceChildren(scores);

  const issues = result.issues || [];
  resultBox.append(el("p", "muted",
    issues.length
      ? `${issues.length} issue${issues.length > 1 ? "s" : ""} found — top ${Math.min(3, issues.length)} below. Open the side panel for rewrites, research, and the full improved posting.`
      : "No gender-coded language found. Nice work."));
  issues.slice(0, 3).forEach(i => resultBox.append(issueCard(i, { compact: true })));

  if (issues.length) {
    const more = el("button", "btn", "See all results in side panel →");
    more.addEventListener("click", openPanel);
    resultBox.append(more);
  }
}

async function openPanel() {
  const win = await chrome.windows.getCurrent();
  await chrome.sidePanel.open({ windowId: win.id });
  window.close();
}

$("analyze-page").addEventListener("click", () => run(extractFromActiveTab));
$("analyze-selection").addEventListener("click", () => run(extractSelection));
$("open-panel").addEventListener("click", openPanel);
