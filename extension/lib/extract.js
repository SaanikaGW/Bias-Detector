/**
 * Page-content extraction (runs on demand, never automatically).
 *
 * chrome.scripting.executeScript injects `pageExtractor` into the ACTIVE tab
 * only, and only after an explicit user click (activeTab grant). Nothing is
 * read in the background; nothing runs on pages the user didn't invoke us on.
 *
 * Site support status (see extension/README.md for the verification list):
 * - LinkedIn job pages: targeted selectors for the job description container.
 * - Google Docs: DOM scraping is impossible (canvas rendering), so we fetch
 *   the document's plain-text export from the SAME origin with the user's
 *   own cookies — the text never touches any third party.
 * - Everything else: selection > <main>/<article> > body text (capped).
 */

const MAX_CHARS = 6000;

// Self-contained function serialized into the page. No closures allowed.
function pageExtractor() {
  const MAX = 6000;
  const clean = (s) => (s || "").replace(/\s+\n/g, "\n").replace(/[ \t]+/g, " ").trim();

  // 1. An explicit selection always wins.
  const sel = window.getSelection && window.getSelection().toString();
  if (sel && sel.trim().length > 30) {
    return Promise.resolve({ source: "selection", text: clean(sel).slice(0, MAX) });
  }

  // 2. Google Docs: text lives in <canvas>; use the same-origin plain-text
  //    export with the user's own session. Works for docs they can open.
  const docMatch = location.hostname === "docs.google.com" &&
    location.pathname.match(/\/document\/d\/([^/]+)/);
  if (docMatch) {
    return fetch(`/document/d/${docMatch[1]}/export?format=txt`,
                 { credentials: "same-origin" })
      .then(r => r.ok ? r.text() : Promise.reject(new Error(`export HTTP ${r.status}`)))
      .then(t => ({ source: "google-docs", text: clean(t).slice(0, MAX) }))
      .catch(e => ({ source: "google-docs", text: "", error: String(e) }));
  }

  // 3. LinkedIn job pages: known description containers, newest first.
  if (location.hostname.endsWith("linkedin.com")) {
    const titleEl = document.querySelector(
      ".job-details-jobs-unified-top-card__job-title, .top-card-layout__title, h1");
    const bodyEl = document.querySelector(
      ".jobs-description__content, .jobs-description-content__text, " +
      "#job-details, .jobs-box__html-content, .description__text, article");
    if (bodyEl) {
      const text = clean(`${titleEl ? titleEl.innerText + "\n\n" : ""}${bodyEl.innerText}`);
      return Promise.resolve({ source: "linkedin", text: text.slice(0, MAX) });
    }
  }

  // 4. Generic fallback: main content region, else the whole body.
  const main = document.querySelector("main, article, [role='main']");
  const text = clean((main || document.body).innerText);
  return Promise.resolve({ source: "generic", text: text.slice(0, MAX) });
}

/** Extract text from the active tab. Returns {source, text} or throws. */
export async function extractFromActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("No active tab.");
  if (/^(chrome|edge|about|chrome-extension):/.test(tab.url || "")) {
    throw new Error("This page can't be analyzed. Try a job posting, or paste the text.");
  }
  let results;
  try {
    results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: pageExtractor,
    });
  } catch (e) {
    throw new Error("Couldn't read this page (" + e.message + "). Paste the text instead.");
  }
  const out = results?.[0]?.result;
  if (out?.error) {
    throw new Error("Google Docs export failed (" + out.error + "). " +
                    "Make sure you can open the document, or paste the text.");
  }
  if (!out?.text || out.text.length < 30) {
    throw new Error("No readable job description found on this page. " +
                    "Select the text first, or paste it manually.");
  }
  return out;
}

/** Extract only the current selection from the active tab. */
export async function extractSelection() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("No active tab.");
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection ? window.getSelection().toString() : "",
  });
  const text = (results?.[0]?.result || "").trim();
  if (text.length < 30) {
    throw new Error("Select at least a sentence or two, then try again. " +
                    "(Selection can't be read inside Google Docs — use Analyze Page there.)");
  }
  return { source: "selection", text: text.slice(0, MAX_CHARS) };
}
