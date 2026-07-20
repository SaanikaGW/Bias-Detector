# BIOS Career Check — Chrome extension (Manifest V3)

Analyzes job descriptions for gender-coded language using the **same
server-side detection pipeline as the web app** (the popup/panel import
`shared/biosClient.js`, auto-synced from `/shared` — run `make ext` at the
repo root after editing, `make check-shared` verifies no drift).

## Load it (unpacked)

1. Start (or deploy) the backend: `python app.py` → http://localhost:5001
2. Open `chrome://extensions`, enable **Developer mode** (top right).
3. Click **Load unpacked** and select this `extension/` folder.
4. Click the ⚖️ icon → open the side panel → set the API address in Settings
   (defaults to `http://localhost:5001`; use your Railway URL in production).

## Features

Popup: Analyze Current Page, Analyze Selected Text, compact scores + top
issues. Side panel: manual paste, both scores (Gender Bias / Inclusive
Language), inline highlighted phrases, per-issue severity + confidence +
research explanations + rewrites, Copy Improved Version, Download Report.
Right-click any selected text → "Analyze selection with BIOS Check".
If the backend is unreachable, a clear offline state with Retry appears —
nothing is silently dropped.

## Site support — verification checklist

Extraction strategies implemented, in order of specificity:

| Site | Strategy | Status |
|---|---|---|
| LinkedIn job pages | Targeted description-container selectors + job title | Implemented — **needs live verification** (selectors current as of writing; LinkedIn changes markup often) |
| Google Docs | Same-origin plain-text export (`…/export?format=txt`) with your session — DOM scraping is impossible (canvas rendering) | Implemented — **needs live verification** |
| Any other page | Selection > `<main>`/`<article>` > body text (capped 6,000 chars) | Implemented — generic fallback |

Greenhouse, Lever, Ashby, Workday, Gmail, Notion, Word Online: **not yet
implemented as targeted extractors** — the generic fallback usually works on
their job-view pages, but per the project plan these get dedicated selectors
incrementally after LinkedIn + Google Docs are verified live.

To verify: load unpacked → open a real LinkedIn job posting → popup →
"Analyze Current Page" (expect scores + issues from the posting text). Then
open a Google Doc containing a JD and repeat. Note: "Analyze Selected Text"
cannot work inside Google Docs (selection lives in canvas) — use Analyze
Current Page there; the popup error message says the same.

**Known MV3 limitation:** the `activeTab` grant is tied to invoking the
extension (toolbar click / context menu). If the side panel stays open while
you navigate to a *new* page, "Analyze Current Page" from the panel may be
denied on that page — the error message will say so. Clicking the toolbar
icon and analyzing from the popup always works; keeping `host_permissions`
empty is worth this trade-off.

## Files

```
manifest.json        MV3, minimal permissions (justified in PERMISSIONS.md)
background.js        service worker: context menu → side panel
lib/extract.js       on-demand page extraction (LinkedIn / Docs / generic)
lib/render.js        safe DOM rendering (no innerHTML for dynamic strings)
shared/biosClient.js AUTO-SYNCED from /shared — API client + report builder
popup/, sidepanel/   UI
PERMISSIONS.md       why each permission exists
PRIVACY.md           what content is read and where it goes
```
