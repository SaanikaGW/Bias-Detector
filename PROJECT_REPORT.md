# BIOS Career Check v2.0 — Full Project Report

Covers all three phases of the v2.0 upgrade: what was built, what was verified,
what was deliberately deferred, and what you need to do next.

---

## Phase 1 — Hybrid detection pipeline + web app upgrade

### What was found (Step 0 inventory)
The v1 code claimed a fine-tuned GPT-2 LoRA as the primary detector with a
sentence-transformers semantic layer — but torch was never in
`requirements.txt`, so in production both silently failed and detection fell
back to **a keyword list**: exact CSV span matching + ~60 regex patterns. It
failed 4 of the 10 spec calibration cases (flagged "aggressive sales
strategy", missed "rockstar", "native English speaker", "work hard, play hard").

### What was built
| Piece | File(s) | Notes |
|---|---|---|
| Layer 1 — rules | `detection/patterns.json` (173 patterns, 9 categories), `detection/layer1.py` | Severity + confidence per pattern, veto guards ("competitive **salary**" never flags), ambiguity marks, inclusive-signal detection, research notes per category. Editable JSON, regenerable via `ml/gen_patterns.py`. |
| Layer 2 — contextual classifier | `detection/layer2.py`, artifacts `layer2_model.npz` + `layer2_vocab.json`, trainer `ml/train_layer2.py` | TF-IDF word 1–3 grams + logistic regression, two heads (biased/none decision + category). Runtime is numpy-only — no torch, no sklearn, no network. Adjudicates ambiguous rule hits from sentence context and scans rule-free sentences at a high threshold. Numpy↔sklearn parity asserted at train time (Δ ≈ 1e-8). |
| Layer 3 — LLM | `detection/layer3.py` | Explanations, research rationale, rewrites. Strictly downstream — never adds/removes flags. Falls back to built-in templates when no `OPENAI_API_KEY`, so detection works fully offline. |
| Orchestrator + scoring | `detection/pipeline.py` | Transparent formulas: bias = Σ severity×confidence×category-weight (capped 100); inclusive = 60 + signals(≤40) − 0.8×penalty. Full derivation returned by the API. |
| API | `app.py` | `/api/bias-reducer/analyze` returns the v2 contract **plus** every v1 field. All 6 routes preserved. `detector.py` is now a compatibility shim. |
| Tests | `tests/test_calibration.py` | 6 groups: calibration through the pipeline, calibration through Layer 2 alone, per-issue contract, score derivability, legacy-field regression, clean-JD sanity. |
| Web app | `frontend/src/App.jsx` | New results view (two score dials, inline highlights, issue cards with severity/confidence/research/rewrites, copy + report download), homepage what/why/how + example result, light/dark theme, keyboard focus styles, reduced-motion support, responsive grids. Branding preserved. |

### Verified
Calibration 10/10 through the pipeline AND through Layer 2 alone (exact
calibration sentences held out of training). All API routes green via test
client, offline end-to-end works, ESLint clean.

### Deviations (flagged at the time)
- Layer 2 is **not** OpenAI embeddings (your original pick): the build
  sandbox couldn't reach api.openai.com or HuggingFace, so I built the spec's
  first-listed option (local trained classifier) — fully verified, zero
  runtime cost. Swapping backends later = retraining only.
- Layer 2 category naming is ~0.66 accurate when the classifier (not a rule)
  originates a flag. Most flags come from rules with exact categories.
- Training deliberately excluded `gender_bias_dataset_final_fixed.csv`
  (templated toy data that teaches bad lessons).

---

## Phase 2 — Evaluation harness

### What was built
- `eval/data/eval_dataset.csv` — 157 labeled examples: calibration,
  fp_trap, fn_trap, subtle_bias, explicit_bias, masculine/feminine-coded,
  neutral, inclusive, plus the held-out test split. Overlap against training
  data is checked at build time (`eval/build_dataset.py`).
- `eval/legacy_detector.py` — faithful reproduction of v1 production behavior.
- `eval/run_eval.py` — runs both systems (LLM disabled), writes checked-in
  artifacts `eval/metrics.json` + `eval/report.md`. Entry point: `make eval`.

### Results (checked into `eval/report.md`)
| Metric | Legacy (v1) | Hybrid (v2) |
|---|---:|---:|
| Accuracy | 0.694 | 0.955 |
| Precision | 0.784 | 1.000 |
| Recall | 0.704 | 0.929 |
| F1 | 0.742 | 0.963 |
| False-positive rate | 0.322 | 0.000 |
| Avg confidence when flagging | n/a | 0.862 |

The eval loop drove 16 new Layer 1 patterns (fixing real gaps: "Male sales
representative required", "Applicants over 40 need not apply", "No mothers of
young children", "delivery boy", height requirements, spelled-out year counts).

### Honest caveats (also inside the report)
- Layer 1 rules were iterated against this dataset → development metrics, not
  a blind test. Layer 2 never trained on any eval row.
- 7 remaining false negatives are listed verbatim in the report — hard
  paraphrases ("your calendar should belong to the company") that need more
  Layer 2 training data, not more regexes.
- The `holdout_test` slice flatters legacy (its span matching memorized parts
  of that data by construction).

---

## Phase 3 — Chrome extension (Manifest V3)

### What was built (`extension/`)
- **Popup**: Analyze Current Page / Analyze Selected Text, compact scores +
  top issues. **Side panel**: manual paste, both scores, inline highlighted
  phrases, issue cards (severity, confidence, research, rewrites), Copy
  Improved Version, Download Report. **Context menu**: right-click a
  selection → analyze. **Offline state** with retry when the backend is down.
- **Extraction** (`lib/extract.js`), on demand only: LinkedIn job-description
  selectors; Google Docs via same-origin plain-text export (DOM scraping is
  impossible — canvas); generic fallback (selection → main/article → body).
- **Shared module, not a fork**: detection stays server-side; both the web
  app and extension import `/shared/biosClient.js` (API client, error
  normalization, report builder). Synced copies via `make ext`, drift-checked
  by `make check-shared`. App.jsx's duplicate code was deleted.
- **Minimal permissions**: `activeTab, scripting, sidePanel, contextMenus,
  storage`; `host_permissions: []`. Justified per-permission in
  `extension/PERMISSIONS.md`. Privacy policy stub in `extension/PRIVACY.md`.

### Verified here
Manifest validates; all modules pass syntax check; esbuild bundling proves
every import resolves; `make check-shared` passes; calibration still 6/6.

### Not verified here (needs you — see checklist)
Live runs on LinkedIn and Google Docs; loading unpacked in your Chrome.
Greenhouse / Lever / Ashby / Workday / Gmail / Notion / Word Online have no
dedicated extractors yet (generic fallback usually works) — deliberately
deferred until the first two are verified live, per the plan.

---

## YOUR TO-DO LIST

### 1. Commit and push
I edited files but can't commit from this environment. Note: there may be a
stale `.git/index.lock` — if `git commit` complains, delete that file first.
```bash
cd ~/Desktop/Bias-Detector
rm -f .git/index.lock          # only if git complains about a lock
git add -A && git commit -m "v2.0: hybrid detection pipeline, eval harness, MV3 extension"
git push
```

### 2. Verify the frontend build locally (2 min)
The sandbox couldn't run vite's native binary; ESLint/esbuild passed but do a
real build before deploying:
```bash
cd frontend && npm install && npm run build && npm run dev  # click around
```

### 3. Deploy the backend (Railway)
- Push triggers the deploy; `requirements.txt` is now leaner (no
  pandas/torch/sklearn at runtime).
- Make sure `OPENAI_API_KEY` is set as a Railway **environment variable**
  (the local `.env` doesn't deploy). Without it, everything still works but
  explanations use templates instead of tailored LLM prose.
- Smoke-test: `curl -X POST https://<your-railway>/api/bias-reducer/analyze
  -H 'Content-Type: application/json' -d '{"text":"Rockstar salesman wanted"}'`

### 4. Deploy the frontend (Vercel)
Auto-deploys on push. Confirm `VITE_API_BASE_URL` is still set in Vercel
project settings to your Railway URL.

### 5. Load and verify the extension (10 min)
1. `chrome://extensions` → Developer mode → **Load unpacked** → select
   `extension/`.
2. Side panel → Settings → set API address to your Railway URL → Save.
3. Open a real LinkedIn job posting → toolbar icon → **Analyze Current Page**.
4. Open a Google Doc containing a JD → **Analyze Current Page**.
5. If either fails, copy the exact error text back to me — most likely fix is
   a selector update in `extension/lib/extract.js`.

### 6. Optional / later
- **Repo cleanup**: `gender-bias-gpt2*/`, `gender-bias-llama/`,
  `gender_bias_dataset/`, `Miniforge3.sh` are large and unused by v2 —
  delete or `.gitignore` them (your call; nothing references them).
- **Recall improvements**: add training rows to `ml/curated_labeled.csv` for
  the 7 documented misses → `make train` → `make eval` to measure.
- **OpenAI-embedding Layer 2** (your original preference): runnable from your
  machine — needs a modified `ml/train_layer2.py` backend; ask me when ready.
- **Chrome Web Store submission** (if ever): needs PNG icons (not required
  for unpacked loading — currently none), a reviewed PRIVACY.md, and dedicated
  extractors for the remaining sites.
- **Maintenance rule**: edit `/shared/biosClient.js` → run `make ext`;
  edit patterns via `ml/gen_patterns.py` → regenerate → `make test && make eval`.
