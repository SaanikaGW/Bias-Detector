# BIOS Career Check

**Grammarly for gender-inclusive hiring.** Paste a job description; BIOS Career Check
detects gender-coded language, explains why each phrase may discourage qualified
applicants (research-backed), and generates an inclusive rewrite that preserves every
real job requirement.

- Web app: https://bios-carear-checkers.vercel.app/ (Vercel, `frontend/`)
- API: Flask on Railway (`app.py`)

## What changed in v2.0

**Detection was rebuilt as a hybrid three-layer pipeline** (`detection/`):

| Layer | What it does | What it is |
|---|---|---|
| 1 — Rules | Flags candidate phrases, applies veto guards ("competitive **salary**" never flags), marks ambiguous terms | 157 curated, severity-tagged patterns in `detection/patterns.json` across 9 bias categories |
| 2 — Context classifier | **The decision-maker for ambiguity.** Adjudicates ambiguous rule hits from full-sentence context and scans rule-free sentences for subtle bias | Trained TF-IDF (word 1–3 grams) + logistic regression; numpy-only runtime (`detection/layer2.py`), retrainable via `python ml/train_layer2.py` |
| 3 — LLM | Explanations, research rationale, inclusive rewrites — **strictly downstream, never adds/removes flags**; falls back to built-in templates when no API key | `detection/layer3.py` (gpt-4.1-nano) |

Honest note on v1: the old `detector.py` claimed a fine-tuned GPT-2 as primary
detector, but torch was never in `requirements.txt`, so in production it silently
fell back to keyword/regex matching. v2.0 replaces that with the pipeline above and
keeps `detector.py` as a thin compatibility shim.

**Why TF-IDF + logistic regression for Layer 2** (not embeddings or an LLM): word
n-grams carry the local context that distinguishes "aggressive personality required"
(flagged) from "aggressive sales strategy" (clean); it is deterministic, testable,
costs nothing per call, and the exported artifact (`layer2_model.npz` +
`layer2_vocab.json`) is plain arrays — no torch, no pickle-version risk, no network.
`ml/train_layer2.py` documents the trade-offs and asserts numpy/sklearn parity.

**Scoring is transparent** (`detection/pipeline.py`): Gender Bias Score = capped sum
of severity × confidence × category-weight per issue; Inclusive Language Score
rewards signals like EEO statements, pay transparency, and flexible work. The API
returns the full derivation so the UI can show its math.

**Per-issue output contract**: phrase, category, severity, confidence, explanation,
why it may discourage applicants, research rationale, suggested rewrite, expected
improvement, and character offsets for inline highlighting.

**API**: `/api/bias-reducer/analyze` returns the v2 contract *plus* every v1 field
(`bias_score`, `bias_level`, `categories`, `highlights`, `suggestions`,
`rewritten_jd`) — no consumer breaks. All other routes are unchanged.

**Web app**: upgraded homepage (what/why/how + example result), results view with
both scores, inline highlights, per-issue cards with research and rewrites, copy
improved JD, downloadable Markdown report, light/dark theme, keyboard-focus styles,
`prefers-reduced-motion` support, responsive grids.

## Running locally

```bash
# Backend (Python 3.10+)
pip install -r requirements.txt
python app.py                        # http://localhost:5001

# Frontend
cd frontend && npm install && npm run dev

# Tests (calibration acceptance + contract + regression)
python tests/test_calibration.py

# Retrain Layer 2 (only if you edit the datasets)
pip install -r ml/requirements-train.txt
python ml/train_layer2.py

# Evaluation harness (Phase 2) — before/after vs the v1 detector
make eval          # or: python eval/run_eval.py
```

## Evaluation (Phase 2)

`eval/` contains a real, runnable evaluation harness — not a claim of accuracy:

- `eval/data/eval_dataset.csv` — 157 labeled examples across calibration,
  false-positive traps, false-negative traps, subtle bias, explicit bias,
  masculine/feminine-coded, neutral, and inclusive types.
- `eval/legacy_detector.py` — faithful reproduction of what v1 actually ran in
  production (dataset span matching + regex keywords).
- `eval/run_eval.py` — runs both systems (LLM disabled; decisions are Layers
  1+2 only) and writes `eval/metrics.json` + `eval/report.md`.

Checked-in results (see `eval/report.md` for confusion matrices, per-type
breakdown, caveats, and the list of remaining misses):

| Metric | Legacy (v1) | Hybrid (v2) |
|---|---:|---:|
| Accuracy | 0.694 | 0.955 |
| Precision | 0.784 | 1.000 |
| Recall | 0.704 | 0.929 |
| F1 | 0.742 | 0.963 |
| False-positive rate | 0.322 | 0.000 |
| False-negative rate | 0.296 | 0.071 |

Caveat, stated plainly: Layer 1 rules were iterated against this dataset, so
these are development metrics, not a blind test. The Layer 2 classifier never
saw any eval row during training.

`OPENAI_API_KEY` in `.env` enables LLM explanations/rewrites (Layer 3). Without it,
the app still works end-to-end using template explanations — detection is identical.

## Data

- `detection/patterns.json` — editable Layer 1 source of truth (patterns, veto
  guards, severities, alternatives, category research notes).
- `ml/curated_labeled.csv` — curated Layer 2 training data (calibration paraphrases,
  false-positive traps). The spec's exact calibration sentences are held out of
  training; they live in `tests/test_calibration.py` as the acceptance bar.
- `bios_check_dataset_full.csv` — 329 labeled JD sentences (v1 dataset, still used
  for training).
- `eval/data/{train,val,test}.csv` — stratified splits written by the training
  script; seed of the Phase 2 evaluation harness.
- `gender_bias_dataset_final_fixed.csv` — v1 toy data (templated sentences);
  intentionally **not** used by v2 training. Kept for reference.

## Chrome extension (Phase 3)

`extension/` is a Manifest V3 extension that reuses the same detection
pipeline as the web app — both import the shared client module
(`/shared/biosClient.js`, synced by `make ext`, drift-checked by
`make check-shared`) and call the same `/api/bias-reducer/analyze` endpoint.
Features: popup + side panel, Analyze Current Page / Selected Text / Manual
Paste, both scores, inline highlights, severity + confidence, research
explanations, rewrites, Copy Improved Version, Download Report, offline
state. Minimal permissions (no host permissions at all — justified in
`extension/PERMISSIONS.md`); privacy policy in `extension/PRIVACY.md`.
Load and verification instructions: `extension/README.md`.

## Repo layout

```
app.py                  Flask API (all v1 routes preserved)
detection/              v2 hybrid pipeline (layer1/2/3, pipeline, patterns.json, model artifacts)
shared/                 shared JS client module (web app + extension; make ext syncs)
extension/              Chrome extension (MV3) — see extension/README.md
ml/                     Layer 2 training script + curated data + metrics
tests/                  calibration acceptance tests
eval/                   evaluation harness + checked-in before/after report
frontend/               React + Vite web app
agents.py               PII stripping + fit evaluation for the Hiring AI pages
detector.py             deprecated shim → detection.pipeline
```
