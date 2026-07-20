#!/usr/bin/env python3
"""
Build eval/data/eval_dataset.csv — the labeled evaluation set for the
detection pipeline (Phase 2).

Composition (per the v2.0 spec):
- calibration:        the 10 spec calibration sentences (never trained on)
- holdout_test:       the stratified test split from training (in-distribution;
                      NOTE: legacy CSV-span matching partially memorized this
                      data in v1 production, so legacy scores are flattered here
                      — the per-type breakdown in the report makes that visible)
- fp_trap:            benign texts containing trigger-adjacent words
- fn_trap:            biased texts phrased to slip past keyword lists
- subtle_bias:        biased persona/lifestyle phrasing without slur words
- explicit_bias:      unambiguous violations
- masculine_coded /
  feminine_coded:     coded trait language
- neutral:            ordinary requirement sentences
- inclusive:          texts with active inclusion signals

All handwritten rows are checked for overlap against train/val data;
any overlap aborts the build.

Columns: Text, expected (biased|none), category, example_type
"""
import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ROWS: list[tuple[str, str, str, str]] = []


def a(example_type: str, expected: str, category: str, *texts: str) -> None:
    for t in texts:
        ROWS.append((t, expected, category, example_type))


# ── calibration (spec acceptance set) ─────────────────────────────────────────
CAL = [
    ("Aggressive sales strategy", "none", "none"),
    ("Aggressive personality required", "biased", "masculine_coded"),
    ("Strong communication skills", "none", "none"),
    ("Rockstar engineer", "biased", "masculine_coded"),
    ("Native English speaker", "biased", "exclusionary"),
    ("Professional English proficiency", "none", "none"),
    ("Chairman", "biased", "gendered_language"),
    ("Chairperson", "none", "none"),
    ("Work hard, play hard", "biased", "exclusionary"),
    ("Collaborative team environment", "none", "none"),
]
for text, expected, cat in CAL:
    ROWS.append((text, expected, cat, "calibration"))

# ── false-positive traps (expected: clean) ────────────────────────────────────
a("fp_trap", "none", "none",
  "Drive aggressive year-over-year growth in the enterprise segment.",
  "Our pricing is highly competitive within the market.",
  "Emotional intelligence is central to this coaching role.",
  "Competitive compensation, equity, and a 401(k) match.",
  "This role supports our 24/7 network operations center on a rotating shift schedule.",
  "Must be able to lift 20 kg as part of warehouse safety requirements.",
  "Nurture our partner ecosystem through quarterly business reviews.",
  "Fluent English and Spanish required for this bilingual support role.",
  "The chairperson emerita will advise the committee.",
  "Alpha and beta releases ship monthly to production.",
  "Experience running competitive analyses against market leaders.",
  "Physical demands include standing for up to six hours per shift.",
  "You will champion our accessibility roadmap across teams.",
)

# ── false-negative traps (expected: flagged; phrased to dodge keyword lists) ──
a("fn_trap", "biased", "caregiver_bias",
  "Ideal for someone without outside obligations competing for their time.",
  "Candidates should have an unbroken record of full-time employment.",
  "We expect total buy-in: nights, weekends, whatever it takes.",
)
a("fn_trap", "biased", "age_coded",
  "Our team skews young and we like it that way.",
  "Preference for those early in their careers, ideally just out of university.",
  "Stamina to keep up with our twenty-something founders is essential.",
)
a("fn_trap", "biased", "gendered_language",
  "Looking for a gal Friday to keep the office running smoothly.",
  "The right guy for this role has deep Kubernetes experience.",
  "This position suits a man of considerable technical depth.",
)
a("fn_trap", "biased", "exclusionary",
  "You'll fit right in if you thrive on late nights at the office bar.",
  "English so natural that clients assume you grew up speaking it.",
)

# ── subtle bias (expected: flagged) ───────────────────────────────────────────
a("subtle_bias", "biased", "feminine_coded",
  "A quiet, agreeable disposition suits our front desk.",
)
a("subtle_bias", "biased", "stereotype",
  "Command-and-control leadership style preferred.",
)
a("subtle_bias", "biased", "qualification_inflation",
  "Only serious contenders who check every box should reach out.",
)
a("subtle_bias", "biased", "appearance_bias",
  "Someone polished and put-together to represent us at galas.",
)
a("subtle_bias", "biased", "caregiver_bias",
  "Your calendar should belong to the company during launch season.",
)

# ── explicit bias (expected: flagged) ─────────────────────────────────────────
a("explicit_bias", "biased", "gendered_language",
  "Male sales representative required for territory expansion.",
  "The foreman will assign his crew daily quotas.",
  "Chairman wanted for the advisory board.",
)
a("explicit_bias", "biased", "age_coded",
  "Hiring a young saleswoman for our boutique.",
  "Applicants over 40 need not apply.",
)
a("explicit_bias", "biased", "appearance_bias",
  "Attractive female hostesses only.",
)
a("explicit_bias", "biased", "caregiver_bias",
  "No mothers of young children, please.",
)
a("explicit_bias", "biased", "masculine_coded",
  "Rockstar ninja developer with killer instinct wanted.",
)

# ── coded language (expected: flagged) ────────────────────────────────────────
a("masculine_coded", "biased", "masculine_coded",
  "A fearless, hard-charging operator who plays to win.",
  "Hungry, relentless hustlers thrive here.",
  "We want assertive dealmakers with real swagger.",
)
a("feminine_coded", "biased", "feminine_coded",
  "A sweet-tempered assistant with a soothing phone manner.",
  "Warm, bubbly energy at the reception desk at all times.",
)

# ── neutral (expected: clean) ─────────────────────────────────────────────────
a("neutral", "none", "none",
  "Design, build, and maintain CI/CD pipelines.",
  "Partner with product managers on quarterly roadmaps.",
  "Five or more years of experience with distributed systems.",
  "Excellent written communication and documentation habits.",
  "Respond to customer tickets within agreed SLAs.",
  "Willingness to travel up to 10% for team offsites.",
  "Experience with React, TypeScript, and GraphQL.",
  "Prepare monthly financial statements and variance analyses.",
)

# ── inclusive (expected: clean, with positive signals) ────────────────────────
a("inclusive", "none", "none",
  "We are an equal opportunity employer; we welcome applicants of all genders.",
  "If you don't meet every requirement, we still encourage you to apply.",
  "Flexible hours, hybrid work, and generous parental leave.",
  "Salary range: $95,000 to $115,000, shared in the spirit of pay transparency.",
  "Reasonable accommodations are available throughout the interview process.",
  "We hire for skills — a degree or equivalent practical experience both count.",
  "Our chairperson hosts open office hours for all staff.",
)


def main() -> None:
    # Guard: handwritten rows must not appear in training/validation data.
    train = pd.read_csv(os.path.join(HERE, "data", "train.csv"))
    val = pd.read_csv(os.path.join(HERE, "data", "val.csv"))
    seen = set(train["Text"].str.lower().str.strip()) | \
           set(val["Text"].str.lower().str.strip())
    overlap = [t for t, *_ in ROWS if t.lower().strip() in seen]
    if overlap:
        sys.exit(f"ABORT — eval rows present in training data: {overlap}")

    # Held-out test split (in-distribution portion, tagged as such).
    test = pd.read_csv(os.path.join(HERE, "data", "test.csv"))
    for _, r in test.iterrows():
        expected = "none" if r["label"] == "none" else "biased"
        ROWS.append((r["Text"], expected, r["label"], "holdout_test"))

    out = os.path.join(HERE, "data", "eval_dataset.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Text", "expected", "category", "example_type"])
        w.writerows(ROWS)

    df = pd.DataFrame(ROWS, columns=["Text", "expected", "category", "example_type"])
    print(f"wrote {out}: {len(df)} rows")
    print(df.groupby(["example_type", "expected"]).size().to_string())


if __name__ == "__main__":
    main()
