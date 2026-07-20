# Detection Evaluation — v1 (legacy) vs v2 (hybrid pipeline)

*Generated 2026-07-08 by `python eval/run_eval.py` over `eval/data/eval_dataset.csv` (157 labeled examples). LLM disabled — decisions come from Layers 1+2 only.*

**Legacy** = what actually ran in v1 production: dataset span matching + a regex keyword list (the declared GPT-2/semantic layers never loaded because torch was not deployed).

**Honesty caveat:** v2's Layer 1 rules were iterated against earlier runs of this same dataset (that is what an eval harness is for), so these are *development* metrics, not a blind test. The Layer 2 classifier was never trained on any eval row. Legacy was not tuned at all, but its span-matching memorized parts of the `holdout_test` slice by construction. Treat the trap/subtle slices as the fairest apples-to-apples comparison.

## Headline metrics

| Metric | Legacy (v1) | Hybrid (v2) | Δ |
|---|---:|---:|---:|
| Accuracy | 0.694 | 0.955 | +0.261 |
| Precision | 0.784 | 1.000 | +0.216 |
| Recall | 0.704 | 0.929 | +0.225 |
| F1 | 0.742 | 0.963 | +0.221 |
| False-positive rate | 0.322 | 0.000 | -0.322 |
| False-negative rate | 0.296 | 0.071 | -0.225 |
| Avg confidence when flagging | — | 0.862 | |

## Confusion matrices

### Legacy (v1)

|                | pred: biased | pred: clean |
|----------------|-------------:|------------:|
| **true: biased** | 69 | 29 |
| **true: clean**  | 19 | 40 |


### Hybrid (v2)

|                | pred: biased | pred: clean |
|----------------|-------------:|------------:|
| **true: biased** | 91 | 7 |
| **true: clean**  | 0 | 59 |


## Accuracy by example type

| Example type | n | Legacy | v2 |
|---|---:|---:|---:|
| calibration | 10 | 0.600 | 1.000 |
| explicit_bias | 8 | 0.875 | 1.000 |
| feminine_coded | 2 | 0.500 | 1.000 |
| fn_trap | 11 | 0.455 | 0.727 |
| fp_trap | 13 | 0.462 | 1.000 |
| holdout_test | 90 | 0.744 | 0.978 |
| inclusive | 7 | 0.714 | 1.000 |
| masculine_coded | 3 | 0.333 | 1.000 |
| neutral | 8 | 1.000 | 1.000 |
| subtle_bias | 5 | 0.600 | 0.600 |

*Note: `holdout_test` rows are in-distribution (drawn from the same datasets the systems were built around; legacy's CSV span matching partially memorized them in v1, flattering its score on that slice). The trap/subtle/neutral/inclusive rows are new, out-of-distribution phrasings — the honest stress test.*

## v2 remaining misses (7 false negatives, 0 false positives)

False negatives (biased, not caught):

- (fn_trap/caregiver_bias) “We expect total buy-in: nights, weekends, whatever it takes.”
- (fn_trap/exclusionary) “You'll fit right in if you thrive on late nights at the office bar.”
- (fn_trap/exclusionary) “English so natural that clients assume you grew up speaking it.”
- (subtle_bias/appearance_bias) “Someone polished and put-together to represent us at galas.”
- (subtle_bias/caregiver_bias) “Your calendar should belong to the company during launch season.”
- (holdout_test/masculine_coded) “We want a champion who will lead from the front and inspire through action.”
- (holdout_test/caregiver_bias) “We will not be able to guarantee a return-to-work role following any maternity absence.”

False positives (clean, wrongly flagged):


## Reproduce

```bash
python eval/build_dataset.py   # rebuild dataset (idempotent)
python eval/run_eval.py        # writes eval/metrics.json + eval/report.md
```
