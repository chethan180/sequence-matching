# When do both people say yes? Pair-level probabilities by feature and by merged features

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Written for: the team deciding which feature combinations are worth modelling.

Script `pair_analysis.py` → `results/pair_analysis.log`, `results/pair_analysis.json`, and `results/pair_cells.csv` (all 3,492 merged cells tested). Data: `results/pair_rows.csv.gz`, one row per introduced pair with the raw attributes of both people as observed on the decision day (`collect_pairs.py`).

## Data and protocol
- 54,717 introduced pairs were collected from 720 rollouts. Identical introductions replayed by different policies in the same world (same pair, same day, same outcome) were removed, leaving **37,094 unique introductions**.
- Split by world: train seeds 1001–1030 (27,653 pairs), test seeds 1031–1040 (9,441 pairs). Merged cells are **discovered on train** (z-test, Benjamini–Hochberg FDR 5%, at least 100 train and 40 test pairs per cell) and **confirmed on test**.
- Target: **both people said yes**. Base rate 14.0% on train, 14.8% on test.
- Pair categories are unordered (zone_a|zone_b is the same as zone_b|zone_a). "Unknown" means at least one of the two had not answered that soft field.

## 1. Single attributes (test worlds; train in the JSON)

Zone. Same zone 14.8% (9,370 pairs), different zone 19.7% (71 pairs, 95% interval 12–30%). Cross-zone pairs are 0.8% of all introductions, because people mostly accept only their own zone.

| Zone pair | Pairs (test) | Both say yes |
|---|---|---|
| a–a | 7,806 | 15.0% |
| b–b | 1,064 | 13.1% |
| c–c | 180 | 18.9% |
| **a–b** | **19** | 15.8% (interval 5.5–37.6%) |
| d–d | 43 | 0.0% (train 0.9%) |

The cross-zone pair a–b has too few examples to say anything. Zone has no usable effect.

Gender pair: every combination sits between 14.6% and 16.2% (man–woman 14.7%, man–man 14.7%, woman–woman 15.0%, non_binary–woman 14.6%, man–non_binary 15.2%). Same versus different gender: 14.9% against 14.7%.

Soft-field agreement (both known):

| Field | Same | Different | Not both known |
|---|---|---|---|
| **Relationship goal** | **18.2%** | **8.7%** | 15.4% |
| **Pace** | **17.6%** | **10.8%** | 15.6% |
| Lifestyle | 17.0% | 12.2% | 15.2% |
| Conversations | 12.3% (train 16.9%) | 14.3% | 15.2% |
| Emotional availability | 14.1% | 14.3% | 15.1% |
| Space / relocate | 16.9% / 13.1% | 12.8% / 13.4% | 15.1% / 15.5% |

Goal pairs: long_term–long_term 19.8%, exploring–exploring 16.4%, **exploring–long_term 8.7%** (n = 1,359). Pace pairs: steady–steady 22.5%, slow–slow 17.6%, quick–quick 13.2%, **slow–steady 8.4%**, quick–steady 10.4%, quick–slow 13.0%.

No stable effect: age-band pair, age gap, schedule overlap (1, 2 or 3 slots: 15.2 / 14.4 / 13.8%), children pair, smoking pair, wants-children pair, history pair, arrival pair and day band (13.2% on days 40–59, 15.0% on days 0–19). The few extreme cells are tiny (has-kids–has-kids 28.9% on 38 pairs; yes-smoker pairs 100% on 8 pairs) and I treat them as noise.

Who ends up in a mutual yes (test): 17.5% of mutual-yes pairs have the same goal against 14.3% of all pairs; 8.4% have different goals against 14.4% of all pairs. Same-zone share is 99.0% against 99.2%.

## 2. Merged features

| | Cells tested | Discovered on train (FDR 5%) | Confirmed on test |
|---|---|---|---|
| 2-way (210 attribute pairs) | 2,343 | 489 | 169 |
| 3-way (56 attribute triples) | 1,149 | 357 | 158 |

Winner's curse: discovered cells differ from the base rate by 5.3 points on train and only 4.2 points on test, so roughly a fifth of the apparent effect disappears on new worlds. Confirmation on test is a 95% interval check without further multiple-comparison control, so some confirmed cells are still noise.

Highest confirmed merged cells (both say yes):

| Cell | n train | Train | n test | Test | Lift |
|---|---|---|---|---|---|
| goal same & pace same & history low\|none | 138 | 31.9% | 47 | 36.2% | 2.4 |
| goal same & pace same & gender same | 450 | 27.6% | 122 | 27.9% | 1.9 |
| goal same & lifestyle unknown & gender same | 354 | 26.8% | 59 | 28.8% | 1.9 |
| goal same & history high\|none & emotional same | 154 | 22.1% | 66 | 30.3% | 2.0 |
| pace same & lifestyle same & emotional same | 359 | 23.1% | 171 | 26.3% | 1.8 |
| man\|non_binary & ages 26–30\|26–30 | 142 | 21.8% | 53 | 34.0% | 2.3 |

Lowest confirmed merged cells:

| Cell | n train | Train | n test | Test | Lift |
|---|---|---|---|---|---|
| goal different & pace different & history low\|none | 305 | 4.9% | 113 | 0.0% | 0.0 |
| lifestyle different & conversations different & history low\|low | 250 | 6.4% | 71 | 0.0% | 0.0 |
| goal different & lifestyle different & history low\|low | 173 | 4.6% | 49 | 2.0% | 0.1 |
| goal different & gender same & history low\|none | 213 | 5.6% | 71 | 1.4% | 0.1 |

The pattern is consistent: the best cells combine agreement on goal and pace, and the worst combine disagreement on several fields, especially when the people have said no before. The cells that mix in gender, age bands or schedule overlap (for example man|non_binary & 26–30|26–30, or goal different & schedule overlap 3 at 1.7% against 9.1% on train) have no main effect behind them and are the likeliest false positives.

Probability table from the merged four soft-field agreements (goal / pace / lifestyle / conversations, each same, different or unknown; smoothed toward the base rate), selected rows:

| Goal / pace / lifestyle / conversations | n train | Train | Smoothed | n test | Test |
|---|---|---|---|---|---|
| same / same / same / different | 226 | 24.3% | 21.2% | 155 | 25.2% |
| same / same / different / different | 658 | 21.7% | 20.7% | 162 | 24.7% |
| same / unknown / unknown / unknown | 658 | 20.8% | 19.9% | 137 | 21.9% |
| different / unknown / unknown / unknown | 641 | 11.7% | 12.0% | 169 | 8.3% |
| different / different / different / different | 1,294 | 4.7% | 5.4% | 363 | 7.2% |

## 3. Does merging features improve the probability? (AUC for both-say-yes, held-out worlds)

| Model | Columns | Train AUC | Test AUC | Gap |
|---|---|---|---|---|
| Lookup table on the four merged agreements (57 cells) | 57 | 0.582 | 0.540 | +0.042 |
| **A. Logistic on all single pair attributes** | 90 | 0.600 | **0.561** | +0.040 |
| B. A + the 327 cells confirmed on test (selection leaks) | 417 | 0.635 | 0.587 | +0.048 |
| C. A + the 846 cells discovered on train only (honest) | 936 | 0.683 | 0.553 | **+0.129** |
| D. A + all 3,492 merged cells, no selection | 3,582 | 0.744 | 0.535 | **+0.209** |

- B looks best, but it picked its cells using the test worlds, so its 0.587 is inflated by selection and cannot be trusted. It shows how easily test-set selection manufactures a gain.
- **Honest merging hurts.** Adding the train-discovered cells (C) raises the train AUC to 0.683 and lowers the test AUC to 0.553, below the single-attribute model (0.561). Adding every cell (D) is worse still. Merged cells fit train noise.
- Calibration of model C on test is poor at the top: pairs predicted at 25% were observed at 18.8%.

## 4. What this means

1. The **zone and gender** effects you asked about are absent. Same-zone pairs are 99% of introductions, and cross-zone cells are too small to estimate (zone a with zone b: 19 test pairs).
2. Every stable effect comes from **goal and pace agreement** (both say yes: about 28% when both agree, 9% when they differ), plus a smaller lifestyle effect. The merged cells largely restate that.
3. Merged features are useful for **describing** where the probability is highest and lowest, and for ranking candidate rules by lift. They do not help **predict**: single attributes generalise better than hundreds of merged cells.
4. The product of two person-level yes models predicted mutual yes better earlier (AUC 0.578 on a related but non-identical dataset), so a person-level model with agreement features remains the best option.
5. If you want to model combinations, use a handful of pre-chosen interactions (goal × pace, agreement × own history) with regularisation, rather than hundreds of discovered cells.

## Limits
Cells with under about 50 test pairs are noisy; the discovery-confirmation procedure still lets some false cells through; and everything describes the simulator.
