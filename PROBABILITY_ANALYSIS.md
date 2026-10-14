# Probability of a yes: feature-by-feature analysis, new features and overfitting checks

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Written for: the team deciding which features go into the yes-model.

Script: `probability_analysis.py` → `results/probability_analysis.log` and `.json`. Data: `results/ext2_rows.npz` (109,434 person-introduction rows, 166 features).

## How the data is split
- A row is one person's response to one introduction. The target is "said yes" (no response counts as not-yes). Overall yes rate: 36.3%.
- A **world** is one simulator seed × scenario variant (200 members, 60 decision days + 40 follow-up days). Features are built from the morning of the decision day; labels are read after the follow-up.
- **Train worlds:** seeds 1001–1030 (180 worlds, 82,340 rows). **Test worlds:** seeds 1031–1040 (60 worlds, 27,094 rows). Splitting is always by world, never by row, because people and even pairs repeat inside a world.
- Tuning inside train: 5-fold cross-validation grouped by seed. The test worlds are used only for the final numbers. Confidence intervals bootstrap over the 60 test worlds.
- This is a frequentist analysis (observed rates, Wilson intervals, bootstrap tests, penalised logistic regression). It is not a Bayesian analysis; the only Bayesian element is the Beta-shrunk yes-rate used as a feature.

## 1. How often people say yes, by feature value (test worlds; train in the JSON)

Agreement on a soft field (both fields known):

| Field | Same | Different | Not both known |
|---|---|---|---|
| Relationship goal | **42.4%** | **29.2%** | 37.4% |
| Pace | **43.2%** | **31.8%** | 37.5% |
| Lifestyle | 42.5% | 33.1% | 37.2% |
| Conversations | 33.3% (train 40.4%) | 37.3% (train 33.2%) | 37.3% |
| Emotional availability | 35.7% | 36.4% | 37.4% |
| Space / Relocate | 38.9% / 35.6% | 35.3% / 36.1% | 37.2% / 37.4% |

Goal and pace agreement have a stable effect of about 11 to 13 points between "same" and "different", with consistent values on train and test. The conversation-style effect changes sign between train and test, so I would not trust it.

Joint goal × pace (test):

| Goal / pace | Yes rate | Yes given they answered |
|---|---|---|
| same / same | **49.2%** | 67.8% |
| same / different | 38.1% | 50.3% |
| different / same | 38.9% | 51.1% |
| different / different | **24.4%** | 33.5% |
| neither known | 37.6% | 49.9% |

The spread from 24% to 49% is the largest effect in the data.

Number of the four core fields that agree: 0 → 36.0%, 1 → 36.2%, 2 → 43.3%, 3 → 44.7%, 4 → 44.4% (only 108 rows). Number that disagree: 0 → 38.0%, 1 → 36.9%, 2 → 41.2%, 3 → 32.3%, **4 → 23.1%**.

The person's own history:

| Group | Yes rate |
|---|---|
| Own shrunk yes-rate, lowest quintile (0.14–0.35) | 32.9% |
| Own shrunk yes-rate, highest quintile (0.48–0.80) | 39.7% |
| Last answer was no | 32.9% |
| Last answer was yes | 40.7% |
| No answer yet | 37.3% |
| 0 / 1 / 2 / 3 / 4+ responses seen | 37.7 / 37.4 / 35.9 / 38.1 / 32.2% |

Flat or negligible: age gap (37.0 / 37.1 / 36.1% for 0–2 / 3–5 / 6–9 years), same zone, the person's gender (36.5–38.4%), the partner's received-yes rate (35.8–38.3%), and day of the episode (33.6–38.7%, with the lowest rate on days 45–59).

Splitting the outcome: whether someone answers at all (about 74–76%) barely changes with any feature; the agreement effects act almost entirely on yes given they answered (for example 67.8% against 33.5% for the goal × pace extremes).

## 2. Screening all 166 features

34 of 166 survive Benjamini–Hochberg false-discovery control at 5% with the same direction on train and test. The strongest single-feature AUCs on the test worlds (0.50 = chance):

| Feature | AUC train | AUC test |
|---|---|---|
| trait_agree_weighted (domain feature) | 0.552 | 0.541 |
| a_yes_vs_segment | 0.529 | 0.532 |
| a_ewma_yes | 0.528 | 0.531 |
| a_last_yes | 0.526 | 0.531 |
| a_p_yes | 0.529 | 0.528 |
| ne_relationship_goal | 0.471 | 0.476 |
| ne_relationship_pace | 0.475 | 0.478 |
| eq_relationship_goal | 0.523 | 0.517 |

Caveat: the bootstrap used 100 resamples, so no p-value can go below about 0.01. Every top feature sits at that floor with an FDR q of 0.043. The screen reliably separates "real and consistent" from "not", but it cannot rank the survivors by significance.

## 3. New features (39, all derived from existing columns)

Counts of agreeing and disagreeing core fields, goal × pace combinations, yes/no/silent counts and logit rates, rates with stronger and weaker priors, hinge terms, interactions of agreement with the person's own yes-rate, selectivity contrasts, and days since the last answer. The best single new features:

| New feature | AUC train | AUC test |
|---|---|---|
| core_net (agreements minus disagreements) | 0.552 | 0.540 |
| core_net × own yes-rate | 0.551 | 0.540 |
| p_combined (answer rate × yes rate) | 0.535 | 0.531 |
| pyes_tau10 (stronger prior) | 0.529 | 0.528 |

core_net is as strong as any single existing feature and is easy to explain.

## 4. Do the new features help, and is the logistic overfitting?

Regularisation chosen by grouped cross-validation inside the train worlds. AUC (0.50 = chance); log-loss lower is better (constant-rate baseline 0.6589 on test):

| Feature set | Columns | Train AUC | Grouped-CV AUC | **Test AUC** [95% over worlds] | Train–test gap | Test log-loss |
|---|---|---|---|---|---|---|
| S0 current policy features | 19 | 0.5771 | 0.5725 | **0.5624** [0.554, 0.570] | +0.015 | 0.6529 |
| S1 core (similarity + own history) | 28 | 0.5750 | 0.5713 | 0.5637 [0.556, 0.571] | +0.011 | 0.6526 |
| S2 current + new | 58 | 0.5773 | 0.5719 | 0.5613 [0.553, 0.569] | +0.016 | 0.6530 |
| S3 core + new | 67 | 0.5766 | 0.5712 | 0.5608 [0.553, 0.568] | +0.016 | 0.6530 |
| L1-selected from S3 | 13 | 0.5753 | n/a | 0.5650 [0.557, 0.572] | +0.010 | n/a |

Paired differences in test AUC against S0 (bootstrap over worlds): S1 +0.0014 [−0.0012, +0.0039]; S2 −0.0010 [−0.0022, +0.0004]; S3 −0.0016 [−0.0037, +0.0003].

Checks on overfitting:

1. **The gap is small and does not grow with feature count.** Going from 19 to 67 columns leaves the train–test gap at about 0.015. Overfitting is mild and is not what is limiting the model.
2. **More features do not help.** None of the new sets beats the current features, and the intervals all include zero. The best result is the L1-selected 13-column model (test AUC 0.565), which is not significantly better than the 19-feature model, but it needs fewer columns.
3. **Heavy regularisation wins.** The best C is the smallest in the grid (0.0003), and the cross-validation AUC falls slightly as C rises. The grid should be extended lower; the signal is weak enough that the optimum is probably more shrinkage still.
4. **More data does not help.** Test AUC stays at about 0.561–0.570 as training worlds increase from 3 to 30, while train AUC falls from 0.608 to 0.577. The model has saturated: the signal is limited, not the sample.
5. **Label-shuffle control** (labels shuffled within each training world): train AUC 0.513, test AUC 0.509, close to 0.50. It sits slightly above 0.50, probably because shuffling within a world keeps each world's base rate, but I did not investigate further. It is not evidence of leakage.
6. **Calibration on the test worlds is good.** Predicted and observed yes rates match within about 1 point for most deciles and within about 4 points at the top; calibration slope 0.93 (1.0 is perfect), intercept −0.01.
7. **Predictions are tightly clustered.** The 1st to 99th percentile of predicted probability is 20.8% to 54.2%, against an average of 36%. The model separates people only modestly, consistent with an AUC of about 0.56.

L1-selected model, standardised coefficients (13 columns kept): core_net +0.120, logit_combined +0.100, eq_goal × own yes-rate +0.068, ne_goal −0.050, ne_pace −0.045, goal_pace_both_eq +0.028, ne_emotional_availability +0.026, eq_pace +0.025, goal_pace_both_ne −0.023, a_silent −0.016, a_recv_n −0.013, a_n −0.006, a_n_intros −0.006.

## 5. What this means

- The yes probability moves from about 24% to about 49% depending on whether goal and pace agree, and by about 7 points with the person's own history. Nothing else has a stable effect.
- A logistic regression on a small number of features is already at the ceiling (test AUC about 0.56–0.565). Extra features, nonlinear terms and interactions add nothing measurable.
- The simplest good model is the L1-selected set: agreements minus disagreements, the combined answer × yes rate, goal agreement × own yes-rate, and goal/pace flags. It is calibrated and has the smallest train–test gap.
- If you want more signal, the remaining lever is a better estimate of each person's own tendency (a hierarchical model with a per-person effect), not more features.

## Limits
- Single-feature p-values are bounded below by the 100-resample bootstrap.
- The effect of conversation-style agreement is not stable between train and test.
- The age-gap 10+ and zone-mismatch groups have under 900 rows each.
- Results describe the simulator. They say nothing about real users.
