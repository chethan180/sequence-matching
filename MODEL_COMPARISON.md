# Model comparison: global matching, logistic, boosting, Thompson sampling and an oracle

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Written for: the team choosing which scoring model and allocation rule to carry into the build phase.

All results are from the public synthetic simulator (release 1.0.0). They describe that invented environment only. Charts: `results/model_comparison.png`, `results/confirmation.png`.

## 1. What was compared

All policies use the same reference clarification rule (ask the first incomplete, non-declined members), so only the pair scoring and allocation differ.

| Policy | How it scores a feasible pair | How it picks the set of pairs | Trained on |
|---|---|---|---|
| greedy (reference) | number of equal soft fields | best pair first | nothing |
| random feasible | nothing | random order | nothing |
| global, soft count | number of equal soft fields (+1 per pair) | exact maximum total weight | nothing |
| global, online rate | each person's yes-rate so far (shrunk) × a small soft bonus | exact maximum total weight | nothing (updates during the episode) |
| **logistic + global / + greedy** | P(a says yes) × P(b says yes), each from a logistic regression on 19 features | exact maximum total weight / best first | 109,434 yes/no labels from 720 rollouts |
| boosted trees + global | the same product, from 80 gradient-boosted trees on the same features | exact maximum total weight | same rows |
| logistic staged + global | adds a second-meeting model after the yes models | exact maximum total weight | same rows + 12,580 date rows |
| Thompson + global | one random draw from each person's Beta belief about their yes-rate | exact maximum total weight | nothing (updates during the episode) |
| ORACLE + greedy / global | the simulator's hidden yes-probabilities | best first / exact | **not a valid policy**: a diagnostic upper bound |

Pure-Python versions of the trained models run inside the container (weights in `model_weights.json`, 58 KiB, matching scikit-learn to 1e-16).

## 2. Offline: how well do the models predict a yes? (held-out worlds, seeds 1031–1040)

| Model | AUC | Log-loss |
|---|---|---|
| constant (population rate) | 0.500 | 0.659 |
| own history only | 0.531 | 0.666 |
| soft-match count only | 0.518 | 0.664 |
| logistic regression | 0.562 | 0.653 |
| gradient boosting | 0.563 | 0.653 |
| second-meeting logistic (rows with a date) | 0.506 | 0.696 (constant: 0.691) |

The logistic model is well calibrated (predicted and observed yes-rates within about 3 points in every decile). The second-meeting model is no better than a constant, so it is not used. The feature study (`feature_study.py`, `FIVE_PEOPLE_WORKED_EXAMPLE.md`) shows that a compact set (similarity agreement + own history) with logistic regression is the best model (AUC 0.566), and that adding 111 more features or 27 domain-knowledge features does not help (`DOMAIN_FEATURES.md`).

## 3. Policy level, study 3: 11 policies × 60 seeds × 6 scenarios (360 paired episodes each)

| Policy | MSMI per 100 | Mutual acc. per 100 | Yes rate | Mutual per introduction | Coverage |
|---|---|---|---|---|---|
| greedy | 0.449 | 5.60 | 0.363 | 0.148 | 0.339 |
| random feasible | 0.453 | 5.80 | 0.364 | 0.153 | 0.339 |
| global, soft count | 0.436 | 5.69 | 0.364 | 0.150 | 0.340 |
| global, online rate | 0.424 | 5.71 | 0.362 | 0.150 | 0.342 |
| logistic + greedy | 0.456 | 5.79 | 0.366 | 0.152 | 0.340 |
| logistic + global | 0.442 | 5.76 | 0.364 | 0.152 | 0.341 |
| logistic staged + global | 0.426 | 5.76 | 0.363 | 0.151 | 0.339 |
| boosted trees + global | 0.446 | 5.88 | 0.367 | 0.154 | 0.341 |
| Thompson + global | 0.431 | 5.82 | 0.365 | 0.153 | 0.341 |
| ORACLE + greedy | 0.456 | 6.00 | 0.375 | 0.161 | 0.338 |
| ORACLE + global | 0.460 | 5.89 | 0.372 | 0.158 | 0.338 |

**The key finding is the oracle.** Even with exact knowledge of every hidden yes-probability, mutual acceptances rise from 11.2 to 12.0 per episode (+7%), and MSMI does not move (0.897 → 0.911 per episode, well inside the noise). Better scoring has very little room to help in this simulator. The trained models capture roughly 40% to 70% of that small head-room on mutual acceptances (logistic +0.32, boosted trees +0.56, oracle +0.79 per episode over greedy) and none of it on MSMI.

Greedy by soft-match count is slightly worse than random on mutual acceptances (−0.39 ± 0.16 per episode): the baseline's tie-breaking by id and its treatment of unknown soft fields as "different" add no value.

## 4. Pre-declared confirmation, study 5: 6 policies × 100 fresh seeds × 6 scenarios (600 paired episodes)

Candidates were fixed before running. Paired differences against greedy (mean ± standard error per episode; rates in percentage points).

| Policy | MSMI | Mutual acceptances | Yes rate (pp) | Coverage (pp) |
|---|---|---|---|---|
| value-ranked asks only | +0.028 ± 0.042 | +0.08 ± 0.13 | +0.02 ± 0.20 | −0.03 ± 0.03 |
| **logistic + global** | **+0.090 ± 0.035** | +0.08 ± 0.10 | +0.21 ± 0.15 | **+0.14 ± 0.03** |
| boosted trees + global | +0.012 ± 0.034 | +0.11 ± 0.11 | +0.23 ± 0.16 | +0.16 ± 0.03 |
| logistic + value asks | +0.023 ± 0.042 | +0.30 ± 0.13 | +0.27 ± 0.21 | +0.18 ± 0.03 |
| boosted trees + value asks | +0.015 ± 0.041 | +0.27 ± 0.14 | +0.33 ± 0.21 | +0.20 ± 0.03 |

Absolute means: greedy 0.745 MSMI and 11.29 mutual acceptances per episode; logistic + global 0.835 and 11.37.

Combining the three independent studies that include each policy (inverse-variance weighted, different seeds each time):

| Policy | MSMI pooled | Mutual acceptances pooled |
|---|---|---|
| logistic + global | **+0.073 ± 0.024 (z = 3.0)** | +0.16 ± 0.07 (z = 2.2) |
| boosted trees + global | +0.034 ± 0.024 (z = 1.4) | +0.28 ± 0.08 (z = 3.7) |
| value-ranked asks only | +0.045 ± 0.033 (z = 1.4) | +0.13 ± 0.10 (z = 1.3) |
| global, soft count | −0.041 ± 0.037 | +0.08 ± 0.12 |

## 5. What this says

1. **Asking is the one big lever.** Without clarification, coverage falls from 0.32 to 0.12 and mutual acceptances by about 70% (study 1). Every policy above asks.
2. **Global matching alone is not the answer.** With soft-count scores it is indistinguishable from greedy and random. Allocation helps only when the weights carry information, and the feasible pairs here are mostly used eventually anyway. The limit is how fast well-connected members can be re-used (each introduction occupies someone for at least 8 days).
3. **Logistic regression is the best model we have.** It is the only one with a consistent (small) positive MSMI difference: +0.09 in the pre-declared study, +0.07 pooled, about +10% relative. The pre-declared study tested five candidates, so its single result is borderline after correction for multiple comparisons; the pooled figure is stronger but includes studies where the policy was also being selected.
4. **Boosted trees add nothing over logistic regression**: equal AUC offline, and a pooled MSMI difference near zero.
5. **Value-ranked asking, the staged second-meeting model and Thompson sampling show no benefit.** An earlier apparent gain from value-ranked asks (+0.21 in one study) vanished on fresh seeds.
6. **Models help the lower-noise metrics more than MSMI**: mutual acceptances (+0.1 to +0.4 per episode) and coverage (+0.14 to +0.20 points, about 0.3 to 0.4 more people served per episode). MSMI gains of 0.05 to 0.1 per episode are at the edge of what 600 episodes can detect.

## 6. The submitted default

`policy.py` now defaults to the `final` configuration: reference asking, logistic yes-model for both directions, exact global matching with a small per-pair constant. Checks run:

- 22 unit tests pass; empty-population requests return valid empty responses.
- Run through the real subprocess protocol (`evaluate.episode`, one episode per scenario variant): all six valid, 120 calls in about 14–15 s per episode (about 0.12 s per call against the 10 s limit).
- The Dockerfile now copies `model_weights.json`. I have not built or run the Docker image, because I did not test container mode here.

## 7. Limitations and next steps

- Matching is exact only for connected groups of up to 20 members; larger groups (the real feasible clusters have 34–65) use a greedy-plus-swaps heuristic. Replace it with an exact general-graph matcher (or compare the heuristic against one offline) before the final build.
- The effect sizes are small and the evidence rests on a simulator whose private worlds differ. Report per-scenario results, not only the mean.
- Untested ideas: a hierarchical model with a per-person effect (the one place the simulator has signal), a direct pair-level mutual-yes model, and waiting thresholds.
- Features and models were tuned on the public simulator. The training seeds (1001–1040), validation seeds (1021–1030) and test seeds (1031–1040, 401–460, 501–560, 601–700) are listed in the scripts.

## 8. Reproduce

```
python experiments.py --collect --configs greedy,random,global_learned --seeds 1001-1040 --output results/train_rows.npz
python train_models.py                     # writes model_weights.json, results/offline_metrics.json
python experiments.py --configs greedy,random,global_only,global_learned,logit_greedy,logit_global,logit_staged,gbm_global,thompson_global,oracle_greedy,oracle_global --seeds 401-460 --output results/study3.json
python experiments.py --configs greedy,value_ask_only,logit_global,gbm_global,logit_value,gbm_value --seeds 601-700 --output results/study5.json
python make_charts.py
```
