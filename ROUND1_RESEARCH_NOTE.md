# Sequential Matching: research note and proposed approach

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Round 1 submission · IIT Madras × Romeo & Juliet · The Sequential Matching Problem · version 2

All results are from the public synthetic simulator (release 1.0.0) and measure that invented environment only. Every number comes from code in this repository. The in-process experiment harness reproduces `evaluate.py` exactly on the case I cross-checked (seed 101, development: MSMI 0.5, mutual acceptances 9.0 per 100, coverage 0.46, ask cost 198).

---

## 1. Problem interpretation

The task is a **sequential allocation problem with a costly information channel**, not a prediction problem. Each day a policy (a) spends up to 12 clarification units and (b) proposes non-overlapping, mutually eligible pairs. Success is Mutual Second-Meeting Intention (MSMI): a date within 30 days of assignment and both people saying yes to meeting again within 3 days of the date.

Three features shape everything else:

1. **Hard reciprocal constraints dominate.** A pair can be introduced only if all eleven hard checks pass both ways and no hard field is unknown. Scores never override this.
2. **Information is bought, not given.** A 3-unit constraints ask turns a member from blocked to usable. A declined answer can never be recovered.
3. **Outcomes are delayed, selective and rare.** Responses arrive up to 7 days later, dates later still, and only pairs the policy chose ever produce labels.

## 2. What the supplied data shows

From the ten public pools (2,000 members, 613 introductions, 1,270 feedback events; `exploration/01_data_exploration.ipynb`):

| Finding | Number |
|---|---|
| Members who declined at least one hard field (permanently unreachable) | 700 of 2,000 (35%) |
| Pairs feasible / needing clarification / infeasible | 0.5% / 12.9% / 86.6% of 199,000 |
| Feasible graph | one connected cluster of 34–65 members per pool, 54–110 edges |
| Historical outcomes | 886 answered responses (419 yes, 467 no), 249 no-response, 84 mutual acceptances, 40 dates, **3 MSMI** |
| Hard answers learned after the introduction was made | 0 of 613 |

Consequences: coverage has a ceiling of about 65% whatever the policy does; MSMI cannot be learned from the static files; outcomes exist only for chosen pairs and propensities are null, so I make no claim of unbiased offline evaluation. Policies are compared only through complete simulator episodes.

## 3. Hypotheses and their status

| # | Hypothesis | Status |
|---|---|---|
| H1 | Clarification is the dominant lever, because it creates the feasible edges. | **Strongly supported** (without asking, coverage 0.32 → 0.12, mutual acceptances −70%). |
| H2 | Choosing the whole set of pairs jointly beats best-pair-first. | True on constructed cases; **not supported with uninformative scores** (global matching on soft counts equals greedy and random). |
| H3 | A learned yes-model gives scores that make allocation worthwhile. | **Weakly supported**: logistic + global is +0.09 ± 0.035 MSMI per episode over greedy in a pre-declared 600-episode test (+0.07 ± 0.024 pooled over three studies), about +10% relative. |
| H4 | A more flexible model (gradient boosting) beats logistic regression. | **Not supported**: equal AUC offline, pooled MSMI near zero. |
| H5 | Ranking who to ask by unlock potential beats list order. | **Not supported** (an early gain did not replicate). |
| H6 | Domain-knowledge features from the dating literature improve prediction. | **Not supported** in the simulator (27 features, all intervals include zero). |
| H7 | Better scoring has large head-room. | **Refuted**: an oracle that knows the hidden yes-probabilities raises mutual acceptances only 7% and MSMI not at all. |

## 4. Proposed approach

Separable layers, each ablatable.

1. **Normalisation.** Per field: value, status (observed / not asked / declined), observed day. Unknown is never zero.
2. **Feasibility.** The supplied `eligibility()` is the only gate. Nothing downstream ever sees an infeasible pair.
3. **Asking.** Spend the 12 daily units on hard-constraint bundles for available, incomplete, non-declined members (reference rule; ranking by unlock potential did not help).
4. **Response model.** A logistic regression estimates, for each direction, the probability that a person says yes to a partner, from 19 features available before the decision: equal/different/unknown on the four core soft fields (goal, pace, lifestyle, conversations), agreement on the other three, the person's own shrunk answer and yes rates and how much history exists, how often others said yes to the partner, age gap, zone, and the day. History uses only feedback already observed (an assertion in the data collector checks that nothing newer than today appears). The pair weight is p̂_a · p̂_b, which assumes the two directions are independent and is used for ranking, not as a calibrated pair probability.
5. **Allocation.** Maximum-weight matching over the feasible, not-yet-introduced pairs among available members: maximise Σ w_ab with each person in at most one pair, plus a small constant per pair so more introductions are preferred to fewer.
6. **Memory.** None needed: state is rebuilt from the observation each call, so no information carries between episodes.

Model families considered: logistic regression (chosen: best offline and in simulation), gradient boosting (equal AUC, no pooled gain), a staged second-meeting model (AUC 0.506, no better than a constant), Thompson sampling over per-person Beta beliefs (no gain), a hierarchical model with per-person effects (not yet tested; it targets the one place the signal is). Full reinforcement learning is not proposed: the reward is about 0.9 MSMI per episode, the action space is very large, calls are limited to 10 s, and the private worlds differ from the public one. RLHF does not apply: no human preference labels and nothing to generate.

## 5. Treatment of uncertainty

- Missing, refused and not-asked stay distinct; only not-asked fields are worth buying.
- Person-level uncertainty is carried by shrinkage toward the population yes-rate (about 0.37); history takes over as it accumulates (history helps from about day 10).
- Recent introductions are censored, never failures; silence is never dislike.
- Hard constraints are never traded off; all learning happens among pairs that already pass `eligibility()`.

## 6. Evidence

**Training and features.** 109,434 person-level labels from 720 rollouts (seeds 1001–1040, split by world). Held-out AUC for "said yes": constant 0.500, own history 0.531, soft-match count 0.518, logistic 0.562, gradient boosting 0.563; well calibrated (deciles within about 3 points). A feature study with logistic regression and XGBoost shows raw attributes alone are near chance (0.49–0.52) while engineered agreement features reach 0.55 and own history adds about 0.013; a compact 28-feature logistic set is best (0.566) and extra features overfit. Pair-level, the product of the two directional yes-probabilities predicts mutual acceptance best (AUC 0.578); MSMI itself (508 positive pairs in 54,717) is not learnable (AUC about 0.53).

**Policies, 360 and 600 paired episodes per policy across all six scenarios** (full tables: `MODEL_COMPARISON.md`). Mutual acceptances per episode against greedy, study 3: random +0.39, global-soft +0.18, logistic + global +0.32, boosted trees + global +0.56, oracle +0.59 to +0.79. Pre-declared confirmation on 100 fresh seeds: logistic + global +0.090 ± 0.035 MSMI per episode and +0.14 ± 0.03 points coverage; boosted trees +0.012 ± 0.034; value-ranked asks +0.028 ± 0.042.

## 7. Planned experiments for the build phase

- **Power.** The paired MSMI difference has a standard deviation of about 1 per episode, so a 0.1 difference needs roughly 400 paired episodes; mutual acceptances (about 11 per episode) are the secondary metric.
- **Ablations:** no clarification; greedy instead of global; soft-count instead of the learned score; no learned history.
- **Untested ideas:** a hierarchical model with a per-person effect; a direct pair-level mutual-yes model; waiting thresholds; an exact general-graph matcher in place of the heuristic used above 20 members.
- **Robustness:** per-scenario results (sparse geography, cold start, delayed dates, shifted weights, drift), not only the mean; report ask cost, first-introduction waiting time, unserved count and missing feedback.

## 8. Expected failure modes

| Failure | Why | Mitigation |
|---|---|---|
| Unreachable members (35%) | Declined fields never resolve | Report separately; do not spend budget on them |
| Sparse supply | Few feasible edges, especially with 12 zones | Accept low coverage; ask early |
| Cold start | Little history in the first ~10 days | Population priors and shrinkage |
| Delayed dates | Dates after day 30 miss the MSMI window | Introduce early; avoid long-blocking pairs |
| Drift | Response rates fall after day 35 | Day feature; recent-history features |
| Overfitting to the public simulator | Private worlds differ | Per-variant reporting; no use of hidden state; held-out seeds |
| Weak signal | Held-out AUC about 0.56 | Rely on shrinkage and calibration; avoid over-claiming |
| Large-component matching | Exact only up to 20 members; heuristic above | Replace with an exact matcher before the final build |
| Residual independence assumption | p̂_a · p̂_b treats the two directions as independent | Used only for ranking; test a direct pair model |

## 9. Limitations

- Gains over the reference are small (about +10% MSMI at best) and rest on a simulator; MSMI is rare enough that 600 episodes only just detect them. The pre-declared study tested five candidates, so its single positive result is borderline after correction; the pooled figure includes studies where the policy was also being selected.
- The oracle result says the ceiling on scoring improvements is low in this environment, so the largest remaining gains, if any, are in allocation and timing.
- Simulator performance says nothing about real relationships or real members.

## 10. Reproducibility

```
python -m unittest -v
python verify_data.py          # on Windows clone with: git clone -c core.autocrlf=false <repo>
python experiments.py --collect --configs greedy,random,global_learned --seeds 1001-1040 --output results/train_rows.npz
python train_models.py
python experiments.py --configs greedy,value_ask_only,logit_global,gbm_global,logit_value,gbm_value --seeds 601-700 --output results/study5.json
```

Policy: `policy.py` (default configuration `final`, weights in `model_weights.json`, pure Python at inference). Training seeds 1001–1030 (validation 1021–1030, held-out 1031–1040); evaluation seeds 401–460, 501–560, 601–700. Inference has no random seed apart from the reference `random` baseline and the Thompson variant, which is deterministic given the day.
