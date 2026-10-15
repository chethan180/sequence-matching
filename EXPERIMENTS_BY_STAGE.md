# Every experiment, stage by stage

**Team:** ML_Wizards_IITM · Round 1 and Round 2 · simulator release 1.0.0

This document lists every experiment we ran, grouped by the stage of the policy it informs, and states which option each stage now ships. The shipped policy itself is described in [BEST_SOLUTION.md](BEST_SOLUTION.md). All results come from the public synthetic simulator and say nothing about real people.

## How the policy is organised

The policy has three stages, **Asking, Matching and Repeat**, split into seven parts:

| Stage | Part | Question it answers |
|---|---|---|
| **A. Asking** | 1. Who to ask, and what | How to spend the 12 daily clarification units |
| **B. Matching** | 2. Who can meet | Which pairs are allowed at all |
| | 3. Scoring a pair | How likely both people are to say yes |
| | 4. Choosing today's pairs | Which non-overlapping pairs to introduce |
| **C. Repeat** | 5. The daily re-run | Arrivals, busy people, departures, waiting |
| | 6. Learning from feedback | Using replies as they arrive |
| | 7. Never invalid, always fast | Validation, fallbacks, run time |

## How to read the results

- **Paired.** Every comparison runs the policies on the same worlds: the same people with the same luck. We take the difference per seed and report the mean ± the 95% interval over seeds. **✱** marks an interval that excludes zero (a proven effect). Without ✱, the difference cannot be told apart from chance.
- **Independent worlds.** Round 2 studies give each scenario family its own world (`--independent`), so the 6 × 60 episodes are independent.
- **Units.**
  - Round 1 reports MSMI and mutual acceptances **per episode**.
  - Round 2 reports them **per 100 arrived members**. Every episode has 200 members, so per 100 = per episode ÷ 2.
  - Coverage is a share (0.340 = 34.0%). Round 1 reported it in percentage points.
- **Diagnostic** experiments read the simulator's hidden values to measure what is possible. They are never inputs to the policy.

---

## Stage A: Asking

### Part 1. Who to ask, and what

**Shipped:** hard-constraint bundles only. Askable members are asked in order of Bayesian expected number of partners, and anyone with less than a 5% chance of any partner is skipped for the day and re-scored daily. Config `submission`, function `smp/clarify.py: bayes_bundle_asks`.

| # | Experiment | Data | Result | Decision |
|---|---|---|---|---|
| A1 | **No clarification at all** (H1) | Round 1, 30 seeds, paired | Mutual acceptances **−7.33 ± 0.33 per episode ✱**, coverage **−20.8 ± 0.6 points ✱**, MSMI −0.41 ± 0.10 per episode ✱ | Clarification is the dominant lever; never switch it off |
| A2 | Organisers' smoke run: greedy vs no asks | Public seed 101, 1 episode per family | Coverage 0.408 → 0.174 | Same conclusion, low counts |
| A3 | Does knowing more soft answers make the yes-model more accurate? (H6) | Round 1, out-of-fold | AUC 0.531 (0 of 4 core answers known) → 0.580 (1–2) → 0.621 (3–4); difference +0.090 [+0.073, +0.108] ✱ | Soft answers carry information |
| A4 | **Buy soft answers with leftover units** (value of information) (H8) | Round 1, 30 seeds × 6, paired | Core answers known at introduction 31.6% → 61.3% ✱; model AUC +0.027 [+0.019, +0.035] ✱; MSMI −0.05 ± 0.06 per episode; mutual −0.17 ± 0.21; **cost +73.6 units ✱** | Off: better prediction, no better outcomes, higher cost |
| A5 | Reserve 3 units a day for soft asks | Round 1, paired | **−1.5 ± 0.5 introductions per episode ✱** | Rejected: hard bundles must come first |
| A6 | Ask order by count of possible partners (`value_asks`) | Round 1 studies 1, 2, 5 (unpaired means) | MSMI per 100 vs greedy: +0.10, −0.01, +0.02; inconsistent | Not proven at the time |
| A7 | **Where do bundles go?** | Round 2, 18 episodes | 47% go to people with **no** compatible partner; only 13 of 559 (2%) could be spotted beforehand by the "zero possible partners" rule | Motivates a better predictor |
| A8 | Fast compatibility check vs organisers' `eligibility` | 59,700 pairs, 3 families | Agrees on every pair | Safe to use for scoring |
| A9 | **Bayesian prediction of who has a partner** | Round 2, 7,920 askable people, 120 worlds; label from true answers (diagnostic) | AUC **0.732** (expected partners), 0.727 (P(at least one)) vs **0.586** for the count rule; better in every family (0.66–0.72 vs 0.53–0.60); over-confident at the extremes | Bayesian score adopted for ordering |
| A10 | **Ask order and skipping, full episodes** | Round 2, 60 seeds × 6, paired, seeds 4001–4060 | Bayesian vs list order: **wait −1.00 ± 0.08 days ✱, introductions +1.28 ± 0.50 ✱**, coverage +0.0003, MSMI +0.019 ± 0.050. Bayesian vs count rule: **wait −0.18 ± 0.07 ✱**. Skip (5%) vs no skip: **cost −7.51 ± 0.49 units ✱**, same 24,664 people served | **Adopted:** Bayesian order + 5% skip |
| A11 | Skip threshold 0.10 instead of 0.05 | Round 2, 60 × 6, paired, fresh seeds 5001–5060 | Cost **−1.66 ± 0.30 ✱**, coverage −0.0001 ± 0.0001 (7 fewer people over 360 episodes), wait **+0.03 ± 0.02 ✱** | Rejected: coverage outranks cost |

Details and a worked example: [ASK_STRATEGY.md](ASK_STRATEGY.md).

---

## Stage B: Matching

### Part 2. Who can meet

**Shipped:** the organisers' `kit.eligibility(a, b) == 'feasible'` is the only gate, together with availability and no repeated pair (`smp/feasibility.py`). It is re-checked on every pair before sending (part 7).

| # | Experiment / measurement | Result | Consequence |
|---|---|---|---|
| B1 | Pair status in the supplied data | Of 199,000 possible pairs: 0.5% feasible, 12.9% need clarification, 86.6% infeasible; 35% of members declined at least one hard field | Most people can never be introduced |
| B2 | Truly feasible pairs per episode (diagnostic, 72 episodes) | 95.8 on average (23.5 sparse, 129.8 shift); greedy uses 78% of them | The pair set is small and fixed |
| B3 | Members with any truly feasible partner (diagnostic) | 34.8% (17.1% sparse); greedy serves 33.7% | This is the real coverage ceiling, not 65% |
| B4 | Why feasible pairs go unused (diagnostic, 1,500 unused pairs) | Competition for the same candidate 57%; a member left or paused first 36%; never free on the same day 7%; not revealed by clarification 0% | Allocation and timing matter, clarification reach does not |

### Part 3. Scoring a pair (the yes-model)

**Shipped:** a logistic model of "says yes" with 14 features (agree / disagree / unknown on goal, pace, lifestyle, conversations; number of unknown core fields; five history terms). It was trained on 110,700 rows from 720 episodes (3 behaviour policies × 40 seeds × 6 families), validated by seed and refitted on all seeds. History priors come from the six training pools (0.797 reply, 0.479 yes). The pair value is the product of the two directions, used for ranking only.

| # | Experiment | Data | Result | Decision |
|---|---|---|---|---|
| B5 | Soft-field agreement effects (H2) | Round 1, 109,434 rows | LOR goal **+0.76**, pace **+0.66**, lifestyle **+0.32**, conversations **+0.24** (all ✱); the other three fields ≈ 0 | Use the four core fields |
| B6 | Agreement acts on accepting, not on answering | Round 1 | Goal on "answered": +0.071 [−0.039, +0.179] | Separate reply-rate and yes-rate terms |
| B7 | Effects add on the log-odds scale | Round 1 | Goal × pace interaction +0.018 [−0.269, +0.290] | No interaction terms |
| B8 | Unknown ≠ mismatch (H3) | Round 1, mixture test | π between 0 and 1 for every core field (goal 0.562 [0.486, 0.642]) | Unknown is a third state |
| B9 | Persistent person effects (H4) | Round 1, 70,946 within-person pairs | Generosity τ² = 0.320 [0.264, 0.376]; reply rate SD 0.119; pair correlation ρ = +0.037 [+0.013, +0.061] | Product is for ranking only |
| B10 | How much of the person effect is learnable (H5) | Round 1, information bound | **5.6%** of the variance; AUC does not rise with history (0.566, 0.572, 0.568, 0.575) | Refuted: history helps little |
| B11 | Model comparison, out-of-fold by seed | Round 1 | Soft-match count 0.506; own history 0.533; agreement only 0.555; **logistic 19 features 0.569** [0.560, 0.579]; calibrated (slope 0.948) | Logistic adopted |
| B12 | Gradient boosting, latent-trait model, full probabilistic model | Round 1 | GBM ≈ logistic; latent trait ΔAUC −0.000; probabilistic +0.001 | No gain from complexity |
| B13 | Perfect-knowledge model (diagnostic) | Round 1 | AUC 0.69 with the simulator's own probabilities | Ceiling for any model |
| B14 | Leave-one-family-out (do learned weights transfer?) | Round 2, round 1 rows | Normal-rule families lose ≤ 0.007 when left out; **shift** 0.533 without it, 0.538 with it, 0.549 for a shift-only model | Model learns the simulator's rules; shift is the stress case |
| B15 | **Only the terms of the simulator's equation** | Round 2, held-out seeds | Core + history 0.5625 vs all 19 features 0.5623; dropping age gap, zone and other soft fields −0.0009 [−0.0023, +0.0005] | Extra features are useless |
| B16 | **Retrain: 14 vs 19 features, new data and priors** | Round 2, 110,700 rows, independent worlds | **14 features 0.5717** vs 19 features 0.5722 (−0.0004 [−0.0019, +0.0010]); GBM 0.5703; calibrated within ~3 points; by family 0.548 (shift) to 0.588 | **14-feature model shipped** |
| B17 | Second-meeting model (H11) | Round 2, 3,212 held-out rows with a date | AUC **0.525** | Not added: the second stage is mostly chance |
| B18 | Within-episode re-learning of soft-field weights (for shift) | Round 2, 60 × 6 paired, seeds 5001–5060 | Finds 3 of shift's 4 changes but moves other families as much; mutual +0.026 ± 0.065, MSMI +0.008 ± 0.024; shift mutual +0.05 ± 0.16 | Rejected |
| B19 | Training-pool priors | Supplied data, pools 01–06 | Reply 0.797, yes 0.479 (all ten pools: 0.781, 0.473) | Adopted to respect the data split |

### Part 4. Choosing today's pairs

**Shipped:** exact maximum-weight matching (Edmonds' blossom algorithm, `smp/matching.py`) with integer tiered weights (`smp/allocate.py`):

```
w = 10^12 (every pair)  +  10^9 × (first-time people in the pair)
  + 10^6 × scarcity level of each first-time person (3, 2, 1 for 1, 2, 3 feasible partners left)
  + round(10^6 × predicted pair value) + 1
```

| # | Experiment | Data | Result | Decision |
|---|---|---|---|---|
| B20 | Learned score + global matching vs greedy (H7) | Round 1, 100 seeds, paired | MSMI +0.090 ± 0.048 per episode (z = 1.9, not proven); **coverage +0.14 ± 0.04 points ✱** | Global matching kept |
| B21 | Greedy vs random feasible | Round 1, 60 seeds | No difference (MSMI +0.01 ± 0.07) | Reference ranking carries no information |
| B22 | Perfect-knowledge allocation (H9, diagnostic) | Round 1, 36 paired episodes | Daily matching **+0.030 [+0.008, +0.054]** per episode; slot-price planning +0.045 [+0.022, +0.069] | Allocation can add only ~4–6% MSMI |
| B23 | Ranking by the simulator's internal probabilities (diagnostic) | Round 1 | Mutual +0.79 ± 0.23 per episode ✱, MSMI +0.01 ± 0.06 | Better ranking moves acceptances, barely MSMI |
| B24 | Blossom matcher vs exhaustive search | 3,000 random graphs, 2–13 nodes, weights up to 10¹² | Optimal on every graph; tiered weights give maximum cardinality on 300 graphs; a 200-node day in < 2 s | Exact matcher adopted |
| B25 | **Tiered weights and their ablations** (H10) | Round 2, 60 × 6, paired vs Round 1 `final`, seeds 2001–2060 | `tiered`: **coverage +0.003 ± 0.001 ✱**, MSMI +0.014 ± 0.031, mutual −0.086 ± 0.095. No value term: **mutual −0.194 ± 0.157 ✱**. No coverage tier: coverage +0.000. Exact matcher on value only: no difference. Greedy: **coverage −0.002 ✱, mutual −0.146 ✱** | Tiered adopted; value term kept |
| B26 | **Scarcity bonus** for first-time people with few partners left | Round 2, 60 × 6, paired vs `submission_v1`, seeds 5001–5060 | **Coverage +0.0007 ± 0.0004 ✱** (+50 people over 360 episodes), **wait −0.08 ± 0.03 days ✱**, MSMI +0.008 ± 0.033, mutual −0.058 ± 0.079, slowest call 0.31 s | **Adopted** |

---

## Stage C: Repeat

### Part 5. The daily re-run

**Shipped:** the policy keeps no memory. Every call rebuilds its view from the observation, so arrivals, busy periods, departures and pauses are handled automatically. Nobody with a feasible partner is held back.

| # | Experiment / finding | Result | Decision |
|---|---|---|---|
| C1 | Slot prices (hold well-connected people for better partners) | With perfect knowledge, +0.015 over daily matching; not distinguishable from zero | Not used |
| C2 | Can departures be predicted? | In `kit.py`, `exit_day = randint(22, 60)` for 12% of people, independent of everything observable | Not predictable; the only defence is introducing early |
| C3 | Earlier first introductions | Bayesian ask order: −1.0 day (A10); scarcity bonus: −0.08 days more (B26) | Adopted |
| C4 | Drift after day 35 | Yes rate 38.2% → 29.9% in drift; a uniform drop does not change which pair is best on a given day (`late`/`day` features add nothing, B15) | Front-loading, no special handling |

### Part 6. Learning from feedback

**Shipped:** each person's reply rate and yes rate are recomputed from visible feedback every call and shrunk toward the population rate (strength 3). Silence lowers the reply rate, never the yes rate. Open introductions are not counted as failures.

| # | Experiment | Result | Decision |
|---|---|---|---|
| C5 | Persistent person effects exist (B9) but are learnable only to 5.6% (B10) | History terms add little but are kept for cold start | Shrinkage kept |
| C6 | Thompson sampling (bandit exploration) | No gain (Round 1) | No exploration |
| C7 | Within-episode re-learning of field weights (B18) | No gain | Rejected |

### Part 7. Never invalid, always fast

**Shipped:**
- an input contract check that sets damaged records aside;
- validation of every ask and pair against the simulator's rules;
- a fallback to greedy, then to empty lists;
- a 6 s internal time budget, with a greedy matcher if fewer than 2 s remain.

| # | Experiment | Result |
|---|---|---|
| C8 | Test suite | **42 tests pass**: validators, fallbacks, damaged requests, 120 randomly corrupted observations, damaged single records, full episodes with zero safety interventions, the command-line modes, the real subprocess |
| C9 | Refactor equivalence | 192 of 192 fingerprinted episodes identical before and after splitting the policy into modules (snapshot tool not part of this build) |
| C9b | Build reproduces the official results | 9 official episodes replayed through `evaluate.episode` (shipped policy in five families, Round 1 policy and greedy in two each) give action fingerprints (`action_sha256`) identical to `results/official/*.json` |
| C10 | Call time | Slowest call in all Round 2 studies: 0.31 s (limit 10 s). A full call takes ~0.16–0.23 s, of which 0.05 s is Python start-up |
| C11 | Removing argparse (#5) | About 3 ms per call: start-up is dominated by `json` and the organisers' `kit`, both required. Kept, since it is harmless |

---

## How much is possible: head-room

| # | Experiment | Data | Result |
|---|---|---|---|
| H1 | Coverage ceiling: members with a usable partner (diagnostic) | 180 worlds, seeds 4001–4030 | Ceiling 0.348; greedy 0.339 (1.72 members short per episode); ours 0.345 (0.64 short); ours − greedy **+0.0054 ± 0.0012 ✱** |
| H2 | Expected MSMI with luck removed (diagnostic) | same | Greedy 0.777 per episode, ours 0.794, **+0.017 ± 0.011 ✱**; slot-capped upper bound 0.972 (greedy 80.0%, ours 81.7%) |
| H3 | Realised MSMI on the same worlds | same | Greedy 0.733, ours 0.844 per episode, +0.11 ± 0.17 (luck dominates) |

Method and per-family tables: [BEST_SOLUTION.md](BEST_SOLUTION.md#how-close-this-is-to-the-maximum).

## The shipped policy against the baselines

| # | Experiment | Data | Result (shipped − greedy) |
|---|---|---|---|
| F1 | Shipped policy vs the three organiser baselines and our earlier versions | 2,160 episodes, 60 fresh seeds × 6 families (6001–6060) | MSMI per 100 +0.019 ± 0.052; **coverage +0.0051 ✱**; **mutual per 100 +0.30 ✱**; **clarification cost −8.0 ✱**; **introductions +0.65 ✱**; **dates +0.41 ✱**; **wait −1.56 days ✱**. vs no clarification: MSMI **+0.303 ✱**, coverage **+0.224 ✱** |

| F2 | Same comparison through the **organisers' harness** (a process per call), organisers' seed layout | 600 episodes, seeds 101–120 × 6 families | All 600 valid. Shipped − greedy: **coverage +0.0042 ✱**, **unserved −0.83 ✱**, **cost −7.4 ✱**, **inference time −3.7 s per episode ✱** (23.5 s, 0.196 s per call), introductions +2.2 ✱; MSMI per 100 −0.079 ± 0.154 (85 vs 104 qualifying pairs, within chance) |

Full tables: [BEST_SOLUTION.md](BEST_SOLUTION.md#what-the-final-policy-achieves).

## Waiting for a better match

**Question:** if pairs below a quality bar are not introduced, so both people wait for a better partner, are outcomes better?

`python experiments.py --configs submission,wait_v_p25,wait_v_p50,wait_v_p50_r40,wait_soft1,wait_soft2 --seeds 8001-8060 --independent`
(2,160 episodes, paired, fresh seeds.)

The value bars are the 25th and 50th percentiles of the model's pair value over all feasible pairs offered to the matcher (0.107 and 0.127, measured by `soft_status_study.py`).

| # | Rule | MSMI / 100 | Coverage | Mutual / 100 | Mutual per introduction | Introductions | Dates | Qualifying pairs per 100 introductions |
|---|---|---|---|---|---|---|---|---|
| — | **Shipped (no waiting)** | **0.411** | **0.344** | **5.72** | 0.150 | **77.8** | **8.9** | 1.06 |
| W1 | Skip pairs with value < 0.107 | −0.042 ± 0.028 ✱ | −0.018 ✱ | −0.32 ✱ | **+0.014 ✱** | −11.4 ✱ | −0.5 ✱ | 1.11 |
| W2 | Skip pairs with value < 0.127 | −0.099 ± 0.045 ✱ | −0.055 ✱ | −1.52 ✱ | **+0.020 ✱** | −28.4 ✱ | −2.3 ✱ | 1.26 |
| W3 | Value < 0.127 until day 40, then accept all | −0.028 ± 0.052 | −0.008 ✱ | −0.48 ✱ | +0.001 | −6.9 ✱ | −0.6 ✱ | 1.08 |
| W4 | Only pairs with ≥ 1 known soft agreement | −0.256 ± 0.045 ✱ | −0.196 ✱ | −3.59 ✱ | **+0.025 ✱** | −53.2 ✱ | −5.6 ✱ | 1.26 |
| W5 | Only pairs with ≥ 2 known soft agreements | −0.332 ± 0.041 ✱ | −0.266 ✱ | −4.65 ✱ | **+0.045 ✱** | −67.2 ✱ | −7.1 ✱ | 1.49 |

Columns W1–W5 are paired differences from the shipped policy (60 seeds × 6 families, 95% interval); the last column is a ratio.

**What happens.** Waiting does raise the **quality of each introduction**: more mutual yeses per introduction, up to 1.49 qualifying pairs per 100 introductions against 1.06. But it removes far more introductions than it improves, so **every total gets worse**: MSMI, coverage, mutual acceptances and dates. The waits are not rewarded because:

- **people leave without warning** (12% of members), and the 60 decision days run out;
- the pair a person waits for **rarely arrives**, because the feasible set is small and fixed (about 96 truly feasible pairs per episode);
- pair values are tightly bunched (the middle 80% lie between 0.075 and 0.167), so a "better" pair is only marginally better, while a lost pair is lost entirely.

**Soft-score bars are the worst** (W4, W5): 72% of introduced pairs have **no** known agreement, because 43% of soft answers are never asked. A soft bar therefore blocks most introductions and collapses coverage from 0.344 to 0.148 or 0.078. (The waiting time falls in W4 and W5 only because the few people still served are those with complete profiles, served early.)

**Releasing the bar at day 40** (W3) limits the damage but still loses coverage and mutual acceptances, and makes people wait 5 days longer.

**Decision:** rejected. Never hold back a feasible pair.

### Soft answers at the moment of introduction

`python soft_status_study.py --seeds 7001-7005` covers 2,366 introductions made by the shipped policy.

| Core soft field (per person) | Observed | Not asked | Declined |
|---|---|---|---|
| Relationship goal | 54.0% | 42.8% | 3.2% |
| Pace | 54.4% | 42.5% | 3.1% |
| Lifestyle | 52.8% | 44.0% | 3.2% |
| Conversations | 54.0% | 42.6% | 3.4% |

| Core fields known for both people | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Share of introductions | 59.1% | 13.6% | 1.8% | 0.1% | 25.4% |

| Known agreements (soft-match count) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Share of introductions | 72.1% | 15.6% | 9.0% | 3.1% | 0.3% |

Missing soft answers come from **not asking** (≈43%), almost never from refusals (≈3%). The yes-model treats unknown as its own state (B8), so missing answers never block a pair; they only weaken the prediction. Buying them was tested (A4): it doubles what is known but does not improve outcomes.

## Commands

```bash
python -m unittest -v
python experiments.py --configs greedy,final,tiered,tiered_nocov,tiered_novalue,exact_value --seeds 2001-2060 --independent --output results/study_r2a.json
python experiments.py --configs greedy,random,global_learned --seeds 1001-1040 --variants all --independent --collect --output results/train_rows_r2.npz
python train_shipped.py
python ask_model_study.py --seeds 3001-3020
python experiments.py --configs tiered,tiered_valueask,tiered_bayes,tiered_bayes_skip --seeds 4001-4060 --independent --output results/study_asks.json
python experiments.py --configs submission_v1,sub_skip10,sub_scarce,sub_adapt --seeds 5001-5060 --independent --output results/study_next.json
python experiments.py --configs submission,wait_v_p25,wait_v_p50,wait_v_p50_r40,wait_soft1,wait_soft2 --seeds 8001-8060 --independent --output results/study_wait.json
python soft_status_study.py --seeds 7001-7005
python run_final_study.py --seeds 101-120 --output-dir results/official
python experiments.py --configs greedy,no_asks,random,final,tiered,submission --seeds 6001-6060 --independent --output results/study_final.json
python headroom_r2.py --seeds 4001-4030 --output results/headroom_r2.json
```

Study `study_next.json` labels the policy as `submission` because it ran before the scarcity bonus was adopted; that configuration is now named `submission_v1`.

**Round 1 rows** (marked "Round 1": A1–A6, B5–B13, B20–B23, C1, C6) are quoted from our Round 1 research note. Their analysis scripts are not part of this Round 2 build. Every Round 2 row is reproduced by the commands above, and the shipped policy depends only on Round 2 results.
