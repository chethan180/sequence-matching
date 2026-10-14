# Sequential Matching: Round 1 research note and proposed approach

IIT Madras × Romeo & Juliet · The Sequential Matching Problem · Round 1 (Research & Approach)

**Team:** _[ML_Wizards_IITM]_

All results come from the public synthetic simulator (release 1.0.0) and describe that invented environment only. Every number is produced by code in the repository (commands in section 13). We state a feature or hidden quantity as important only where the data proves it; each such claim below carries its test statistic and interval. Where an analysis reads the generator's hidden values (true answers, person traits), it is labelled *diagnostic*: it bounds what any policy could achieve and is never an input to the policy.

## Summary

Five findings, each tested below:

1. **Clarification is the dominant lever.** Without it, coverage falls by 20.8 points and mutual acceptances by 7.3 per episode (H1).
2. **The feasible pair set is small, fixed and mostly used.** Hard answers never change and a pair can be introduced only once, so each episode has a fixed budget of about 96 truly feasible pairs (23 in sparse geography). The greedy reference introduces 78% of them and serves 33.7% of members against a ceiling of 34.8% (diagnostic, section 12).
3. **Most of what decides a yes is hidden and cannot be learned in time.** Persistent person effects exist (H4), but the available history recovers only 5.6% of their variance (about 8% even with the generator's true spread), so the best yes-model reaches AUC 0.57 (H5, section 5.3).
4. **Better allocation has a real but small ceiling.** A slot-capped ceiling lies 25% above greedy, yet allocation given the true value of every pair gains only +0.030 to +0.045 MSMI per episode, about 4–6% (H9).
5. **The ranking is decided by counts and tie-breakers.** Every episode has exactly 200 arrived members, so the primary score equals the total number of qualifying pairs over the 120 private episodes divided by 240. Exact ties between competent teams are likely, and coverage, the first tie-breaker, decides them.

The plan follows from these findings: (i) a policy that cannot produce an invalid episode; (ii) prioritised hard-constraint clarification; (iii) exact global matching with a coverage-first objective; (iv) a calibrated yes-model, and optionally slot prices, to order the remaining choices; (v) soft-answer buying kept as a tested option, off by default.

## Where each requested item is covered

| Requested in the brief | Section |
|---|---|
| Problem interpretation, technical assumptions | 1 |
| Method of proof (how claims are tested) | 2 |
| Hypotheses | 3 |
| Reciprocal feasibility | 4 |
| Modelling approach and optional probability analysis | 5 |
| Decision / allocation method | 6 |
| Clarification strategy | 7 |
| Treatment of uncertainty, missing and delayed data | 8 |
| Learning strategy | 9 |
| Baselines | 10 |
| Planned experiments, ablations, experimental design | 11 |
| Preliminary policy evidence and head-room diagnostics | 12 |
| Expected failure modes and limitations | 13 |

---

## 1. Problem interpretation and technical assumptions

Each day a policy spends up to 12 clarification units and proposes non-overlapping, mutually eligible pairs; an empty list (waiting) is valid. The outcome is Mutual Second-Meeting Intention (MSMI): a date within 30 days of assignment and both people answering yes to meeting again within 3 days of it, scored per 100 arrived members and averaged over six scenario families. We read this as a **sequential allocation problem with a paid information channel, over a small, fixed and largely exhausted set of feasible pairs**: a pair used today is unavailable for ever, a person used today is busy for at least eight days, and answers can be bought.

What the statement expects of a system, and the pipeline stage that provides it:

| Expectation | Pipeline stage | Status |
|---|---|---|
| Read the observable population and feedback; distinguish confirmed, missing and declined | Normalisation: every field as value + status + observed day; unknown is never zero | In place |
| Assess both directions; hard constraints never overridden | Feasibility: the supplied `eligibility()` is the only gate | In place |
| Select clarification within the daily budget | Clarification: hard-constraint bundles first; soft answers by value of information as an option (section 7) | In place; bundle ordering planned |
| Estimate outcomes with explicit uncertainty | Response model: directional yes-probabilities (section 5) | In place |
| Allocate across the pool, each person once; update as people arrive or become busy | Allocation: maximum-weight matching recomputed every day (section 6) | In place (exact up to 20 members); exact general matcher planned |
| Reproducible outputs and evidence to explain decisions | Deterministic, memory-free policy; per-decision explanation of observed evidence and unresolved information | Explanation planned |
| Keep normalisation, modelling and allocation separate | Separate modules behind the JSON interface | Planned refactor |
| A valid response in every assessed episode (one invalid episode disqualifies) | Output validator and safe fallback (section 13) | Planned |

Facts from the supplied data (10 pools of 200 members, 613 introductions, 1,270 feedback events): 700 of 2,000 members (35%) declined at least one hard field and can never be introduced; of 199,000 possible pairs 0.5% are feasible, 12.9% need clarification and 86.6% are infeasible; 886 answered responses (419 yes, 467 no) and 249 non-responses produced 84 mutual acceptances, 40 dates and **3 MSMI**. Declined fields alone cap coverage at 65%, and MSMI cannot be learned from the static files.

**Two structural facts shape everything that follows.**

- *The pair set is fixed and small.* Hard answers never change, so the set of truly feasible pairs is fixed when an episode starts: clarification only reveals it, and no pair can be used twice. Diagnostic over 72 independent episodes (section 12): 95.8 truly feasible pairs per episode on average (23.5 in sparse geography, 129.8 in shift), and only 34.8% of members have even one truly feasible partner (17.1% in sparse geography). That, not 65%, is the real coverage ceiling.
- *The score is a count.* All members arrive by day 20, so N = 200 in every episode and `primary score = (qualifying pairs summed over the 120 episodes) / 240`. Two teams tie exactly whenever their totals are equal, and the tie-breakers (coverage, then mutual acceptances, then lower clarification cost, then lower inference time) then decide the order.

Technical assumptions: (1) an answer is a noisy draw, so one "no" does not prove incompatibility; (2) outcomes of introduced pairs carry over to feasible pairs that were not introduced (not verifiable); (3) clarification is immediate and truthful unless declined, as the statement specifies; (4) whole-day timing; (5) members can leave without notice, so a pair postponed may be lost.

## 2. Method of proof

- **Data.** 720 simulated episodes: 40 seeds × 6 scenarios × 3 behaviour policies, giving 54,717 introductions and 109,434 person-level rows (yes 36.3%; answered 75.4%; yes among those who answered 48.2%). Every feature is built from the observation at the start of the decision day; an assertion checks that no feedback newer than that day is present.
- **Statistics.** Effects are log-odds ratios, `LOR = ln[p1/(1-p1)] - ln[p0/(1-p0)]`, with `z = LOR / SE`. Person-level hidden quantities are tested through the covariance between two answers of the same person.
- **Intervals.** 95% bootstrap over the **40 seeds**. The seed is the only independent unit: the simulator draws an introduction's randomness from the seed, the two people and the day, so the same pair on the same day has identical luck under every policy that shares a seed. Four scenarios (development, delayed, shift and drift) even generate identical members from the same seed; sparse and cold start diverge after the first few members. Treating scenario runs as independent understates uncertainty, and we corrected our own earlier intervals for this.
- **Held-out accuracy.** Every prediction is out-of-fold by seed (5-fold cross-validation grouped by seed).
- **Expected-value diagnostic.** To compare allocation rules without outcome noise, we compute the exact expected MSMI of each introduced pair from the generator's hidden values (quadrature over the shared pair noise, exact enumeration of the date delays). On independent seeds it predicts 0.736 MSMI per episode for greedy, against 0.745 observed in our runs. It is an analysis tool only, never a policy input, and realised outcomes remain the reported result.

## 3. Hypotheses

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Clarification is the dominant lever | Reference policy with vs without asks, 30 seeds | **Proven.** Without asks: mutual acceptances −7.33 ± 0.33 per episode (z = −22), coverage −20.8 ± 0.6 points |
| H2 | Agreement on a soft field raises the chance of a yes | LOR per field | **Proven for 4 of 7 fields** (section 5.1) |
| H3 | An unknown answer is not a mismatch | Mixture test | **Proven** (section 5.1) |
| H4 | People differ persistently in how readily they say yes and in whether they answer | Within-person covariance | **Proven** (section 5.2) |
| H5 | Those person effects can be learned well from history | Information bound, and a latent-trait model | **Refuted**: 5.6% of the variance is recoverable (section 5.3) |
| H6 | Knowing more soft answers raises prediction accuracy | AUC by number of known answers; AUC with and without buying answers | **Proven**: 0.531 → 0.621 across known answers, and +0.027 [+0.019, +0.035] when answers are bought (section 7) |
| H7 | A learned score with global matching beats the reference on MSMI | 600 paired episodes, 100 seeds | **Not proven**: +0.090 ± 0.048 (z = 1.9); perfect-information allocation gains at most +0.045 (H9), so the true effect is likely smaller. Coverage +0.14 ± 0.04 points is proven |
| H8 | Buying soft answers with the idle budget improves outcomes | 180 paired episodes, 30 seeds | **Not proven**: MSMI −0.05 ± 0.06, although known answers double (section 7) |
| H9 | The remaining head-room lies in allocation over the horizon, not in re-ranking | Slot-capped ceiling; allocation fed the true pair values, 36 paired episodes (diagnostic) | **Partly.** The ceiling is +0.18 per episode above greedy, but perfect-information allocation captures only +0.030 [+0.008, +0.054] (daily matching) and +0.045 [+0.022, +0.069] (slot-price planning) (sections 6.1, 12) |
| H10 | A coverage-first matching objective raises coverage without lowering MSMI | Paired episodes, coverage tier on vs off | **Planned** (Round 2) |
| H11 | Modelling the second stage improves the MSMI ranking of pairs | Out-of-fold on dates from rollouts; paired episodes | **Planned** (Round 2) |

## 4. Reciprocal feasibility

A pair (a, b) is introduced only if the supplied `eligibility(a, b)` returns *feasible*: for each direction x → y, y's gender is in x's `who_to_meet`, y's age is inside x's range, y's zone is acceptable to x, and x's smoking and children-present preferences are not violated; the relationship structures match; an explicit yes/no conflict on wanting children excludes the pair (unsure does not); and the schedules share a slot. A known hard failure takes precedence over missing information, and an unresolved hard field blocks the pair. Availability, past introductions and one use per person per batch are also enforced. No score can override this gate, and all modelling and exploration happens among pairs that pass it. Because a known failure already excludes a pair before every field is known, the same function also tells us which unclarified members still have any possible partner (section 7).

## 5. Modelling approach and probability analysis

**Targets and windows.** Directional: person X returns yes by the response deadline, 7 days after assignment (no response counts as "not yes", never as dislike). Joint: both return yes. Later stages (date within 30 days; both second-meeting answers within 3 days) are the evaluation outcome and not yet a training target: the supplied static files hold only 3 MSMI. Our own rollouts, however, contain several thousand dates with second-meeting answers, enough to model that stage, and Round 2 will add it (H11). Two quantities can matter more for MSMI than for a first yes: the answer rate, which counts twice (the first response and the timely second-meeting answer), and relationship-goal agreement, which the generator's source also uses in the second-meeting answer. We will test this from rollouts rather than assume it.

**Model.** A logistic regression for "said yes" with 19 inputs, all available before the decision: for each of the four core soft fields a same / different / unknown state; a count of agreements on the other three fields and a count of unknown core fields; the person's own answer rate and yes-given-answer rate so far (two separate terms, each shrunk toward the population rate) and how many answers we have seen; two partner-history terms; age gap, zone match and the day. The pair weight is the product of the two directional predictions. The form is justified below: agreement enters as one additive term per field, unknown is its own state, and answering and accepting are carried by separate terms.

### 5.1 Observable features: which matter, and why the model has this form

Agreement on each soft field, among rows where both answers are known and the person answered:

| Field | Yes if same | Yes if different | LOR | 95% interval | z |
|---|---|---|---|---|---|
| Relationship goal | 56.7% | 38.1% | **+0.759** | [+0.636, +0.893] | 11.4 |
| Pace | 57.7% | 41.3% | **+0.662** | [+0.533, +0.789] | 10.1 |
| Lifestyle | 51.9% | 44.1% | **+0.316** | [+0.217, +0.412] | 6.4 |
| Conversations | 51.8% | 45.9% | **+0.237** | [+0.078, +0.400] | 2.9 |
| Emotional availability | 47.0% | 47.9% | −0.037 | [−0.162, +0.074] | −0.6 |
| Space for relationship | 46.3% | 47.3% | −0.040 | [−0.162, +0.077] | −0.7 |
| Relocate | 48.5% | 46.2% | +0.091 | [−0.029, +0.229] | 1.4 |

Four fields have intervals that exclude zero, and they remain significant after correcting for seven tests. The other three show no evidence of an effect; the current model carries them as a single count with almost no weight, and the final build will drop them. These are pooled over the six scenarios; in the shift scenario the weights differ (section 13).

- **Agreement acts on accepting, not on answering.** The same test with "answered at all" as the outcome gives LOR +0.071 [−0.039, +0.179] for goal and −0.046 [−0.142, +0.047] for pace. This is why answering and accepting are modelled by separate terms.
- **The effects add on the log-odds scale.** With goal and pace both known, the yes rates are 66.5% (same, same), 50.8% (same, different), 47.9% (different, same) and 32.8% (different, different). The interaction `L(s,s) − L(s,d) − L(d,s) + L(d,d)` is +0.018 [−0.269, +0.290]. We find no interaction, so one additive term per field is adequate; the interval is wide enough that a moderate interaction cannot be excluded.
- **Unknown is neither "same" nor "different".** If an unknown pair is secretly one or the other, then `p_unknown = π·p_same + (1−π)·p_diff`, so `π = (p_unknown − p_diff)/(p_same − p_diff)`. Measured π is 0.562 [0.486, 0.642] for goal, 0.454 [0.346, 0.581] for pace, 0.609 [0.409, 0.916] for lifestyle and 0.461 [0.169, 0.897] for conversations. Every interval lies strictly between 0 and 1, so an unknown answer must not be scored as a mismatch (π = 0), which is what the reference soft-match count does. For goal, π also agrees with the chance that two random people share an answer (0.500); for pace and lifestyle it is somewhat above chance (0.334), so we claim only the weaker statement.

### 5.2 Hidden quantities: proof that they exist, from observable answers only

**A persistent "generosity" of each person.** Among people with a past answer, the chance of a yes is 51.8% if their last answer was yes and 42.6% if it was no: LOR +0.370 [+0.298, +0.447], z = 9.5. To separate this from compatibility, let `m` be the probability of a yes from compatibility alone and `r = y − m`. For two answers of one person with a person effect `g`, `Var(g) = τ²`, a first-order expansion of the logistic gives

`E[r1·r2] ≈ m1(1−m1) · m2(1−m2) · τ²`, hence `τ² = Σ r1·r2 / Σ m1(1−m1)m2(1−m2)`.

Over 70,946 within-person answer pairs, `E[r1·r2]` = +0.0191 [+0.0157, +0.0223] (it would be 0 with no person effect), giving **τ² = 0.320 [0.264, 0.376]**, a standard deviation of 0.57 on the log-odds scale. That is nearly as large as the goal effect.

**A persistent response rate.** A person who answered last time answers 77.3% of the time, against 70.6% after a silence: LOR +0.348 [+0.258, +0.431], z = 8.0. The covariance of two answered-indicators of the same person equals the variance of the response rate: 0.0141 [0.0116, 0.0169], a standard deviation of 0.119 around a mean of 0.754.

**A pair-specific component shared by both people.** Within one introduction, B says yes 39.7% of the time when A said yes and 34.7% when A did not: LOR +0.216 [+0.115, +0.314], z = 4.2. After removing compatibility, the residual correlation is `ρ = Σ rA·rB / Σ √(pA(1−pA)·pB(1−pB))` = +0.037 [+0.013, +0.061]. So the two directions are not independent: multiplying them predicts 13.5% "both yes" against 14.4% observed. With ρ estimated on training seeds, `pA·pB + ρ·√(pA(1−pA)pB(1−pB))` predicts 14.3% on held-out seeds against 14.8% observed. We therefore use the product only to rank pairs and report it as uncalibrated.

**Check against the generator (diagnostic).** The generator draws response rates uniformly on [0.55, 0.98], a mean of 0.765 and a standard deviation of 0.124; our estimate from answers alone (0.754, 0.119) recovers both. Its generosity has a standard deviation of 0.70; our first-order estimate (0.57) is attenuated, as expected from linearising the logistic. The estimators find what is there, and the attenuation is conservative.

### 5.3 How much of the hidden person effect can be learned

One yes/no answer carries Fisher information `I = m(1−m)` = 0.244 about `g`. With a `N(0, τ²)` prior, the share of the variance of `g` that `n` answers can recover is

`R(n) = n·I·τ² / (1 + n·I·τ²)`: R(1) = 0.073, R(2) = 0.135, R(3) = 0.190, R(5) = 0.281, R(10) = 0.439.

At decision time the person has 0 past answers in 51.8% of cases, 1 in 26.1%, 2 in 12.9%, 3 in 6.0% and 4 or more in 3.2%. Averaged over decisions, **only 5.6% of the generosity variance is recoverable**. Using the generator's own spread (0.70) instead of our attenuated estimate raises this to about 8%; the conclusion does not change. This is a limit of the data, not of the model, and two tests agree with it:

- A latent-trait model that infers generosity after adjusting for the compatibility of each past partner, and a full probabilistic model that also infers response rates and unknown soft answers by passing evidence between people, both match the simple model: change in held-out AUC −0.000 and +0.001, neither distinguishable from zero.
- Accuracy does not rise with history: AUC is 0.566, 0.572, 0.568 and 0.575 for 0, 1, 2 and 3+ past answers.

*Diagnostic, clearly separate from the model:* using the simulator's internal probabilities for analysis only (never as an input), a perfect model would reach AUC 0.69, and knowing each person's true generosity alone would give 0.64. The bound above explains why that gap cannot be closed from the answers available.

### 5.4 Accuracy and calibration of the yes-model (out-of-fold by seed)

| Predictor of "said yes" | AUC | 95% interval |
|---|---|---|
| Reference soft-match count | 0.506 | [0.499, 0.513] |
| Person's own history only | 0.533 | [0.526, 0.540] |
| Agreement features only | 0.555 | [0.545, 0.565] |
| Proposed yes-model (19 features) | **0.569** | [0.560, 0.579] |

The gain over the reference score is +0.063 [+0.054, +0.074]; the reference score is statistically indistinguishable from chance. The model is calibrated: slope 0.948 [0.850, 1.065] against a perfect 1, and predicted and observed yes rates differ by at most 1.3 points across ten deciles (24% to 48%). Log-loss is 0.6474 against 0.6553 for a constant rate. An AUC of 0.57 is modest, and section 5.3 shows why: most of what decides an answer is hidden and not learnable from this much history.

## 6. Decision and allocation method

**Current policy.** Each day, among available members, we build the graph of feasible, not-yet-introduced pairs and solve a maximum-weight matching: maximise `Σ w_ab` with each person in at most one pair, where `w_ab = p̂_a · p̂_b + ε`. The constant ε makes the solution prefer more introductions to fewer. The matching is exact for connected groups of up to 20 members; larger groups (the real feasible clusters have 34–65) currently use a greedy-plus-swaps heuristic. Waiting is expressed by omitting people; it occurs when no feasible pair exists. The policy keeps no memory and rebuilds its state from the observation, so nothing carries between episodes.

### 6.1 What the diagnostics say about allocation

- **Where pairs are lost.** Greedy leaves 22% of truly feasible pairs unused. Of 1,500 unused pairs across 72 episodes, 57% were feasible with both people free on some day, but the matcher chose other pairs (competition for the same candidate); 36% lost a member to an unannounced departure or a pause after success first; 7% never had both people free on the same day. None was lost to clarification: under the reference ask order, every truly feasible pair is revealed by day 59.
- **Slots bind for well-connected members.** Most worlds contain members with more feasible partners than eight-day slots (up to 21 partners against at most 8 slots), so not every pair can be used. A slot-capped ceiling, a linear-programming relaxation that gives each member only the slots between arrival, departure and day 59, is 0.918 MSMI per episode against 0.736 for greedy.
- **Most of that ceiling is not reachable.** Fed the true value of every pair, daily maximum-cardinality matching gains +0.030 [+0.008, +0.054] per episode, and planning with slot shadow prices +0.045 [+0.022, +0.069]. The ceiling ignores that both people must be free on the same day and that a mutual yes keeps both busy until after the date.
- **Consequence.** Ordering choices are worth at most about 5% MSMI even with perfect knowledge. The allocation therefore aims first at what is dependable, never wasting a feasible pair and maximising coverage, and uses predicted value for the remaining choices.

### 6.2 Round 2 allocation design

1. **Exact matcher.** Edmonds' blossom algorithm for maximum-weight matching on a general graph (preferences are not bipartite: non-binary members and same-gender preferences exist), implemented in the standard library. The daily feasible graph has at most about 150 pairs, far inside the 10-second limit. This removes the 20-member exact limit.
2. **Tiered weights aligned with the ranking.** `w_ab = C + B·(u_a + u_b) + v̂_ab`, with `C ≫ B ≫ max v̂`. The constant `C` gives maximum cardinality: a median pair carries only about 0.7% MSMI, but a pair not used is usually lost for good. `u = 1` for a member never introduced, scaled up when the member has few feasible partners left; this targets coverage, the first tie-breaker. `v̂` is the predicted pair value and decides among the remaining options.
3. **Slot prices (option).** A daily LP over all revealed, unused pairs with each member's remaining eight-day slots gives each member a shadow price per slot; `v̂_ab − y_a − y_b` is a pair's value net of the slots it consumes, which lets a heavily connected member wait for a better partner. With true values it added +0.015 over daily matching, a difference we cannot yet distinguish from zero, so it is an ablation rather than the default.
4. **Front-loading.** A member whose remaining feasible partners fit into their remaining slots is never held back. Departures are unannounced (36% of lost pairs), and in the drift scenario acceptance falls after day 35.
5. **Value term.** `v̂ = p̂_a · p̂_b` today; Round 2 adds the second-stage term (H11). It is used only to rank and is reported as uncalibrated (section 5.2).

## 7. Clarification strategy: how the 12 daily units are spent

**What the statement provides.** 12 units per day; the hard-constraint bundle of one member costs 3 units and one named soft field costs 1; "clarification is immediate in v1.0 and returns exact simulated self-reports when disclosed. Declined answers remain unavailable"; only available members can be asked.

**Why it matters.** Two facts from the data:

1. *Hard bundles create every introduction.* Without them, coverage falls by 20.8 points (H1).
2. *Soft answers decide how accurate the model can be.* Out-of-fold AUC by how many of the four core answers are known for both people:

| Core answers known for both | Share of introductions | AUC | 95% interval |
|---|---|---|---|
| 0 of 4 | 55.5% | 0.531 | [0.520, 0.543] |
| 1–2 of 4 | 18.5% | 0.580 | [0.559, 0.599] |
| 3–4 of 4 | 25.9% | 0.621 | [0.604, 0.638] |

The difference between the top and bottom rows is +0.090 [+0.073, +0.108]. More than half of all introductions are made with no core answer known for both people, and the reference rule leaves most of the budget unused: about 196 of the 720 units available in an episode (27%).

**Which answer to buy.** Knowing whether a pair agrees on field k moves its log-odds by `a_k` with probability `π_k` and by 0 otherwise, so the variance of the score that one answer removes is `V_k = a_k² · π_k · (1 − π_k)`. With the effects of section 5.1 and the chance of agreement measured from observed answers: goal 0.144 (52.5% of the total), pace 0.097 (35.5%), lifestyle 0.022 (8.1%), conversations 0.011 (3.9%). Goal and pace carry 88% of the value. (The implementation tested below takes `a_k` from the fitted yes-model instead; it gives the same order for goal and pace, which is what matters.)

**The algorithm.**

1. *Hard bundles first, most useful first.* For every available member with a missing, not-declined hard field, ask the bundle (3 units), up to the budget. Never ask a declined member. Round 2 orders these asks by the number of potential partners not yet ruled out: `eligibility()` already returns *infeasible* on a known failure, so public age, gender and zone plus partners' known answers show who could still match. Members with no possible partner are skipped and re-checked daily, since arrivals continue to day 20. Under the reference order every feasible pair is already revealed by day 59 (section 6.1), so this ordering changes timing and clarification cost (the third tie-breaker), not reach; it matters most in cold start.
2. *Then soft answers by value of information, with the units left.* A soft answer matters only through pairs we could still introduce, so the value of asking member m about field k is
   `V_k × Σ over m's feasible, not-yet-introduced partners j of { 1 if j's answer on k is known; 0.5 if it is unknown but still askable; 0 if j declined it }`.
   Pick the highest value, update the partners' values (an ask makes that answer "known"), and repeat. Only available members with a not-asked field are asked.
3. *Stop when no ask has positive value.* Unused units stay unused, because clarification cost is a ranking tie-breaker.

Hard bundles come first because an introduction that cannot be made is worth nothing, whereas a soft answer only reorders introductions that will mostly be made anyway.

**Measured effect.** 30 seeds × 6 scenarios, paired with the same policy without soft asks:

| Quantity | Without soft asks | With the algorithm | Paired change |
|---|---|---|---|
| Core answers known for both people at introduction | 31.6% | 61.3% | +29.6 ± 0.7 points |
| Clarification units per episode | 197 | 271 | +73.6 ± 2.9 |
| Mutual acceptances per episode | 11.3 | 11.2 | −0.17 ± 0.21 |
| MSMI per episode | 0.97 | 0.92 | −0.05 ± 0.06 |
| Coverage | 0.340 | 0.339 | −0.04 ± 0.05 points |
| Yes-model AUC, out-of-fold (measured on the 40 analysis seeds) | 0.568 | 0.595 | **+0.027 [+0.019, +0.035]** |

The algorithm does what it is designed to do: it nearly doubles the information available when a pair is chosen, and the yes-model becomes measurably more accurate. It does not change the outcomes, which agrees with section 6.1: even perfect pair values move MSMI by only about 5%. A variant that holds 3 units a day back from hard bundles to buy soft answers earlier loses 1.5 ± 0.5 introductions per episode (z = −2.9), which confirms that hard bundles must come first.

**Decision.** Buying soft answers is proven to improve prediction and not proven to improve outcomes, and clarification cost is a ranking tie-breaker. So it stays an option that is off in the default policy. We keep the algorithm for two reasons: it is the principled way to spend the idle budget if a later test shows an outcome gain, and known answers make each decision easier to explain (section 8).

## 8. Treatment of uncertainty, missing data and delayed feedback

- **Missing, refused and not-asked** stay distinct. Only not-asked fields are bought; declined answers are never inferred.
- **Unknown is never zero.** It is a third state with its own (neutral) effect, as proven in section 5.1.
- **Person-level uncertainty** is carried by shrinkage: `(yes + 3 × 0.47) / (answered + 3)` and `(answered + 3 × 0.78) / (seen + 3)`. A person with no history gets the population rate. The two constants were computed on all ten supplied pools; on the six training pools alone they are 0.48 and 0.80, and the final build will use those to respect the pool split.
- **Silence is not rejection.** It lowers the estimated response rate, not the estimated willingness to accept.
- **Delayed and censored feedback.** Features use only events whose observed day is on or before the decision day; recent introductions are open, never failures.
- **Three sources of uncertainty kept apart** in any explanation: unresolved information (unknown or declined fields), model uncertainty (AUC 0.57, calibrated) and outcome chance (a 40% prediction still ends in "no" most of the time). No latent simulator value is presented as an observation.

## 9. Learning strategy

- **Offline:** the yes-model is trained on our own simulator rollouts with declared seeds, split by seed, using only pre-decision observations and observed answers.
- **Online:** each person's answer and yes rates update as their feedback arrives.
- **Selective labels:** outcomes exist only for introduced pairs and propensities are null, so we make no claim of unbiased inverse-propensity evaluation; policies are compared through complete simulator episodes.
- **Cold start:** population priors, with shrinkage; section 5.3 shows that history adds little even later.
- **Exploration:** none is built in. Most feasible pairs are eventually introduced (78% under greedy, section 12), so there is little to explore among pairs, and a bandit variant gave no gain. Information is acquired deliberately through clarification instead (section 7). Any exploration would act only among pairs that pass eligibility.

## 10. Baselines

The three supplied baselines on identical seeds and all six scenario families: greedy (reference clarification, best soft-match pair first), no clarification, and random feasible. In our runs greedy and random feasible show no significant difference in MSMI, mutual acceptances or coverage over 60 seeds (section 12), consistent with the soft-match score carrying no information (AUC 0.506).

## 11. Planned experiments, ablations and experimental design

- **Independent worlds.** Use a different seed for every scenario so that episodes are independent, and compute every interval over seeds.
- **Power.** The paired MSMI difference has a standard deviation of about 0.5 per seed, so resolving a difference of 0.1 at two standard errors needs about 100 seeds per comparison; mutual acceptances (about 11 per episode) are the lower-noise secondary metric, and the expected-value diagnostic (section 2) is a third, near-noiseless check for allocation rules.
- **What the private ranking can resolve.** At the reference level about 88 pairs qualify across 120 episodes; with Poisson variation that total has a standard deviation of about 9.4, or ±0.04 on the primary score. The largest allocation gain we measured with perfect information is about 5.4 pairs (+0.023). Between competent policies the primary score will often tie or differ by noise, so coverage is a design target, not an afterthought, and we report total counts alongside rates.
- **Ablations (hypothesis-driven):** no clarification; soft answers not bought; greedy instead of global matching; soft-match count instead of the yes-model; yes-model without the person's history; coverage tier off (H10); slot prices on vs off; hard-bundle order by potential partners vs list order; exact blossom matcher vs greedy-plus-swaps.
- **Reported for every method:** the outcome funnel (assignments, mutual acceptances, dates, MSMI), distinct-member coverage, first-introduction waits, unserved members with reasons, clarification cost, missing feedback and runtime, per scenario.
- **Build work:** module separation, an output validator with a safe fallback, a per-decision explanation, an exact matcher for large groups, priors from training pools only, and a container run before submission.

## 12. Preliminary policy evidence and head-room diagnostics

Paired against the greedy reference; the unit is the seed.

| Policy | Seeds | MSMI per episode | Mutual acceptances per episode | Coverage (points) |
|---|---|---|---|---|
| No clarification | 30 | −0.41 ± 0.10 | −7.33 ± 0.33 | −20.8 ± 0.6 |
| Random feasible | 60 | +0.01 ± 0.07 | +0.39 ± 0.25 | +0.05 ± 0.08 |
| Yes-model + global matching | 100 | +0.090 ± 0.048 | +0.08 ± 0.14 | **+0.14 ± 0.04** |

For reference, greedy scores 0.745 MSMI and 11.3 mutual acceptances per episode. The yes-model with global matching serves slightly more people in all three studies that include it (coverage is the first tie-breaker), but its MSMI gain is not established (z = 1.9).

*Diagnostic, not a valid policy:* an allocation that ranks pairs by the simulator's internal yes-probabilities gains +0.79 ± 0.23 mutual acceptances per episode (about 7%) and +0.01 ± 0.06 MSMI.

**Head-room diagnostic** (12 independent seeds per scenario; expected MSMI per episode from the generator's hidden values; perfect-information columns use the first 6 seeds, paired with greedy on the same worlds):

| Scenario | Truly feasible pairs | Used by greedy | Greedy | Slot-capped ceiling | Perfect-information daily matching | Perfect-information slot prices |
|---|---|---|---|---|---|---|
| Development | 106.2 | 83.9 (79%) | 0.819 | 0.977 | +0.035 | +0.044 |
| Sparse geography | 23.5 | 22.5 (96%) | 0.241 | 0.255 | +0.005 | +0.005 |
| Cold start | 91.8 | 71.2 (78%) | 0.705 | 0.891 | +0.041 | +0.051 |
| Delayed dates | 115.9 | 87.7 (76%) | 0.889 | 1.142 | +0.063 | +0.054 |
| Shifted weights | 129.8 | 98.8 (76%) | 0.870 | 1.114 | +0.044 | +0.103 |
| Drift | 107.7 | 85.8 (80%) | 0.890 | 1.127 | −0.005 | +0.015 |
| **Mean** | 95.8 | 75.0 (78%) | **0.736** | **0.918** | **+0.030** [+0.008, +0.054] | **+0.045** [+0.022, +0.069] |

Members with at least one truly feasible partner: 34.8% on average; greedy serves 33.7%. Causes of the 1,500 unused pairs: chosen against while both people were free 57%, a member left or paused first 36%, never free on the same day 7%, never revealed by clarification 0%.

Reading the table: sparse geography is supply-bound (96% of pairs already used); the scenarios with the most room are delayed dates and shifted weights, where a slot is held longer or the pooled model is most wrong. The largest single effect of perfect values appears in shift (+0.103), which is consistent with pooled soft-field weights mis-ranking pairs there.

## 13. Expected failure modes, limitations and reproducibility

| Failure | Why | Mitigation |
|---|---|---|
| Invalid episode | One invalid assessed episode makes the whole entry ineligible | A final validator re-checks every ask (budget, availability, duplicates, field names) and every pair (eligibility, availability, repeats, overlap) with the simulator's own rules and drops offenders; any internal error falls back to empty lists, which are always valid; a time guard keeps each call well under 10 s; container run before submission |
| Unreachable members (35%) | Declined hard fields never resolve | Reported separately; no budget spent on them |
| No feasible partner at all | 65% of members on average (83% in sparse geography) | Reported as unserved with the reason; coverage measured against this ceiling |
| Competition for the same candidate | 57% of lost pairs | Exact global matching; coverage tier; optional slot prices |
| Unannounced departures | Members can leave without notice; 36% of lost pairs | Introduce early; no holding without a slot price |
| Sparse supply | 23.5 feasible pairs per episode with 12 zones; greedy already uses 96% | Ask early; accept low coverage |
| Cold start | No history early, and little later (section 5.3) | Population priors and shrinkage; asks ordered by potential partners |
| Delayed dates | Dates after day 30 miss the MSMI window, and slots are held longer | Introduce early |
| Drift | In the drift scenario the yes rate falls from 38.2% to 29.9% after day 35 | A day term and front-loading; we have no working detector yet |
| Shifted weights | Pooled soft-field effects mis-rank pairs when the weights change | Per-scenario reporting; the value term only orders choices after cardinality and coverage |
| Weak signal | AUC 0.57; most variation is hidden and unlearnable | Calibration and shrinkage; no over-claiming |
| Dependence between the two directions | ρ = +0.037 | Product used for ranking only |
| Large-component matching | Exact only up to 20 members today | Blossom matcher (section 6.2) |
| Exact ties on the primary score | The score is a count divided by 240 | Coverage-first tier |
| Overfitting to the public simulator | Private worlds differ | Per-scenario reporting; no use of hidden state |

**Limitations.** The only large, proven lever is clarification. Even with perfect knowledge of every pair, allocation moves MSMI by only about 5%, so the proposed scoring and allocation give a small, proven gain in coverage and an unproven gain in MSMI. The head-room diagnostics read hidden generator values and say what is possible in this simulator, not how to achieve it with observable data. Simulator results say nothing about real people; real use would need an adapter that maps authorised observations without fabricating defaults, re-checks constraints, and returns suggestions for human review.

**Reproducibility.**

```
python -m unittest -v
python verify_data.py                 # on Windows, clone with: git clone -c core.autocrlf=false <repo>
python collect_seq.py --seeds 1001-1040 --output results/seq_rows.npz
python proofs.py                      # sections 5.1-5.3 and the value-of-information weights
python oof_auc.py                     # section 5.4
python auc_models.py                  # latent-trait model and the diagnostic ceiling (section 5.3)
python collect_world.py && python prob_model.py     # full probabilistic model (section 5.3)
python experiments.py --configs greedy,value_ask_only,logit_global,gbm_global,logit_value,gbm_value --seeds 601-700 --output results/study5.json
python headroom.py --seeds 2001-2012 --output results/headroom.json      # sections 1, 6.1, 12 (diagnostic)
python oracle_alloc.py development 2001 2002 2003 2004 2005 2006         # repeat for each scenario: H9 (diagnostic)
```

Diagnostic seeds are 2001–2012 plus 1,000 × the scenario index (development 0, sparse 1, cold start 2, delayed 3, shift 4, drift 5), so every scenario uses independent worlds. The two diagnostic scripts read hidden generator values and use NumPy, SciPy and NetworkX; they are analysis tools and are never part of the policy or its image.

The policy uses only the Python standard library at inference, with parameters we trained ourselves; the blossom matcher will also be written in the standard library. Analysis used NumPy, SciPy, pandas, scikit-learn and NetworkX. No external data or pretrained model is used.
