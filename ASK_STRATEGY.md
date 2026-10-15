# Who to ask: a Bayesian ordering of clarification questions

**Team:** ML_Wizards_IITM · Round 2 · simulator release 1.0.0

This note explains how the policy decides whose deal-breakers to ask about each day, why we replaced the reference rule, and what the experiments show. All results come from the public synthetic simulator and describe that invented environment only.

---

## 1. The problem

Every day the policy has **12 clarification units**. Asking one person's hard-constraint bundle (all 11 deal-breakers at once) costs 3 units, so the policy can ask about **4 people per day**. A person whose deal-breakers are unknown cannot be introduced at all, so these questions decide who becomes matchable and when.

Each day the policy therefore has to choose which 4 of the askable people to ask first. Askable means available, with at least one deal-breaker unknown, and none declined.

Two facts frame the choice:

- **Almost everyone askable is asked by about day 17–20.** The budget is fully used early and idle afterwards. So changing the order mainly changes **when** people are unlocked, not **whether** they are.
- **47% of bundles go to people who turn out to have no compatible partner at all** (measured over 18 episodes). Only skipping an ask lowers the clarification cost, which is the third tie-breaker.

---

## 2. A worked example

Day 0. Three people have already answered all their deal-breakers:

| Person | Visible facts | Deal-breakers |
|---|---|---|
| **Asha** | woman, 28, zone_a | wants men; ages 25–33 (3 below, 5 above her own age); zone_a only; no smokers; weekends |
| **Ben** | man, 30, zone_a | wants women; ages 26–36 (4 below, 6 above); zones a and b; any smoking; weekends and weekday evenings |
| **Chen** | man, 35, zone_c | wants women; ages 30–40 (5 below, 5 above); zone_c only; no smokers; weekday evenings |

Six people have unknown deal-breakers and could be asked:

| Person | Visible facts | Position in the member list |
|---|---|---|
| Kiran | man, 44, zone_d (a rare zone, 5% of people) | 1st |
| Dev | non-binary, 38, zone_b | 2nd |
| Meera | woman, 24, zone_c | 3rd |
| Priya | woman, 29, zone_a | 4th |
| Ravi | man, 31, zone_a | 5th |
| Sana | woman, 33, zone_a | 6th |

The scores in this example are illustrative. The measured results are in section 6.

---

## 3. The methods

### 3.1 Before: the reference rule (list order)

This is the organisers' `kit.baseline_asks`, which every earlier configuration of ours (`final`, `tiered`) used:

```python
eligible = [m for m in state['members'] if m['available'] and any(m['fields'][k] is None for k in HARD)
            and not any(m['field_status'][k] == 'declined' for k in HARD)]
return [{'member_id': m['member_id'], 'field': 'constraints'} for m in eligible[:state['ask_budget_remaining'] // 3]]
```

- **Day 0:** Kiran, Dev, Meera, Priya.
- **Day 1:** Ravi, Sana.

The member list is in the order the simulator created people, which is **unrelated to anyone's chance of matching**. Kiran is asked first although almost nobody lives in or accepts zone_d. Ravi and Sana, who are very likely to match, wait a day.

### 3.2 Round 1 alternative: count possible partners

`experimental.value_asks` counts, for each askable person, everyone not **already ruled out** by a known fact (`eligibility(...) != 'infeasible'`), and asks the highest counts first.

- **Kiran:** Asha, Ben and Chen rule him out (age range; both men want women), but all 5 people with unknown answers still count as possible. Count = 5.
- **Priya:** Asha and Chen rule her out; Ben and the 5 unknown people remain. Count = 6.

**The flaw:** an unknown answer is treated as fully compatible, so Kiran looks almost as promising as Priya. As a predictor of who really has a partner, this count reaches only AUC 0.59 (section 5).

### 3.3 New: Bayesian, by imagining the missing answers

**Idea.** We don't know Priya's deal-breakers, but we know what complete sets of deal-breakers look like in this pool, because some people have already answered. We borrow a whole set from one of them and fit it to Priya's own age and zone. Answers that go together therefore stay together, for example a narrow age range with a single acceptable zone.

**Prior: the observed complete answer sets** (`smp/clarify.py`, `_templates`). Each set is made relative to its owner:
- the age range becomes "years below and above my own age";
- the acceptable zones become "extra zones beyond my own";
- every other deal-breaker is copied as is.

**One draw** (`_imagine`). Priya's known answers stay as they are. Each unknown one is taken from the borrowed set. Here she borrows **Asha's** set:

| Deal-breaker | Asha's answer | Priya's imagined answer |
|---|---|---|
| Wants | men | men |
| Ages | 3 below to 5 above own age | 26–34 |
| Zones | own zone only | zone_a only |
| Smoking partner | no smokers | no smokers |
| Schedule | weekends | weekends |

In the same draw, every other person with unknown answers gets an imagined set too. We then count Priya's compatible partners with `compatible()`, a fast copy of the organisers' eligibility check. It checks both directions and agrees with `kit.eligibility` on all 59,700 test pairs:

| Possible partner | Compatible in this draw? | Why |
|---|---|---|
| Ben (man, 30, zone_a) | yes | each accepts the other |
| Ravi (man, 31, zone_a, imagined from Ben's set) | yes | ages, zone and schedule fit |
| Chen (zone_c) | no | she accepts zone_a only |
| Kiran (44) | no | outside 26–34 |

This draw gives 2 partners.

**Twelve draws** (`BAYES_DRAWS = 12`, deterministic for a given day):

| Draw | Set Priya borrowed | Priya's partners |
|---|---|---|
| 1 | Asha | 2 |
| 2 | Chen (wants women, zone_c only) | 0 |
| 3 | Ben (zones a and b) | 3 |
| … | … | … |
| 12 | Asha | 2 |

The two outputs:

- **P(at least one partner)** = draws with at least one partner ÷ 12, here 11/12 = **0.92**.
- **Expected partners** = average count over the draws, here **2.4**.

Scores for everyone (illustrative):

| Person | P(at least one partner) | Expected partners | Why |
|---|---|---|---|
| Priya | 0.92 | 2.4 | common zone, typical age |
| Ravi | 0.92 | 2.1 | same |
| Sana | 0.75 | 1.5 | same zone, slightly older |
| Meera | 0.50 | 0.9 | zone_c is less common, and she is young |
| Dev | 0.25 | 0.4 | few people want to meet non-binary people |
| **Kiran** | **0.00** | **0.0** | nobody in the pool accepts zone_d, and he is 44 |

**Bayesian order:** Priya, Ravi, Sana and Meera on day 0, then Dev, with Kiran last.

```python
def bayes_bundle_asks(state, skip_below=None):
    candidates = [m for m in state['members'] if m['available'] and not is_complete(m) and not is_blocked(m)]
    chances = match_chances(state, candidates)                     # {id: (P(at least one), expected partners)}
    order = sorted(candidates, key=lambda m: (-chances[m['member_id']][1], m['member_id']))
    if skip_below is not None:
        order = [m for m in order if chances[m['member_id']][0] >= skip_below]
    return [{'member_id': m['member_id'], 'field': 'constraints'} for m in order[:state['ask_budget_remaining'] // 3]]
```

The model uses only observable data: known answers, visible age, gender and zone, and past introductions. It never reads the simulator's hidden values. Scoring a whole day takes about 0.07 s.

---

## 4. Skipping, and when a skip ends

**Skipping means not asking someone today because their chance of any partner is below 5%** (`skip_below=0.05`). It is decided again every day; nobody is excluded for good.

Following Kiran:

| Day | What happens | Kiran's chance | Asked? |
|---|---|---|---|
| 0 | better candidates are asked first | 0.00 | no (last in order, and below 5%) |
| 1 | only Dev and Kiran remain and the budget is free | 0.00 | **no, skipped** |
| 2–6 | nothing changes | 0.00 | skipped every day |
| 7 | **Tara arrives**: woman, 45, zone_d, with known answers that accept men aged 40–50 | 0.60 | **yes** |

A skip ends in one of two ways:

1. **The chance rises above 5%**, because someone arrives or more answers become known. The person is then asked as usual, as Kiran was on day 7.
2. **The decision days run out with the chance still below 5%.** The person is never asked, which saves 3 units. In our experiments such people would not have been served even if asked: coverage was identical (section 6).

---

## 5. Experiment 1: does the Bayesian score predict who really has a partner?

`python ask_model_study.py --seeds 3001-3020 --output results/ask_model_study.json`

- **Sample.** 7,920 askable people from 120 independent worlds (20 seeds × 6 families). Each is scored on the first day they could be asked, using only observable data.
- **Label (diagnostic).** Does the person truly have at least one compatible partner among people who can ever be introduced? This reads the generator's true answers and is used only to evaluate, never by the policy. 52% do.

| Score | AUC for "has a partner" |
|---|---|
| Reference list order | no relation (arbitrary order) |
| Count of possible partners (Round 1 rule) | 0.586 |
| **Bayesian expected partners** | **0.732** |
| Bayesian P(at least one) | 0.727 |

The Bayesian score wins in every family:

| Family | Bayesian | Count rule | Share with a partner |
|---|---|---|---|
| development | 0.686 | 0.560 | 58% |
| sparse | 0.662 | 0.527 | 24% |
| cold_start | 0.674 | 0.573 | 58% |
| delayed | 0.698 | 0.559 | 56% |
| shift | 0.722 | 0.603 | 60% |
| drift | 0.705 | 0.566 | 57% |

**Calibration.** The score ranks well but is over-confident at the extremes:

| Predicted P(at least one) | Actually had a partner | People |
|---|---|---|
| below 0.05 | 13% | 829 |
| 0.05–0.2 | 29% | 1,122 |
| 0.2–0.4 | 45% | 1,191 |
| 0.4–0.6 | 59% | 2,191 |
| 0.6–0.8 | 69% | 1,510 |
| 0.8 and above | 77% | 1,077 |

Likely causes:
- only 12 draws;
- the borrowed answer sets come from people who already answered, who may differ from the people still unknown;
- people who arrive later add partners the model cannot foresee.

**What Experiment 1 suggested about skipping.** Scored once, on the first askable day, a 5% threshold would skip 829 people (6.9 per episode). 110 of them did have a partner (0.9 per episode at risk). Because coverage outranks clarification cost, this looked like a bad trade. Experiment 2 tests it properly, with daily re-scoring.

---

## 6. Experiment 2: full episodes, paired

```
python experiments.py --configs tiered,tiered_valueask,tiered_bayes,tiered_bayes_skip \
       --seeds 4001-4060 --variants all --independent --output results/study_asks.json
```

The four configurations ran on the **same 360 worlds** (60 seeds × 6 families, a separate world per family). Matching is the exact tiered matcher and the yes-model is the shipped 14-feature model in all four. Only the asking differs:

| Config | Who is asked first | Skip |
|---|---|---|
| `tiered` | list order (reference rule) | no |
| `tiered_valueask` | count of possible partners | no |
| `tiered_bayes` | Bayesian expected partners | no |
| `tiered_bayes_skip` | Bayesian expected partners | below 5%, re-scored daily |

### Means per episode

| Config | MSMI / 100 | Mutual / 100 | Coverage | Ask cost | Introductions | Wait (days) |
|---|---|---|---|---|---|---|
| tiered | 0.382 | 5.70 | 0.342 | 198.7 | 75.5 | 6.31 |
| tiered_valueask | 0.418 | 5.71 | 0.343 | 198.7 | 76.7 | 5.49 |
| tiered_bayes | 0.401 | 5.59 | 0.343 | 198.7 | 76.8 | 5.31 |
| **tiered_bayes_skip** | 0.400 | 5.58 | 0.343 | **191.2** | 76.8 | 5.32 |

### Paired differences (95% interval over 60 seeds; ✱ = excludes zero)

| Comparison | Wait (days) | Introductions | Coverage | MSMI / 100 | Mutual / 100 | Ask cost |
|---|---|---|---|---|---|---|
| count rule − list order | −0.82 ± 0.08 ✱ | +1.23 ± 0.61 ✱ | +0.0003 ± 0.0006 | +0.036 ± 0.044 | +0.006 ± 0.177 | 0 |
| Bayesian − list order | −1.00 ± 0.08 ✱ | +1.28 ± 0.50 ✱ | +0.0003 ± 0.0006 | +0.019 ± 0.050 | −0.117 ± 0.147 | 0 |
| **Bayesian − count rule** | **−0.18 ± 0.07 ✱** | +0.06 ± 0.48 | −0.0000 ± 0.0005 | −0.017 ± 0.053 | −0.122 ± 0.163 | 0 |
| **skip − no skip** (both Bayesian) | +0.008 ± 0.003 ✱ | +0.006 ± 0.011 | +0.0000 ± 0.0000 | −0.001 ± 0.003 | −0.004 ± 0.012 | **−7.51 ± 0.49 ✱** |

Bayesian vs list order, by family:

| Family | Wait (days) | Coverage |
|---|---|---|
| development | −0.60 ± 0.17 ✱ | +0.0010 ± 0.0015 |
| sparse | −1.51 ± 0.28 ✱ | +0.0005 ± 0.0008 |
| cold_start | −1.56 ± 0.26 ✱ | +0.0011 ± 0.0015 |
| delayed | −0.66 ± 0.20 ✱ | +0.0000 ± 0.0020 |
| shift | −0.80 ± 0.16 ✱ | −0.0004 ± 0.0014 |
| drift | −0.83 ± 0.18 ✱ | −0.0003 ± 0.0017 |

**Served people.** The skip version and the no-skip version served **exactly the same total, 24,664 people over 360 episodes**.

**Speed.** The slowest single call across all 1,440 episodes took 0.17 s, against a 10 s limit.

---

## 7. The metrics, read with the example

| Metric | Meaning | In the example | Measured result |
|---|---|---|---|
| **Wait** | days from a person's arrival to their first introduction, averaged over people who got one | Priya is asked first and introduced on day 0 instead of a day or two later | 6.31 → 5.31 days (−1.0, proven) |
| **Introductions** | total pairs introduced in the episode | starting earlier gives Priya more 8-day slots before day 59 | +1.3 per episode (proven) |
| **Coverage** | share of the 200 members with at least one introduction | the order changes *when* Ravi and Sana are served, not *whether*; Kiran depends on Tara arriving under every method | +0.0003 (not distinguishable from zero) |
| **MSMI / 100** | qualifying second-meeting pairs per 100 members, the primary score | more introductions can only help a little; the outcome is mostly luck | 0.382 → 0.400 (±0.05, not proven) |
| **Mutual / 100** | pairs where both said yes, per 100 members | | −0.12 ± 0.15 (not proven) |
| **Ask cost** | clarification units spent in the episode (3 per bundle, 1 per soft question), the third tie-breaker | skipping Kiran saves 3 units | 198.7 → 191.2 (−7.5, proven) |

**How to read "proven".** For each seed we take new method − old method on the same world, the same people with the same luck. If the 95% interval of the mean difference over 60 seeds excludes zero, the effect is real. Otherwise it cannot be told apart from chance.

---

## 8. Conclusions

1. **The order of questions matters for timing.** Asking likely matches first gets people their first introduction about one day sooner in every family, most in cold start and sparse geography, and adds about 1.3 introductions per episode.
2. **The Bayesian score is the better order.** It predicts who has a partner far better than the count rule (AUC 0.73 against 0.59) and shortens waits by a further 0.18 days.
3. **Skipping is free in full episodes.** Experiment 1 scored people once and over-stated the risk. With daily re-scoring, people whose chance rises are asked later. Those never asked would not have been served anyway: the same 24,664 people were served, at 7.5 units less per episode.
4. **MSMI and coverage do not move measurably.** They are capped by luck and by the fixed set of feasible pairs. The gains are on waiting time, introductions and clarification cost.

**Recommendation:** ship `tiered_bayes_skip`.

**Risk:** the 5% threshold was fixed before the test, not tuned on it. Private worlds may follow different rules, and the skip is the only part of the policy that can withhold a question, so its worst case is a few asks' worth of coverage.

---

## 9. Files

| File | Content |
|---|---|
| `smp/clarify.py` | `compatible`, `_templates`, `_imagine`, `match_chances`, `bayes_bundle_asks` |
| `smp/pipeline.py` | configurations `tiered_valueask`, `tiered_bayes`, `tiered_bayes_skip` |
| `ask_model_study.py` | Experiment 1 |
| `experiments.py` | Experiment 2 runner; adds the waiting-time metric |
| `results/ask_model_study.json` | Experiment 1 results |
| `results/study_asks.json`, `results/study_asks.log` | Experiment 2 results |
