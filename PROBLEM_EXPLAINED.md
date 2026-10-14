# The Sequential Matching Problem — explained with a tiny example

Written for: a participant who is new to the dating vocabulary and wants to understand the data, the budget and the goal before choosing an approach.

Everything below is based on the repo files (`PROBLEM_STATEMENT.md`, `docs/DATA_CONTRACT.md`, `docs/POLICY_INTERFACE.md`, `kit.py`, `policy.py`) and on real records from `data/public_01`. The 8 people in the toy world (Priya, Arun, Sam, Kiran, Meera, Rohan, Zoya, Dev) are invented by me for teaching. The real data has 200 people per pool. The toy values are written in exactly the real file format.

---

## 1. Vocabulary

| Term | Plain meaning |
|---|---|
| **Member** | A person who signed up. All are synthetic (invented). |
| **Pool** | One independent group of 200 members. Matches only happen inside a pool. |
| **Introduction** | The system picks two members and tells both "would you meet this person?". It is **not a date**. |
| **Response** | Each person answers `yes`, `no`, or nothing within **7 days** of the introduction. |
| **Mutual acceptance** | Both answered yes. |
| **Date happened** | After a mutual acceptance, the simulator records that they actually met. |
| **Second-meeting intention** | After the date, each person says whether they want to meet again (`yes`/`no`/nothing). |
| **MSMI** (success) | **M**utual **S**econd-**M**eeting **I**ntention. Date happened within 30 days of the introduction **and** both said yes to meeting again within 3 days of the date. One pair counts once. |
| **Hard constraint** | A rule that must hold, such as age range, gender, smoking, zone. No score can override it. |
| **Soft field** | A trait such as pace or lifestyle. It can inform a guess but never blocks a pair. |
| **Clarification / ask** | Spending budget to learn a missing answer. |
| **Censoring** | Recent introductions that haven't finished yet. They are *not* failures. |

A `null` answer means **unknown**. It is never "no". `declined` means the person refused to answer, and you can't force it.

---

## 2. What you are building

A **policy**: a program that plays matchmaker for 60 simulated days. Every day it answers two questions in order:

1. **Ask phase.** Which members do I spend clarification budget on?
2. **Match phase.** Which pairs do I introduce today? An empty list (introduce nobody) is allowed and means "wait".

Then the simulator moves one day forward and releases any feedback that has now arrived.

### Budget and rules (exact numbers)

| Item | Value |
|---|---|
| Clarification budget | **12 units per day** (does not accumulate) |
| Ask for the whole hard-constraint bundle of one member | **3 units** (field name `constraints`) |
| Ask for one soft field | **1 unit** |
| Decision days | **60**, then **40 follow-up days** with no new asks or introductions (feedback keeps arriving) |
| Introduction response deadline | **7 days** after assignment |
| Concurrent introductions | **At most 1 per person**. Busy people are `available: false` |
| Repeat pair | Never allowed in the same episode |
| Members who arrive late | `arrived_day` is 0 to 20, so new people keep appearing |
| Some members leave | They disappear after an exit day |
| Runtime limits (assessed) | 10 s per call, 2 CPUs, 1 GiB RAM, offline, Docker |

Over 60 days that is at most 720 units, or about 240 hard-constraint bundles. With 200 members, you can't afford to ask everyone everything. The baseline spent 198 units in my test run.

An introduction is valid only if **all** of these hold:
- Both are available and in the same pool.
- Each is inside the other's age range and `who_to_meet` list.
- Same `relationship_structure`.
- Neither's smoking or children constraint is violated.
- Each accepts the other's zone, and their schedules overlap.
- Not previously introduced, and not used twice in the same batch.

If any hard field is `null` on either side (and nothing else already fails), the status is `needs_clarification` and you may not introduce them yet. If any hard rule is already broken, the status is `infeasible`. In `kit.py`, `eligibility(a, b)` returns exactly `feasible`, `infeasible` or `needs_clarification`.

---

## 3. Which file holds what

All files are in `data/public_01/` … `public_10/`.

| File | One line = | What it contains |
|---|---|---|
| `state.json` | one pool | Full observable state at day 30: `day`, `ask_budget_remaining`, `members`, `introductions`, `feedback`, `ask_log`, `schema_version`, `synthetic`. **This is exactly what your policy receives each call.** |
| `members.jsonl` | one member | Age, gender, zone, arrival day, availability, all 18 `fields`, plus `field_status` and `field_observed_day` per field. |
| `questionnaires.jsonl` | one member | The same answers as human-looking text (`"age_range": "29-39"`, `"smoking": "No"`). Just a different representation. |
| `conversations.jsonl` | one member | A short scripted chat (4 questions and answers). Template text, not real speech. |
| `introductions.jsonl` | one introduction | `introduction_id`, `user_a`, `user_b`, `assigned_day`, `response_deadline_day`, `logging_policy`, `propensity` (always null). |
| `feedback.jsonl` | one event | `introduction_response`, `date_happened`, `second_meeting_intention`, `pause_after_mutual_interest`. Each has `occurred_day`, `observed_day`, `value`, optional `missing_reason`. |
| `members_flat.csv` | one member | Easy-to-open table with age, gender, zone and the soft fields only. **It omits the hard constraints**, so don't build your policy from it. |
| `data_manifest.json` (root) | — | Which pools are train (01–06), validation (07–08), development-test (09–10). |

Static files describe day 30 only. The simulator in `kit.py` is what lets you replay episodes from day 0.

---

## 4. Toy world: 8 members

Soft traits are in the order goal / pace / lifestyle / conversations. Zones and schedules are shown the way the data stores them.

| Member | Gender, age, zone | Arrives | Wants to meet | Structure | Smoking | Kids (has / wants) | Accepts zones | Schedule | Soft traits |
|---|---|---|---|---|---|---|---|---|---|
| **Priya** | woman 28, zone_a | day 0 | man, 28–36 | monogamous | no; partner must not smoke | no / yes | a, b | weekday_evening, weekend_evening | long_term / steady / quiet / ideas |
| **Arun** | man 30, zone_a | day 0 | woman, 25–35 | monogamous | no; any | no / yes | a, b | weekend_evening, weekend_day | long_term / steady / quiet / ideas |
| **Sam** | man 40, zone_b | day 0 | woman, 30–45 | monogamous | occasionally; any | yes / no | a, b, c | weekday_evening | long_term / slow / mixed / practical |
| **Kiran** | woman 33, zone_a | day 0 | man, 28–40 | monogamous | **not asked**; any | no / **not asked** | a, b | weekday_evening, weekend_evening | long_term / steady / social / stories |
| **Meera** | woman 35, zone_c | **day 5** | man, 32–45 | monogamous | no; any | yes / no | b, c | weekday_evening | long_term / slow / mixed / practical |
| **Rohan** | man 38, zone_b | day 0 | woman, 30–40 | monogamous | no; any | no / unsure | a, b, c | weekday_evening, weekend_day | exploring / quick / social / playful |
| **Zoya** | non_binary 27, zone_a | day 0 | woman, man or non_binary, 24–33 | **non_monogamous** | no; any | no / unsure | a | weekend_evening | exploring / steady / mixed / playful |
| **Dev** | man 29, zone_a | **day 12** | woman or non_binary, 24–33 | **non_monogamous** | no; any | no / unsure | a | weekend_evening | exploring / steady / mixed / playful |

(Zoya's own acceptable ages are 25–32. Her `age_min`/`age_max` in the record is 25 and 32.)

### How one member looks in `members.jsonl`

Priya on day 0. Every field is known (`observed`, seen on day 0):

```json
{"member_id": "syn_priya0000000000000000000000000", "pool_id": "toy_01", "synthetic": true,
 "age": 28, "gender": "woman", "zone": "zone_a", "arrived_day": 0, "available": true,
 "source": "synthetic_questionnaire",
 "fields": {
   "age_min": 28, "age_max": 36, "who_to_meet": ["man"],
   "relationship_structure": "monogamous",
   "smoking": "no", "partner_smoking": "no_smoking",
   "has_children": false, "partner_children": "any", "wants_children": "yes",
   "acceptable_zones": ["zone_a", "zone_b"],
   "schedule": ["weekday_evening", "weekend_evening"],
   "relationship_goal": "long_term", "relationship_pace": "steady",
   "lifestyle": "quiet", "conversations": "ideas",
   "emotional_availability": "ready", "space_for_relationship": "moderate", "relocate": "unsure"},
 "field_status":       {"...all 18 keys...": "observed"},
 "field_observed_day": {"...all 18 keys...": 0}}
```

Kiran on day 0. Two hard fields are missing:

```json
"fields":       {"smoking": null, "wants_children": null, "age_min": 28, "age_max": 40, "...": "..."},
"field_status": {"smoking": "not_asked", "wants_children": "not_asked", "age_min": "observed", "...": "..."},
"field_observed_day": {"smoking": null, "wants_children": null, "age_min": 0, "...": 0}
```

After you spend 3 units on `{"member_id": "<Kiran>", "field": "constraints"}`, the simulator returns this record (the same shape as the entries of `ask_log` in the real `state.json`):

```json
{"member_id": "<Kiran>", "field": "constraints", "observed_day": 0,
 "values":   {"smoking": "no", "wants_children": "yes", "age_min": 28, "age_max": 40, "...": "..."},
 "statuses": {"smoking": "observed", "wants_children": "observed", "...": "observed"}}
```

Her record is then fully known. If she had answered `declined`, the value would stay `null` and she could never be introduced to anyone.

### What the policy receives (`state`, day 0, before any ask)

```json
{"schema_version": "1.0.0", "synthetic": true, "day": 0,
 "ask_budget_remaining": 12,
 "members": [ Priya, Arun, Sam, Kiran, Rohan, Zoya ],
 "introductions": [], "feedback": [], "ask_log": []}
```

Meera (day 5) and Dev (day 12) are **not in the list yet**, and you can't see future arrivals.

---

## 5. Who can meet whom (checked with the `eligibility` rules)

| Pair | Status | Why |
|---|---|---|
| Priya – Arun | **feasible** | Ages, genders, zones, structure, smoking, children plans and a shared weekend_evening slot all pass both ways. |
| Priya – Sam | infeasible | Sam is above her age_max (36) and she is below his age_min (30). She wants children and he doesn't. She requires a non-smoker and he smokes occasionally. |
| Priya – Rohan | infeasible | Rohan (38) is above her age_max (36). |
| Priya – Dev | infeasible | Structure mismatch (monogamous vs non_monogamous). |
| Arun – Kiran | **needs_clarification** | Everything known passes. Kiran's `smoking` and `wants_children` are `null`. |
| Arun – Meera | infeasible | Meera's zone_c is not in Arun's zones. Meera wants no children, Arun wants yes. Meera's age_min is 32 and Arun is 30. |
| Kiran – Sam | needs_clarification, then **infeasible if she says she wants children** | He answered `wants_children: no`, so an explicit yes/no conflict excludes the pair. |
| Kiran – Rohan | **needs_clarification** | Passes everything known. Rohan is `unsure` about children, which does not block. |
| Sam – Meera | **feasible** (from day 5) | Same slow/practical profile, zones overlap, both want no children, both on weekday_evening. |
| Meera – Rohan | **feasible** (from day 5) | Passes both ways. |
| Zoya – Dev | **feasible** (from day 12) | Both non_monogamous, and ages, zones and schedule match. |
| Zoya – everyone else | infeasible | Only Dev has the same relationship structure. |

Draw it as a graph, where an edge means "could be introduced":

```
Priya — Arun — Kiran — Rohan — Meera — Sam        Zoya — Dev
        (Arun–Kiran and Kiran–Rohan wait for the ask; Meera arrives day 5; Dev arrives day 12)
```

Priya's only option is Arun, and Sam's only option is Meera. That chain is the heart of the problem.

---

## 6. Why "best pair first" can be wrong

Suppose a model estimates the chance each pair ends in MSMI. **These numbers are invented to illustrate the idea**, not computed from data:

| Pair | Illustrative value |
|---|---|
| Arun – Kiran | 0.60 |
| Sam – Meera | 0.55 |
| Priya – Arun | 0.50 |
| Kiran – Rohan | 0.30 |
| Meera – Rohan | 0.20 |

- **Greedy** (take the highest pair, remove those two people, repeat): Arun–Kiran (0.60), then Sam–Meera (0.55). Priya and Rohan are left with no partner. Total is 1.15 with 2 pairs.
- **Global best** (a maximum-weight matching on the whole graph): Priya–Arun (0.50) + Kiran–Rohan (0.30) + Sam–Meera (0.55). Total is 1.35 with 3 pairs, and everyone in the chain is served.

The real baseline in `kit.py` (`baseline_match`) is greedy, using the number of matching soft fields as the score.

There are two more decisions on top of the pairing:
- **Waiting.** Sam on day 0 has nobody. The right move is to wait for Meera (day 5), not to ask Sam anything. Zoya waits for Dev (day 12).
- **Asking.** Only Kiran has missing hard data, so the best use of today's 12 units is one 3-unit ask on Kiran. It unlocks two edges (Arun and Rohan) and she is the middle of the chain. Everyone else has nothing worth asking, so the other 9 units can go unspent.

---

## 7. One full run of the toy world (what the files would show)

| Day | What the policy does | What the data records |
|---|---|---|
| 0, ask | Spends 3 of 12 units on Kiran's `constraints`. | `ask_log` entry for Kiran. |
| 0, match | Submits `[[Priya, Arun], [Kiran, Rohan]]`. | Two lines in `introductions.jsonl`, both `assigned_day: 0`, `response_deadline_day: 7`. All four are now busy (`available: false`). |
| 1–3 | Waits. Nothing to match. | Priya yes (d1), Arun yes (d2), Kiran yes (d1), Rohan no (d3). Priya–Arun is a mutual acceptance. Kiran–Rohan is dead. |
| 4 | — | `date_happened: true` for Priya–Arun. |
| 5 | Meera arrives. Submits `[[Sam, Meera]]`. Rohan could also take Meera, but she goes to Sam. | New intro, deadline day 12. |
| 5–6 | — | Priya yes and Arun yes on second meeting (days 5 and 6, within 3 days of the day-4 date). **That pair is one MSMI.** |
| 6 | — | Meera yes and Sam yes (mutual acceptance for Sam–Meera). |
| 9–11 | — | Sam–Meera: date on day 9, Sam says yes (d10) but Meera says no (d11), so no MSMI. |
| 12 | Dev arrives. Submits `[[Zoya, Dev]]`. | Zoya yes (d13), Dev never answers: `missing_reason: "no_response"`, observed day 19. |

Scoring this toy episode (N = 8 members arrived):
- MSMI per 100 = 100 × 1 / 8 = **12.5**
- Coverage = 8 members introduced / 8 = **1.0**. Kiran–Rohan and Zoya–Dev count even though they failed, because coverage measures people served, not success.
- Mutual acceptances per 100 = 100 × 2 / 8 = **25** (Priya–Arun and Sam–Meera)
- Clarification cost = **3** units

The real score uses 200 members, so one success is worth 0.5 points. A good policy gets several.

---

## 8. What is to be solved in the end

Build a policy (`policy.py`, run in Docker through the JSON protocol in `docs/POLICY_INTERFACE.md`) that maximises:

**MSMI per 100 arrived members**, averaged over 6 scenario families, each with 20 private seeds (120 episodes in total). The families are standard, sparse geography, cold start, delayed dates, shifted outcome weights and changing response conditions.

Tie-breakers, in order: coverage, mutual acceptances per 100, lower ask cost, lower inference time.

Hard requirements:
- Never output an invalid pair. One invalid action makes the episode invalid, and the whole submission is then ineligible.
- Use only what is in the observation. No reading hidden simulator state and no reconstructing answers from seeds.
- Stay within the time and memory limits.
- Deliver the code, Dockerfile, pinned dependencies, scenario-level results, a technical report, comparison with the three baselines (greedy, no_asks, random) on the same seeds, and at least one ablation.

Dates (IST): Round 1 research note is due 9 Oct 2026 23:59. Round 2 final build is due 18 Oct 2026 23:59.

---

## 9. Potential solutions (you decide which to pick up)

These are options only. They can be combined, and the simplest is a valid submission.

### A. Replace greedy with global matching
Build the graph of `feasible` pairs and solve a maximum-weight matching (for example the Blossom algorithm, or the Hungarian algorithm after splitting into two sets). This fixes the Priya–Arun–Kiran example directly. It is low risk, because validity comes from `eligibility()`.

### B. Smarter asking
Rank members by how much an ask is worth. For example: members with missing hard fields who would gain many `needs_clarification` edges, and who are not already busy. Skip members who can't be unlocked because they are `declined`. Optionally spend 1-unit soft asks only where they change the pair ranking.

### C. Better edge scores
The baseline counts equal soft fields. You could instead:
- Learn a model from the 6 training pools' feedback (features of both people → probability of mutual yes, of a date, or of MSMI).
- Use simple hand-made rules (similar goal and pace score higher).
- Treat the two directions separately (A says yes to B, B says yes to A) instead of multiplying them.

Caution: you only see outcomes for the pairs the old policy chose (selection bias), and propensities are null. Treat this as a modelling assumption and say so in the report. Compare whole policies by running simulator episodes.

### D. Timing: when to wait
Keep a person unmatched if a much better candidate is likely to arrive (members arrive up to day 20), or if the available option has low value. Estimate this from arrival patterns in the simulator. Remember that Zoya's only partner arrives on day 12, so waiting is sometimes correct.

### E. Learn from your own rollouts
Run the simulator under different policies and log what you observed **before** each decision. Don't attach the final day-30 profile to an earlier outcome, because answers may have been learned later.

### F. Fancier approaches
Bandits (explore which kinds of pairs work), reinforcement learning, or integer programming over several days. These are allowed but need more experiments and care with the runtime limit.

### Suggested order of work (if useful)
1. Run all three baselines on multiple seeds and variants (`python evaluate.py --baseline greedy --seeds 101,102,103 --variants all`).
2. Add option A and measure the difference.
3. Add option B and measure the difference (this is also your ablation).
4. Only then consider C/D.

### Practical note
On Windows, clone with `git clone -c core.autocrlf=false <url>`. With the default `core.autocrlf=true`, `verify_data.py` fails with a checksum mismatch on the data files because their line endings change.

---

## 10. Things to be careful about
- `null` is unknown, not no. No response is not a rejection. A pair without a date has no second-meeting label.
- Recent introductions at a snapshot are right-censored and not failures.
- Hard constraints are checked in **both** directions.
- Relocation interest does not override acceptable zones.
- Don't split rows of the same people across train and test. Use the pool-level splits in `data_manifest.json`.
- Probability estimates are optional and do not affect ranking. If you report them, define the target and show calibration.
