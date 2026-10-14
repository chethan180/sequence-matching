# Five real members, worked end to end

Everything in this document is computed from the real files in `data/public_05/`, using the real code in `kit.py` and `policy.py`. The five members are synthetic, like every record in the dataset. Nothing here is invented, apart from the aliases P1 to P5.

Reproduce it with:

```
python exploration/build_five_people.py     # writes exploration/five_people.json and prints every number below
```

---

## 1. Where the data comes from and how it was merged

Each member is spread over five files. To work with one person you have to join them on `member_id`.

| Source file | One row is | What it gives for a member | Join key |
|---|---|---|---|
| `members.jsonl` | a member | Age, gender, zone, arrival day, availability, and every field with its **value**, **status** (`observed`, `not_asked` or `declined`) and **observed day** | `member_id` |
| `questionnaires.jsonl` | a member | The same answers as text (`"age_range": "27-39"`, `"smoking": "No"`). Used to cross-check, adds nothing new | `member_id` |
| `conversations.jsonl` | a member | A four-question scripted chat. The replies map one-to-one onto soft fields | `member_id` |
| `introductions.jsonl` | an introduction | The two endpoints, `assigned_day`, `response_deadline_day` | `user_a`, `user_b` |
| `feedback.jsonl` | a feedback event | Response (yes/no/none), date happened, second-meeting intention | `member_id`, `introduction_id` |
| `state.json` | a pool | The day-30 view of all of the above. Used to run `kit.eligibility` | — |

### Cleaning and merging rules I applied

1. **One record per person.** Each field becomes one object `{value, status, observed_day}` instead of three parallel dictionaries.
2. **Hard and soft fields are separated** using `kit.HARD` (11 fields) and `kit.SOFT` (7 fields).
3. **Null stays null.** `null` means unknown. It is never turned into zero or "no".
4. **Aliases.** The five members are `P1` to `P5`. Anyone introduced to them who is not in the five is labelled `outside:<6 hex chars>`.
5. **Introductions are attached only if they touch one of the five**, and their feedback events are sorted by day.
6. **Cross-check passed.** The questionnaire text agrees with the field values (for example P4: `"age_range": "27-39"` against `age_min 27, age_max 39`).
7. **Derived sections added**: each person's response history, the 10 pairwise eligibility results, and the matchings.

### Shape of the merged file `exploration/five_people.json`

```
meta                                  sources, hard/soft field lists, alias -> member_id
people[5]
  alias, member_id, pool_id
  profile                             age, gender, zone, arrived_day, available_at_day_30
  hard_fields   {field: {value, status, observed_day}}   11 fields
  soft_fields   {field: {value, status, observed_day}}   7 fields
  questionnaire_text                  raw text answers
  conversation                        [(role, text), ...]
  introduction_ids, own_feedback_events
  response_history                    answered, yes, shrunk probabilities
introductions_touching_these_people[] endpoints, deadline, events with who/value/days
pairs[10]                             eligibility, fail_reasons, soft comparison, scores, weights
matchings                             greedy vs global results
```

A real excerpt (P4, shortened):

```json
{"alias": "P4", "member_id": "syn_d1ce45f75d93596c81ae980e72fec48f",
 "profile": {"age": 35, "gender": "woman", "zone": "zone_a", "arrived_day": 11, "available_at_day_30": false},
 "hard_fields": {"schedule": {"value": ["weekday_evening"], "status": "observed", "observed_day": 11}, "...": "..."},
 "soft_fields": {"conversations": {"value": "stories", "status": "observed", "observed_day": 11}, "...": "..."},
 "questionnaire_text": {"age_range": "27-39", "smoking": "No", "who_to_meet": "women, men, non-binary people", "...": "..."},
 "own_feedback_events": [{"introduction_id": "intro_46a8...", "event": "introduction_response", "value": "yes", "occurred_day": 14, "observed_day": 14},
                         {"introduction_id": "intro_79c2...", "event": "introduction_response", "value": "no", "occurred_day": 25, "observed_day": 25}],
 "response_history": {"answered": 2, "yes": 1, "p_yes": 0.4184}}
```

---

## 2. The five people

All five are in `zone_a`. All of their hard and soft fields are `observed`.

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Gender, age | non_binary, 29 | woman, 35 | woman, 30 | woman, 35 | woman, 32 |
| Arrived (day) | 10 | 0 | 3 | 11 | 0 |
| Available at day 30 | no | yes | no | no | no |
| Wants to meet | woman | woman | woman, man, non_binary | woman, man, non_binary | woman, man, non_binary |
| Accepts ages | 22–36 | 27–41 | 25–36 | 27–39 | 25–40 |
| Accepts zones | a | a, c | a | a | a |
| Smokes / partner must not smoke | yes / any | yes / any | no / **no_smoking** | no / any | no / any |
| Has children / partner must have none | yes / any | yes / **no_children** | no / **no_children** | no / any | no / any |
| Wants children | unsure | no | yes | yes | unsure |
| Schedule | weekend_evening, weekday_evening, weekend_day | weekend_day, weekend_evening | weekday_evening, weekend_evening | weekday_evening | weekday_evening, weekend_evening |
| Goal / pace | long_term / quick | exploring / quick | long_term / quick | exploring / quick | exploring / steady |
| Lifestyle / conversations | mixed / practical | social / playful | quiet / ideas | social / stories | social / stories |
| Emotional availability | taking_time | taking_time | ready | taking_time | taking_time |
| Space for relationship | ample | moderate | limited | moderate | ample |
| Relocate | unsure | yes | no | yes | yes |

---

## 3. Who can meet whom (hard eligibility)

A pair is **feasible** only if every hard check passes in **both** directions. For a pair (a, b), `kit.eligibility` checks, writing `x → y` for "x's preferences about y":

| Check | Passes when |
|---|---|
| who_to_meet | y.gender ∈ x.who_to_meet |
| age | x.age_min ≤ y.age ≤ x.age_max |
| geography | y.zone ∈ x.acceptable_zones |
| smoking | not (x.partner_smoking = no_smoking and y.smoking ∈ {yes, occasionally}) |
| children present | not (x.partner_children = no_children and y.has_children = true) |
| relationship structure | a.structure = b.structure |
| children plans | {a.wants_children, b.wants_children} ≠ {yes, no} (unsure never conflicts) |
| schedule | a.schedule ∩ b.schedule ≠ ∅ |

All five are `monogamous` and have no missing hard field, so none of their pairs is `needs_clarification`.

| Pair | Status | Exact reasons |
|---|---|---|
| P1–P2 | infeasible | P2 only wants a woman and P1 is non_binary (who_to_meet). P2 wants no children and P1 has them |
| P1–P3 | infeasible | P3 needs a non-smoker and P1 smokes. P3 wants no children present and P1 has children |
| **P1–P4** | **feasible** | |
| **P1–P5** | **feasible** | |
| P2–P3 | infeasible | P3 needs a non-smoker and P2 smokes. P3 wants no children present and P2 has them. Children plans: P2 no, P3 yes |
| P2–P4 | infeasible | Children plans: P2 no, P4 yes. Schedule: P2 has weekends only, P4 only weekday_evening |
| **P2–P5** | **feasible** | |
| **P3–P4** | **feasible** | |
| **P3–P5** | **feasible** | |
| **P4–P5** | **feasible** | |

**Worked check of P4–P5** (feasible): ages 35 ∈ [25, 40] and 32 ∈ [27, 39]; P5 is a woman and P4 accepts women, and P4 is a woman and P5 accepts women; both zone_a; both monogamous; partner_smoking is `any` for both; partner_children is `any` for both; children plans yes/unsure do not conflict; schedules share weekday_evening.

The feasibility graph is therefore six edges: P1–P4, P1–P5, P2–P5, P3–P4, P3–P5, P4–P5.

```
P1 — P4 — P3        P5 is joined to P1, P2, P3 and P4
 \    |    |        P4 is joined to P1, P3 and P5
  \   |    |        P2 is joined only to P5
   \  |    |
    — P5 ——          edges: P1–P4, P1–P5, P2–P5, P3–P4, P3–P5, P4–P5
```

---

## 4. How the soft score is counted

The baseline's score for a feasible pair is the number of soft fields on which the two people **have the same known value**. For soft fields k ∈ S (the 7 soft fields):

$$\text{soft}(a,b)=\sum_{k\in S}\mathbf 1\big[a_k \ne \text{null}\ \wedge\ b_k \ne \text{null}\ \wedge\ a_k = b_k\big]$$

It ranges from 0 to 7. A null on either side scores 0. That is a modelling choice, and it makes unknown behave like "different". In this example nothing is null, so it does not matter here. In the full dataset about 60% of soft fields are null, so it does.

### Worked example: P4–P5, soft score = 5

| Soft field | P4 | P5 | Same? |
|---|---|---|---|
| relationship_goal | exploring | exploring | **1** |
| relationship_pace | quick | steady | 0 |
| lifestyle | social | social | **1** |
| conversations | stories | stories | **1** |
| emotional_availability | taking_time | taking_time | **1** |
| space_for_relationship | moderate | ample | 0 |
| relocate | yes | yes | **1** |
| **Total** | | | **5** |

### All six feasible pairs

| Pair | Same fields | Soft score (of 7) | Matches among the 4 core fields (goal, pace, lifestyle, conversations) |
|---|---|---|---|
| P4–P5 | goal, lifestyle, conversations, emotional, relocate | **5** | 3 |
| P2–P5 | goal, lifestyle, emotional, relocate | **4** | 2 |
| P1–P4 | pace, emotional | **2** | 1 |
| P1–P5 | emotional, space | **2** | 0 |
| P3–P4 | pace | **1** | 1 |
| P3–P5 | none | **0** | 0 |

The four-field count is used by my learned score in section 7.

---

## 5. What "people served" means

**Within one batch.** A batch is the set of pairs submitted on one day. Each person can be in at most one pair. So, for a batch of m pairs:

$$\text{people served in the batch}=2m,\qquad \text{share served}=\frac{2m}{\text{people considered}}$$

With these five people, a batch can hold at most 2 pairs, so at most 4 of the 5 are served. One person is always left over because five is odd.

**Across an episode (the official coverage metric).**

$$\text{coverage}=\frac{\text{distinct arrived members who received at least one introduction}}{N},\quad N=\text{members who arrived by day 59}$$

Someone introduced three times counts once. Members who later left or paused stay in the denominator. This is the metric used as a tie-breaker in the ranking.

**Related definitions** from `PROBLEM_STATEMENT.md`:

$$\text{MSMI per 100}=100\times\frac{M}{N},\qquad \text{mutual acceptances per 100}=100\times\frac{\text{pairs where both said yes}}{N}$$

where M is the number of introductions with a date within 30 days of assignment and both second-meeting answers "yes" within 3 days of the date. If these five people were a whole pool (N = 5) and one pair reached MSMI, the score would be 100 × 1 / 5 = 20.

---

## 6. Greedy vs global vs the other baselines on these five

**Objective.** Each policy chooses a set of pairs M ⊆ E (edges), where no person appears twice:

$$\max_{M}\ \sum_{(a,b)\in M} w_{ab}\qquad\text{s.t. each person is in at most one pair}$$

Greedy takes the largest remaining edge each time. Global solves the whole maximisation.

### With the baseline score w = soft score

| Policy | Pairs | Soft total | People served (of 5) |
|---|---|---|---|
| **greedy** | P4–P5 (5) | 5 | **2** |
| **no_asks** | P4–P5 (5) — nobody here has a missing hard field, so no_asks is identical to greedy in this case | 5 | **2** |
| **random** | exact average over all 6! = 720 orderings of the six edges: 1.83 pairs on average (2 pairs in 600 orderings, 1 pair in 120) | 3.92 | 3.67 |
| **global** | P1–P4 (2) + P2–P5 (4) | **6** | **4** |

Why greedy loses: it takes P4–P5 first, which uses up P4 and P5. P1, P2 and P3 have no feasible edge among themselves (P1–P2, P1–P3 and P2–P3 are all infeasible), so they are left with no partner. Global gives up the best single pair (5) and wins on the total (2 + 4 = 6) and on people served (4 against 2).

Global's other 2-pair options all score less: {P1–P5, P3–P4} = 2 + 1 = 3, {P2–P5, P3–P4} = 4 + 1 = 5, {P1–P4, P3–P5} = 2 + 0 = 2.

In my policy, a constant ε = 1 is added to each edge weight (w′ = soft + 1), so the optimiser also prefers more pairs. It picks the same matching here: (2 + 1) + (4 + 1) = 8, against 6 for P4–P5 alone.

### Why no_asks is the same here, and when it is not
`no_asks` only differs when someone has missing hard data. All five are fully known, so asking has nothing to unlock. In Example 1 of the earlier toy world (Kiran with unknown fields), it differed.

---

## 7. A learned score: using what people actually did

The baseline score ignores behaviour. The data includes each person's responses to past introductions. A simple estimate of "will this person say yes?" is shrunk toward the population rate so that one or two events don't dominate.

For a person with T responses or timeouts, A answers and Y yeses, and prior strength τ = 3:

$$\hat p_{\text{answer}}=\frac{A+\tau\cdot 0.78}{T+\tau},\qquad \hat p_{\text{yes}\mid\text{answer}}=\frac{Y+\tau\cdot 0.47}{A+\tau},\qquad \hat p=\hat p_{\text{answer}}\ \hat p_{\text{yes}\mid\text{answer}}$$

(0.78 and 0.47 are the answer rate and the yes-given-answer rate over the 613 historical introductions in the supplied data.)

| Person | T | A | Y | p̂_answer | p̂_yes\|answer | p̂ |
|---|---|---|---|---|---|---|
| P1 | 2 | 2 | 0 | 0.868 | 0.282 | **0.245** |
| P2 | 1 | 0 | 0 | 0.585 | 0.470 | **0.275** |
| P3 | 3 | 2 | 2 | 0.723 | 0.682 | **0.493** |
| P4 | 2 | 2 | 1 | 0.868 | 0.482 | **0.418** |
| P5 | 4 | 4 | 1 | 0.906 | 0.344 | **0.312** |

Worked: P3 has 2 answers, both yes, out of 3 responses/timeouts, so p̂_answer = (2 + 3 × 0.78) / (3 + 3) = 4.34 / 6 = 0.723 and p̂_yes|answer = (2 + 3 × 0.47) / (2 + 3) = 3.41 / 5 = 0.682, so p̂ = 0.493.

The edge weight multiplies the two directions and adds a small bonus for matching core soft fields (m = matches among the 4 core fields, bonus coefficient 0.15, found by experiments rather than estimated from data):

$$w_{ab}=\hat p_a\,\hat p_b\,(1+0.15\,m_{ab})$$

The product assumes the two directions are independent. The statement warns this may not hold, and I use it only for ranking, not as a calibrated probability.

| Pair | p̂_a × p̂_b | m | Weight |
|---|---|---|---|
| P3–P4 | 0.493 × 0.418 = 0.206 | 1 | **0.237** |
| P4–P5 | 0.418 × 0.312 = 0.130 | 3 | 0.189 |
| P3–P5 | 0.493 × 0.312 = 0.154 | 0 | 0.154 |
| P1–P4 | 0.245 × 0.418 = 0.102 | 1 | 0.118 |
| P2–P5 | 0.275 × 0.312 = 0.086 | 2 | 0.112 |
| P1–P5 | 0.245 × 0.312 = 0.076 | 0 | 0.076 |

Under this score, **greedy and global agree**: both pick P3–P4 (0.237) and P2–P5 (0.112), total 0.349 and 4 people served. So global is not always better. It wins when a locally best pair blocks other people, and not here, where the best edge is also in the best set. This is why the real comparison needs many episodes, not one example.

### Where bandits and online learning fit
Treat each person's yes-rate as unknown, with a Beta belief. After past events, P3 (2 yes in 2 answers) has Beta(1 + 2, 1 + 0) = Beta(3, 1), mean 0.75. P5 (1 yes in 4 answers) has Beta(1 + 1, 1 + 3) = Beta(2, 4), mean 0.33. Thompson sampling draws one value from each belief and acts on the larger draw. P3 usually wins, but P5 is sometimes tried, and that is how a wrong belief gets corrected. The shrunk estimates in the table above are the same "update with every new answer" idea in its simplest form. Both only ever act on pairs that passed the hard checks.

---

## 8. What actually happened to these people

Three of the six feasible pairs were actually tried by the earlier policy, and they show why a yes from one side is not a success. All days are simulation days.

| Introduction | Assigned | Events | Result |
|---|---|---|---|
| P2–P5 | day 8 | P5 yes (day 12). P2 no response (timed out day 15) | not mutual |
| P1–P4 | day 11 | P4 yes (day 14). P1 no (day 15) | not mutual |
| P3–P4 | day 19 | P4 no (day 25). P3 yes (day 26) | not mutual |

Reading these correctly:
- A missing answer (P2) is **not** a rejection. It is simply no response.
- One yes is not a mutual acceptance. All three left a person who said yes and nobody to meet.
- None reached a date, so none has a second-meeting label. They are not "failed dates" either.
- These pairs are exactly three of the feasible edges, which shows the earlier policy already used the same graph.

Other introductions touch these members with people outside the five (for example P5 had four of its own, and P3 and P4 each have one introduction assigned on day 27 whose answers had not arrived by day 30). Those still-open introductions are right-censored: they are neither successes nor failures.

---

## 9. Caveats

1. **Availability.** At day 30 only P2 is available, because the others have outstanding introductions. The matchings above assume all five are free, as they would be on an earlier day, so they illustrate the mechanism rather than a real day-30 decision.
2. **Five people is too small to prove anything.** One example shows global matching can beat greedy, and the learned-score table shows it need not. The evidence must come from many simulator episodes, which the experiment harness (`experiments.py`) runs.
3. **Null soft fields count as no match.** In the full data this matters, because most soft fields are unknown. An ask for a soft field costs 1 unit.
4. **Learned weights come from a handful of events.** The shrinkage keeps them near the population rate (0.37) until a person has history, but they are still noisy.
5. **The independence assumption** in p̂_a × p̂_b is a ranking device, not a probability estimate of a mutual yes.
6. **Selection bias.** The responses we learn from exist only for pairs the earlier policy chose.

---

## 10. Files

| File | Purpose |
|---|---|
| `exploration/five_people.json` | The merged, cleaned data for the five people |
| `exploration/build_five_people.py` | Builds it and prints every number used here |
| `exploration/toy_examples.py` | The invented 8-person toy world (see `PROBLEM_EXPLAINED.md`) |
| `policy.py` | `greedy_matching`, `global_matching`, `propensities` used above |
