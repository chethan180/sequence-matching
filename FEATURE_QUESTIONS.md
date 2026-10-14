# Testing the candidate-feature questions and the noise idea

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Written for: the team deciding whether any of these ideas goes into the model.

Scripts: `feature_questions.py` (data `results/ext3_rows.npz`, 181 features per person-introduction row, 109,434 rows, collected by `collect_ext3.py`), `noise_pair.py` and `noise_analysis.py`. Results: `results/feature_questions.json`, `results/noise_pair.json`, `results/noise_analysis.json`.

Protocol: split by world (train seeds 1001–1030, test seeds 1031–1040). Each idea is a yes-rate table on the test worlds plus the change in test AUC when its features are added to the core model (similarity agreement + own history, logistic regression, test AUC 0.5637). Intervals bootstrap over the 60 test worlds. 14 model tests were run, so about 0.7 would look "supported" at a 95% interval by chance alone.

## Verdicts

| # | Idea | Yes-rate pattern (test) | Change in test AUC | Verdict |
|---|---|---|---|---|
| 1 | Zone flexibility (zones accepted) | 1 zone 36.9%, 2 zones 37.3% | −0.0006 [−0.0022, +0.0010] | no effect |
| 2 | Zone thickness (zone a, distinct zones in pool) | zone a 37.0%, other 36.8%; 9+ zones 35.1% | +0.0001 [−0.0009, +0.0013] | no effect |
| 3 | Openness score | least open 38.6%, most open 35.7% (partner: 36.0% → 38.5%) | −0.0013 [−0.0029, +0.0005] | no effect |
| 4 | Near-miss count | 37.1 / 36.5 / 36.6 / 38.6% for 0 / 1 / 2 / 3+ | +0.0014 [+0.0002, +0.0024] | marginal, flat table: likely chance |
| 5 | One-sided age fit | 36.5 / 36.4 / 37.3% | −0.0027 [−0.0044, −0.0009] | hurts |
| 8 | Personalised taste (past partners' traits) | −: 33.1%, 0: 37.7%, +: 35.1% (people with 2+ answers) | −0.0004 [−0.0014, +0.0006]; on 2+ answers −0.0022 | no effect |
| 9 | Stated vs revealed (age gap of past yes-partners) | not tabulated | −0.0026 [−0.0040, −0.0012] | hurts |
| 10 | Silence vs "no" | see below | −0.0001 [−0.0004, +0.0002] | no predictive gain, useful finding |
| 11 | Wait time (days since arrival / last answer) | 36.7 / 37.9 / 37.7 / 35.1% by arrival | −0.0003 [−0.0011, +0.0008] | no effect |
| 12 | Drift detector (recent pool yes-rate) | see below | −0.0006 [−0.0021, +0.0011]; drift scenario after day 35 +0.0014 [−0.0010, +0.0034] | no effect |
| 13 | Pool shape (share complete, zones, size) | variants 34.7–38.2% | −0.0021 [−0.0037, −0.0006] | hurts |
| 14 | Weekday (day mod 7) | 34.3–38.9%, train and test patterns disagree | +0.0023 [+0.0003, +0.0041] | marginal, simulator has no weekdays: likely chance |
| 15 | Date stage | see below | n/a | see below |

Ideas #6 (children and smoking combinations) and #7 (age position) were already covered by the pair tables in `PAIR_PROBABILITY.md` and showed no stable effect.

Two ideas show a "supported" interval (#4 and #14), each worth about +0.002 AUC. With 14 tests, about 0.7 are expected by chance, and neither has a plausible mechanism or a consistent yes-rate pattern, so I do not treat either as real. Three ideas (#5, #9, #13) significantly lower test AUC, meaning the model fits train noise.

## Findings that are not about prediction

**#10 Silence vs "no".** Yes rate on the next introduction:

| Past behaviour | Yes rate |
|---|---|
| no history | 37.7% |
| one past answer: yes | 43.9% |
| one past answer: no | 32.4% |
| one past silence only | 35.8% |
| two past silences | 33.6% |
| a past no plus a past silence | 33.0% |

A single silence is nearly neutral (2 points below no history), and a past "no" is clearly negative (5 points below). Silence is not rejection, which matches the problem statement. Repeated silence looks like a mild negative. Giving silence its own features adds no accuracy.

**#12 Drift.** In the drift scenario the yes rate drops from 38.2% (days 0–34) to 29.9% (days 35–59), an 8-point fall, against 37.2% to 36.1% elsewhere. The effect is real, but a detector built from the pool's recent yes-rate does not improve predictions. The drift-after-day-35 test set has only 1,024 rows, so the test has limited power.

**#15 Date stage** (test worlds, unique introductions): 1,398 mutual-yes pairs; a date happened in 77.0%, consistent with a fixed date chance. Date rate by shared schedule slots: 75.7% (1 slot), 74.9% (2), 89.1% (3, n = 165); same zone 76.9%, different zone 92.9% (n = 14). The 89.1% for three slots is an outlier relative to a fixed chance of about 78%, but it comes from 165 pairs; I would not rely on it without more data. Among pairs that dated, MSMI occurred in 6.3% (goal same 9.8%, goal different 7.8%): the second-meeting stage is the main reason MSMI is so rare.

## The noise idea (high yes-rate over several introductions)

Tested at person level (`noise_analysis.py`) and pair level (`noise_pair.py`):

- Person level: the effect of agreement on the log-odds is about 37% weaker for people with 2+ answers and yes-rate ≥ 60% (interaction −0.072, 95% interval −0.129 to −0.023), but they are 6.2% of rows and no way of using it improves test AUC (+0.0002).
- Pair level: the interaction changes sign with the definition of "generous" (+0.055 for 2+ answers ≥ 50%; −0.142 for 3+ answers ≥ 70%, interval −0.234 to −0.008). Flag-plus-interaction, reliability scaling and training noise all leave test AUC unchanged (+0.002, −0.000, +0.000).
- Conclusion: not robust, not worth building into the policy.

## What this means
No candidate feature improves the yes-model in a way I would trust. The model is already at the ceiling the data allows (test AUC about 0.56), and the oracle test showed that even exact hidden probabilities would give little extra. Keep the compact feature set; the useful outputs of this round are the findings above (silence is nearly neutral, drift is real but not detectable from the pool's recent yes-rate, and the date stage is close to a fixed chance), which belong in the report's analysis of failure modes.
