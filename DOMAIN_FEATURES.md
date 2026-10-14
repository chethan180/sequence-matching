# Domain-knowledge features from the online-dating literature

> **Correction (9 October 2026).** The intervals and standard errors in this document resampled each seed-and-scenario run as if it were independent. Runs that share a seed are not independent (the simulator draws an introduction's randomness from the seed, the two people and the day), so these intervals are too narrow and some results marked significant are not. Point estimates are unaffected. The corrected, seed-level analysis is in `ROUND1_RESEARCH_SUBMISSION.md`.

Written for: the team deciding which features and models to carry into the build phase.

## Question
Do features suggested by published studies of dating apps and speed dating improve the prediction of whether a person says yes to an introduction? Everything here is built from fields we already have, using only information available before the decision.

## Literature → feature

| Finding | Source | Features built (27 in 6 groups) |
|---|---|---|
| Desirability is a hierarchy; being chosen by choosy people signals desirability; people pursue partners above themselves | [Bruch & Newman](https://www.futurity.org/online-dating-desirability-1834912/) | `desirability`: received-yes rate adjusted for how generous each responder is, for the actor and partner, plus the gap |
| Reciprocal recommenders combine interest similarity and attractiveness similarity from interactions | [Reciprocal Recommendation System for Online Dating](https://arxiv.org/pdf/1501.06247) | Same desirability features. True interaction-similarity needs many interactions per person; ours have 1–3 |
| Replies go to people closest to the receiver's *stated* preferences, yet stated preference and behaviour differ a lot | [Predicting User Replying Behavior](https://ojs.aaai.org/index.php/ICWSM/article/view/14516) | `range_fit`: how central the partner's age is inside the actor's stated range, both directions |
| Popular users are more selective; selectivity rises with choice-set size | [Computational Courtship](https://arxiv.org/pdf/1809.10032), [Fisman et al.](https://business.columbia.edu/faculty/research/gender-differences-mate-selection-evidence-speed-dating-experiment) | `demand_supply`: share of the pool whose stated preferences accept you; share you accept; their product |
| Strong similarity preferences on many attributes | [Hitsch, Hortaçsu & Ariely](https://scholars.duke.edu/publication/859289) | `trait_logic`: ordinal pace/space distance, both-long-term, both-exploring, goal mismatch, weighted agreement |
| Perceived similarity predicts attraction better than actual similarity | [Montoya et al.](https://researchportal.murdoch.edu.au/esploro/outputs/journalArticle/Is-actual-similarity-necessary-for-attraction/991005635870107891) | Weighted agreement score in `trait_logic` |
| Traits predict an individual's own tendency and desirability, but not pair-specific compatibility | [Joel, Eastwick & Finkel](https://journals.sagepub.com/doi/10.1177/0956797617714580) | `recency_segment`: recency-weighted yes-rate, and gender/age-band segment priors |
| Reply behaviour depends on engagement | [Predicting User Replying Behavior](https://ojs.aaai.org/index.php/ICWSM/article/view/14516) | `latency`: mean days to answer, share of fast answers |

## Result (held-out simulator worlds, target = said yes, 27,094 test rows, bootstrap over 60 worlds)

AUC of each domain group on its own (0.5 = chance):

| Group | Logistic | XGBoost |
|---|---|---|
| desirability (4) | 0.512 | 0.507 |
| demand_supply (6) | 0.510 | 0.501 |
| range_fit (3) | 0.509 | 0.508 |
| latency (3) | 0.493 | 0.497 |
| trait_logic (7) | 0.537 | 0.533 |
| recency_segment (4) | 0.532 | 0.528 |

Change in AUC when added to the core set (similarity + own history, 28 features: logistic 0.5657, XGBoost 0.5594). Brackets are 95% intervals.

| Added group | Logistic | XGBoost |
|---|---|---|
| desirability | +0.0000 [−0.002, +0.002] | +0.0002 [−0.002, +0.003] |
| demand_supply | −0.0004 [−0.003, +0.002] | −0.0054 [−0.011, −0.001] |
| range_fit | −0.0010 [−0.003, +0.001] | −0.0043 [−0.008, −0.001] |
| latency | −0.0019 [−0.003, −0.001] | +0.0010 [−0.001, +0.003] |
| trait_logic | −0.0005 [−0.002, +0.000] | −0.0015 [−0.004, +0.001] |
| recency_segment | +0.0003 [−0.001, +0.002] | +0.0021 [+0.000, +0.005] |
| all 27 | −0.0020 [−0.005, +0.001] | +0.0039 [−0.001, +0.009] |

Added to the earlier full 139-feature set, the 27 domain features move logistic by −0.0006 (interval spans zero) and XGBoost by +0.006 (interval [+0.001, +0.013], the only hint of a gain, and it only brings XGBoost back to the level logistic reaches on the 28-feature core).

The best single added feature gained +0.0003 AUC.

## What this means

1. **None of the domain features helps in this simulator.** Every interval includes zero, or is negative for the logistic model.
2. **The compact 28-feature logistic model is the best model we have** (AUC 0.566). It beats the 139-feature set (0.560) and every XGBoost variant. More features only add noise.
3. **The strongest drivers are the same as before**: disagreement on relationship goal and pace, and a person's own yes/answer history.
4. **Why the literature does not transfer here** (explaining the result, not used to build features): the real-world findings describe mechanisms such as partner attractiveness, selectivity driven by choice-set size, and response speed. Reading the public generator confirms that a simulated person's chance of saying yes comes from their own tendency, agreement on a few soft traits, and pair-specific randomness. The partner's desirability and response speed play no role. So features built to capture those mechanisms have nothing to find.
5. **The literature still agrees with our finding in one respect**: the speed-dating study found that traits predict an individual's own tendency far better than pair compatibility. Our ceiling of about 0.56 AUC is consistent with that.

## What to do with this
- Keep the compact core (similarity agreement + own history) as the model's feature set.
- Do not spend Round 2 time on more features of this kind. If you want a stronger model, the only lever left is better estimation of each person's own tendency (a hierarchical model with a per-person effect), because that is where the signal is.
- If this work moves to real data, the domain features are worth retesting. They are cheap to compute, and real platforms do show the effects the literature describes.
- All numbers: `feature_study2.py` → `results/feature_study2.json`. Features: `features_domain.py`.
