"""Sequential matching policy, split into the layers the problem statement asks to keep separate.

normalise     input handling: what is known, not asked or declined; history counts
feasibility   reciprocal eligibility: which pairs may be introduced
clarify       how the daily clarification budget is spent
model         the yes-model: features and probabilities
allocate      which feasible pairs to introduce today
safety        validation of every action, and safe fallbacks
pipeline      wires the layers together for one request
experimental  research-only variants, kept out of the default path

Everything is standard-library Python. Only the observable JSON state is ever read.
"""
