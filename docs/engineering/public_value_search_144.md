# Experiment 144: public-value search gameplay

Owner authorized the 143 next proposal with "ok, please test that too".
This is one fixed gameplay test; no learning, promotion or successor is part of it.
All intentionally dirty work and earlier frozen dependencies are preserved.

## Question and estimand

Does real information-set PUCT, using the exact 035 canonical-max action priors
and the frozen 143 public score/resource value, beat exact greedy raw 035?
Primary is candidate score share (win 1, draw .5, loss 0) in 400 fresh games,
200 starting seeds 1440000000--1440000199, each played with candidate in both
seats. Index 2k uses candidate seat 0; 2k+1 seat 1. No replacement or extension.
The environment is the existing two-player Splendor engine. Both policies act
without opening exploration. Seed pairs are the independent statistical units.

This compares the combined system with raw 035. Search spends extra compute:
it is not an equal-consumed-compute comparison or an isolated causal test of
the value evaluator. No neutral-search or neural-search comparison is included.
The 143 held-out test is consumed and is not reused for tuning or selection.

## Frozen treatment

035 checkpoint SHA256 4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd.
143 simple.json SHA256 a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c.
Use its selected 15 coefficients, zero intercept, actor-minus-opponent public
features, float32 feature rounding then float64 dot product, clipping [-1,1].
Do not refit. Terminal values and perspective/sign changes use existing PUCT.
035 attention receives the complete concrete legal set; canonical action priors
are maxima of its logits within successor classes. No root cap or policy mix.

Use BudgetSearch with c_puct 1.5, max_depth 96, real observer-legal determinizations,
no noise, no move temperature, fresh tree every decision. Each search receives
.4 seconds, including root encoding/expansion; caller legal enumeration and
artifact logging are outside this measurement. Complete at least one simulation;
start no further simulation after deadline. One simulation plus result assembly
can overrun. Fixed-config simulations=64 is used only by tests, not as a timed cap.
Search RNG = game_seed*10000 + candidate_seat*1000 + turn + 144013.
Eight worker processes, one Torch thread each, CPU inference. Actual simulation
counts and CPU/wall timings are saved for every decision. Resource sharing can
change completed counts; interpretation is specific to this host and allowance.
Raw reference calls original LegalActionAttentionScorerBot.choose_action exactly.

## Integrity, inference and decisions

All games are retained. Natural completion and the existing two-sole-PASS deadlock
are valid; record the latter separately. Repetition limit 4, no-progress limit 60,
max turns 400. Arbitrary cutoff or stalled games have no valid winner and receive
.5 only in a labeled operational summary. Any such game invalidates a positive
scientific strength conclusion. Never use its score-adjudicated winner as a win.

Budget gate: candidate mean decision time <= .44 seconds, p95 <= .50 seconds,
and every search actually runs its .4-second allowance. Report all overshoot,
actual raw latency and the ratio; no claim of time matching to raw 035.
Any model/source identity or trajectory/root audit failure invalidates the run.

Bootstrap 50,000 whole-seed-pair resamples with NumPy default_rng(144999), percentile
95% CI. Each bootstrap observation is the mean of the two seat-swapped scores.
Primary success requires CI lower bound > .5 and point share >= .55. This proves
positive strength with a material point estimate, not that the gain exceeds 5pp
with 95% confidence. If upper bound < .5, reject the exact system as weaker.
If lower > .5 but point < .55, label positive below practical threshold.
Otherwise improvement is not established. Separately flag upper CI < .55 as
excluding the prespecified practical target. Seat results, score margin, timings,
simulation counts and depths are descriptive, not additional success tests.

## Preparation and independent verification

Unit tests cover policy identity/full attention, hidden-information invariance,
public features/sign/clipping, timed versus fixed simulation equivalence,
pairing, cutoff and budget gates, trace tampering and complete JSON serialization.
Run the full repository suite before launch. Two full probes use seed 1449000000
with both seats and the actual allowance; they are never protected outcomes.
Freeze protocol, all source files, test/verifier/wrapper, config and input hashes
before the first protected game. Preflight must bind exactly to that identity.
Inspect only progress/integrity during collection; reveal results once all 400
games finish. No outcome-dependent stopping, tuning or schedule changes.

A separate verifier reconstructs every complete trajectory, legal action, state
hash, raw-035 decision, exact public root value, canonical root set and policy
prior, visit accounting, selected row and budget record. Recompute the first
and last search decision of every game from its recorded seed and completed
simulation count, requiring exact root edges/values/action/expansion counts.
These are fixed-count reproductions, not reruns of the timed arena. Middle
searches receive structural and prior/value audits, not full-tree reproduction.
Recompute outcomes, seat-pair CIs and final summary independently; check all
game hashes and frozen dependencies. This shares the engine and search code,
so does not constitute an independent implementation of the game rules.
No champions change: 084 research, 035 fast reference.
