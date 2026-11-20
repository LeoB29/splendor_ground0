# Recommendation after159: test useful sampling-budget scaling directly

Execution update2026-09-15: the owner subsequently authorized this recommendation
with "ok lets do it". It is implemented as Experiment160; read
`docs/engineering/world_budget_160.md` for the sealed design and `AGENT_CURRENT.md`
for execution status. The recommendation and confidence review below retain their
original historical status and reasoning.

2026-09-15. Recommendation only, following the owner's next-step question.
Read AGENT_CURRENT.md,159's closed results/protocol and the154-158 strategy and
result evidence. An independent read-only reviewer challenged the alternatives
and reached the same recommendation. No new simulation, inference, training,
arena, implementation, monitoring or launch was performed. Preserve all dirty
work and all frozen predecessors. Confidence in this choice is moderate.

## Confidence review requested by the owner

The owner asked for a stronger challenge to this recommendation, without
expressing a preferred alternative. The practical128-versus32 question remains
the recommendation; agreement from another reviewer is not a guarantee that it
is optimal. The main reservation is expected information relative to two days
of compute, especially if the true gameplay advantage is modest.

A new calculation uses only existing saved outcomes, with no neural inference
or simulated games. In159's180 selection roots, nested1/2/4/8/16-world choices
were evaluated on a disjoint16-world block, averaging512 random partitions per
root. GG held-out gains over candidate0 at4/8/16 samples were1.567/2.214/3.022
score percentage points. In154's768 discovery roots, the analogous figures
were0.897/1.358/1.867pp. Thus two saved panels show a useful lower-budget trend
in the continuation policy's own objective. This is more pertinent than action
disagreement alone. It is exploratory reused evidence, and resplits are not
additional independent source games. It cannot establish the32-to128 slope,
absence of saturation, real-game benefit or a cause of158's result.159's128-world
evaluation blocks contain only the two already chosen actions, so they cannot
be reused as a complete four-action128-world selector test.

Reproducible artifacts: `scripts/analysis/review_sampling_curve_after_159.py`
and `outputs/planner_preference_159/review_sampling_curve_20260915/summary.json`
plus `roots.json`. All948 input root records were checked against their saved
locks. The20,000-resample exploratory root bootstrap corrects each stratum's
small-sample variance by sqrt(n/(n-1)). For GG, the16-minus8-world increments
were+0.808pp (pointwise95% interval[0.476,1.149]) on159 and+0.509pp
([0.353,0.669]) on154. These reused-data summaries do not create a new
confirmatory gate. No simulations or inference were performed.

A planning calculation using158's provisional paired outcomes gives a source-
pair SD of0.3413. At200 pairs that implies SE2.413pp and a two-sided95% interval
halfwidth near4.73pp, subject to a different variance in the new matchup. Under
a normal approximation, the proposed>=55% plus positive-lower-bound gate has
pass probability about20%/50%/80% at true candidate shares53%/55%/57%. The>=55%
point-estimate condition itself keeps pass probability around50% when the true
share is exactly55%; simply enlarging the panel does not remove that boundary.

Therefore retain400 games as a bounded screen for a sizeable benefit, not a
well-powered promise to detect a small one or a guarantee of decisive exclusion
after a null. An early futility design is a possible cost refinement, but must
have its stopping/inference/power tradeoffs specified prospectively; none is
adopted informally here. No extra paid conditional gate is recommended: the
saved-data trend already supports the sampling hypothesis, while a new local
proxy would still leave the gameplay question unanswered. No new experiment
has been implemented or launched as part of this confidence review.

## One next experiment

Recommend a fresh complete-game match of **the proven155 planner with128 legal
hidden-world samples per move versus unchanged155 with32 samples**. The only
behavioral intervention is the number of sampled worlds. This returns to the
practical question of improving the strongest demonstrated player.

Keep the035 proposer, maximum four canonical proposals and smaller supports,
greedy035 continuations for both simulated actors, terminal-WDL objective,
lawful hidden-world distribution and first-maximum tie handling unchanged.
At matching positions, preserve common candidate proposals and nest the32
reference world draws in the candidate's128-world stream. Fresh exogenous
streams must remain independent of hidden information. Preserve exact155 as
the reference; do not globally mutate its frozen configuration.

Propose400 fresh games from200 initial seeds, each played with seats reversed.
Freeze the whole schedule before outcomes. Use the seed pair as the statistical
unit, with paired score intervals. Proposed positive screen: candidate score
at least55% and paired95% lower bound above50%. This establishes an observed
useful advantage with evidence of superiority; it does not prove the true
advantage exceeds55%. An upper bound below55% would argue against a five-point
benefit from this particular scaling. Other results remain inconclusive, with
no automatic sample extension.400 games are a bounded screen, not guaranteed
power for an advantage close to five points. Seat and margin results should be
reported without choosing a favorable subgroup. Final implementation must seal
the precise endpoint and integrity rules before its timing or main results.

No learned continuations, new training, extra root candidates, adaptive budget
allocation or phase filter belongs in this comparison.

## Why this follows from the evidence

155 scored95.5% against035 and156 scored65.125% against084 with complete audits.
That operator is the programme's strongest directly demonstrated improvement.
157 independently established a useful fast-player gain, but the remaining
compression gap is large. Its direct improvement remains valid.

158's learned-continuation substitution scored48.01% against155, with interval
43.28%-52.74%; it remains provisional because the owner stopped full main replay
at19/402.159 completed all its prespecified checks, with sampled inner replay.
Across180 independent source pairs, LL-selected minus GG-selected moves scored
-1.441pp under freshGG continuations and+0.245pp under freshLL continuations.
The simultaneous intervals were[-2.519,-0.363] and[-0.721,+1.211]pp. Precision
was adequate, but the preset material classification remained inconclusive.

An additional read-only summary of159's already saved rows gives apparent
selection-sample differences of-4.748pp underGG and+4.878pp underLL. Their fresh
evaluation differences are the figures above. The shrinkage supports studying
finite-sample selection, but is not a causal diagnosis of158.154 likewise found
substantial selection optimism alongside a real held-out benefit, and155's
subsequent gameplay success shows that optimism alone does not invalidate search.

159's positive exploratory interaction and cross-half interaction product
suggest repeatable policy-dependent differences. They do not tell us which
simulation policy ranks moves better for actual repeated replanning. Therefore
159 has not supplied a demonstrated useful role effect to exploit.

Four times as many independent samples halves Monte Carlo standard error for
a fixed action-difference estimator with unchanged variance. This mathematical
property neither removes continuation-model bias nor guarantees stronger play.
The direct match is valuable precisely because it tests the missing practical
consequence. It will not identify the unique reason158 failed to improve.

## Why the alternatives wait

- A2x2 self/opponent continuation-role probe could localize the observed
  simulator disagreement, but would still lack an independent action-quality
  target. Earlier134/140/141 caution against treating conditional evaluation as
  a gameplay oracle. The proposed direct match has a more actionable endpoint.
- More157 fitting is cheaper, but its final training pass improved validation
  CE by only0.26%; endpoint checkpoint selection alone does not establish
  substantial undertraining. Better loss or teacher agreement is not strength.
- Learned proposals are promising for broader coverage. However,157's existing
  restricted choice already belongs to155's proposal support. Useful expansion
  requires a new support construction or an untested unrestricted learned policy.
- Adaptive allocation or variance reduction could save compute, but changes
  another mechanism. First measure the value of a simple count increase on the
  successful operator. Best-arm identification research provides principled
  allocation alternatives, not a guarantee in this correlated game setting.
  [Karnin, Koren and Somekh, ICML2013](https://proceedings.mlr.press/v28/karnin13.html).

Simulation-balancing research separately motivates treating simulation quality
as a different learning objective from direct playing strength. Its experiments
on small Go boards do not establish our Splendor mechanism or select this next
experiment for us. [Silver and Tesauro, ICML2009](https://research.ibm.com/publications/monte-carlo-simulation-balancing--1).

## Cost and operational preparation

158's402-game collection took23.08hours, with mean per-rootCPU15.61seconds forGG
and34.24seconds forLL. Replacing that cost mix with128GG+32GG gives a rough
multiplier(4*15.61+15.61)/(34.24+15.61)=1.57, or about36hours of collection.
Scaling155's8.30-hour collection by five gives about41.5hours. Both extrapolate
across differing root counts and state distributions. Budget roughly two days
including a proportionate verification plan; this is not a firm ETA.

Before main, use a fixed off-panel timing/integrity preflight and inspect timing
and validity only. Fully reconstruct actual game traces, check all world/proposal/
score/hash bindings, and prospectively select a fixed inner-trajectory replay
subset using the tested frozen kernels. This audit scope must be explicit;
it is not a full replay of every simulated branch. Fix the verification
commitment before results and complete it on either statistical outcome.

The159 detached parent still caused each Windows-spawn worker to open a console.
The next implementation must test genuinely windowless worker creation across
all stages before launch, without changing159 or killing another task's console.
Scheduled monitoring stays disabled. No automatic retries, promotion or successor.

## What the result would change

A pass would establish useful compute scaling for this planner and justify
testing more efficient allocation or stronger teacher decisions for later
distillation. It would not establish equal-time superiority, repeated learning
improvement or superhuman strength. A precise negative screen would lower the
priority of this fourfold uniform-world increase. It would leave proposal
coverage, learned objectives and other search algorithms open. An inconclusive
result must remain inconclusive; no unapproved continuation follows automatically.
