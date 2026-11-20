# Independent-world validation of frozen resolver choices 138

Frozen before selection or new continuation outcomes on 2026-09-06. The owner
explicitly authorized this experiment after reviewing the independent review
following 134/137. Preparation, verification, and launch are authorized; no
training, promotion, installation, full-game arena, or automatic successor is.

## Question and fixed inputs

Does the advantage of the original 086 discovery action over the original
four-root 084-contract discovery action attenuate when evaluation holds out
worlds as well as continuation-action randomness?

Source selection:
`outputs/expanded_root_regret_stability_replication_086/selection.json`, SHA256
`7bebc07fede835be023c986701fd813e4134374f1a4559fdcaf9439910d7e46f`.
Original decisions:
`outputs/expanded_root_regret_stability_replication_086/result.json`, SHA256
`8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.
Frozen legal 035 checkpoint SHA256:
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
Neither 087 outcomes nor human records are inputs.

Select ten states per `seat0/seat1 x early/mid/late` cell, 60 distinct source
games total. Within each cell sort by SHA256 of the UTF-8 string
`138-selection-v1|SOURCE_SEED|TURN_INDEX|ACTOR_SEAT|PHASE`, then source seed and
turn. Selection reads only source identifiers/cells and snapshots. Only after
the IDs are fixed, attach the two original discovery-selected action keys from
086 result rows. No payoff, effect, action disagreement, or terminal outcome
influences inclusion. Identical choices remain included and share computation.
The resulting manifest, protocol, checkpoint, implementation/dependencies, and
preflight receive hashes before launch and must match on resume.

## Crossed evaluation

Each state has four blocks in each of two conditions:

- `original_worlds`: the exact three worlds reconstructed using 086's original
  `deterministic_belief_range` schedule, with entirely new continuation seeds;
- `fresh_worlds`: independently reconstruct three observer-legal worlds for
  each block using root index `1_380_000_000 + sample_index*100 + block*10` and
  the unchanged belief function's seed start 49,000,000.

Continuation seed is
`1_380_000_000_000_000_000 + sample_index*100_000 + block*1_000 + world*10 + replicate`.
It is paired across conditions, root alternatives and replies. The four
replicates per leaf are independent seed slots; all are disjoint from 086.
Each fresh world resamples the complete legal unseen pool and deck order,
including opponent private reserve possibilities. Repeated draws are legal;
do not reject worlds for matching identities by chance.

Evaluate only the two fixed semantic actions. For every world, apply each root
through the exact engine, regenerate the opponent's top min(4, legal distinct
actions) 035 replies, and run four unchanged stochastic-035 loop-safe terminal
continuations per reply (epsilon .1, temperature .5, top-3, 400-turn cap,
64 loop-fallback triggers, score margin /20). No old reply IDs are carried into
different refill states. Identical opponent information sets must have identical
semantic reply sets and one shared minimizing reply. Reuse `resolve_two_ply`.
An information-set/action inconsistency is an integrity failure, never a reason
to discard or replace a world. Terminal root actions contribute exact scores
and WDL through a singleton terminal branch, requiring no continuation.

The existing continuation routine is unchanged. An observational environment
wrapper records its final state to obtain WDL and scores without consuming RNG
or changing transitions. Preserve its existing loop contract; loop-adjudicated
returns receive score/card-count WDL and are explicitly distinguished from
natural terminal outcomes. Proven sole-PASS deadlocks can be labeled at the
adjudicated endpoint; they are not silently treated as normal wins. Every
continuation must resolve, hard caps are forbidden, and loop adjudication must
be <=0.1% both overall and within each condition. Fallback *triggers* are telemetry.

Resolve each three-world/four-continuation block independently, then average
the four block values within a state. Do not pool fresh worlds before the
nonlinear opponent minimum. This tests the deployed four-continuation estimator,
not the old headline 086 pooled-twelve-continuation estimator. No new root choice
is selected or fitted. Maximum primary budget is
60*2*4*2*3*4*4 = 46,080 terminal continuations, reduced only by coincident fixed
actions, fewer legal replies, or terminal roots. A resumable result is saved
after each complete state; incomplete in-flight states use the same seeds.

## Endpoints and decisions

For each state and condition, let gain be fixed expanded-action value minus
fixed champion-action value. Primary endpoint D is mean fresh-world gain minus
mean original-world gain. Negative D indicates attenuation. Material attenuation
is 0.025 normalized margin. Every state represents one distinct game; use the
same equal-state/equal-game weighting for means and confidence intervals.
Bootstrap 50,000 paired samples using NumPy default_rng seed 138300000,
resampling within each of the six fixed cells (ten games per cell). Report
percentile CI95 for both gains and D. Blocks and rollouts are not independent
population observations. Report phase/seat/private-card/action-family breakdowns
descriptively, with no subgroup-driven successor.

Prespecified interpretation, after all integrity checks:

1. Original-world point gain <=0: `conditional_gain_not_reproduced`.
2. Original-world CI lower <=0: `original_world_gain_uncertain`.
3. D point <=-.025 and D CI upper <0:
   `material_world_attenuation_supported`. Also report whether fresh gain is
   positive, uncertain, or negative; attenuation need not explain the whole loss.
4. Fresh gain CI lower >0 and D CI lower >-.025:
   `conditional_gain_survives_fresh_worlds`.
5. Otherwise: `independent_world_validation_inconclusive`.

These are diagnostic labels, not strength or optimal-value claims. No decision
promotes 086, authorizes distillation, or launches another experiment.

Record secondary WDL in [-1,+1] and scores/card counts from the same trajectories.
Report (a) WDL under the margin-minimizing opponent reply, and (b) separately
WDL-minimized values. Use the same deterministic semantic-key tie-break. These
sample-selected minima are descriptive policy-conditioned estimates, not
unbiased opponent best-response values or exact game values. Record reply
choice differences and both fixed-action WDL gaps. The primary objective stays
margin so the world-independence contrast does not also change the objective.

## Verification, precision, and runtime

Before launch: validate source hashes; schedule and fixed action bindings;
snapshot round trips; original-world reproduction against 085's exact branch
builder; root-observation/unknown-card-pool conservation; invariance to hidden
deck order; accounting; grouped-minimum and WDL tests; and focused regression
tests. Freeze all src Python dependency hashes and the runner/test hashes.
Run the full test suite before protected evaluation. No test outcome controls
state inclusion or scientific thresholds.

This 60-game mechanism panel is not represented as a powered gameplay study.
For a two-sided 5% normal approximation with 80% power against |D|=.025,
60 independent games require state-difference SD approximately <=.0692.
Prospectively record CI half-width estimates 1.96*SD/sqrt(60) for SD=.05,.10,.20:
approximately .0127,.0253,.0506. The unknown fresh-world variance cannot be
estimated from 086's fixed-world evaluation halves. If precision is insufficient,
report that limitation; do not add states or change the materiality threshold.

Preflight may perform eight timed continuations on one source state outside the
60-state panel, with a disjoint 1389 probe namespace, to establish operational
feasibility. Report runtime/integrity only. Cap projected serial primary work
at 12 CPU-hours and use ten CPU workers, one Torch thread each. Projection is
eight-rollout timing times the maximum budget, with a factor of two engineering
allowance. If this feasibility ceiling fails, stop before protected evaluation
and report it; do not quietly reduce the scientific workload. Runtime during
the primary experiment is telemetry; genuine hangs/integrity failures stop it.

Launch requires `--launch` plus a verified preflight and matching preparation
identities. A process lock prevents concurrent copies. Atomic per-state files
and deterministic manifests support crash resume. Results are sealed until all
60 states complete, apart from completion counts, runtime and integrity status.
Keep exact 084/035 and all existing checkpoints unchanged.
