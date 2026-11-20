# Experiment 141: opponent-response calibration

Prospective protocol, 2026-09-08. The owner authorized implementation and execution
of the proposed cheap evaluation calibration. No training, strong-player search,
arena, promotion, or automatic successor is included. Preserve the dirty worktree.

## Question and estimands

For the original frozen 084/086 move pairs, does replacing 139's sampled opponent
minimum with 140's actual stochastic-035 response change the estimated advantage?
All subsequent decisions use the same stochastic 035 policy: epsilon .1,
temperature .5, top three. The checkpoint and historical source artifacts are
unchanged. This is a calibration of weak-policy evaluation, not optimal values or
evidence of stronger gameplay. A difference need not explain 134's failure.

There are two arms on identical sampled worlds and fixed root actions:

1. `sampled_min`: 139's top-four distinct canonical opponent replies, four
   stochastic continuations per reply. Minimize the mean within each shared
   opponent information set across its worlds, with lexical semantic-key ties.
   Retain 139's semantic-set guard; ordering alone is irrelevant. Resolve each
   three-world block separately, then average blocks. Do not pool all worlds
   before the nonlinear minimum.
2. `policy_reply`: force the root once, then let stochastic 035 choose the
   immediate reply and every later move, four full continuations per world.
   This is 140's weak continuation law, with greater replication.

Both arms use the unchanged loop-safe continuation routine used by 139. All
protected paths must finish naturally, with zero fallback, adjudication or hard
cap, before absolute turn 400. Independently track repetition/no-progress limits
including the forced action. With these strict conditions the different cutoff
implementations in 139/140 do not determine any outcome. Off-panel probes compare
the actual-policy action sequence and endpoint against 140 from the same seed.
Terminal roots are exact endpoints, with no unnecessary continuations. Fewer
than four legal replies are retained and counted; no state/world replacement.

## Population and fixed budget

Select **120 positions from 120 distinct existing 086 source games**, twenty in
each phase/seat cell. Exclude all 138, 139 and 140 protected games and their timing
probe games. This leaves 23 eligible games per cell. Rank identifiers by SHA256
of `141-selection-v1|SOURCE_SEED|TURN_INDEX|ACTOR_SEAT|PHASE`, then seed/turn ties;
select the first twenty. Original outcomes, gains, action agreement and private
information status do not enter selection. These positions are fresh to these
calibrations, not fresh source gameplay: all come from the historical 035 corpus
used by 086. Bind the two old decisions only after selection. Coincident choices
stay as zero-gain observations, with shared computation.

Use **16 independent blocks of three worlds** per position (48 worlds), with
four continuation RNG replicates. Upper bounds: 184,320 sampled-min continuations
and 46,080 policy-reply continuations, total **230,400**. Actual counts account for
coincident roots, terminal roots and reply availability. No additional sampling
after results; no subgroup-driven selection or tuning.

World root index: `1_430_000_000 + sample_index*1000 + block*10`, belief seed base
49,000,000, world offset 0--2. Probe world index substitutes 1,439,000,000.
Continuation seed: `1_430_000_000_000_000_000 + sample_index*1_000_000
+ block*1000 + world*10 + replicate`; probes add 9,000,000,000,000,000.
The seed is shared across roots, replies and arms. It starts at the first
unforced action, so the two arms are not claimed to consume RNG at identical
plies. All namespaces are distinct from 138--140. No root is selected again.

## Precision and analysis

The primary endpoint is **policy-reply gain minus sampled-min gain**, where each
gain is expanded-minus-default clipped final score margin divided by 20.
Report both absolute gains alongside it. Average replicates within world/reply,
resolve the block, average 16 blocks within state, then equally weight all 120
states/games. Bootstrap 50,000 paired source-game resamples within six cells,
sorted by cell/seed, NumPy seed 141300000. Worlds and continuations are not
independent population samples. Primary CI95 is unadjusted for the one primary
contrast; absolute-gain and WDL CIs are prespecified secondary estimates, not a
family of independent confirmatory claims. WDL uses true card-count tiebreaks;
for sampled-min it follows the margin-minimizing reply.

The precision target is primary CI half-width <= .025 (half a score point).
This is a target, not guaranteed power and not an execution-validity gate.
For state contrast SD .05/.10/.20, normal CI half-width at N=120 is
.00895/.01789/.03578. The 140 pilot's three-world covariance decomposition is
unstable (a negative estimated between-state contrast variance), so it cannot
justify precise power claims or a smaller panel. Its weak-arm moment estimates
suggest that 48 worlds could yield about .019 CI half-width for the absolute
weak gain at N=120, ignoring any benefit of additional continuation replicas;
this rough planning calculation is not promised precision for the primary
operator contrast. New data will report within-block and between-state variation
for subsequent design, retaining negative moment estimates rather than silently
turning them into biological/statistical facts.

Classify the primary CI independently of whether either absolute gain is positive:
- wholly above +.025 or below -.025: material response-rule effect supported;
- otherwise excluding zero: response-rule difference detected, materiality uncertain;
- wholly within [-.025,+.025]: response-rule difference bounded below material size;
- otherwise: response-rule contrast inconclusive.
Separately report each absolute gain as positive, negative or uncertain from its
secondary CI. Report achieved precision. No point-estimate-only material claim.
An uncertain control does not invalidate a clean paired operator comparison.

## Preparation, execution and integrity

Hash source, checkpoint, exclusion artifacts, protocol and all Python dependencies.
Verify all 1,920 block structures before launch: observer information, full card
inventory, invariance to actual hidden deck ordering, root legality, shared reply
membership, and exact expected work. Select one off-panel timing probe per cell
from the remaining eligible games using `141-probe-v1` and run block zero under
both arms. Retain only time, counts, integrity and opaque result digests; discard
payoffs/winners. Repeat the late-seat0 probe from scratch for determinism.

Conservative serial wall projection is twice the sum over cells of
`20 states * 16 blocks * probe block seconds`. The prelaunch ceiling is 24
aggregate serial hours. Fixed sample size is not reduced for timing. If the
ceiling fails, stop before protected outcomes and review feasibility. Once
launched, runtime is telemetry and cannot stop a valid run selectively.

Use ten CPU workers with one Torch thread each, atomic per-state checkpoints,
identity-checked resume and an exclusive process lock. Before all 120 states
complete inspect only timing, progress and integrity, not gains or winners.
Preserve failures, never drop/replace a difficult case. Require exact expected
continuation accounting, finite returns, correct source bindings, all planned
blocks, all hash guards, and at least twelve differing fixed move pairs for
meaningful intervention dose. A dose failure is reported, never fixed by selection.

Before execution run focused scientific/seed/identity tests and the repository
suite. After completion independently reconstruct terminal scores/WDL, every
leaf mean, grouped minimum, policy average, state contrast and bootstrap endpoint
without using experiment aggregation/resolution helpers. Check saved world/reply
structures and hashes. This endpoint reconstruction is not full trajectory replay
or an independent game-rules implementation. Preserve all previous artifacts and
leave 084/035 as champions whatever the calibration result.
