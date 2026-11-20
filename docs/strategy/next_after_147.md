# Recommendation after 147: establish search-teacher headroom

Authorization update: the owner subsequently approved this proposal with
"ok, please do it". Execution protocol:
`docs/engineering/search_teacher_headroom_148.md`. Follow AGENT_CURRENT for
status; original proposal and not-authorized statements below are historical.

Completion update: 148 finished and passed independent audit. In 400 fresh
natural games, 1.6-second search scored 53.75% against .4-second search,
CI95 [49.5%,58.0%]. Both strength requirements failed; the 55% target is not
excluded. Actual mean time ratio was 3.963922, with 221.283 versus 56.357
simulations. The conditional raw-035 panel did not run. See
`docs/engineering/search_teacher_headroom_148_result.md`. No extension,
training, promotion or successor is authorized by this completion.

2026-09-08. Recommendation only; no launch authorization. No training, inference,
new games, searches or tests ran during this review. Preserve the intentionally
dirty worktree, frozen experiment artifacts and champions 084 research / 035 fast.

## Decision

Recommend one bounded search-budget experiment, provisionally 148: **does the
existing 035/public-value planner become a materially stronger player when its
decision allowance increases from .4 to 1.6 seconds?** Keep its policy, value,
search rules and all other settings fixed. Measure full-game strength directly.
If that comparison passes, run one prespecified fresh comparison with raw 035.
Both stages belong to the same proposed experiment; neither is authorized yet.

This is a capability check before another learning cycle. It is not a new
neural value fit, a rescue of 147, or an open-ended budget sweep. The longer-term
objective remains a planner/learner that improves across generations, rather
than accumulating refinements around a permanently frozen 035.

## Why this question now

147 gave both learners 16,384 updates and covered 99.959% of available unique
training positions. Its selected trainable model improved fresh early/middle
MSE over the public baseline by .6977%, below the 3% requirement; superiority
over frozen-long and trainable-short was not established. The final trainable
checkpoint's training/validation MSE was .126827/1.658038. More updates to this
fixed recipe are therefore a poor default investment. This is not evidence
that value learning generally fails, nor that the small observed benefit is zero.

The more fundamental open question is whether the current search can generate
a useful improvement signal around the strong policy. At .4 seconds, 144
scored 52.75% against raw 035, CI95 [48.5%,57.0%]; 145's public-versus-neutral
comparison scored 52.375%, CI95 [49.25%,55.5%]. Neither establishes the relevant
strength gain. They do not exclude useful effects, and they must not be pooled.

Those searches completed roughly 60-65 simulations per decision. The mean of
each search's maximum depth was about 2.3. This describes a shallow operating
regime; it does not establish depth as the cause of the weak results. A fourfold
time allowance is a substantial, directly interpretable intervention with
manageable measured-host cost. It tests the total effect of additional search,
including exploration and chance sampling, rather than isolating depth alone.

113 already established 1,024-versus-256-simulation headroom on generation 11
(57.375%, CI95 [52.5%,62.25%]). That was a different, much weaker policy/value
system. It does not establish headroom for 035/public-value search. Likewise,
084 establishes that an expensive rollout resolver can improve 035, but is
not evidence that this selective shared-tree operator supplies a good teacher.

## What the accumulated evidence deprioritizes

- Plainly extending the fixed value-training recipe: 142 and 147 show overfit;
  146/147 did not establish the required representation-learning gain.
- Merely averaging more terminal outcomes: 130 measured substantial conditional
  outcome variance, and 132's compact frozen-head repeated-return treatment
  gained only about .195% in protected MSE. Those do not rule out better targets
  or a different learner, but another generic noise audit would repeat evidence.
- Generic imitation of the proven resolver: 052/053 and subsequent teacher
  audits already explored that direction. It is not an untried alternative.
- Treating offline root gains or stable search agreement as strength evidence:
  134 rejected expanded-root 086, while 138-141 showed that conditional move
  estimates still leave important deployment questions unresolved. 137 did not
  establish a ranking benefit from its refill-coverage intervention.

The strongest remaining hypotheses concern useful search improvement, learning
targets and state distributions that support decisions, and an integrated
learning cycle that transfers improvements into actual play. Fully learned
representations, stronger regularization and afterstate architectures remain
open, but none is uniquely indicated by 147. Test the current planner's ability
to improve first, before choosing another learner intervention.

## Proposed bounded design

1. **Primary compute comparison:** 200 new starting seeds, each played with
   swapped seats: 400 games of public-value PUCT at 1.6 seconds versus the same
   planner at .4 seconds. Freeze exact 035 canonical priors, the selected 143
   public evaluator, c_puct=1.5, depth cap 96, legal determinizations, terminal
   handling, action selection and fresh-tree behavior. No noise or temperature.
   Neither 146 nor 147 checkpoint enters this experiment. Budgets include root
   inference, as in 144/145. Keep eight CPU workers and one Torch thread each.
2. **Conditional competence comparison:** only after the complete first stage
   and its audit pass, play the unchanged 1.6-second candidate against exact raw
   035 on another 200 disjoint starting seeds / 400 swapped-seat games. Freeze
   this stage before collecting either panel. Do not infer this matchup from
   the primary result or from historical 144; strength can be non-transitive.
3. **Decision:** for each panel require score share >=55% and paired CI95 lower
   bound >50%, plus integrity and timing checks. A teacher-headroom pass requires
   both panels to pass. Draws count .5. Use 50,000 whole-seed-pair bootstrap
   replicates, fixed RNG seeds, and no exclusions, extensions or alternative
   checkpoint selection. This establishes superiority with a material point
   estimate; it does not establish a gain of at least five points with 95%
   confidence. Report the full interval and whether it excludes the 55% target.

Use separate deterministic seed namespaces for both panels, preflight and
statistics. Turn this proposal into a fully specified protocol before execution;
do not select the time allowance, sample size or gate from protected outcomes.
Historical timing supports feasibility but is not a fresh runtime measurement.

Retain natural termination and explicit forced-PASS deadlock accounting from
144/145. Arbitrary cutoffs cannot support a positive scientific result. Require
the full time allowance per search, with mean <=1.10 times allowance and p95
<=1.25 times allowance in each timed arm. Save CPU and wall time, simulations,
depths and root traces. A fourfold allowance does not promise exactly fourfold
simulations or measured CPU consumption. This intentionally tests extra compute,
not equal-time efficiency. Prior ranks, action families and depth are diagnostics,
not substitute success endpoints or proof of a causal bottleneck.

Prepare focused tests and outcome-blind probes, freeze code/config/input hashes,
and independently reconstruct all trajectories, root policy/value records,
outcomes and paired statistics. Reproduce first and last search of each timed
arm per game using saved RNG and completed counts, as in 145. Preserve earlier
dependencies using their explicit manifest lists; do not rewrite frozen source.
Do not examine running win rates or change the design during collection.

## Interpretation and cost

If both panels pass, there is a validated stronger shared-tree teacher at the
tested allowance. The next proposal should address learning from it in a bounded
live cycle. A pass does not prove the student can absorb the gain: 113 followed
by 114/127 already warns against that inference. It does not justify promotion
over 084, nor establish the public evaluator's isolated benefit at 1.6 seconds.

If the first panel fails or is inconclusive, stop this experiment. Do not run a
series of larger budgets or add games to pass. The current operator remains
unvalidated as a practical improvement teacher, motivating a substantive review
of evaluator targets, search allocation and representative states. A negative
result at 1.6 seconds would not rule out every budget or architecture. If the
first panel passes but raw-035 superiority is unestablished, report exactly that
and do not begin a teacher-learning cycle automatically.

Planning estimate on this host: about 55 minutes for the first arena and another
45 minutes if the second runs, plus preparation and audits. This is scaled from
145's 43.37-minute 800-game .4/.4 arena and 144's approximately 11-minute 400-game
.4/raw arena with eight workers. Allow roughly 1-1.5 hours after launch if it
stops after stage one, or about 2-2.5 hours for both stages and verification;
implementation/testing are additional. Actual cost depends on game length,
resource contention and audit replay work. Refine from preflight without
changing the scientific design based on outcomes.

No automatic training, promotion or successor. This review adds a proposal and
handoff notes only. All previous negative results remain closed and unchanged.

Sources: `docs/engineering/value_training_exposure_147_result.md`,
`docs/engineering/public_value_search_144_result.md`,
`docs/engineering/public_value_ablation_145_result.md`,
`docs/engineering/alphazero_search_headroom_113_result.md`,
`docs/engineering/alphazero_value_uncertainty_audit_130_result.md`,
`docs/engineering/alphazero_deadlock_consistent_repeated_return_value_ab_132_result.md`,
`docs/strategy/independent_superhuman_review_after_134_137.md`, and
`docs/strategic_pivot_after_089.md` (historical strategic intent, read alongside
the later authorized strong-initialization work).
