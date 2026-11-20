# Experiment 140: fixed moves under different continuation policies

Frozen 2026-09-07 before timing probes or protected outcomes. The owner authorized
preparation and execution after the 139 result. No training, model change,
promotion, new source games, candidate tuning or automatic successor is included.

## Question and estimand

Does the advantage of a frozen expanded 086 proposal over the original 084-contract
choice shrink when subsequent play uses exact 084 rather than stochastic 035?
Each move is forced once, then both players use the assigned continuation policy
to the end. Starting worlds, root choices, scoring and outer termination rules
are paired and identical across conditions.

Primary conditions are `stochastic_035` (epsilon .1, temperature .5, top-3, as in
the earlier rollout evaluator) and `resolver_084` (the exact frozen four-root
resolver, including its native structural fallback). A cheap `greedy_035` control
uses the native 035 argmax policy. This distinguishes a change in action sampling
from adding the stronger resolver without additional 084 continuations. The two
primary contracts differ in both planning and stochasticity; do not attribute a
primary change uniquely to planning without considering that control.

This experiment deliberately has no sampled opponent minimum at the outer root.
After the forced move, the assigned policy chooses the immediate opponent reply
and every later move. The stochastic control is therefore not an exact replication
of 139's grouped two-ply estimator. A positive stochastic-control gain must be
reproduced before attributing a loss of advantage to stronger subsequent play.

## Fixed inputs, selection and timing-only panel size

Use the unchanged 086 selection and original discovery decisions, with SHA256s
`7bebc07fede835be023c986701fd813e4134374f1a4559fdcaf9439910d7e46f` and
`8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.
Legal 035 checkpoint SHA256 is
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
No 087 outcome or human record is read. No 138/139 returns or screening decisions
are inputs; their ID-only selectors determine excluded source games.

Exclude the 60 games in each of 138 and 139. Rank the remaining positions within
each phase/seat cell by SHA256 of UTF-8
`140-selection-v1|SOURCE_SEED|TURN_INDEX|ACTOR_SEAT|PHASE`, then source seed/turn.
The first ten per cell form a 60-state maximum panel. Attach original champion
and expanded discovery action keys only after the IDs are fixed. Coincident
choices remain and share computation. Panels of 48, 36 and 24 are nested prefixes
of eight, six and four states per cell, respectively. Every state is a distinct
game. No outcome, action disagreement or estimated gain selects a position.

Select six probe states outside 138, 139 and the entire maximum 140 panel: one
per phase/seat cell, lowest SHA256 under the same string with prefix
`140-probe-v1`. Each probe forces its original default action in one new probe
world, then completes once under each of the three policies. Retain runtime,
termination/integrity and resolver-work counters only, never payoff, winner,
scores, selected actions or full trajectories. Repeat the late seat-0 resolver
probe from zero and compare a digest of the complete deterministic result with
timing excluded. No probe becomes a protected observation.

For each phase, let T be the slower of its two probe-state sums of elapsed times
across all three policies. Project a panel of N states as
`1.5 * (N/3) * 2 root actions * 3 worlds * sum(T_early,T_mid,T_late)` serial
seconds. The factor 1.5 is an engineering allowance. Select the largest N in
60,48,36,24 with projection <=129,600 seconds (36 serial CPU-hour approximation,
about 3.6 hours across ten workers before overhead). Record actual process CPU
times as well as wall times; the projection uses conservative serial wall times.
If no panel fits or probe integrity fails, stop before protected evaluation.
Do not change the policies, world count, screen, thresholds or panel after seeing
outcomes. Once selected, elapsed primary runtime is telemetry, not optional
stopping. No time-based extension of the panel is allowed.

## Worlds and policy randomness

Each selected position has three fresh full observer-legal worlds, sampled by
the unchanged `deterministic_belief_range` with root index
`1_400_000_000 + sample_index*10`, count 3 and seed start 49,000,000. Probes use
root index `1_409_000_000 + sample_index*10`, count 1. Sample the full unknown
pool and deck orders; preserve the root actor's information and known cards.
Worlds are not rejected for coincident cards or outcomes. Both root actions and
all conditions use the same actual starting world and independent policy seeds.

Primary policy slot is `sample_index*3 + world_index`. Probe slot adds 10,000.
The stochastic-035 RNG seed is `1_410_000_000_000_000_000 + slot*1_000_000`,
shared between the two forced-root alternatives; one RNG serves both players.
Each 084 continuation constructs fresh bots for both seats with engine seed
`1_400_000 + slot`, smoke seed 1,400,000, belief-root base 1,420,000,000 and
inner-continuation base 1,400,000,000,000,000,000. Preserve the native formulas
using actor-seat/decision/world/replicate offsets. The two alternative roots share
the same slot. All namespaces are disjoint from 086/138/139 and between probe and
primary evaluation. Different branches may diverge after their common seed slots;
do not claim their later trajectories are identical.

The 084 bot is reused unchanged: 3 worlds, 4 roots, 4 opponent replies, 4 inner
stochastic-035 continuations, score-margin scale 20, inner cap 400 turns and
64 fallback triggers. It uses serial inner continuations. Preserve every native
structural fallback, including insufficient replies or inconsistent semantic
reply sets; these are part of the deployed policy. Do not apply the 139 adapter
inside 084 or alter its ranking, random draws, horizon or decision rule.

## Common outer game driver and outcomes

Resume the exact engine from the sampled state. Force the specified legal root
action once without querying either continuation bot. Reset policy decision
counters at the start of each continuation. Then use the assigned policy at
every turn, including the first opponent reply. There is no repeated 086 policy
intervention. Use the existing match rules: absolute turn cap 400, repetition
limit 4, no-progress limit 60, and a proven two-sole-PASS deadlock scored by the
established score/card-count tiebreak. Counters start at the snapshot; prior game
history is not reconstructed. Both conditions use identical outer rules.

Only natural completion or the existing forced-pass-deadlock endpoint is valid.
No score from a turn/repetition/no-progress cutoff is accepted as evidence.
Illegal actions, engine failures, unresolved inner continuations or hard caps
are integrity failures. Stop and preserve evidence on such failures; never drop
or replace an inconvenient state/world. Inner 084 adjudications must be <=0.1%
in aggregate. Native structural fallbacks are telemetry, not excluded cases.
Require the 084 operator to complete a search in every source phase and to switch
away from its raw-035 default at least once overall before interpreting a policy
contrast. Report search/fallback rates without claiming equal computation.

Save complete starting worlds, canonical root keys, per-turn canonical actions,
before/after state hashes, terminal scores/card counts, WDL, normalized clipped
score margin, final state and per-seat 084 telemetry. Save each completed
trajectory atomically. Test/evaluation never reselects root actions.

Maximum protected work is N*3 worlds*2 actions*3 policies <=1,080 outer
continuations, at most 360 using 084; identical roots share trajectories. The
many inner 084 rollouts are separately counted and are not independent samples.
Record full-game and inner-search work, CPU and elapsed times. No direct arena
between 086 and 084, self-play training corpus or network update is included.

## Estimates, interpretation and precision

For each state and condition, average expanded-minus-default normalized margin
over its three paired worlds. Primary D is the mean resolver_084 gain minus the
mean stochastic_035 gain. Bootstrap 50,000 paired source games within the six
fixed cells with NumPy default_rng seed 140300000, ordered by cell/source seed.
Use the same equal-state/game weights for means and intervals. Keep policy,
root and world results paired. Individual worlds, turns and inner rollouts are
not independent population observations.

Prespecified primary decision after integrity and operator-dose checks:

1. Stochastic-control point gain <=0: `stochastic_control_gain_not_reproduced`.
2. Its CI95 lower bound <=0: `stochastic_control_gain_uncertain`.
3. D point <=-0.025 and CI95 upper <0:
   `continuation_policy_mismatch_supported` (describe the strong gain separately).
4. Strong gain CI95 lower >0 and D CI95 lower >-0.025:
   `advantage_survives_stronger_continuations`.
5. Otherwise: `continuation_policy_crossover_inconclusive`.

Report both absolute gains, D, the greedy control gain and strong-minus-greedy
and greedy-minus-stochastic contrasts. Greedy contrasts and WDL are secondary;
report intervals descriptively without a separate confirmatory claim. The primary
change compares complete continuation contracts, not an isolated proof about
planning strength. A missing stochastic control gain cannot be blamed on 084.
Even a surviving strong-play gain does not validate repeated 086 play or its
on-policy state distribution. No result promotes a bot or authorizes training.

This is a mechanism pilot with unknown effect variance, not a promised powered
strength test. Freeze N using timing only and report uncertainty honestly. For
state-contrast SD .05/.10/.20, report prospective normal CI half-widths
`1.96*SD/sqrt(N)`. Do not add worlds or states to cross a statistical gate.
Phase/seat/private-state breakdowns are descriptive, never tuning signals.

## Verification and execution

Before protected launch, verify source hashes, state/action binding, nested
balanced selection and probe separation, snapshot round trips, information/card
inventory conservation and hidden-deck permutation invariance. Test common
driver parity with native match logic, forced-root behavior, deterministic replay,
seed separation, stochastic sampling, score/tiebreaks, cutoff rejection, paired
bootstrap and resume validation. Run focused and full tests before launch.

Freeze protocol, input dependencies, runtime-selected manifest and preflight.
Use ten CPU workers, one Torch thread each, no nested process pools, a kernel
process lock, atomic trajectory files and identity-checked resumes. Do not expose
partial scores, winners or contrasts. Display only counts, runtime and integrity
until the full selected panel is complete. Independently replay every saved
trajectory through the engine and reconstruct outcomes, primary means and
bootstrap intervals without calling the experiment's driver or summary helpers.
Preserve the intentionally dirty worktree and all existing artifacts.
