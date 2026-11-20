# Bounded teacher-variability diagnostic

13 November 2026. The user authorized implementing and running the recommended
saved-position diagnostic. This separately versioned study reopens six-world
work only for this diagnostic; it authorizes no new games, training, model
promotion, external testing or automatic strength comparison. The original EF8A
checkpoint and production players remain unchanged.

## Question and prior evidence

At larger search budgets, does increasing the root-world pool from three to six
reduce variability in the teacher's move recommendations and values? How much
sensitivity remains when root worlds or neural-input randomness are held fixed?
The previous 64-simulation six-world study improved repeatability but scored
50.5% in 400 live games (paired 95% interval 46–55%). Stability is therefore a
development measurement, not evidence of stronger play. The recently accepted
calibration is separately recovered verification; its original failed attempt
remains incomplete and every prior artifact is preserved.

## Fixed sample and matrix

Reuse the authenticated 512-position calibration panel from 128 corrected-rules
held-out games. Those positions have already been examined in development.
Select **128 primary positions, exactly one per source game**, without using
actions, outcomes, values or timing. Within each 64-game source partition, hash
order the games and assign chronological band `ordinal % 4`; take that game's
existing saved position in the assigned band. This gives 16 positions in each
partition/band cell. Select the eight smallest independent hashes per cell for
the **64-position deep subset**. Freeze all identities before fresh execution.

Every primary position gets 512 simulations. Every deep position additionally
gets 2,048 simulations. At each budget compare pools three and six, using:

- Four original shared-stream runs, seed labels A–D.
- A separate crossed diagnostic: two root-world seeds by two encoding/residual
  seeds, labels A/B independently domain separated.

This is **3,072 ordinary decisions**: 2,048 at 512 and 1,024 at 2,048. Rotate
the 16 conditions deterministically within each root/budget. Repeat 32 decisions
exactly on 16 prospectively selected deep roots in a new service lifecycle,
covering both quotas, modes and pools. No seed selection, replacement, retry,
sample extension or optional stopping is allowed.

## Player, randomness and rules

Both pools use EF8A, the accepted optimized float32 CPU graph, depth64, batch1,
one inference thread, cpuct0.4, unchanged auxiliary handling, and the same TOTAL
simulation quota (not a quota per world). Three/six hypothetical root states
are drawn once and cycled. In the separate split mode only the root-world RNG
is separated; the original stream retains neural encodings and residual uses.
The canonical route's chance-universe path is inactive and state-transition
RNG use must be inspected and disclosed. Contrasts are controlled sensitivities,
not an additive causal decomposition of variance.

Nested root draws require the three-world arm to use exactly the first three
states of the six-world draw. Retain full sampled-state debug representations,
cryptographic hashes and RNG boundary evidence. Independently verify that the
worlds are identical when only encoding seeds change, first-three worlds match
across pools, and split world generation does not advance the encoding RNG.
Original shared-stream mode must reproduce the original native actions,
scientific traces and float32 callbacks exactly. Production binaries and source
remain untouched; the changed native crate and adapter are separate versions.

The mandatory identity is `splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, turn contract
`atomic_prereveal_completed_turn_v1`. Every evaluator and audit worker calls
`verify_runtime_sources()` once at startup. Source positions are authenticated
corrected observations; only lawful observations enter model requests. Preserve
the inherited pre-reveal compound-turn and internal cutoff conventions; this
study generates no terminal training labels.

## Qualification and verification

First pass source-bound model-free tests and three independent scoped reviews,
then freeze the complete protocol, source/runtime dependencies, release build,
profiles, input/evidence bindings, request and reproduction matrices. Require a
detached owner fixture before launch. Actual-model qualification comprises the
eleven exact graph fixtures; the full corrected-rule/cost/noble/return/payment/
private-reserve entrypoint suite for every mode/pool; original native parity;
telemetry on/off invariance; and direct nested-world/stream-isolation checks.
Explicit annotation projections used to reuse historical fixture checkers must
retain all raw new identities and leave actions, callbacks and numerical work
unchanged. A projected historical checker alone is never qualification.

Prelaunch correction: the first65 model-free tests passed, but their source
preservation wrapper rejected Cargo's `.global-cache` SQLite bookkeeping, which
compilation had updated after the build's initial inventory. Preserve that
failed wrapper receipt, initial preparation and source inventory. Exclude only
that exact non-executable cache-bookkeeping file from the immutable build-input
closure; retain all Rust/crate sources, Cargo manifests/locks, compiler tools,
release binary and original-source preservation checks. Freeze the separately
refreshed preparation after source-bound tests and renewed reviews.

Retain completed raw evidence before checking it. Audit each complete atomic
action, all phases, legal support, requested/completed work, callback bits,
source/profile/mode/seed identities, visits, missing Q support and real clock.
Independent acceptance rereads all records and exact reproductions, recomputes
the statistics through independent arithmetic, and verifies frozen-input
preservation and actual process exits. A completion file alone is insufficient.
After graceful native/service shutdown, allow up to five seconds for observed
asynchronous exact-process disappearance, inside the original stage clock. Save
the observed grace and absence; never treat a surviving identity as success.

## Measurements and interpretation fixed before execution

The single primary endpoint is the paired pool6-minus-pool3 difference in mean
pairwise Jensen–Shannon divergence of the normalized visit distributions, at512
in original shared-stream mode. Four seeds supply six dependent pairs per root;
average those pairs within the root. A negative upper95% bound supports reduced
variability on this development panel; a positive lower bound supports increased
variability; otherwise the result is unresolved. This does not select a player.

Use 20,000 paired source-game bootstrap draws, stratified by the two original
partitions, fixed seed202611133072. Seeds/pairs from one root are not independent
sample units. Secondary exploratory intervals describe total variation,
actual-action disagreement (including the true native tie-break), root-value
variation, fixed-action Q-contrast variation, and the deeper subset. For the
crossed design report same-world/different-encoding, same-encoding/different-world
and both-changed pairs separately. Do not sum them into variance percentages.
For the secondary budget comparison use only the same64 deep roots at both512
and2,048, preserving paired games and requiring joint Q support across both
budgets. Do not infer a budget effect by comparing the128-root primary summary
against the64-root deep summary. Fixed encoding seeds control the random stream;
different visited paths can still receive different neural-input completions.

For each primary root freeze two distinct safe Main actions ranked by the pooled
visit counts from its existing accepted 2,048 R/S searches, breaking ties by
canonical index. Q contrasts require both actions to have been visited in every
repeat of both pools for the comparison; report common coverage and exclude
unsupported roots without imputation. These Qs remain same-network internal
estimates, not objective move quality. The historical `network_value` telemetry
is a post-search dynamic-FPU mean, not the initial network prediction.

Report costs and all over-one-second responses; neither one second nor700ms is
an admission or selection gate. Do not call twice-JSD an irreducible noise floor,
convert stability into Elo, or treat reference self-agreement as a ceiling.
Produce at most two summary charts after independent acceptance. The next
recommendation will weigh teacher stability against broader-data collection;
no live comparison follows automatically.

## Bounded execution

Three serial model services maximum: qualification, diagnostic, reproduction.
Model CPU0 and controller CPU2 where available; no concurrent model experiments.
Every complete response has an external120-second arrival deadline and internal
119-second allowance; nested qualification decisions retain a real watchdog.
No active clock reset or failed-decision retry. Startup240seconds; graph300seconds;
entrypoint batch6000seconds. All are clipped to original stage/global clocks.

Overall24hours, model cutoff18hours. Stage caps: qualification2hours,
diagnostic12hours, reproduction2hours, analysis4hours, acceptance4hours, all
clipped to overall/model deadlines. Actual model-tree RSS4GiB, audit/root8GiB,
new evidence16GiB, actual disk reserve5GiB. No available-RAM admission gate and
no forecast-based refusal. Current planning estimate is roughly4–6hours of
search plus qualification and audits; it is an estimate, not a stop rule.
No scheduled monitor. Preserve every failure and distinguish technical
incompletion from an unfavorable scientific result.
