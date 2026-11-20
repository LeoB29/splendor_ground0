# Bounded same-position proposal diagnosis at 128 worlds

2026-09-27. The owner authorized implementing and starting the complete diagnostic
recommended in docs/strategy/next_after_proposal_iteration_128.md. This new study
does not alter the completed generation or any frozen ancestor. New code is in
tools/proposal_diagnostic_128; entrypoint run_proposal_diagnostic_128.py; output
outputs/proposal_diagnostic_128. All previous dirty work is preserved.

## Fixed question and source

Does the selected second-generation model supply better complementary candidates,
and does this improve independently assessed choices under the unchanged planner?
This diagnoses one update on saved arena states. It does not measure full-game
win rates, causal isolation of teacher quality, equal-time strength or human play.

Source: completed outputs/proposal_iteration_128, binding
1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace.
Require its final audit and all model seals, the unchanged frozen dependencies,
and the completed 402-game Stage A. The observed51.4925% score did not establish
practical progress; the skipped Stage B is not resumed by this study.

## Panel fixed before new outcomes

Main: one pre-decision state from each of all201 Stage A source pairs,67 per fit.
Hash-order pairs within fit and hash-order six source-role x phase cells; cycle
assigns12 pairs to one cell and11 to the other five. Source role is treatment or
anchor; phases are0--19,20--39,40+. Hash-select an eligible root across both seat
games in the pair. Keep private reserves and short supports. Assert cell availability
before freezing; no replacements, extra games, filtering by outcome or best-fit choice.

Probe:12 positions from the separate parent probe_a games, one per game, four
per fit, with role/phase cells allocated by fixed identity rules. Probe scientific
outcomes never select models, state counts or eligibility. Main and probe source
pairs are disjoint and the probe is excluded from the primary estimate.

Rebuild selection independently from sealed source metadata, check source game/
audit/trace/root links, and replay actual moves to each selected state. No new
source games or inner source replays are needed. Bind both full panels, their
source artifacts and sampling metadata in the manifest. data.py RULES and all
selection code are included in the frozen identity.

## Counterfactual arms and independent worlds

Retain the common first three candidates from the source position's original
search stream, preserving candidate nonce, canonical equivalence, concrete action
representatives, ordering and ties. Ask the corresponding sealed anchor, treatment
and replay model for its highest-ranked remaining canonical move. Random fourth
is the identity-hash minimum of eligible canonical keys. When support is below
four, keep the complete original set for all arms.

Union order is common first three followed by anchor, treatment, replay and random
fourths, deduplicated. At most seven candidates. Record full legal scores, model
identities, per-arm ordered support and exact representatives. Two additional
choices use common first-three-only support and the full union; they add no new
continuations. The larger union is a headroom diagnostic, not an equal-budget arm.

Each state uses128 fresh selection worlds and128 fresh assessment worlds. Seeds
are disjoint integer namespaces based on2080000000, stage offset, state index,
block offset and world index, fully checked before freeze. They are disjoint from
saved source string seeds. Couple candidates within each reconstructed observer-
legal world; use the existing serial greedy035 terminal-WDL continuation routine.
Never preserve discovery deck orders in the assessment block.

Select each arm's first maximum over its ordered support using selection outcomes.
Persist its complete selection artifact, scientific identity, outcome sums and
locked choices BEFORE generating assessment outcomes. Assessment judges those
fixed choices and fixed neural proposals; it never selects another choice.

Recorded WDL is loss-1/draw0/win1. Report terminal score=(WDL+1)/2, so losses0,
draws.5 and wins1. Gains are score percentage points, not complete-game win rates.
Each arm score has denominator256 after128 worlds. Main ceiling360,192 recorded
continuations; separate probe ceiling21,504. Actual deduplication may reduce work.

## Descriptive saved-prefix context

At launch, inspect all201 opening pairs and common pre-decision prefixes, including
the first differing decision. Check exact snapshot, legal-input, stream/world and
common-three-action/outcome equality. Match root hashes to source traces/audits.
Report proposal and final-choice disagreement and fixed-proposal outcome differences;
opening and later-prefix summaries remain separate and pair-accounted.

Both seat games use the SAME worlds. They are not independent assessment of a
search-selected maximum. Prefixes emphasize early and agreeing trajectories and
do not represent all later states. The cache description is not an outcome gate
and cannot change the fixed representative panel or main execution.

## Statistics and limits

Primary: treatment-minus-anchor independently assessed selected-move score.
Secondary: treatment-minus-replay, random-minus-anchor, first-three-minus-anchor,
and union-minus-anchor. Report all; only the first is primary. Also describe
fixed-proposal quality, selection/disagreement rates, and fit/phase/source-role
coverage. Small-support rows remain in primary analysis and have zero proposal
effect where all supports coincide. Missing fourth-proposal values are reported
as ineligible, never as zero-valued bad moves.

Equal weight for each of201 source pairs, hence equal weight for the three fixed
fits. Bootstrap67 source-pair units within each fit,50,000 replicates, NumPy
default_rng seed2083000000:50 chunks of1000, ascending fit order, common index
draws across all five contrasts, linear2.5/97.5% quantiles. An independently
implemented indexed-sum reference reconstructs raw assessment endpoints and must
agree with the verifier's separate accumulation and final statistics.

Primary interval entirely above/below zero is labeled a positive/negative
conditional proposal effect; otherwise inconclusive. These labels do not promote
a model, establish a full-game effect, or authorize any follow-up. Subgroup and
secondary intervals are descriptive without multiplicity-adjusted claims.

## Prescribed verification

Apply docs/engineering/verification_policy.md to the mature simulation path.
Check every selected actual source state, model identity, lawful model input and
hidden-information invariance, candidate/group construction, representative and
tie rule, all declared worlds, observer reconstruction, outcome-field validity,
recorded budgets, selection locks and aggregate/assessment arithmetic.

Probe: replay all candidates in all selection and assessment worlds at all12
positions. Main: hash-select8 of128 whole worlds per block at EVERY position and
replay every union candidate. Add full replay at the identity-minimum position in
each fit x source-role x phase cell (18), each fit x privacy cell (12), and each
fit x canonical-support1/2/3 cell (9), deduplicating. Maximum39 full stress roots;
missing strata are reported without adding cases. Privacy categories are neither,
own, opponent and both. Sampling depends only on frozen IDs and lawful metadata.

This diagnostic-specific stress design evaluates every conceptual arm at each
chosen root, while protecting per-fit privacy and small-support coverage. It is
frozen before outcomes. Persist both sample manifests and seals before collection.
The verifier independently rebuilds them and reconciles exact selected identities,
coverage, model/root/selection hashes and all recorded/replayed counts. Targeted
preflight covers independence, ties, hidden information, short supports, corruption,
selection-lock timing, mismatch detection, termination and windowless workers.

No discrepancy is ignored or replaced. An unexplained mismatch stops the study
and preserves all partial evidence; no automatic retry or unbounded audit follows.
Reports distinguish complete structural checks from sampled continuation replay.
Rules, deterministic encoders and neural primitives remain shared trusted dependencies.

## Autonomous execution and operational gate

One hidden pythonw supervisor, eight hidden CPU workers, one deterministic Torch
thread each. Execution is independent of the active chat. Supervisor-scoped system
sleep inhibition permits display sleep and releases on exit. No recurring monitor.

Order: saved-prefix description, complete12-state probe plus its full verification,
outcome-blind timing/disk gate, main201-state collection, prescribed main audit,
independent statistics and final report. There is no scientific gate between these
parts. Failures and operational holds retain evidence and do not retry automatically.

The EIGHT-HOUR execution budget begins at the supervisor launch receipt and
includes cache work, probe, collection, audit and finalization. Preparation and
preflight are additional. Workers check limits at world boundaries; a currently
executing world/trajectory may finish before noticing the limit. Queued jobs cancel
on failure, active jobs preserve partial records. Five-GiB disk reserve throughout.

Forecast main collection from measured probe worker-seconds per recorded
continuation times the360,192 ceiling, divided by eight. Separate measured audit
structural time from replay time; scale structural time by201/12 and replay time
by the sealed main maximum replay count, using probe replay cost per continuation.
Apply35% time allowance and five minutes finalization allowance. Reserve50% storage
allowance plus512MiB. Proceed only if elapsed plus forecast fits eight hours and
projected storage preserves the disk reserve; otherwise hold before main work.

Write a final report even for a scientifically uncertain completed result. No
additional states/worlds, checkpoint sweep, seed selection, training, promotion,
full-game follow-up or scheduled monitoring is authorized by this protocol.
