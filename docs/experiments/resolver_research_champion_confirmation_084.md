# Experiment 084: Resolver Research-Champion Confirmation

**Preregistered:** 2026-08-06, before `084` implementation, smoke outcomes,
or strength outcomes.

## Question and governance correction

Experiments `050` and `051` established that the frozen public-belief resolver
is much stronger than raw legal checkpoint `035`: it scored 69.0% in the
direct 200-game comparison and 74.5%--79.0% across three broader opponents.
Those protocols authorized the resolver as a teacher but did not authorize a
champion promotion. Consequently, later bookkeeping continued to call `035`
the research champion even though it was no longer the strongest demonstrated
agent.

`084` asks one narrow prospective question: on a larger fresh paired block,
does the exact frozen resolver again clear the existing 55% promotion standard
against raw `035` strongly enough to become the research champion? This is a
governance confirmation, not a search-tuning experiment. The fast champion is
not at stake.

## Frozen candidate and comparator

The candidate is the exact serial public-belief resolver used by `050`/`051`,
wrapping frozen checkpoint
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
The comparator is raw `035` using the same checkpoint. Construct new candidate
and comparator bots for every game.

The resolver contract is unchanged: three observer-relative
determinizations, four root successors, four opponent successors, and four
stochastic-`035` discovery continuations for every retained leaf. Each
completed search consumes exactly 192 continuations. Continuation epsilon is
0.1, temperature 0.5, top-k 3, terminal continuation limit 400, loop fallback
limit 64, and score-margin scale 20. The resolver is attempted on every turn
and uses the exact structural fallbacks and canonical tie rule from `050`.
No human replay, outcome-conditioned routing, parallel within-decision path,
parameter change, tuning, training, or GPU work is permitted.

Candidate provenance requires the untouched passing evidence:

- `outputs/paired_public_belief_resolver_gameplay_050/result.json`, SHA-256
  `68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195`;
- `outputs/broad_public_belief_resolver_confirmation_051/result.json`, SHA-256
  `1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4`.

## Fresh namespaces and deterministic smoke

Use engine seed `840000` for a deterministic integrity smoke. Play one paired
run and an independent from-zero repeat, each containing the candidate once
from each seat. Excluding wall-clock timing, the two projected pairs must be
byte-identical. All four games must terminate cleanly, all continuations must
resolve with exact branch accounting, hard caps must be zero, and the loop-
adjudicated continuation fraction must be at most 0.1%. Strength is not
inspected or gated in the smoke.

For all `084` games, freeze belief root-index start `840000000` and terminal
continuation seed start `8400000000000000000`. These namespaces are distinct
from `050`/`051` and all later resolver audits. Retain the `050` formulas using
the `084` smoke seed as the engine-seed origin. CPU worker count is 10 and
PyTorch uses one thread in each worker.

The strength block is forbidden unless the complete smoke passes.

## Fixed fresh paired strength block

Run exactly 200 paired engine seeds `840100--840299`, producing 400 games.
For each seed the resolver plays once from seat 0 and once from seat 1 against
raw `035`. Treat each seed pair as the independent sampling unit. Do not
inspect interim outcomes, replace seeds, tune, or stop early. Progress may be
written only for crash recovery. After all pairs finish, compute the established
paired summaries and a 20,000-sample paired bootstrap with seed `840100`.

The primary endpoint is resolver paired score share. Mean paired final-score
margin and seat-specific score shares are mandatory secondary safeguards.
Resolver completed-search and switch counts are the intervention dose.

All gates are conjunctive:

1. The checkpoint, resolver contract, protocol, and prerequisite hashes match.
2. Exactly 400 strength games terminate through ordinary `completed` or the
   corrected two-sole-`PASS` `forced_pass_deadlock`, with zero timeout, stall,
   invalid cutoff, illegal action, or unexpected resolver failure.
3. Every completed search consumes exactly 192 continuations; every
   continuation resolves; hard caps are zero; and loop adjudications are at
   most 0.1%.
4. The resolver completes at least one search and switches at least one action
   in every phase (early `0--19`, mid `20--49`, late `50+`). Its total switch
   rate among candidate decisions must be at least 15%, establishing material
   intervention.
5. Resolver paired score share is at least 55% and its paired-bootstrap 95%
   confidence-interval lower bound is strictly above 50%.
6. Mean paired resolver final-score margin is strictly positive.
7. Resolver score share from each seat is at least 50%.

## Effect size, power, and compute

The conservative design alternative is 65% score share, below the prior 69%.
The relevant minimum effect is +10 percentage points over parity because the
promotion gate is 55%. With 400 games / 200 paired units, a conservative
independent-game normal approximation gives more than 99.9% probability of
clearing both strength gates at a true 65% share. Even at a true 58% share it
gives roughly 88% power; the fixed paired bootstrap supplies the reported
interval rather than this planning approximation.

Based on `050`, expect about 2.2 million continuation branches. The preregistered
budget is one complete 200-pair run on 10 CPU workers, estimated at 12--24 wall
hours with high uncertainty. There is no retry budget except resuming the same
identity-matched progress artifact after interruption.

## Fixed decision and falsification

If every gate passes, designate the frozen public-belief resolver as the
**research champion**. Frozen raw `035` remains the **fast champion**. Future
research-champion candidates must ultimately beat the resolver; clearing `035`
alone can establish only fast-policy progress.

A valid strength block with score share below 55%, bootstrap lower bound at or
below 50%, nonpositive margin, seat collapse, or inadequate switch dose
falsifies promotion and leaves the research-champion registry unchanged. An
operational-integrity failure makes `084` invalid and authorizes diagnosis only,
not a strength claim or altered rerun. No threshold relaxation, extra games,
seed pooling with `050`, or post-hoc pass is allowed.

No training, corpus generation, architecture choice, or target-selector
reopening is authorized by either outcome.
