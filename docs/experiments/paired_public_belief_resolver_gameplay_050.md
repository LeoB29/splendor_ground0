# Paired Public-Belief Resolver Gameplay Gate (`050`)

Frozen before implementing the gameplay wrapper, playing any `050` game, or
inspecting any `050` outcome. This CPU-only experiment is the first full-game
test authorized by the complete `049b` pass. It performs no training,
distillation, checkpoint promotion, broad-opponent test, or GPU work.

## Question and frozen candidate

`049` and fresh replication `049b` established that the bounded public-belief
resolver produces stable, positive local action targets on uncertain states.
`050` asks the separate causal question: does deploying that resolver improve
complete games against raw legal `035`?

The candidate wraps frozen checkpoint
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
The opponent is a fresh raw `035` bot using the same checkpoint. Construct new
candidate and opponent bots for every game.

On every candidate turn:

1. Rank distinct canonical public successors with raw `035`. The deployed
   parent is the first successor.
2. Attempt the exact `049b` resolver: top four root successors, three
   observer-relative determinizations, top four `035` opponent successors, and
   one minimizing response shared inside each opponent information set.
3. Evaluate every leaf with exactly four stochastic-`035` terminal
   continuations: the frozen discovery half that selected actions in
   `049`/`049b`. Freeze epsilon `0.1`, temperature `0.5`, top-3 sampling, final
   score margin divided by `20`, maximum 400 additional turns, and at most 64
   loop-fallback actions. Using the untouched validation half during gameplay
   is forbidden.
4. Choose the discovery table's best resolved root action, with the frozen
   canonical tie rule. A choice differing from the raw parent is a switch.

The resolver is attempted on every turn, with no phase, logit-gap, outcome, or
resource-dependent routing. Fall back to raw `035` for that turn only when the
state has fewer than four distinct canonical root successors, a retained root
action ends before an opponent response, an opponent node has fewer than four
distinct successors, root mappings change across determinizations, or an
opponent information set exposes unstable response actions. All fallbacks are
counted by reason. An unexpected exception, missing semantic action, illegal
chosen action, or nonfinite leaf is an integrity failure rather than a fallback.

## Frozen random namespaces and budgets

- Valid engine terminations are exactly ordinary `completed` and corrected
  two-sole-`PASS` `forced_pass_deadlock`. Repetition limit is 4, no-progress
  limit 60, and game turn limit 400; every cutoff or stall is invalid.
- For engine seed `s`, candidate seat `p`, and zero-based candidate decision
  `d`, the deterministic belief-range root index is
  `1,000,000 + (s-500,000)*1,000 + p*400 + d`. The existing sorted unseen-pool
  belief sampler and its `49,000,000 + root_index*10,003 + det` formula remain
  unchanged. Decision counts above 399 are an integrity failure.
- Terminal continuation seed is
  `56,000,000 + (s-500,000)*1,000,000,000 + p*400,000,000 +`
  `d*1,000,000 + det*10,000 + continuation`. It is identical across root and
  opponent-response alternatives and distinct across games, seats, decisions,
  determinizations, and the four continuation indices.
- Every completed resolve therefore consumes exactly
  `3*4*4*4 = 192` continuation branches. CPU worker count is 10. PyTorch uses
  one thread inside each worker. No adaptive budget or early strength stopping
  is allowed.

## Deterministic integrity smoke

Play engine seed `500000` twice from each candidate seat: one paired run and a
from-zero repeated paired run. The repeat must match exactly after excluding
wall-clock timing. Neither smoke outcome contributes to strength.

Require all four games to have valid clean terminations; zero timeout, stall,
illegal action, unexpected error, or hard-capped continuation; exact branch
accounting; loop-adjudicated continuations at most 0.1%; and byte-identical
projected games plus resolver telemetry between the original and repeat. The
direct block runs only after this complete smoke pass. No smoke strength
criterion exists.

## Fresh paired strength gate

Run exactly 100 fresh paired seeds `500100--500199`, 200 games total. For every
seed, play the candidate once from seat 0 and once from seat 1 against raw `035`.
Do not inspect outcomes, tune, replace seeds, or stop early. Bootstrap the 100
paired units 20,000 times.

All conditions are required:

1. Frozen checkpoint/configuration and resolver contract pass.
2. Exactly 200 games end through valid clean terminations, with zero timeout,
   stall, repetition/no-progress/turn cutoff, illegal action, or unexpected
   resolver failure.
3. Every completed resolve uses exactly 192 continuations; all continuations
   resolve, hard caps are zero, and loop adjudications are at most 0.1%.
4. The resolver completes at least one search and switches at least one action
   in each of early (`0--19`), mid (`20--49`), and late (`50+`) phases, with at
   least eight switches overall.
5. Candidate paired score share is at least 55% and its paired-bootstrap CI95
   lower bound is strictly above 50%.
6. Mean paired candidate final-score margin is strictly positive.
7. Candidate score share from each seat is at least 50%, preventing an
   aggregate pass caused by collapse in one seat.

## Decision

A complete pass establishes the resolver as a stronger full-game teacher than
raw `035` and authorizes only separately preregistered broad strong-opponent
confirmation. It does not yet authorize self-play generation or distillation.
Any direct strength failure rejects deployment at this frozen budget; do not
add games, tune widths, route selectively, deepen search, or reuse outcomes.
Operational integrity failure may be diagnosed, but cannot be called a strength
pass. Raw legal `035` remains champion throughout `050`. No GPU is authorized.

## Result (2026-07-21)

Pass every preregistered gate. Protocol commit `a3e2aee` and implementation
commit `0588f4e` preceded all evidence; the full suite passed 446/446.

The smoke passed with four clean games and a byte-identical from-zero repeat.
The direct block then completed all 100 paired seeds / 200 games through
ordinary `completed`, with zero invalid termination, timeout, stall, hard cap,
or model loop fallback. All 1,098,240 continuation branches resolved. Six
branches used frozen loop-safe adjudication, a fraction of 0.000546%, well below
the 0.1% limit.

Resolver `035` went 138-62-0 against raw `035`: 69.0% score share, paired
bootstrap CI95 [62.5%,75.0%], and mean final-score margin +3.595 with CI95
[+2.70,+4.48]. Score share was 73% from seat 0 and 65% from seat 1. The
resolver completed 5,720 searches, switched 3,868 actions, and was active in
every phase. All frozen gates passed.

Decision: authorize only a separately preregistered broad resolver confirmation.
No training, self-play, distillation, promotion, or GPU rental is authorized by
`050` alone. Evidence is under
`outputs/paired_public_belief_resolver_gameplay_050`; `result.json` SHA-256 is
`68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195`.
