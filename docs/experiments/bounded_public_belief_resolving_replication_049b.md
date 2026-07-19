# Eligibility-Safe Bounded Public-Belief Resolving Replication (`049b`)

Frozen before implementation, source collection, continuation evaluation, or
inspection of any `049b` outcome. This CPU-only experiment is a fresh replication
of `049`; it changes only the source-eligibility contract and random namespaces.
It performs no gameplay strength test, neural training, promotion, or GPU work.

## Purpose and independence

`049` passed every resolver, leaf, and target-quality gate, but formally failed
because one of 192 games ended normally before its preassigned actor could act
in its preassigned late phase. The balanced 48-root sample was still filled, but
the frozen source gate required every attempted game to contain such a state.

`049b` removes that assumption prospectively. It collects a fixed number of
valid candidate-bearing games from disjoint, preallocated seed bands. It does
not reuse any `049` source state, outcome, root, or continuation. A full pass
replicates the local resolver result and authorizes only a separately
preregistered paired resolver-versus-`035` gameplay gate.

## Frozen checkpoint and eligibility-safe source bands

- Freeze raw legal `035` checkpoint
  `outputs/legal_action_scorer_035_legal_observation/`
  `legal_action_attention_scorer_best.pt`, SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Preallocate these nonoverlapping inclusive engine-seed bands, scanned in
  ascending order and never reassigned between cells:
  - seat 0 / early `491000--491999`;
  - seat 0 / mid `492000--492999`;
  - seat 0 / late `493000--493999`;
  - seat 1 / early `494000--494999`;
  - seat 1 / mid `495000--495999`;
  - seat 1 / late `496000--496999`.
- In each band, construct fresh `035` bots for every game and scan until exactly
  32 games are admitted for that cell. A game is admitted only if it terminates
  by ordinary completion or corrected two-sole-`PASS` forced deadlock and
  contains at least one state where the assigned seat acts in the assigned
  phase with at least four distinct canonical public successors.
- Early is turns `0--19`, mid is `20--49`, and late is `50+`. At every eligible
  state, record the raw `035` top-two logit gap. Retain the minimum-gap state per
  admitted game, breaking ties by earlier turn. Selection may use no terminal
  result, future action, or future state.
- A normally completed game that lacks an assigned-phase candidate is recorded
  as skipped and the scanner advances. Repetition, no-progress, turn-400,
  stall, or any other cutoff is recorded as invalid and skipped. Reaching a
  band's end before 32 admissions fails collection; no seed may spill into a
  different band.
- From the 32 admitted games in each cell, examine candidates in ascending
  `(logit gap, seed)` order and retain the first eight that satisfy the frozen
  `049` subgame structural contract. The result must contain 48 unique fresh
  games, exactly eight per seat/phase cell.

## Frozen resolver and leaf contract

Everything below is identical to `049` except the fresh continuation namespace.

- Retain `035`'s top four distinct canonical root successors, including its
  deployed top action. Use exactly three information-legal determinizations,
  the same sorted-unseen-pool sampler, and the same deterministic belief formula
  from `049`. Fresh roots make these belief samples disjoint in state content.
- Apply each root action and retain the opponent's top four distinct canonical
  successors. Group opponent histories by the same opponent-relative canonical
  public-belief key plus public root action; require a shared response set and
  one shared minimizing response per information set.
- Evaluate exactly eight loop-safe stochastic-`035` terminal continuations for
  every `(determination, root action, opponent response)` leaf. Freeze epsilon
  `0.1`, temperature `0.5`, top-3 sampling, final score margin divided by `20`,
  maximum 400 additional turns, and at most 64 loop-fallback actions before the
  existing score/card-count adjudication.
- Use the fresh continuation seed formula
  `52,000,000 + root_index*1,000,003 + determinization_index*10,007 +`
  `continuation_index`, matched across every root/response alternative.
- Continuations `0--3` are discovery and `4--7` untouched evaluation. Solve
  both tables independently exactly as in `049`, including canonical tie breaks.
- Keep 10 CPU workers and 20,000 game-cluster bootstrap samples. Repeat the
  entire first eight-root evaluation from zero and require byte identity.

## Frozen gates

All gates are required and all numerical thresholds are copied from `049`.

1. Source collection completes in every band with exactly 32 admitted valid
   candidate-bearing games per cell, no admitted invalid game, and selection is
   48 unique games with exactly eight roots per cell.
2. Checkpoint hash/configuration, root mappings, determinization invariance,
   opponent information-set grouping/action stability, and all applications
   pass with zero illegal action or hidden-order dependence.
3. All `48*3*4*4*8 = 18,432` continuations resolve, hard caps are zero, and loop
   adjudications are at most 0.1%.
4. Leaf discovery/evaluation Pearson correlation is at least `0.60`, MAE at
   most `0.10`, and sign agreement at least 70% where both half-means have
   absolute magnitude at least `0.025`.
5. First-eight repetition is byte-identical.
6. Four-action resolved-value Pearson correlation is at least `0.60` and best
   actions agree in at least 60% of roots.
7. At least 24 roots have both best-minus-parent half-effects of absolute
   magnitude at least `0.01`, with at least 70% sign agreement.
8. At least eight roots change the `035` action and at least 25% of all roots
   resolve with same-sign effects and evaluation absolute effect at least
   `0.025`.
9. The 20,000-sample game-cluster bootstrap CI95 lower bound on untouched gain
   of the discovery-selected action over `035` is strictly above zero.
10. Every seat and phase has at least one changed root and mean untouched gain
    no worse than `-0.025`.

## Decision

A complete pass authorizes a new, separately preregistered paired gameplay gate
for this exact frozen resolver against raw `035`. It does not authorize training
or promotion. A failure of any substantive replicated gate rejects the bounded
resolver at this budget; do not tune after seeing the result. A source-band
exhaustion is an operational failure and still cannot be declared a pass.

Raw legal `035` remains champion. No GPU is needed or authorized.
