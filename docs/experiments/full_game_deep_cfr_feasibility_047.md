# Full-Game Neural Deep CFR CPU Feasibility Pilot (`047`)

Frozen before implementing the full-game adapter, history tensorization,
reservoirs, traversal probe, or producing any `047` evidence. This is the
bounded CPU prerequisite authorized by `046`. It performs no neural training,
creates no deployable checkpoint, runs no strength match, and cannot promote a
policy.

## Questions

1. Can a perfect-recall, information-legal full-game Splendor information state
   and variable legal action set be represented and stored with bounded memory?
2. Can standard terminal external-sampling Deep CFR traversals finish on the
   production game at a plausible per-traversal CPU/node budget?
3. If not, is the blocker representation/memory, termination semantics, or the
   exponential traverser-action tree?

## Fixed representation

- Reuse the canonical observer-legal entity and semantic action encoders that
  support the strongest clean entity representation, checkpoint `023`:
  `outputs/canonical_entity_coherent_023/`
  `canonical_entity_action_scorer_coherent_width64_best.pt`, SHA-256
  `29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070`.
- Preserve the `044` perfect-recall contract by adding a chronological public
  history tensor. Each event stores semantic action type, any public card/noble
  identity, public token flows, blind-reserve tier only, actor relative to the
  observer, and normalized turn. It must never store hidden deck order or an
  unrevealed opponent blind-reserve identity.
- A memory row contains canonical state entities, public-history entities,
  semantic legal-action entities, a target vector exactly matching the legal
  action count, acting player, iteration, reach weight, and provenance.
- The smoke may initialize/test the existing `023` state/action representation
  on these tensors. It may not claim that the history branch is trained.

## Fresh schema corpus and reservoirs

- Generate exactly eight fresh loop-safe `023` versus `023` games at engine
  seeds `470000--470007`, constructing new bots for every game.
- Use `max_turns=400`, repetition limit `4`, and no-progress limit `60` for
  operational protection.
- Valid clean terminals are exactly `completed` and an adjudicated
  `forced_pass_deadlock` after two consecutive states whose sole legal action
  is `PASS`. The latter is not a stall. True stalls, repetition/no-progress
  cutoffs, and maximum-turn cutoffs invalidate the schema-corpus gate.
- Encode every acting state before its selected action and retain chronological
  public history. Use uniform finite placeholder vectors solely to exercise the
  advantage/strategy row schemas; label every such row `schema_only` and forbid
  it from training.
- Exercise two deterministic Algorithm-R reservoir memories, capacity `128`
  each and seed `470047`, over the same stream. Repeat from zero and require
  identical contents/hashes. Serialized combined payload must be at most
  `32 MiB`.
- Require at least 400 encoded states, at least 32 each in turns `0--19`,
  `20--49`, and `50+`, legal-action counts in `[1,512]`, finite tensors/targets,
  exact target/action alignment, and successful CPU collation plus a frozen
  `023` forward pass for batches of eight.

## Privacy and semantic invariance

The following are mandatory:

- permuting hidden deck order does not change the uninformed observer's state
  or history tensor;
- after blind reserves of different top cards, the owner's state tensor differs
  while the opponent state and both public histories are identical;
- public history contains no blind-reserved card id until that card is bought
  and publicly revealed;
- permuting visible cards, reserves, nobles, or legal action list order preserves
  the state representation and semantic action rows after descriptor matching;
- no privileged entity type is accepted by a deployable memory row.

## Fixed full-game external traversal probe

- Use the unchanged production `SplendorEnv`, root deck/noble chance samples at
  seeds `470100` and `470101`, and uniform regret-matched strategies.
- Run one terminal external-sampling traversal for each traverser/player at each
  seed: four attempts total. Enumerate every legal traverser action; sample the
  opponent uniformly with deterministic RNG seed `root_seed * 10 + traverser`.
- Cap each attempt at exactly `25,000` visited decision/terminal nodes. Record
  nodes by depth, legal-action distribution, maximum depth, completed terminal
  leaves, forced-deadlock leaves, repeated public-state visits, and elapsed CPU
  time.
- Ordinary engine completion and two-forced-pass deadlock are the only terminal
  utilities. Repetition, no-progress, turn, or node-budget cutoffs are not game
  outcomes and must never emit advantage or strategy targets.
- An attempt is complete only if the entire recursively enumerated traverser
  tree reaches valid terminals before the node cap. Partial subtrees are
  diagnostics and contribute zero training-eligible samples.

## Frozen gates

All are required to authorize neural fitting:

1. Exact `023` checkpoint hash/config load and successful finite CPU forward
   passes on the new state/action batches.
2. All eight schema games have clean valid terminals under the corrected
   forced-deadlock contract and meet count/phase/action bounds.
3. Every privacy, semantic-invariance, target-alignment, finiteness, and
   no-privileged-entity audit passes.
4. Both repeated capacity-128 reservoirs are byte-identical, contain only
   `schema_only` rows, round-trip losslessly, and serialize within `32 MiB`.
5. All four full-game external traversal attempts complete their entire trees
   within `25,000` nodes with zero invalid cutoff or illegal action.
6. Completed traversals alone produce correctly aligned advantage rows at
   traverser information sets and corrected-`046` strategy rows at sampled
   opponent information sets; partial attempts produce exactly zero eligible
   rows.
7. Repeating the first traversal attempt from zero is byte-identical after
   excluding elapsed time.

## Decision

A complete pass authorizes a separately frozen neural-fit stage. At that point
a rented GPU is justified for two width-64 models initialized from compatible
`023` representation weights: per-player advantage prediction and corrected
opponent-node average-strategy prediction. Neural fitting still cannot authorize
gameplay without held-out target, rare-state, legality, and iteration-stability
gates.

If representation/memory passes but terminal external traversal fails, do not
rent a GPU or add a value bootstrap. Standard full-game external Deep CFR is
computationally mismatched to Splendor's horizon; the next branch must
preregister an outcome-sampling/continual-resolving CFR formulation with
unbiased importance weighting and an explicit finite-game/loop contract. If
privacy or representation fails, repair that interface first. Raw legal `035`
remains champion, `044/045` remain failed, `046` remains the tabular reference,
and BGA data is not used in `047`.
