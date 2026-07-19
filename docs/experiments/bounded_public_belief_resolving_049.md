# Bounded Public-Belief Resolving Target Diagnostic (`049`)

Frozen before source collection, implementation of the resolver, continuation
evaluation, or inspection of any `049` outcome. This CPU-only diagnostic is the
bounded-subgame branch authorized after `048`. It performs no neural training,
gameplay strength test, policy promotion, or GPU work.

## Distinct question

`005` already showed that a one-shot comparison between `035`'s top two root
actions has reproducible local terminal returns but does not improve full games.
`036--037` rejected an unqualified learned scalar value as a search leaf, and
`047--048` rejected both terminal external enumeration and terminal trajectory
importance weighting for full-game CFR.

`049` does not repeat any of those mechanisms. It asks whether a small Bayesian
public-belief subgame, with an information-set-consistent adversarial opponent
response before the frozen continuation policy resumes, produces stable local
regret targets. Passing authorizes only a separately preregistered resolver
gameplay gate; it cannot authorize neural fitting or promotion by itself.

## Fresh outcome-blind roots

- Freeze raw legal `035` checkpoint
  `outputs/legal_action_scorer_035_legal_observation/`
  `legal_action_attention_scorer_best.pt`, SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Generate exactly 192 fresh deterministic `035` mirror games at engine seeds
  `490000--490191`, constructing new bots for every game. Assign each game in
  advance to one of six actor-seat/phase strata by game index: actor seat 0/1
  crossed with early turns `0--19`, mid `20--49`, and late `50+`, exactly 32
  games per stratum.
- A source game is valid only through ordinary completion or the corrected two-
  sole-`PASS` forced deadlock. Repetition, no-progress, turn-400, stall, or any
  other cutoff invalidates collection.
- Within each assigned game/actor/phase, select the acting state with the
  smallest raw `035` logit gap between its top two distinct canonical public
  successors, without reading the result or future trajectory. Retain the eight
  smallest-gap unique games per stratum: 48 roots, exactly eight per cell.

## Fixed two-ply public-belief subgame

- At each root, retain `035`'s top four distinct canonical public successors,
  including the deployed top action. Fewer than four invalidates that root.
- Draw exactly three observer-relative determinizations with seeds
  `49,000,000 + root_index * 10,003 + determinization_index`. The sampler must
  sort the unseen pool before randomization; permuting the true hidden deck or
  an opponent blind-reserve list cannot change the sampled belief range.
- The same four semantic root actions must be legal in every determinization.
  Apply each root action. At the opponent node retain its top four distinct
  canonical successors under the same frozen `035` scorer.
- Opponent decisions are constrained by their legal information sets. Group
  histories by the opponent-relative canonical public-belief key plus the
  public root action. Histories in one group must expose the same semantic
  response set, and the minimizing response must be shared across all histories
  in that group. This prevents the resolver from conditioning on hidden deck
  order.
- For each `(determination, root action, opponent response)` leaf, evaluate
  eight loop-safe stochastic-`035` terminal continuations. Use epsilon `0.1`,
  temperature `0.5`, top-3 sampling, score margin divided by `20`, maximum 400
  additional turns, and at most 64 loop-fallback actions before score/card-count
  adjudication. Continuation seed is
  `50,000,000 + root_index*1,000,003 + determinization_index*10,007 +
  continuation_index`; the identical continuation seed is matched across every
  root/response alternative.
- Continuations `0--3` form the discovery leaf table and `4--7` form the
  untouched evaluation table. Solve each table independently. For a fixed root
  action and opponent information set, choose the response minimizing mean root-
  player payoff; average those group values over the three equally weighted
  determinizations. The root target is the four resolved action values and their
  regret relative to the table's best action. Ties use canonical representation
  order.

## Integrity and leaf gates

All are required:

1. All 192 source games are valid, every assigned phase exists, and selection
   contains exactly 48 unique games with exact eight-state seat/phase balance.
2. The checkpoint hash/config, root action mapping, determinization invariance,
   opponent information-set grouping/action stability, and all action
   applications pass with zero illegal action or hidden-order dependence.
3. All `48*3*4*4*8 = 18,432` continuation branches resolve; hard caps are zero
   and loop adjudications are at most 0.1%.
4. Across all leaves, discovery/evaluation payoff Pearson correlation is at
   least `0.60`, mean absolute difference at most `0.10`, and sign agreement is
   at least 70% among leaves where both absolute half-means are at least 0.025.
5. The entire first eight-root evaluation repeats byte-identically from zero.

## Resolved-target gates

All are required before any gameplay may be proposed:

1. Discovery/evaluation four-action resolved-value Pearson correlation is at
   least `0.60`, computed across all root/action pairs.
2. Discovery and evaluation choose the same best root action in at least 60% of
   roots.
3. At least 24 roots have discovery and evaluation best-minus-parent effects
   of absolute magnitude at least `0.01`; their signs agree at least 70%.
4. The discovery resolver changes `035`'s root action on at least eight roots,
   and at least 25% of all roots are resolved, defined as same-sign half-effects
   with evaluation absolute effect at least `0.025`.
5. Evaluating the discovery-selected action in the untouched evaluation table,
   its mean gain over `035` has a 20,000 game-cluster bootstrap CI95 lower bound
   strictly above zero.
6. Every seat and phase contains at least one changed root and has mean untouched
   gain no worse than `-0.025`.

## Decision

A full pass establishes a stable local public-belief resolver target and
authorizes a separately preregistered, fresh paired gameplay gate for this exact
resolver. The gameplay gate must precede any target distillation, following the
project's AlphaGo-loop lesson that the teacher must first beat the policy.

If leaf consistency fails, the matched continuation leaf contract is too noisy
at this budget. If leaves pass but resolved targets fail, short Bayesian
resolving adds no reliable decision signal beyond `035`; do not deepen the
horizon, tune top-k, reuse the learned value, or rent a GPU after observing the
result. Any further CFR work would require a new leaf source, potentially
authorized external expert/BGA data. Raw legal `035` remains champion.
