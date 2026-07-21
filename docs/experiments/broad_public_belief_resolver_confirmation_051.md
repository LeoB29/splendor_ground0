# Broad Public-Belief Resolver Confirmation (`051`)

Frozen before implementing the `051` runner, generating any `051` game, or
inspecting any `051` outcome. This CPU-only experiment is the broad gameplay
confirmation authorized by the complete `050` pass. It performs no training,
distillation, checkpoint promotion, self-play collection, or GPU work.

## Question and evidence boundary

`050` established on 100 fresh paired seeds that the exact `049b` public-belief
resolver improves raw legal `035`: 138-62-0, 69.0% score share with CI95
[62.5%,75.0%], and mean final-score margin +3.595. `051` asks whether the same
frozen resolver is a broad full-game teacher rather than a mirror-match
specialist.

Require `outputs/paired_public_belief_resolver_gameplay_050/result.json` to have
SHA-256
`68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195`, report
`passed=true`, and authorize broad confirmation. No `049`, `049b`, or `050`
outcome may select an `051` seed, opponent, parameter, or numerical threshold.

## Frozen candidate

Deploy exactly `FrozenPublicBeliefResolverBot` and the default
`ResolverGameplayConfig` from `050`, using legal-observation checkpoint
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

On every candidate turn, attempt the unchanged discovery resolver: top four
distinct canonical root successors, three observer-relative determinizations,
top four `035` opponent successors shared inside each opponent information set,
one minimizing response, and four matched stochastic-`035` terminal
continuations. Epsilon `0.1`, temperature `0.5`, top-3 sampling, margin scale
`20`, continuation cap 400, loop-fallback cap 64, canonical tie rules, and all
structural fallback reasons remain literal `050`. Every completed search must
consume exactly 192 branches. No routing, opponent adaptation, or budget change
is permitted.

Preserve the `050` namespaces for arbitrary engine seed `s`: belief root index
`1,000,000 + (s-500,000)*1,000 + candidate_seat*400 + decision`, and
continuation seed
`56,000,000 + (s-500,000)*1,000,000,000 + candidate_seat*400,000,000 +`
`decision*1,000,000 + determinization*10,000 + continuation`. Decision index
400 or greater is an integrity failure.

## Frozen broad opponents

Use the established three-opponent broad panel. Construct a fresh candidate and
opponent for every game.

1. `shallow`: gate-`030` information-legal `ShallowSearchBot`, depth 1,
   maximum branching 6, buy/reserve/take caps 4/2/2, seed 23, one
   determinization, determinization seed 29.
2. `016`: `LegalActionScorerBot` using
   `outputs/legal_action_scorer_016_policy_guided_mix/`
   `legal_action_scorer_best.pt`, SHA-256
   `876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056`.
3. `023`: loop-safe `CanonicalEntityActionScorerBot` using
   `outputs/canonical_entity_coherent_023/`
   `canonical_entity_action_scorer_coherent_width64_best.pt`, SHA-256
   `29ef7c13c5553ec88c8bc98f7378c925f0e600e2fcf3d944df5926efed7a070`.

## Fresh paired blocks and blinding

Run exactly 50 paired seeds / 100 games independently against each opponent:

- shallow: `551000--551049`;
- `016`: `561000--561049`;
- `023`: `571000--571049`.

For every seed, play the resolver once from seat 0 and once from seat 1. Use
CPU, ten workers across the combined pending work, PyTorch one thread per
worker, game turn cap 400, repetition limit 4, and no-progress limit 60.

Run all 150 pairs even if an earlier opponent would fail. During execution,
persist only resumable pair records and expose completion counts, errors, and
ETA; do not compute or inspect opponent outcomes, score shares, margins, or
interim gates. Summarize all three opponents only after every pair completes.
Bootstrap each opponent's 50 paired seeds independently 20,000 times using its
block seed start as the fixed bootstrap seed. Do not pool opponents for a
promotion claim.

## Integrity gates

Every condition is required independently in each opponent block:

1. Exact candidate, opponent class/configuration, checkpoint hashes, `050`
   result hash, seed band, and random namespaces match this protocol.
2. Exactly 100 games end through valid `completed` or corrected two-sole-`PASS`
   `forced_pass_deadlock`; timeout, stall, repetition/no-progress/turn cutoff,
   illegal action, exception, or unknown termination is invalid.
3. Every completed resolver search uses exactly 192 continuations, all
   continuations resolve, hard caps are zero, and loop adjudications are at
   most 0.1%.
4. The resolver completes and switches at least one action in early (`0--19`),
   mid (`20--49`), and late (`50+`) phases, with at least eight switches in the
   block.

## Independent strength gates

Every opponent block must independently satisfy all of:

- candidate paired score share at least 55%;
- paired-bootstrap CI95 score-share lower bound strictly above 50%;
- mean paired candidate final-score margin strictly positive;
- candidate score share from each seat at least 50%.

The broad gate passes only if every integrity and strength condition passes for
all three opponents. Aggregate panel statistics are descriptive only and cannot
rescue a failed opponent.

## Decision

A complete pass establishes the frozen resolver as a broad stronger search
teacher and authorizes only a separately preregistered resolver-every-move
teacher-corpus and distillation experiment. That next protocol must specify how
complete root action/value targets are stored for games won and lost, how fresh
collection alternates with supervised updates, and how raw `035` remains the
paired promotion control. `051` itself authorizes no training, corpus
generation, checkpoint promotion, or GPU rental.

Any substantive failure rejects broad deployment of this exact resolver at its
frozen budget. Do not add games, tune an opponent-specific route, alter widths,
or pool opponents after seeing results. Raw legal `035` remains champion
throughout `051`.
