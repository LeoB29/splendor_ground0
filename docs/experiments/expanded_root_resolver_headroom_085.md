# Experiment 085: Expanded-Root Resolver Headroom Audit

**Preregistered:** 2026-08-07, before implementing the `085` runner,
generating any `085` source game, or evaluating any `085` continuation.

## Question

Experiment `084` promoted the exact `050`/`051` public-belief resolver as the
research champion after a 75.875% paired score share against raw `035`. That
resolver nevertheless selects exclusively among `035`'s top four distinct
canonical root successors. The top-four truncation has never been isolated:
all previous belief-breadth experiments varied determinizations while holding
the same four root actions.

`085` asks a local causal headroom question before committing several days to
gameplay:

> Holding determinizations, opponent responses, continuation policy, utility,
> and tie rules fixed, does adding every structurally evaluable public root
> outside `035`'s top four produce a stable, independently evaluated payoff
> gain over the current research champion's selected root?

This is an audit, not a new champion evaluation. It performs no training,
human-data use, gameplay promotion, or GPU work.

## Frozen source sample

Generate exactly 600 fresh raw-`035` self-play games at engine seeds
`850000--850599`, with state snapshots and established limits: 400 turns,
repetition limit 4, and no-progress limit 60. Both seats use checkpoint
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
All source games must terminate through `completed` or
`forced_pass_deadlock`, with zero timeout, stall, or loop fallback. Game winner,
final score, and later actions are never selection inputs or summarized as
strength evidence.

Assign each source seed prospectively to one of six cells using
`(seed - 850000) % 6`: seat 0 early, seat 0 mid, seat 0 late, seat 1 early,
seat 1 mid, seat 1 late. Phases are turns `0--19`, `20--49`, and `50+`.

Within each game, take steps for the assigned seat and phase and order them by
absolute distance from that game's median eligible turn, then by turn. Choose
the first step on which the exact champion four-root subgame is structurally
evaluable under the frozen `085` belief range. Structural evaluability uses no
continuation payoff or game outcome. In increasing seed order, retain the first
50 selected games in each cell: exactly 300 unique states, 100 per phase and
150 per actor seat. If any cell supplies fewer than 50 states, stop invalid
before continuation evaluation. No game can supply more than one state.

## Frozen root sets

At each selected public state, score all legal actions with frozen raw `035`
and deduplicate by the existing canonical successor key.

The **champion root set** is exactly the first four ranked roots. The state is
selectable only if all four reproduce the current resolver's contract across
all determinizations: stable root mapping, nonterminal successor, at least four
distinct `035` opponent successors, and a stable shared opponent action set
inside every opponent public information set.

The **expanded root set** begins with those same four roots in the same order.
Then consider every lower-ranked distinct root in `035` rank order. Exclude
omitted `RESERVE_DECK` roots because their acting-player-private drawn identity
does not define a stable public root alternative. Add every other omitted root
that individually satisfies the same stable mapping, nonterminal-successor,
four-opponent-action, and opponent-information-set consistency checks. An
ineligible omitted root is counted by reason and skipped; it does not invalidate
an otherwise evaluable state. This structural filtering is deterministic,
outcome-blind, and part of the candidate—not a post-result rescue.

Thus expansion may add visible or own-reserved purchases, visible reserves,
token actions, and other public legal alternatives, but never hidden card
identity or a different leaf model. If no omitted root is evaluable, the two
selectors are identical with zero gain on that state.

## Matched discovery and untouched evaluation

Use exactly three observer-relative determinizations, the top four distinct
raw-`035` opponent successors per information set, and eight stochastic-`035`
terminal continuations per leaf. Continuations `0--3` form the discovery table;
continuations `4--7` form the untouched evaluation table. Freeze epsilon 0.1,
temperature 0.5, top-k 3, 400 additional turns, at most 64 loop-fallback
actions, and final-score margin divided by 20.

The belief-root namespace is `850000000 + (engine_seed-850000)*400 + turn`.
The continuation namespace is
`8500000000000000000 + sample_index*1000000000 +`
`determinization*10000 + continuation`. Seeds are deliberately shared across
root and response alternatives for matched variance reduction and are disjoint
from every earlier experiment.

Solve the champion and expanded discovery tables separately with the existing
information-set-consistent minimizing-response resolver and canonical tie rule.
Solve their corresponding evaluation tables independently. The primary row
effect is the expanded discovery-selected root's evaluation value minus the
champion discovery-selected root's evaluation value, both read from the same
untouched expanded evaluation table. A changed state means the expanded
discovery selector chose a root outside the champion four.

Use ten CPU state workers, each loading frozen `035` once. Write progress after
every completed state and permit only identity-matched crash resumption. Repeat
the complete first selected-state evaluation from zero and require a
byte-identical deterministic projection.

## Endpoints, effect size, and power

The primary endpoint is mean state-cluster evaluation gain. Compute a
20,000-resample state-cluster bootstrap CI95 with seed `850085`.

The minimum practically useful effect is `+0.05` scaled payoff, approximately
one final-score point. All signal gates are required:

1. mean evaluation gain is at least `+0.05`;
2. its bootstrap CI95 lower bound is strictly above zero; and
3. expanded-root selection changes at least 15% of the 300 states, with at least
   one change in every phase.

The declared design alternative is `+0.075`. With 300 independent game-state
clusters and paired-effect SD 0.30, standard error is about 0.0173: the +0.05
point threshold gives about 92.6% power, while the CI-lower-above-zero condition
is weaker. Even at SD 0.40, power remains about 86%. Exactly +0.05 is the
practical decision boundary and therefore has 50% probability of clearing the
point threshold; it is not the powered alternative.

Phase effects are collapse guardrails, not three additional sign tests. For
each 100-state phase, bootstrap its mean gain separately; fail the guardrail
only if that phase's CI95 upper bound is at or below `-0.05`, establishing a
material phase-specific regression.

## Integrity and reliability gates

All are required:

1. Exact source seeds, clean source integrity, exact 50-state balance in all six
   cells, 300 unique games/states, frozen checkpoint/configuration, and no human
   path or hash.
2. Every champion root and every retained expanded root satisfies the frozen
   public-belief structural checks; excluded roots are fully accounted by
   reason; no selected or evaluated action is illegal.
3. Exact variable branch accounting:
   `retained_roots * 3 * 4 * 8` continuations per state. Every continuation
   resolves, hard caps are zero, rollout fallback triggers are zero, and loop
   adjudications are at most 0.1%.
4. Across all retained leaves, discovery/evaluation payoff Pearson correlation
   is at least 0.60, mean absolute half-difference at most 0.10, and sign
   agreement is at least 70% where both absolute half-means are at least 0.025.
5. Across all root action values, discovery/evaluation Pearson correlation is
   at least 0.60 and best-root agreement is at least 60% for both champion and
   expanded tables.
6. Hidden-deck order reversal leaves the public key, root construction, and
   deterministic belief range invariant on the first selected state.
7. The first-state from-zero repeat matches byte-for-byte after excluding timing.

## Cost and stopping

Expected evidence cost is roughly 200,000--350,000 terminal continuations and
1--3 wall hours on ten CPU workers. The fixed operational ceiling is six wall
hours after selection. Exceeding it stops invalid without inspecting or
summarizing payoff outcomes; no reduced root set, sample, or continuation count
is authorized.

## Frozen decision

- Integrity, reliability, privacy, determinism, or operational failure:
  `expanded_root_085_invalid_no_headroom_decision`.
- Valid audit passing every signal and phase guardrail:
  `expanded_root_085_headroom_pass_authorize_gameplay_preregistration`.
- Valid audit with change dose below 15%:
  `expanded_root_085_low_dose_close_exact_expansion`.
- Any other valid audit:
  `expanded_root_085_no_reliable_headroom_close_exact_expansion`.

A pass authorizes only a separately preregistered 200-pair gameplay comparison
of this exact expanded-root resolver against the current resolver research
champion. That later panel must declare a 56.5% expected share, retain at least
15% gameplay action-change dose, and use the established 55%/CI-lower-above-50%
promotion standard. Expected gameplay cost is 40--60 wall hours on ten CPUs.

No `085` outcome promotes a bot, changes the fast champion, authorizes training
or corpus generation, reopens `081--083`, or permits threshold/root-set tuning.
Failure closes only this exact public-root expansion; it does not prove that a
different continuation policy or strategically independent candidate generator
has no value.

## Evidence

Write selection, resumable progress, and final result under
`outputs/expanded_root_resolver_headroom_085`. Record protocol,
implementation, test, checkpoint, selection, and final-result SHA-256 hashes.
