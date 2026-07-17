# Outcome-Blind Plan Persistence Diagnostic (`038`)

Frozen before fresh collection or inspection of future actions. This diagnostic
does not train a model, inspect game outcomes, authorize gameplay, or promote a
checkpoint. It tests whether hierarchical card-directed options have an
observable behavioral basis.

## Corpus and information contract

- Generate 256 fresh games at seeds 390000--390255. Frozen greedy `035` is the
  measured actor against `035`, `016`, the operational shallow bot, and `023`.
- Divide games into game-disjoint discovery and confirmation halves. Each half
  contains exactly 16 games in every opponent/actor-seat cell.
- Store only the actor's public pre-action state and chosen action. Remove true
  deck order and opponent-private reserved-card identities before writing.
- Never store or read winner, final score, final margin, or replay final value.

## Observable plan definition

An anchor is a `TAKE_TOKENS` action with at least two legal take choices. Its
future target is the first card the same actor buys or visibly reserves within
four own turns, provided that card was already visible or self-reserved at the
anchor. Blind-deck reservations and newly revealed cards censor the anchor.
At least one legal take must reduce the target's remaining colored cost.

For every anchor, compare the chosen take with every alternative legal take
from the exact same public state. Gold is applied after colored tokens when
computing remaining cost. The primary statistics are:

- chosen progress minus mean alternative progress;
- chosen progress rank percentile among the legal alternatives;
- progress advantage in episodes containing at least two takes directed toward
  the same later target.

Bootstrap equal-weight per-game means, so overlapping anchors and long games do
not masquerade as independent evidence.

## Frozen gate

Both discovery and confirmation must independently have:

- at least 200 anchors;
- mean progress advantage at least 0.10 and game-cluster CI95 lower bound above
  zero;
- mean progress-rank percentile at least 0.60 and CI95 lower bound at least
  0.55;
- at least 40 multi-take episodes;
- multi-take mean progress advantage at least 0.08 and CI95 lower bound above
  zero.

Across the pooled corpus, every opponent/seat cell must contain at least 30
anchors and have positive mean progress advantage. At least three phases must
contain 50 anchors, with no covered phase at or below zero.

A complete pass authorizes only the design of a separate hierarchical option
representation and predictive probe. Failure closes the proposed hierarchy.
Thresholds, horizon, target definition, and splits may not be changed after
collection.
