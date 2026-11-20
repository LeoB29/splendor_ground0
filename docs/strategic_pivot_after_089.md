# Strategic Pivot After `089`

Recorded 2026-08-09 by agreement between the project owner and Codex.

## Decision

The programme spent too much time optimizing policy `035` and increasingly
lost sight of the broader objective: building a system that can discover and
learn stronger Splendor strategy through an AlphaGo-style search/self-play
loop.

This does not erase the value of the resolver programme. It established that
public-belief search can improve substantially on raw `035`, produced the
current research champion (`050`/`051`, confirmed by `084`), found real
expanded-root headroom (`085`/`086`), and showed that optional token-taking
actions contain most of that headroom (`088`/`089`). Those are useful results.
The strategic mistake was allowing that successful local line of work to
become the programme itself.

The post-hoc owner-authorized `087` panel may finish unchanged and will be
reported under its existing evidential limitations. Regardless of its result,
the default next direction is **not** another resolver-root, rollout, cap,
action-family, or `035`-distillation refinement.

## What is being left behind

The current resolver is a strong local policy-improvement operator around
`035`:

- `035` ranks or supplies the actions considered;
- `035` selects opponent responses and plays the continuation policy;
- each candidate is evaluated with many independent terminal rollouts;
- successive experiments improve the candidate set, rollout use, targets, or
  learners around that fixed strategic centre;
- most iterations are treated as confirmatory experiments with promotion-grade
  gates.

This can find better choices inside and near `035`'s strategic mode, but it is
expensive, inherits its teacher's blind spots, and does not create the repeated
search-to-network-to-self-play improvement cycle that motivated the project.

## Pivoted direction

The new target is a minimal, genuine policy/value iteration loop:

1. Train a policy/value network from scratch on the acting player's legally
   available observation and a structured set of legal action entities. `035`
   is not its encoder, prior, continuation policy, or training target.
2. Use one selective, shared search tree per decision. The policy guides which
   actions to explore and the value network evaluates leaves; search does not
   run thousands of independent `035`-played terminal games for every move.
3. Handle Splendor's large token-action branching with policy-guided expansion
   or progressive widening rather than a static top-four prefix or a fixed
   action-family filter.
4. Generate self-play examples containing the information legally available to
   the acting player, the search visit distribution over legal actions, and the
   final outcome or score margin.
5. Train the next policy/value network on those examples, then repeat with the
   new network inside search.

The first engineering proof should use a perfect-information debug mode so the
search, targets, batching, and learning loop can be validated without mixing in
observation or chance-handling errors. The debug mode is scaffolding, not the
final research claim.

## Hidden-information scope decision

The final architecture should remain close to AlphaZero rather than allowing
Splendor's very narrow hidden-information mechanism to dictate a radically
different design.

- An unrevealed future deck card is unknown to both players. Treat market
  replenishment as a stochastic chance transition over possible remaining
  cards, not as strategically private information. A fixed simulator deck
  order must never be exposed to the acting policy or search.
- `RESERVE_DECK` is the one genuine asymmetric-information action: its owner
  knows the card while the opponent knows only its public provenance/tier.
- The project owner expects `RESERVE_DECK` to be a very rare and marginal
  strong-play action. It must remain legal so the agent can discover a useful
  application, but it should be handled as a localized observation/search
  exception rather than as the organizing principle of the architecture.
- The intended final system is therefore stochastic AlphaZero: the usual
  policy/value network, PUCT-style shared tree, visit targets, outcomes, and
  iterative self-play, augmented with chance handling for refills and
  information-legal player-relative observations for blind reserves. If blind
  reserves prove consequential, lightweight root sampling or information-set
  handling can be added specifically where needed.

This decision forbids two shortcuts: exposing the hidden shuffled deck order
to search, and exposing an opponent's blind-reserved card identity. It does not
require a wholesale CFR-style or deeply belief-centric replacement for the
AlphaZero loop.

## Relationship to earlier AlphaZero-like work

The repository already contains many relevant components and bounded partial
attempts: information-legal PUCT, batched inference, policy priors, heuristic
and learned value leaves, search self-play corpora, exported root-visit
distributions, and policy/value training. The closest attempt was the
generation-1 root-visit pilot: 300 clipped-PUCT035 self-play games produced
visit targets, and a student initialized from `035` was trained on those games
plus broad `035` behaviour data. It did not beat `035` or its search teacher.

That was close in components but was not the defining AlphaZero process. Search
remained centred on frozen `035`; the student was warm-started from and anchored
to `035`; the value learner was not accepted into a continuing live loop; and
the trained student was never installed as the new search policy/value model to
generate a stronger second search generation. No sequence of the form
`network N -> search -> self-play -> network N+1 -> search with N+1` has run,
and no from-scratch joint policy/value generation has completed that cycle.

Past PUCT, leaf-value, and visit-distillation failures remain valid negative
evidence about those exact implementations. They reduce confidence and warn
against repeating a frozen-`035` search engine, weak terminal-outcome values,
or anchored visit imitation. They do not constitute a test of the integrated
from-scratch iterative system now proposed.

## How progress will be judged

Early iterations belong in an **engineering sandbox**, separate from the locked
promotion/evidence lane. They require reproducible seeds, logs, tests, and held-
out development games, but not a new 200-pair preregistered panel for every
implementation choice.

The minimal proof of a working loop is:

1. search improves on its own raw network;
2. a newly trained network learns a material part of that search improvement;
3. the new raw network beats its parent;
4. search still improves the new network, leaving room for another generation.

Only after that mechanism works reproducibly should a candidate be frozen and
enter formal confirmation against the fast and research champions on untouched
seeds.

## Boundaries and retained evidence

- Raw `035` remains the fast champion and a useful external baseline, not an
  architectural dependency.
- `050`/`051` remains the research champion until a valid promotion changes it.
- The tested historical PUCT variants remain negative results for those exact
  designs. They do not establish that an integrated, from-scratch policy/value
  self-play loop has been tested, because those experiments lacked the full
  mutually improving cycle described above.
- Existing human games remain evaluation and diagnostic evidence, not training
  data.
- The sandbox/promotion separation relaxes iteration overhead, not integrity:
  exploratory results must be labelled exploratory and may not be presented as
  formal promotion evidence.

The immediate next deliverable is a small architecture and milestone design
for this loop, including the perfect-information debug interface, structured
action representation, search budget, replay format, training schedule, and
the criteria for advancing to public-belief search.
