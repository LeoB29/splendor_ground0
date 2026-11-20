# Stochastic Refill Versus Blind-Reserve Architecture Correction

Date: 2026-09-04  
Status: active programme-level correction; outcome-blind with respect to experiment 134

## Experiment 136 update (2026-09-04)

The replay-only blind-reserve/refill-coverage audit strengthens and narrows this
correction. Across 63,882 reconstructed generation-8--11 positions, the public
entity encoding always retained enough information to reconstruct the
observer's possible refill-card set. At 1,024 simulations, priority refill
edges averaged 98.635% expected identity coverage and visit winners 99.951%, so
there is no broad high-budget refill identity-coverage failure.

At 256 simulations, priority alternatives averaged only 79.786--84.117%
coverage, with 32--42% below 80%, while eventual winners remained well covered.
This supports a bounded matched-compute stratified-refill search test as an
efficiency experiment, not an architectural commitment.

Generation-11 blind reserves were raw-network top on 51.953% of states where
legal but visit top on 43.003%, with a negative mean Q gap versus non-reserves.
The learned policy/self-play ecosystem, not MCTS, is the main source of the
unusual opening preference. Keep blind-reserve handling information-legal, but
do not promote it to the central architecture without a targeted exploit test.

Full result:
`docs/engineering/alphazero_blind_reserve_refill_coverage_audit_136_result.md`.

## Decision

Do **not** let blind reserves from a deck determine the programme's primary
learning and search architecture.

The main uncertainty in ordinary Splendor play is the public refill mechanism:
after a visible card is bought or reserved, neither player knows which card will
replace it, and the replacement is then revealed to both players. This is a
chance event, not strategically asymmetric private information.

The primary architecture should therefore first be tested as:

**AlphaZero-style policy/value learning + stochastic, chance-aware MCTS + the
exact Splendor simulator.**

Blind deck-reserves remain part of the rules and must remain information-legal,
but they should initially be handled as a bounded secondary case. Escalate to a
full public-belief/CFR architecture only if targeted evidence shows that this
rare private-information mechanism materially limits strength or creates a
practical exploit.

This corrects the earlier recommendation in
`docs/strategy/whole_programme_direction_review_during_134.md`, which promoted
a public-belief GT-CFR mechanics bridge directly to the next experiment. That
review remains useful for its audit of the project and current resolver, but its
architecture choice and proposed experiment 135 are superseded by this note.

## The distinction that must be preserved

### Refill uncertainty

The ordinary sequence is:

`public state -> player action -> random refill -> new public state`

- Neither player knows the future replacement.
- Neither player has privileged knowledge of it.
- Once drawn, the card is public.
- Search should average over the possible replacements according to the
  remaining deck.

This is a stochastic game transition. It calls for chance nodes, afterstates,
or statistically correct chance sampling. It does not by itself require an
information-set equilibrium solver.

### Blind deck-reserve uncertainty

When a player reserves directly from a tier deck:

- that player sees the reserved card;
- the opponent sees its tier and source but not its identity;
- later actions can reveal information about the card.

This is genuine private information. In principle it can require beliefs about
the hidden identity and strategically consistent play across information sets.
In practice, architectural importance must be established from its frequency,
strength effect, and exploitability—not inferred merely from its presence in
the rules.

## Existing evidence about materiality

The current project evidence supports treating blind reserves as secondary.

1. Experiment 071 removed blind deck-reserve actions from the candidate. The
   mask changed 4,641 of 63,424 decisions, or 7.317%, but the candidate's paired
   score share across 1,000 pairs / 2,000 games was 50.575%, CI95
   [48.5%, 52.65%]. The decision was
   `deck_reserve_mask_neutral_at_measured_dose`. Removing the agent's ability to
   blind-reserve was therefore strength-neutral at the measured precision.
2. In the ten-game human-versus-resolver calibration, the human made only two
   blind deck-reserves in 278 decisions, or 0.72%. The resolver made 27 in 278,
   or 9.71%, including 24 in the opening. The resolver's higher frequency may
   be a weakness inherited from its policy family rather than an essential
   strategic mode. The small human panel is descriptive, not a definitive
   population estimate.
3. Other experimental bots used blind deck-reserves around 6--10% of decisions.
   This proves that the current bot family encounters the mechanism; it does
   not prove the action is useful.
4. Experiment 130's large value uncertainty must not be generalized to ordinary
   public states as evidence for a belief-centric architecture: 23 of its 24
   selected states contained an opponent blind reserve, leaving only one state
   in the absent subgroup.

Important limitation: experiment 071 shows that taking away the candidate's
own blind-reserve option was neutral against 035. It does not prove that an
adversary can never exploit weak reasoning after the adversary blind-reserves.
That question should receive a targeted robustness test later.

## Consequences for AlphaZero

Canonical AlphaZero was demonstrated in deterministic perfect-information
games, but MCTS can be extended to stochastic perfect-information games. With
Splendor's exact simulator, there is no need to learn the refill dynamics.

For a buy or visible-reserve action, search should conceptually separate:

1. the deterministic result of the chosen action;
2. the random replacement card drawn from the correct remaining-tier
   distribution;
3. the opponent's decision after observing that card.

The project already implements a basic chance-sampling approximation. Each
AlphaZero simulation resamples an information-legal unseen deck order, applies
the action in the exact engine, and keys later nodes by the resulting visible
state. An action's backed-up value therefore averages sampled refill outcomes.

The open question is not whether stochastic AlphaZero is logically possible.
It is whether the current implementation covers important refill outcomes
efficiently and learns sufficiently accurate expected values at affordable
compute.

## Representation implications

In the absence of an opponent blind reserve, the current public card identities
are sufficient to infer the unordered remaining card pool:

- market card identities are public;
- purchased card identities are public and included in the canonical entity
  representation;
- visible-reserved cards and the observing player's own reserves are known;
- deck counts are known.

The network may nevertheless benefit from a more direct legal encoding of the
remaining-card distribution. Candidate representations include one entity per
possible unseen card with a probability or availability weight, or compact
tier-wise statistics. This would make refill risk easier to learn without
exposing the actual deck order.

Public action history or a recurrent belief model is not required merely to
predict refills when the current public state already identifies all publicly
removed cards. History becomes relevant mainly when inferring an opponent's
blind-reserved identity from later behavior.

## Revised immediate experimental crux

Do not automatically start training, GT-CFR, or a new long gameplay run after
experiment 134.

The next proposed experiment should be a small, no-training, chance-search
mechanism audit on positions with no blind reserves. Its purpose is to decide
whether stochastic AlphaZero-style search is the right primary foundation.

Suggested contract:

1. Select an outcome-blind, phase-balanced panel of ordinary production states
   with no player holding a blind deck-reserve.
2. Build a high-budget reference for root actions using controlled samples or
   enumeration of legal refill identities from the remaining tier deck.
3. Compare the current root-determinization PUCT with an explicit or stratified
   chance-aware variant.
4. Where valid, use coupled refill samples across competing actions to reduce
   comparison noise.
5. Measure action agreement, expected-WDL error, independent-run stability,
   important-refill coverage, and wall-clock cost.
6. Verify that no search ever uses the actual future deck order.
7. Open no gameplay or training until the mechanism comparison is understood.

If the chance-aware AlphaZero route works on ordinary states, use it as the main
architecture. Then add a bounded blind-reserve layer and test it against an
opponent deliberately designed to blind-reserve often. A full public-belief or
CFR solver becomes justified only if the simpler treatment shows a material and
repeatable weakness.

## Action-representation ideas and audit `135`

Two related action-side hypotheses were recorded after this correction:

1. organize complete actions hierarchically into a main card/token intent and
   conditional payment, return-token, and noble details;
2. encode the deterministic public afterstate produced by an action before the
   refill chance event.

The owner considered both potentially relevant, especially the first. A frozen
no-training audit over all 15,916 high-budget experiment-114 replay positions
tested the first as experiment `135`. It found that broad fragmentation is not
material: only 7.778% of positions had a multi-variant core intent, those
groups received 2.888% of visits, and grouping improved raw-versus-search
agreement by only 0.804 percentage points. Keep the current structured
legal-action encoding; do not train a hierarchical head from this evidence.

The second, deterministic-afterstate hypothesis remains open and fits naturally
with the chance-aware refill audit. Full result:
`docs/engineering/alphazero_action_space_fragmentation_audit_135_result.md`.

## Effect on interpretation of earlier experiments

- Experiment 133 remains a clean failure of its particular public-belief PUCT
  mechanism. It should no longer be presented as evidence that ordinary PUCT is
  fundamentally unsuitable for the dominant, public stochastic portion of
  Splendor.
- The compact AlphaZero plateau remains evidence against that exact small CPU
  recipe. It is not a clean test of a deliberately chance-aware AlphaZero search
  with explicit refill coverage.
- Experiment 130 demonstrates noisy targets on its selected mostly
  blind-reserve states. It does not establish the noise level of ordinary states
  without private cards.
- Exact resolvers 084/086 remain valuable strength baselines and teachers.
  Their result in experiment 134 does not decide between stochastic MCTS and
  public-belief CFR as the long-term architecture.

## Guardrail for future handoffs

Whenever a future agent proposes a belief-centric architecture because
"Splendor has hidden information," it must first acknowledge all four facts:

1. the strategically dominant uncertainty appears to be public random refill;
2. human blind deck-reserves were rare in the available descriptive panel;
3. masking the candidate's blind-reserve action was strength-neutral in the
   powered 071 ablation;
4. rare adversarial private information still needs a later robustness test.

The architecture should be selected by measured strategic importance, not by
the mere existence of a rarely used rule.

## References

- Experiment 071 evidence:
  `outputs/deck_reserve_mask_ablation_071/result.json`
- Human/resolver behavior:
  `docs/experiments/human_resolver_ceiling_calibration_074.md`
- Experiment 130 evidence:
  `outputs/alphazero_value_uncertainty_audit_130/result.json`
- Current AlphaZero chance sampling:
  `src/splendor_ai/alphazero/search.py`
- Antonoglou et al., *Planning in Stochastic Environments with a Learned
  Model* (Stochastic MuZero), ICLR 2022:
  https://openreview.net/forum?id=X6D9bAHhBQ1
- Lanctot et al., `Monte Carlo *-Minimax Search`, 2013:
  https://arxiv.org/abs/1304.6057
