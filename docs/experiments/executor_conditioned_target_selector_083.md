# Experiment 083: Executor-Conditioned Target-Advantage Selector

**Preregistered:** 2026-08-06, before `083` implementation, model fitting,
fresh corpus generation, mechanics outcomes, or gameplay outcomes.

## Question

Can a target-conditioned relational model that observes the exact frozen
executor action learn the stable card-target rankings that `082` could not fit,
and then beat raw legal checkpoint `035`?

Post-result read-only audit of `082` ruled out the two simplest explanations.
Target labels remained reliable (historical/fresh split-half Spearman
0.733/0.709 and cross-half uplift +0.225/+0.230), while the frozen model barely
fit even its historical training rows (rank Spearman 0.242 and uplift +0.0069).
It captured about 5% of reducible historical listwise cross-entropy and learned
a systematic late-game tier-3 bias. This audit motivates `083` but cannot gate
it. No `082` validation or confirmation row may be used by any `083` pipeline.

## Frozen candidate

The candidate is an `ExecutorConditionedTargetBot`. It considers `CLEAR`, every
visible development card, and every card reserved by the acting player. For
each target it first computes exactly one action using the frozen mechanics
contract:

- `CLEAR`: raw `035`'s top canonical action;
- non-clear target: the exact `081`/`082` target executor action.

The public model receives the canonical state entities, target identity and
face, and that exact canonical action. A 128-wide, two-layer public state
transformer encodes the entities. Separate 128-wide target and action encoders
form one query per target/action candidate. One four-head target-conditioned
cross-attention layer queries the state entities independently for each of the
roughly 15 candidates. A 192-wide two-layer head scores each candidate. The
`CLEAR` score is subtracted from every score so the deployed output is direct
advantage and `CLEAR` is exactly zero.

When no target is active, choose the maximum advantage with canonical target
identity as deterministic tie break. `CLEAR` delegates that turn to raw `035`.
A non-clear target remains active until purchase or loss; while active, the
frozen executor supplies actions without model switching or clearing. This is
the same persistence and executor contract as `082`; only the learned
representation and loss are new.

The model observes no hidden deck identity, opponent blind reserve identity,
human row, `035` scalar score, prior model parameter, or `082` fresh row.

## Stage A: frozen historical architecture screen

Use only
`outputs/card_target_value_selector_082/training/historical_target_rows.json`,
whose frozen SHA-256 is
`7a4c080f6a50aa179ea7afa2fe08d7b4f936c139d703959e9fc65752888ab3b6`.
These are the 1,000 already observed `081` states reduced to their exact
target/executor-action labels. They are an architecture screen, not new
evidence.

Split whole source games once with seed `783083` into 800 fit and 200 historical
screen states. Train one randomly initialized model for 50 epochs, batch size
16, AdamW learning rate `3e-4`, weight decay `1e-4`, gradient norm 1.0, and seed
`783083`. For every state define label advantage as target return minus `CLEAR`
return. Loss is equal-weight per-state listwise cross-entropy to
`softmax(advantage / 0.10)` plus 1.0 times MSE between predicted and labelled
advantages. Select the earliest epoch with minimum historical-screen mean
regret. There is no sweep, retry, warm start, continuation, or fresh-row epoch
selection.

The architecture screen passes only if all hold:

- fit-state mean target-rank Spearman is at least 0.50;
- historical-screen mean target-rank Spearman is at least 0.35;
- fit-state selected-minus-`CLEAR` return is at least +0.10;
- historical-screen selected-minus-`CLEAR` return is at least +0.05;
- on historical-screen non-clear target pairs, predicted-positive recall is at
  least 20%; and
- predicted-positive precision is strictly above the observed beneficial-pair
  prevalence.

Failure closes this exact architecture/training recipe before any new corpus.

## Stage B: fresh target-only corpus

Only after Stage A passes, generate up to 600 source games at seeds
`789500--790099` with the frozen `082` source policy: public epsilon 0.25 over
canonical successors, otherwise raw `035`. Do not store or inspect source-game
scores, winners, or margins. Select exactly 400 game-disjoint states: 200 fresh
validation and 200 untouched confirmation. Each split has 100 states per actor
seat and phase counts 67 early, 67 mid, and 66 late. Selection seed is `783083`.

Evaluate every public target at its one frozen executor action using two public
determinizations and four matched continuations each. Continuations use
stochastic `035` with epsilon 0.10, temperature 0.50, top-k 3, at most 120
additional plies, and the normal repetition/no-progress safeguards. Labels are
mean terminal score share over eight continuations.

Require exact state/split/phase/seat/target/determinization coverage, one
`CLEAR` baseline per state, zero mapping/illegal/null errors, cutoff fraction at
most 1%, and positive game-clustered lower CI95 for mean split-half target-rank
Spearman. Failure stops before fresh model evaluation.

## Stage C: fresh offline gate

Evaluate the already frozen Stage-A checkpoint; do not retrain it. Validation
is evaluated first, followed by untouched confirmation. Stage C passes only if:

- validation mean selected-minus-`CLEAR` return is positive;
- confirmation mean selected-minus-`CLEAR` return is at least +0.05 and its
  20,000-resample game-clustered bootstrap CI95 lower bound is above zero;
- confirmation regret is at most 80% of `CLEAR` baseline regret;
- at least 15% of confirmation executor actions differ from raw `035`; and
- at least 20% of confirmation selections use a non-clear target.

Failure closes the exact candidate before games.

## Stage D: outcome-blind mechanics and deployability

Only after Stage C passes, run 40 fresh paired seeds `791000--791039` against
raw `035`, candidate once from each seat. Outcomes are not exposed or used.
Replay the first pair byte-identically after the panel, excluding wall-clock
latency fields.

Integrity requires exact seed/seat coverage and deterministic replay, valid
`completed` or `forced_pass_deadlock` terminals, zero stalls, timeouts,
illegal/null actions, or loop fallbacks, and exact raw/model hashes. Mechanics
requires at least 15% action movement, an active target on at least 20% of
decisions, at least half of selected targets purchased before loss or game end,
and at least 5.0 purchases per candidate game. Inference p95 must be at most
150 ms, the mechanics panel at most 2 hours, and its linear 200-pair projection
at most 10 hours.

## Stage E: fresh strength panel

Only after Stage D passes, run 200 paired seeds `792000--792199`, candidate once
from each seat: 400 games. Limits are `max_turns=400`, `repetition_limit=4`, and
`no_progress_limit=60`; no game may be replaced or excluded. Use a
20,000-resample paired-seed bootstrap with seed `792083`.

Promotion requires paired score share at least 55% and CI95 lower bound
strictly above 50%, plus integrity and at least 15% action movement. Other
telemetry cannot rescue a failed primary endpoint.

## Decisions

- Historical/fresh corpus or mechanics integrity/operational failure:
  `executor_conditioned_target_083_invalid_no_strength_decision`.
- Historical architecture-screen failure:
  `executor_conditioned_target_083_fit_failed_close_recipe`.
- Fresh corpus signal failure:
  `executor_conditioned_target_083_counterfactual_signal_failed_close_recipe`.
- Fresh offline failure:
  `executor_conditioned_target_083_offline_failed_close_recipe`.
- Mechanics failure:
  `executor_conditioned_target_083_mechanics_failed_close_candidate`.
- Strength share at least 55% and CI95 lower bound above 50%:
  `executor_conditioned_target_083_stronger_promote_fast_and_research_champion`.
- Valid strength with CI95 upper bound below 50%:
  `executor_conditioned_target_083_weaker_close_candidate`.
- Any other valid strength panel:
  `executor_conditioned_target_083_not_decisively_stronger_close_candidate`.

A pass replaces `035` as both champions. A non-pass changes neither champion,
does not authorize repair or threshold relaxation, and cannot pool gameplay
with earlier panels. Human replays remain diagnostic-only and must not be read.

## Evidence

The final manifest will be
`outputs/executor_conditioned_target_selector_083/result.json`, with every
generated artifact referenced by SHA-256. Record the final decision and hashes
in `AGENT_CURRENT.md` after the single frozen run.

## Result

The single frozen run completed on 2026-08-06 with decision
`executor_conditioned_target_083_fit_failed_close_recipe` after 227 seconds.
It stopped at the historical architecture screen, so no fresh corpus,
mechanics panel, or strength panel was generated.

The one model selected epoch 5. Executor conditioning and target cross-attention
improved positive-pair classification, but not enough to fit or generalize the
target ranking:

- fit-state rank Spearman was 0.2975 versus the required 0.50;
- historical-screen rank Spearman was 0.2065 versus the required 0.35;
- fit-state selected-minus-`CLEAR` return was +0.05680 versus +0.10;
- historical-screen selected-minus-`CLEAR` return was +0.00563 versus +0.05;
- historical-screen positive-target recall was 36.25%, passing 20%; and
- historical-screen positive precision was 30.06%, above the 22.50% beneficial
  target prevalence and therefore passing.

Thus two of six gates passed. The model identified positive target pairs better
than `082`, but its argmax ranking remained near-flat and did not generalize
across already observed historical states. The exact architecture/training
recipe is closed. Raw `035` remains both champions.

Artifact SHA-256 values:

- fit result: `52fae32bb0b2f4ece15e5b6b382edcff53479ef4841aac705e6641c798ee226b`;
- frozen checkpoint: `2b4a63544f7290f0dbe4908c65a108f7e538631566cd1341952841b525b49ed3`;
- final manifest: `b9bea507007e3eb80dce39b38e2e34447750fe620eeffb80174f77e5f7a30a0b`.
