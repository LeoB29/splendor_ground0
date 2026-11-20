# Experiment 082: Card-Target Value Selector

**Preregistered:** 2026-08-06, before any `082` implementation, fresh corpus
generation, training, mechanics outcome, or gameplay outcome.

## Question

Can a from-scratch model that ranks only public card targets, while a frozen
mechanics executor supplies the action, beat raw legal checkpoint `035`?

Experiment `081` established a reproducible counterfactual option signal but
failed offline because its joint target/action scorer selected `CLEAR` on
99.5% of confirmation states. Post-result diagnosis found that the failure was
specific to the flat joint scorer: on the already observed `081` confirmation
states, target values evaluated at the frozen executor action had split-half
Spearman 0.739, and a cross-half target oracle improved on `CLEAR` by 0.212.
Those values motivate this architecture but are not evidence for `082`.

## Frozen candidate

The deployable candidate is a `CardTargetValueBot` with a newly initialized,
observer-legal canonical-entity network. It scores only these target choices:

- `CLEAR`;
- every visible development card; and
- every card reserved by the acting player.

The model has two 128-wide canonical state-entity attention layers, four heads,
256-wide feed-forward blocks, 16-wide ID embeddings, a 128-wide target encoder,
and a 192-wide two-layer independent target score head. It observes no action,
hidden deck identity, opponent blind-reserve identity, recurrent state, `035`
score, human row, or prior model parameter.

When no target is active, the model selects the maximum-scoring target, with
canonical target identity as the deterministic tie break. `CLEAR` delegates
that turn to raw `035` and leaves no active target. A non-clear target becomes
active and remains active until it is purchased or is no longer visible or in
the candidate's reserve. While active, the frozen executor supplies every
action; the model cannot switch or clear it. After completion or loss, the
model selects afresh on its next turn. Deck-reserve actions remain available to
raw `035` under `CLEAR` but cannot create a target.

The executor is exactly the `081` mechanics-only executor, in this order:

1. buy the target if legal;
2. reserve it if visible and a reserve slot is free;
3. buy the raw-`035`-highest affordable card whose bonus reduces the target's
   post-bonus colour deficit;
4. take the action with the largest reduction in that deficit, breaking ties by
   raw `035` score and canonical action identity; and
5. otherwise take raw `035`'s top canonical action.

## Stage A: historical training rows

Use all 1,000 states in the completed `081` corpus at
`outputs/card_option_q_controller_081/corpus.json`, whose frozen SHA-256 is
`f368b462b71bf02154ed00ce524606a8142e238ad55b0d2689026d9d8abb373e`.
The old train, validation, and confirmation labels have all been observed, so
all 1,000 states are training data only and none can gate or select `082`.

For each historical state, reconstruct canonical legal actions and raw `035`
scores. The `CLEAR` label is the stored return for raw `035`'s top action. Each
non-clear target label is the stored return for the action chosen by the frozen
executor at that state. Require an exact unique stored target/action match,
eight continuations per label, and exact corpus hash. Any mismatch stops
invalid. Human replays at `outputs/human_games/replays.jsonl` must not be read.

## Stage B: fresh target-only corpus

Generate up to 600 source games at seeds `785500--786099`. Both seats use the
same deterministic public epsilon source policy as `081`: probability 0.25 of
a uniform canonical legal successor, otherwise raw `035`'s top action. Do not
store or consult final scores, winners, or margins. Select exactly 400
game-disjoint actor states, 200 fresh validation and 200 untouched confirmation.
Each split must contain 100 states from each actor seat and phase counts
67 early (`turn < 20`), 67 mid (`20--39`), and 66 late (`>=40`). Selection is
fixed by seed `782082`; failure to fill every cell stops invalid.

For every state evaluate each public target exactly once at its frozen action:
raw `035`'s top action for `CLEAR`, otherwise the executor action. Use two
public-observation determinizations and four matched continuations per
determinization. After the root action, the executor pursues a live non-clear
target on the candidate's turns until purchase or loss; otherwise both seats
use stochastic `035`. Stochastic continuation has epsilon 0.10, temperature
0.50, top-k 3, at most 120 additional plies, and the normal repetition and
no-progress safeguards. Random namespaces are matched across targets.

The target label is mean terminal score share (`1/0.5/0`) over the eight
continuations. Stage B requires exact state/split/phase/seat/target/
determinization coverage, exactly one `CLEAR` baseline per state, zero
mapping/illegal/null errors, continuation cutoff fraction at most 1%, and a
positive game-clustered lower 95% confidence bound for mean per-state
split-half Spearman target ranking. Failure stops before training.

## Stage C: one frozen training run and fresh offline gate

Train one model from random initialization on the 1,000 historical states for
25 epochs with batch size 16, AdamW learning rate `1e-4`, weight decay `1e-4`,
gradient norm 1.0, and seed `782082`. For each state, form a target distribution
as `softmax((return - state_mean_return) / 0.10)`. Loss is listwise cross-entropy
from that distribution to the model target logits plus 0.25 times MSE between
the mean-centered target returns and mean-centered logits. Each state
contributes equally. There is no action loss or cross-state target-pair loss.

Select the earliest epoch with minimum fresh-validation mean regret. There is
one model, no sweep, retry, warm start, continuation, threshold change, or
post-confirmation repair. Only after epoch selection evaluate the untouched
fresh confirmation split. Stage C passes only if:

- validation mean selected-minus-`CLEAR` return is positive;
- confirmation mean selected-minus-`CLEAR` return is at least `+0.05` and its
  20,000-resample game-clustered bootstrap CI95 lower bound is above zero;
- confirmation regret is at most 80% of `CLEAR` baseline regret;
- at least 15% of confirmation executor actions differ from raw `035`; and
- at least 20% of confirmation selections use a non-clear target.

Failure closes this exact target model/training recipe and stops before games.

## Stage D: outcome-blind mechanics and deployability

Only after Stage C passes, run 40 fresh paired seeds `787000--787039` against
raw `035`, candidate once from each seat. Do not summarize, gate on, or expose
wins, scores, or margins. Replay the first pair byte-identically after the
panel, excluding nondeterministic wall-clock latency fields.

Integrity requires exact seed/seat coverage and deterministic replay, valid
`completed` or `forced_pass_deadlock` terminals, zero stalls, timeouts,
illegal/null actions, or loop fallbacks, and exact raw/model hashes. Mechanics
requires at least 15% action movement from raw `035`, an active non-clear target
on at least 20% of decisions, at least half of selected targets bought before
loss or game end, and at least 5.0 purchases per candidate game. Report phase
movement, target selections/completions/losses, action mix, reserve/purchase
tiers, deck reserves, gold spent, and inference latency.

Operationally, inference p95 must be at most 150 ms, the 40-pair panel at most
2 hours, and its linear 200-pair projection at most 10 hours. Failure stops
without a strength decision.

## Stage E: fresh strength panel

Only after Stage D passes, run 200 fresh paired seeds `788000--788199` against
raw `035`, candidate once from each seat: 400 games. Limits are
`max_turns=400`, `repetition_limit=4`, and `no_progress_limit=60`; no game may
be replaced or excluded. Use a 20,000-resample paired-seed bootstrap with seed
`788082`.

Primary endpoint is paired score share. Promotion requires observed share at
least 55% and CI95 lower bound strictly above 50%. Integrity and action
movement at least 15% are conjunctive. All other telemetry is diagnostic and
cannot rescue a failed primary endpoint.

## Decisions

- Historical/fresh corpus or mechanics integrity/operational failure:
  `card_target_value_082_invalid_no_strength_decision`.
- Fresh corpus signal failure:
  `card_target_value_082_counterfactual_signal_failed_close_recipe`.
- Training/offline failure:
  `card_target_value_082_offline_failed_close_recipe`.
- Mechanics failure:
  `card_target_value_082_mechanics_failed_close_candidate`.
- Stage-E share at least 55% and CI95 lower bound above 50%:
  `card_target_value_082_stronger_promote_fast_and_research_champion`.
- Valid Stage E with CI95 upper bound below 50%:
  `card_target_value_082_weaker_close_candidate`.
- Any other valid Stage E:
  `card_target_value_082_not_decisively_stronger_close_candidate`.

A pass replaces `035` as both fast and research champion. A non-pass never
changes either champion and never permits pooling with prior gameplay panels.

## Evidence

The final manifest will be
`outputs/card_target_value_selector_082/result.json`, with historical-row,
fresh-corpus, model, mechanics, and optional strength artifacts referenced by
SHA-256. Record all result hashes in `AGENT_CURRENT.md` after the single frozen
run.

## Result

The single frozen run completed on 2026-08-06 with decision
`card_target_value_082_offline_failed_close_recipe` after 1,500 seconds (25.0
minutes). It stopped before mechanics or strength games, as required.

Historical extraction passed: all 1,000 `081` states produced 15,468 exact,
unique target/executor-action rows (123,744 stored continuations), and the old
validation and confirmation splits were used only as training rows.

The fresh corpus passed every gate. It contained 400 game-disjoint states,
6,202 targets, and 49,616 matched continuations with zero mapping,
illegal-action, or null-action errors. There were 82 cutoffs (0.165%, below the
1% ceiling). Mean split-half target-ranking Spearman was 0.7087 with CI95
[0.6799, 0.7366], independently confirming that target-only return rankings are
stable.

The one frozen model selected epoch 10. It fixed `081`'s `CLEAR` collapse but
did not improve returns:

- validation selected-minus-`CLEAR` return was +0.00156, CI95
  [-0.02969, +0.03313];
- untouched confirmation selected-minus-`CLEAR` return was -0.00281, CI95
  [-0.02875, +0.02250], missing both the +0.05 and positive-lower-bound gates;
- confirmation regret was 0.34250 versus baseline 0.33969, ratio 1.0083 rather
  than at most 0.80;
- action movement was 31.5%, passing the 15% dose gate; and
- 52.0% of confirmation selections were non-clear targets, passing the 20%
  target-use gate (82 visible, 22 reserved, and 96 `CLEAR`).

Thus target-only listwise training solved the gross calibration failure without
turning the reproducible counterfactual target signal into held-out policy
uplift. The exact model/training recipe is closed. Raw `035` remains both fast
and research champion.

Artifact SHA-256 values:

- fresh corpus: `14da773f06f044c3242221f981dadab008055f43dffd5322e8169895a49bb073`;
- historical target rows: `7a4c080f6a50aa179ea7afa2fe08d7b4f936c139d703959e9fc65752888ab3b6`;
- training result: `2044e598a0b766061990f9145344283babccd44460dd99c95414ca3ea9347edb`;
- frozen checkpoint: `59bdf6061a8afedf218479ce24e4ed4731c15d8819d5565a61f00d652e73ffab`;
- final manifest: `768572189d99d55cdf5379351f8e50da3fb15fea67db4de47c31c437f3cc1a4e`.
