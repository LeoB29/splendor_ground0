# Experiment 081: From-Scratch Card-Option Q Controller

**Preregistered:** 2026-08-05, before corpus generation, training, mechanics
outcomes, or gameplay outcomes.

## Question

Can a fast, from-scratch policy represent the identity and delayed value of a
card target directly, without the rigid lexicographic search that failed in
`080`, and beat raw legal checkpoint `035`?

`078` established that funded prestige for a high-efficiency reserved target
causally helps an otherwise identical one-ply search. `080` then showed that
the exact persistent depth-3 option planner is not a viable controller: it
failed the purchase mechanic and produced a repetition cutoff even when its
compute limits were expanded. `081` is a new architecture, not a repair,
reweighting, depth change, or behavioral clone of `077`--`080`.

## Outcome-blind sizing audit

Before this protocol, 12 raw-`035` trajectories at seeds `781000--781011` were
advanced for at most 60 plies without reading or reporting scores or winners.
Across 718 public states there were a mean 15.33 target choices (visible cards,
own reserved cards, and `CLEAR`) and 21.55 canonical legal actions, or 324.61
target/action pairs. Median/p90/max pair counts were 320/476/864. Raw `035`
median decision latency was 3.34 ms on this CPU. These shape and timing values
were used only to select a batched, independent-pair network and its operational
limits; they are not gameplay evidence.

## Frozen candidate

The deployable candidate is a `CardOptionQBot` with a newly initialized
observer-legal canonical-entity network. It receives:

- the public canonical state entities;
- one target candidate: `CLEAR`, any visible card, or any card in the acting
  player's reserve; and
- one canonical legal action.

It scores every target/action pair in one batched forward pass and chooses the
maximum with canonical target/action identity as the deterministic tie break.
The chosen non-clear card becomes the current target. On each later own turn the
same complete choice is repeated, so keeping, switching, or clearing the target
are all learned choices with no fixed switching threshold. Deck-reserve actions
remain legal but never create a target because their card identity is not public.

The model has independent pair scoring, not quadratic attention over pairs:
two 128-wide canonical state-entity attention layers, 4 heads, 256-wide
feed-forward blocks, 16-wide ID embeddings, 128-wide target and action encoders,
and a 192-wide two-layer score head. It is trained from random initialization;
no `035` parameter, logit, KL anchor, recurrent state, `077`/`080` action, or
human row is used.

## Stage A: fresh counterfactual option corpus

Generate 1,200 source games at seeds `781500--782699`. Both seats use frozen
`035` scores with a deterministic public epsilon policy: with probability 0.25
choose uniformly among canonical legal successors and otherwise choose the top
score. Selection is outcome-blind: do not store or consult scores, winners, or
final margins. Select exactly 1,000 game-disjoint actor states, balanced 334/333/333
over early (`turn < 20`), mid (`20--39`), and late (`>=40`) phases and 500/500
over actor seat. If exact coverage is unavailable, stop invalid.

Split whole source games once with seed `781081`: 600 train, 200 validation, and
200 untouched confirmation states. For every state enumerate the Cartesian
product of every public target candidate and every canonical legal action. No
pair is selected by `035`, a heuristic, or an observed return.

Evaluate every pair under two public-observation determinizations and four
matched continuations per determinization. Apply the root action, then on the
candidate's turns use a frozen mechanics-only target executor until the target
is purchased or lost; the opponent uses stochastic `035`. After completion,
loss, or `CLEAR`, both seats use stochastic `035`. The executor orders rules as:

1. buy the target if legal;
2. reserve it if visible and a reserve slot is free;
3. buy the raw-`035`-highest affordable card whose bonus reduces the target's
   post-bonus colour deficit;
4. take the action with the largest reduction in that deficit, breaking ties by
   raw `035` score and canonical action identity; and
5. otherwise take raw `035`'s top canonical action.

Stochastic continuation uses epsilon 0.10, temperature 0.50, and top-k 3, with
matched random namespaces across pairs. Each continuation has at most 120
additional plies and uses the normal repetition/no-progress safeguards. The
target is only a public card identity; determinizations may never expose a hidden
deck card or an opponent blind reserve to the model or executor.

The label is mean terminal score share (`1/0.5/0`) over the eight continuations.
Corpus gates are exact state/split/target/action/determinization coverage, zero
illegal/null actions, continuation cutoff fraction at most 1%, and positive
game-clustered lower 95% confidence bound for the mean split-half Spearman
correlation of target/action returns. Failure stops without training.

## Stage B: one frozen training run

Train for 25 epochs, batch size 8 states, AdamW learning rate `1e-4`, weight
decay `1e-4`, gradient norm 1.0, and seed `781081`. Loss is target-value MSE plus
0.5 times a best-versus-worse logistic ranking loss over pairs whose label is at
least 0.10 below the state's best label. Select the epoch with minimum validation
mean regret. There is one model, no sweep, retry, warm start, or continuation.

On validation and untouched confirmation, compare the model-selected pair with
the exact raw-`035` baseline pair (`CLEAR` plus raw `035`'s top canonical action)
using the already generated matched labels. Stage B passes only if:

- validation mean selected-minus-baseline return is positive;
- confirmation mean selected-minus-baseline return is at least `+0.05` and its
  20,000-resample game-clustered bootstrap CI95 lower bound is above zero;
- confirmation regret is at most 80% of baseline regret;
- at least 15% of confirmation actions differ from raw `035`; and
- at least 20% of confirmation selections use a non-clear card target.

Training or offline-gate failure closes this exact model/training recipe and
stops before mechanics games.

## Stage C: outcome-blind mechanics and deployability

Only after Stage B passes, run 40 fresh paired seeds `783000--783039` against raw
`035`, candidate once from each seat. Do not summarize, gate on, or expose wins,
scores, or margins. Replay the first pair byte-identically after the panel,
excluding nondeterministic wall-clock latency fields.

Integrity requires exact seed/seat coverage and deterministic replay, valid
`completed` or `forced_pass_deadlock` terminals, zero stalls, timeouts,
illegal/null actions, or loop fallbacks, and exact raw/model hashes. Mechanics
requires at least 15% action movement from raw `035`, a non-clear target on at
least 20% of decisions, at least half of newly selected non-clear targets bought
before being switched, cleared, or lost, and at least 5.0 purchases per candidate
game. Report overall and phase movement, target keeps/switches/clears/selections,
completion/loss rates, action mix, reserve/purchase tiers, deck reserves, gold
spent, and inference latency.

Operationally, inference p95 must be at most 150 ms, the 40-pair panel at most
2 hours, and its linear 200-pair projection at most 10 hours. Failure stops with
no strength decision.

## Stage D: fresh strength panel

Only after Stage C passes, run 200 fresh paired seeds `784000--784199` against
raw `035`, candidate once from each seat: 400 games. Limits are `max_turns=400`,
`repetition_limit=4`, and `no_progress_limit=60`; no game may be replaced or
excluded. Use a 20,000-resample paired-seed bootstrap with seed `784081`.

Primary endpoint is paired score share. Promotion requires observed share at
least 55% and CI95 lower bound strictly above 50%. Integrity and movement at
least 15% are conjunctive. All other telemetry is diagnostic and cannot rescue
a failed primary endpoint.

## Decisions

- Corpus or Stage-C integrity/operational failure:
  `card_option_q_081_invalid_no_strength_decision`.
- Corpus signal failure:
  `card_option_q_081_counterfactual_signal_failed_close_recipe`.
- Training/offline failure:
  `card_option_q_081_offline_failed_close_recipe`.
- Mechanics failure:
  `card_option_q_081_mechanics_failed_close_candidate`.
- Stage-D share at least 55% and CI95 lower bound above 50%:
  `card_option_q_081_stronger_promote_fast_and_research_champion`.
- Valid Stage D with CI95 upper bound below 50%:
  `card_option_q_081_weaker_close_candidate`.
- Any other valid Stage D:
  `card_option_q_081_not_decisively_stronger_close_candidate`.

A pass replaces `035` as both fast and research champion. A non-pass never
changes either champion, never authorizes a threshold/model/data correction,
and never permits pooling with `077`--`080`. Human replays at
`outputs/human_games/replays.jsonl` remain diagnostic-only and must not be read
by any `081` pipeline.

## Evidence

The final manifest will be
`outputs/card_option_q_controller_081/result.json`, with corpus, model, mechanics,
and optional strength artifacts referenced by hash. Record all result hashes in
`AGENT_CURRENT.md` after the single frozen run.

## Result

The single frozen run completed on 2026-08-06 with decision
`card_option_q_081_offline_failed_close_recipe` after 65,172 seconds (18.10
hours). It stopped before mechanics or strength games, as required.

Stage A passed every corpus gate. The 1,000 game-disjoint states contained
335,436 target/action pairs and 2,683,488 matched continuations. There were zero
mapping/illegal/null errors. Only 4,238 continuations cut off (0.158%, below the
1% ceiling), and split-half target/action rankings were strongly reproducible:
mean Spearman 0.7632, CI95 [0.7504, 0.7755]. The new option-value signal exists
and is stable.

The one frozen model selected epoch 15, but failed five of six offline gates:

- validation selected-minus-raw-`035` return was -0.02594, CI95
  [-0.07188, +0.02000];
- untouched confirmation selected-minus-raw-`035` return was -0.03031, CI95
  [-0.07563, +0.01594], missing both the +0.05 point and positive-lower-bound
  requirements;
- confirmation regret was 0.43344 versus the baseline's 0.40313, ratio 1.0752
  rather than at most 0.80;
- action movement was high at 92.0% and passed the 15% dose gate; but
- only 0.5% of confirmation selections used a non-clear target, failing the 20%
  target-use gate.

Thus the learned scorer changed actions at high dose but almost entirely
collapsed to `CLEAR`; it did not learn the intended card-option mode and did not
beat the raw-`035` offline reference. The exact model/training recipe is closed.
No mechanics or gameplay-strength conclusion was made, and raw `035` remains
both fast and research champion.

Artifact SHA-256 values:

- corpus: `f368b462b71bf02154ed00ce524606a8142e238ad55b0d2689026d9d8abb373e`;
- training result: `e1c61118e50f2b9305a466ed6eb53f55e6fe9abc2c67cd9cf6b354bd31e980b0`;
- frozen checkpoint: `10512b1247b043c5348b7b22fa5ff1cda735ded5bc92c3bcd73c585df108e0ef`;
- final manifest: `021475e1503aa8bd4833931bffd5ce84fbd40cb6e363a06cc569efa9acb3de80`.
