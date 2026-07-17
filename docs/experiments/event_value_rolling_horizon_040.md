# Event-Value Rolling-Horizon Diagnostic (`040`)

Frozen before implementing the planner, running gameplay, or inspecting any
new outcome. This is an information-legal, non-training diagnostic. It cannot
promote a checkpoint or authorize self-play by itself.

## Question

Can short open-loop sequence evolution extract useful Splendor decisions from
transition events where minimax/PUCT leaves and state-value models did not?

This is deliberately not another PUCT, rollout-leaf, value-model, AWR, or
hierarchical-option variant. It follows the Branching Mutation Rolling Horizon
(BMRH) and event-value formulation studied directly in Splendor-like games.

## Frozen planner contract

- Root policy and simulated opponent policy: legal-observation `035`.
- The root observer receives a fresh legal determinization for each planning
  sample. Unknown deck order and opponent-private blind reserves may never be
  read from the live state.
- A genome contains two decisions by the root player. Intervening opponent
  turns are chosen by frozen deterministic `035`; the planner never chooses a
  cooperative opponent action.
- Actor candidates are distinct canonical successors among the top four `035`
  actions. If a stored prefix action is no longer legal, the suffix is rebuilt.
- Use a `(1+1)` branching-mutation loop. Each offspring copies the incumbent
  before one uniformly sampled actor-decision mutation point and resamples the
  suffix. The first population covers every root candidate before mutation.
- Each decision evaluates 32 sequences over two determinizations, with fixed
  seed derivation from environment seed, public turn index, and evaluation
  index. Ties are resolved by frozen `035` root score and canonical action key.
- Both variants receive exactly the same determinizations, initial root-covering
  genomes, random mutation-point/suffix schedule, policy scores, candidate
  caps, and simulated-action budget. Later incumbent genomes may differ because
  accepting an offspring is the evaluator-dependent quantity under test.

## Frozen evaluators

`score_delta` is the control. It values the root-relative change in prestige
score over the simulated sequence, with terminal win/loss overriding all
nonterminal values.

`event_value_hc` uses the published five-event hand-grouping and fixed linear
weights, applied root-minus-opponent:

- ordinary or gold token received: `+0.2` per token;
- blind/deck reserve: `+0.2`;
- visible reserve: `-0.4`;
- noble received: `-0.6`;
- a prestige-producing card or noble event: `+0.8` per event, independent of
  point magnitude.

The evaluator counts transition events; it does not read final outcomes before
a simulated terminal state and does not use a learned state value.

## Verification and bounded CPU screen

Before evidence, require tests for:

- exact event extraction for take, reserve, buy, and noble transitions;
- root-minus-opponent sign symmetry;
- invariance to the true ordering of hidden decks and opponent-private cards;
- deterministic decisions under fixed seeds;
- legal returned actions and exact sequence-evaluation budgets;
- identical sampled genomes/determinizations between evaluator variants.

Then run, in order:

1. Two fresh paired seeds (four games) at seeds `402000--402001` as an
   operational smoke. Stop if any game stalls/times out, the budget contract is
   violated, or elapsed time exceeds 60 minutes.
2. Only after a clean smoke, run 30 fresh paired seeds (60 games) at seeds
   `402100--402129`, event-value versus score-delta, with seats reversed per
   seed. Require zero failures, event-value score share at least `55%`, paired
   bootstrap CI95 lower bound above `50%`, and positive mean paired final-score
   margin.
3. Only if step 2 passes, run 30 fresh paired seeds (60 games) at seeds
   `402200--402229`, event-value versus raw `035`, under the same strength
   thresholds.

Failure in step 2 rejects only this fixed published event evaluator at this
budget. It does not authorize tuning after the result. Failure in step 3
rejects this planner as a stronger teacher. A complete pass authorizes a
separately preregistered broad confirmation and possible teacher use; it does
not itself promote anything.

CPU is required for this diagnostic. GPU use is not authorized because exact
environment simulation and many small policy calls dominate this bounded run.

## Result

Rejected on 2026-07-17 at the frozen event-versus-score control gate.

Pre-evidence verification passed all 397 tests. The operational smoke completed
all four games in 43.5 seconds with zero stalls, timeouts, or non-completed
games, and exact `32 x planning decisions` budgets for both variants.

The complete control block then ran all 30 paired seeds / 60 games without any
operational failure. `event_value_hc` went 15-45-0 against `score_delta`:

- paired score share `25.0%`, CI95 `[15.0%, 35.0%]`;
- mean paired final-score margin `-3.9833`, CI95 `[-5.3000, -2.6667]`;
- 1,899 candidate planning decisions, 60,768 sequence evaluations, and 179,295
  simulated actions;
- 1,897 control planning decisions, 60,704 sequence evaluations, and 179,728
  simulated actions.

Only the zero-failure gate passed. The minimum score share, lower confidence
bound, and positive-margin gates all failed. The conditional raw-`035` block
was not generated.

This decisively rejects the published fixed five-event weighting at the tested
two-decision BMRH budget. It does not show that the rolling-horizon machinery is
invalid: the matched score-delta control was much stronger, and its strength
against raw `035` was not tested. It also does not authorize post-result weight,
horizon, or budget tuning. Exact evidence is under
`outputs/event_value_rolling_horizon_040`; `control.json` SHA256 is
`a28c793f822881bdf736bbb0194ef2dbfb55bd25f2a4dabd406108d15fe61eea` and
`gate_final.json` SHA256 is
`60f57de76973d5faf9b80df11f5460985ad11056f86d1125bfd23225f5de3608`.
