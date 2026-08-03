# Deck-Reserve Mask Ablation (`071`)

Frozen before implementing the masked bot, running any game, or inspecting any
outcome. This is a no-training inference-time ablation of a single action class
on the frozen champion. It cannot promote a checkpoint.

## Origin

Two independent sources converged on `RESERVE_DECK` as a suspect action class.

1. An experienced human player, after ten serious games against `035`, reported
   never reserving from the deck and judged that a good Splendor player is very
   unlikely to take that action. That is external knowledge the local opponent
   league cannot supply: every neural league member (`035`, `016`, `023`) and the
   shallow search bot trace their strategic knowledge to one hand-written
   heuristic evaluator, so none of them punishes the class.
2. The disjoint non-human reserve generalization audit
   (`outputs/diagnostics/reserve_generalization_policy035_full.json`) resolved 12
   mistakes out of 60 audited states under matched terminal continuations.
   Three of those 12 are `RESERVE_DECK` tier 1 -> `RESERVE_VISIBLE` tier 1.
   The continuation policy has no reason to favour visible reserves, so this is
   independent evidence in the same direction.

Measured frequency of the class, recorded before this protocol was frozen:

- deterministic `035` across the ten human games: 4 / 271 decisions = `1.48%`;
- epsilon-greedy `035` self-play, `data/corpus_generation1_policy035_selfplay_038`:
  4,586 / 63,622 decisions = `7.21%`, i.e. 33% of all its reserves;
- both reserve classes take a gold token at comparable rates (`RESERVE_DECK`
  91.2%, `RESERVE_VISIBLE` 95.3%), so masking the deck class does not remove
  the gold-acquisition route.

## Question

Does removing `RESERVE_DECK` from `035`'s legal candidate set at inference time
change its paired strength against unmodified `035`, and on what fraction of
decisions does the mask actually bind?

## Declared primary output: the dose

The fraction of candidate decisions where masking changes the greedy action is a
**primary reported quantity, not a gate**. Prior experiments in this repository
repeatedly evaluated interventions that changed 1-4% of decisions against
strength gates requiring several points of score share. This protocol records
the dose so that a null result is interpreted at its measured dose rather than
as a refutation of the hypothesis.

Expected dose from the frozen pre-run measurement above is approximately
`1.5%`. That is low, and it is declared here in advance that a null at that dose
is weak evidence. The properly dosed test of the same hypothesis is the resolver
arm (the resolver chooses `RESERVE_DECK` on `9.4%` of all decisions and `19.8%`
of early decisions in the `052` corpus); it requires a separate preregistration
and roughly six hours of CPU, and is authorized only by the decision rules
below.

## Candidate and baseline

Both sides load the identical frozen champion:

`outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`
SHA-256 `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`

- **Baseline:** `LegalActionAttentionScorerBot`, unmodified.
- **Candidate:** `DeckReserveMaskedAttentionScorerBot`. One forward pass per
  decision over the complete legal action set; the greedy choice is taken over
  the legal actions whose `action_type` is not `RESERVE_DECK`. If every legal
  action is a `RESERVE_DECK`, the full set is used and the decision is counted
  as a `deck_only_fallback`. No weight, encoder, feature, or score is altered.

Because the weights are identical, any difference is attributable to the masked
action class alone.

## Panel

1,000 fresh paired seeds, `740000`--`740999`, each played twice with the
candidate in seat 0 and then in seat 1: **2,000 games**. The seed band is
disjoint from every prior experiment. Engine limits are the established
`max_turns=400`, `repetition_limit=4`, `no_progress_limit=60`. Valid terminals
are exactly `completed` and `forced_pass_deadlock`, consistent with the frozen
engine contract. No game may be replaced or excluded on the basis of its
outcome.

Paired score share is computed per seed cluster (mean over the seed's two games)
and the CI95 comes from a paired-cluster bootstrap with 20,000 draws and seed
`740071`. Individual games are never treated as independent observations.

## Declared power

The per-pair score-share standard deviation measured in the `043` robustness
gate is approximately `0.32`, so 1,000 clusters give a standard error near
`0.0101`. The primary gate below therefore resolves a true effect of about
`+2.8` points at 80% power and `+3.3` points at 90%. Effects smaller than
`+2` points are not resolvable by this panel and will be reported as
inconclusive rather than as null.

## Integrity gates

All required:

1. exactly 1,000 completed clusters and 2,000 games, all seeds in the reserved
   band and each used exactly once per seat;
2. every game ends in `completed` or `forced_pass_deadlock`; zero stalls, zero
   `max_turns`, zero `repetition_cutoff`, zero `no_progress_cutoff`;
3. zero illegal actions; the candidate never returns a null action;
4. the candidate's mask never removes the final legal option without the
   `deck_only_fallback` path recording it;
5. the baseline chooses `RESERVE_DECK` at least once, otherwise the ablation is
   vacuous on this panel;
6. both checkpoint hashes equal the frozen value above;
7. the first cluster replays byte-identically in a repeat invocation.

## Strength and decision rules

Primary metric: candidate paired score share against the baseline.

- Dose below `1.0%`:
  `deck_reserve_mask_dose_too_small_inconclusive`.
  Do not interpret the score share as evidence about the hypothesis. Authorizes
  only a separately preregistered resolver-arm ablation.
- Dose at least `1.0%` and score-share CI95 lower bound above `0.50`:
  `deck_reserve_mask_improves_authorize_confirmation_preregistration`.
  Authorizes only a separate fresh untouched confirmation panel. Does **not**
  promote, and does not by itself change the champion.
- Dose at least `1.0%` and score-share CI95 upper bound below `0.50`:
  `deck_reserve_class_is_load_bearing_close_mask_direction`.
  The class carries real value in this engine despite expert judgement; close
  the masking direction and record the disagreement.
- Otherwise:
  `deck_reserve_mask_neutral_at_measured_dose`.
  Report the dose-conditional minimum detectable effect. Authorizes only the
  separately preregistered resolver arm.

## Prohibitions

Do not tune the mask (for example to a phase window or a tier subset) after
seeing this result, do not pool this panel with any future confirmation, do not
promote either configuration from this experiment, do not run the resolver arm
without its own frozen protocol, and do not rent a GPU. The bottleneck is engine
simulation; the neural forward pass is unchanged from the frozen champion.

## Evidence

`outputs/deck_reserve_mask_ablation_071/result.json`, with the result SHA-256
recorded in `AGENT_CURRENT.md` after the run.
