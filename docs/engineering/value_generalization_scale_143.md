# Experiment 143: value generalization at two data scales

Prospective protocol, 2026-09-08. The owner authorized the proposed 5,120-game
value-learning comparison. This is a new experiment; do not resume or modify 142.
Keep all previous work and frozen dependencies. No search, searched self-play,
policy update, arena, installation, promotion or automatic follow-up is included.
Research and fast champions remain 084 and 035.

## Question and primary test

Does the 035 value head learn reliable outcome predictions from a larger number
of independent games while every policy-affecting parameter stays unchanged?
142 fitted 33,025 value-head parameters using 205 games and overfit. Its 51-game
validation set is development evidence and is not reused here. Failure does not
establish whether data, features, regularization or optimization is responsible.

The prespecified primary candidate is the **4,096-game head**. Primary estimand:
its whole-game mean squared W/D/L error minus that of a neutral zero predictor
on 512 fresh test games. Lower is better. The 256-game head tests data scale under
the same initialization, learning rule and maximum optimizer budget. A small
public score/resource evaluator provides a reference independent of 035's learned
state representation. Neither test outcomes nor comparison with that reference
select which neural candidate is primary.

## Data and information boundary

Collect exactly 5,120 scheduled games with game seed `1_430_000_000 + index`:

| Index | Role | Games |
|---|---|---:|
| 0--4095 | Large training pool | 4,096 |
| 4096--4607 | Shared validation | 512 |
| 4608--5119 | Protected test | 512 |

The small arm uses 256 nested training indices selected by sorting all 4,096
indices by SHA256 of `143-small-v1|INDEX`, taking the first 256. This selection
uses no observations or outcomes. All splits are by complete game, never by
position. The schedules are fresh to the 142 game/probe corpus.

Both seats use the exact 035 checkpoint, SHA256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
Use 142's behavior distribution: for the first 20 turns, probability .2 of
sampling from the top three concrete actions at softmax temperature .5;
otherwise choose the first maximum-scoring action. Behavior RNG is
`random.Random(game_seed + 143003)`. Score the complete legal action list with
035's original public observation/action features. No search is involved.

The shared replay collector uses absolute turn cap 400, repetition limit four,
and no-progress limit 60. Valid outcomes are normal completion and a verified
two-sole-pass deadlock, using points then fewest purchased cards. All attempted
games remain saved. Cutoff games receive no training target and no replacement.
Stop before fitting if more than 2% of the small, large or validation split is
non-natural; otherwise use all naturally completed games and report effective
counts. Any non-natural test game invalidates confirmatory claims; a complete
descriptive evaluation of the naturally completed subset may still be recorded.
Engine failures stop execution. Never substitute an adjudicated cutoff winner.

Save the 295 observer-legal input features, frozen 256-dimensional state-encoder
output, 15 public baseline features, actor, turn and actual final W/D/L
(-1/0/+1 from the actor's view). Cache frozen encoder outputs by evaluating each
game's observations as one batch. Do not store full legal-action feature arrays
that value fitting does not need. Save each full action trace, all pre-move
complete-state hashes and the shared collector's diagnostic final state.
Hidden state is available only for engine/replay verification; it never enters
the value input or the simple evaluator.

**Collect only development games first.** Train/select both heads and the simple
baseline, then seal all checkpoints, selection reports and the development
summary by hash. Generate the 512 protected test games only after that lock.
This order prevents test outcomes from affecting selection. Test evaluation
happens once with those sealed models; verification may reproduce the same fixed
predictions without creating a new comparison or selecting a different model.

## Neural fits and validation selection

Both conditions copy the same original 035 model, freeze all 633,089
policy-affecting parameters, and train only its 33,025-parameter value head.
Initialize its last linear layer to zero, retaining the same inherited first
linear layer in both arms. Thus step zero is the neutral predictor and is an
eligible validation-selected checkpoint. Do not reuse 142's failed head.

Use AdamW LR .001, weight decay .0001, batch 128, MSE on W/D/L, gradient norm
cap 1. Both runs use training seed 143010 and **at most 4,096 optimizer updates**.
For each batch sample games uniformly with replacement, then one position
uniformly from each sampled game. This balances games of different lengths.
The maximum number of updates and validation opportunities is matched; it is
not a matched-epoch comparison. Actual steps can differ because the same
stopping rule applies independently. Report the distinct games and positions
seen by each selected checkpoint as well as the available corpus size.

Validate at step zero and every 128 updates. Validation metric is the mean of
per-game position MSEs, giving every valid game equal weight. Select an update
only if it improves the best validation error by more than .001. Keep the earlier
checkpoint on ties or smaller changes. Stop after four consecutive checks without
such improvement, or at update 4,096, and restore the selected checkpoint.
Persist every validation observation and improved head before proceeding.
Persist complete final diagnostics even if step zero remains selected. A lack
of validation improvement does not suppress the preplanned protected evaluation.
Nonfinite loss/gradient or changed policy parameters stops the experiment.

## Simple reference

Fit a zero-intercept ridge regression to actor-minus-opponent public features:
score/15; each of five permanent color bonuses/7; each of five colored-token
counts/7; gold/5; purchased-card count/20; reserve count/3; noble count/3. These
are 15 antisymmetric features with no card IDs, hidden cards, outcomes or future
information. The coefficient vector minimizes whole-game-balanced squared error
plus lambda times squared coefficient norm, using the large training pool.
Choose lambda from the fixed grid [.001, .01, .1, 1, 10] by the same 512-game
validation metric, keeping the earliest grid entry on ties. Clip predictions
to [-1,1] for validation and test, matching the neural head's output range.
Dense linear algebra uses Torch float64 to avoid this host's incompatible
simultaneous NumPy/Torch OpenMP runtimes; no environment workaround is enabled.

## Protected statistics and decision

For each valid test game and each model, average squared prediction error across
that game's positions. Average those 512 game means for the reported MSE. The
neutral predictor is exactly zero; its error accounts for actual draws. Models
are compared on identical test positions/games. Bootstrap 50,000 resamples of
complete games, with the same resampled indices for all models, using NumPy
default_rng seed 143999 and percentile 95% intervals.

Primary contrast: large minus neutral. Secondary contrasts: large minus small,
large minus simple, simple minus neutral. Secondary intervals are unadjusted
diagnostics, not a family-wide claim. Also report descriptive errors for turns
0--19, 20--39 and 40 onward; no subgroup selects a model or determines success.

Integrity and natural test completion take precedence over all statistical flags.
For a valid test:

- Large-minus-neutral CI upper bound below zero establishes detectable value
  improvement. Call it material only if its point MSE is also at least 5% below
  the neutral predictor.
- Large-minus-small CI upper below zero supports a data-scale benefit under this
  training/selection recipe. Report it separately from absolute prediction quality.
- Material primary improvement with or without a supported scale benefit receives
  the corresponding distinct classification. A smaller positive primary result is
  reported below the material threshold.
- If the primary CI lower is above zero, report regression. Otherwise improvement
  is not established. A step-zero selection is allowed to yield an exact neutral
  result. There is no outcome-driven extension or alternative checkpoint rescue.

Comparison with the simple reference helps interpret representation/complexity,
but does not by itself isolate those mechanisms. Offline accuracy is not playing
strength: even success would only justify proposing a fresh test of whether a
value model improves actual search decisions. No such run is included here.

## Execution and independent audit

Use eight local CPU workers, Torch one thread each. Preserve all 271 dependencies
frozen by 142. Freeze every current source Python file, runner, tests, verifier,
this protocol, configuration, software versions, original checkpoint and old
manifest hashes before protected collection. Per-game records bind the frozen
manifest; tensors and JSON are saved atomically. Sealed collection summaries and
model locks detect changed artifacts. An incomplete fit must be inspected, not
silently restarted. Save failure diagnostics and do not launch another recipe.

Run the full suite and independent off-schedule full-game probes first. Probes
use the separate 1,439,000,000 range and do not enter the experiment. Preflight
requires an estimated serial collection-plus-audit time below 24 hours and corpus
size below 20 GB; these are engineering gates, not outcome selection.

The verifier independently replays all 5,120 games using the original 035 scorer,
reproduces every behavior RNG choice, legal move and state hash, checks endpoints,
and reconstructs every observation, frozen latent, public feature and outcome.
It checks the validation stopping/selection history, selected head tensors,
unchanged policy, ridge normal equations and validation losses. It recomputes
all sealed test predictions and paired statistics. It uses the repository's
engine; it is not an independent implementation of Splendor's rules. No partial
test errors, outcomes or comparisons are inspected while collection is running.
