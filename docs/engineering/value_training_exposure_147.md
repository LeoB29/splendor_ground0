# Experiment 147: fixed longer training exposure over the 146 value recipe

Owner authorization: "ok great, can you do it" on 2026-09-08, following the
proposal in `docs/strategy/next_after_146.md` and the conditional runtime estimate.
This includes the bounded prediction experiment and its conditional arena.
No automatic successor, promotion or sample extension. Preserve the dirty tree.

## Question and controls

Does a fixed longer training schedule improve value prediction enough to open
the gameplay gate, and is there an advantage for training the value encoder?
146 selected update 128 after 16,384 sampled positions (15,855 unique), then
stopped both fits at 640. Limited selected exposure is not proof of undertraining.
This experiment varies the schedule while retaining the architecture, optimizer,
sampling and selection rule. Existing 146 short models stay frozen controls.

Four cells: frozen encoder / short, trainable encoder / short (the original 146
selected models), and frozen encoder / long, trainable encoder / long (new fits).
Neutral zero and the frozen 143 public evaluator are additional references.
The only candidate for conditional gameplay is long-schedule trainable encoder.

## Exact training contract

Reuse 143 development only: training IDs [0,4096), validation [4096,4608).
Validate every sealed JSON/tensor identity and natural endpoint. Bind prior
146 dependencies, inputs, selected files and completion audit before launch.
No old protected examples or labels enter fitting or checkpoint selection.

Both new value models copy the original 035 encoder and 256 -> 128 GELU -> 1
value head, remove final Tanh, zero the last linear layer, and add the residual
to the frozen public baseline with final [-1,1] clipping. Freeze the copied
encoder in the control only. The actual 035 policy stays in its separate,
unchanged model; canonical priors retain the complete legal action set.

Same precision contract as corrected 146: training/evaluation use float32
observation/features, float32 residual computation converted to float64, and
clipped float64 public matrix-vector predictions. Deployment reuses the exact
PublicValueInference scalar before adding the residual, avoiding scalar/batched
last-bit differences at zero initialization. No private information is added.

AdamW lr .0003, weight decay .0001, gradient clip norm 1, batch 128, seed
146010, one Torch thread. Same uniform-game then uniform-position sampling,
actor-relative final W/D/L squared loss over all training turns. Each new fit
must complete exactly **16,384 updates**, 2,097,152 sampled positions. Sampling
is with replacement: about eight corpus-equivalents, not guaranteed full epochs.
Record distinct games/positions at every validation and the selected dose.

Validation every 128 updates, game-balanced early/middle MSE (turns <40), step
zero eligible. Select a checkpoint only when loss improves the previously
selected loss by more than .001. Keep the earlier checkpoint otherwise. The
four-stale-validation termination is disabled; no other selection change.
Early checkpoints can remain selected after the longer schedule. Do not force
the final checkpoint to become the candidate. Save it separately as ineligible
diagnostic evidence, with selected/terminal training and validation errors.

At update 640, require exact reproduction of all six 146 validation-history
entries (including step zero), the selected step, and all selected tensors.
The batch stream and update operations are unchanged. This prefix check does
not claim that 146 saved every intermediate optimizer state; those did not
exist. The independent audit reruns each entire new 16,384-update fit and
checks history, selected and final tensors, plus coverage counted separately.
On any prefix/integrity failure, persist evidence and stop before fresh tests.

## Selection lock and fresh protected prediction

Seal both new selected models, ineligible terminal diagnostics, reports and
cost measurements, and hashes of both original 146 short models, before
generating protected games. No candidate substitutions after the lock.

512 new games, seeds **1470000000..1470000511**, eight workers, one Torch thread
each. Same 143 exact raw-035 behavior: epsilon .2 in first 20 turns, top three
temperature .5, RNG offset 143003, greedy afterwards. Max turns 400, repetition
4, no-progress 60. Completed and forced-pass-deadlock are natural. No cutoff
outcome becomes a training/test win; save invalid games without replacement.

Evaluate all six models on the same positions. Primary is per-game mean MSE
on turns <40, then equal mean across games. Report overall, late >=40, and
descriptive early <20 / middle 20..39. Paired whole-game bootstrap, 50,000
resamples, seed **147999**, two-sided percentile 95% intervals.

Prespecified contrasts (negative favors the first named model):

- Trainable-long minus public.
- Trainable-long minus frozen-long.
- Trainable-long minus trainable-short (exposure effect in trainable arm).
- Frozen-long minus frozen-short (exposure effect in frozen arm).
- Trainable-short minus frozen-short (short-arm trainability contrast).
- Interaction: (trainable-long minus trainable-short) minus (frozen-long minus
  frozen-short). This is secondary; it is not an alternative success criterion.

Every condition below must pass to open gameplay:

- All 512 games natural; all contribute to early/middle, overall and late.
- Early/middle trainable-long MSE at least 3% below public, difference CI upper <0.
- Early/middle trainable-long at least 1% below frozen-long, difference CI upper <0.
- Early/middle trainable-long beats trainable-short: difference CI upper <0.
  No separate practical margin is imposed on this additional exposure check.
- Overall and late difference CI upper <.01 absolute MSE against each of public,
  frozen-long and trainable-short. This is a declared tolerance, not equality.
- Combined policy/value inference trainable/public median <=1.5x and p95 <=2x.
  Before test collection, benchmark every position from two fixed probe games,
  three repetitions with alternating order, two warmup calls per model, and
  record all timings. All three models' policy logits must remain identical.

All conditions are conjunctive. A successful secondary contrast cannot rescue
a failed primary gate. Keep diagnostics even on failure. No retuning or retries.

## Conditional gameplay (only after independent prediction audit passes)

800 games / 400 seat pairs, seeds **1471000000..1471000399**. Long-trainable
residual search versus frozen public-value search; identical 035 priors and
real PUCT. Each receives .4 seconds/move including encoding/root expansion,
minimum one completed simulation, no new simulation started after deadline.
No equal-simulation ceiling or padding. c_puct 1.5, max depth 96, zero noise and
exploration temperature, original canonical visit/Q/prior selection. Search
seed = game_seed*10000 + player*1000 + turn +147013. Eight single-thread workers.

All games natural. Both mean decision times <=.44 seconds, p95 <=.50, every
search >=.4 seconds, mean time ratio in [.95,1.05]. Primary success: score share
>=55% and seat-paired bootstrap CI95 lower >50%. 50,000 resamples, seed 147998.
Otherwise report invalidity, regression or uncertainty without extension.

Independently reconstruct every trajectory, final state/outcome, original 035
prior, scalar, visit totals, selected edge and timing. Reproduce first/last
search of each arm in every game at the recorded simulation count. Independently
recompute paired statistics. No champion promotion, even if this duel succeeds.

## Preflight, freezing and stop rules

Focused tests cover unchanged recipe, fixed-dose continuation through stale
validation, preservation of selection threshold, exact prefix rejection,
short-control identity, six-model contrast/interaction arithmetic and each gate.
Run full suite before protected launch. Two raw-behavior probes use seeds
1479000000/1479000001; reconstruct all behavior/features/labels and exact initial
predictions. Two zero-residual arena probes use **1479100000**, swapping seats,
with all-root and four-search/game audit. Probes are engineering checks only.

Freeze this document, implementation, verifiers, tests and inputs before fitting
and protected outcomes. No changes to prior frozen source or output. Never call
old dynamic identity inventories after adding source; validate their explicit
manifest lists instead. All dense linear algebra uses Torch on this Windows
host to avoid the previously observed duplicate NumPy/Torch OpenMP conflict.

Independent prediction audit must finish before any arena: reconstruct all
512 behaviors, features, latents, outcomes, six-model predictions and contrasts;
reproduce both complete new fits, prefix checks, selected/final tensors and dose.
If any gate fails, stop with a complete recorded negative result and no protected
gameplay. No alternate schedule, learning-rate search, holdout extension,
post-hoc checkpoint substitution, new targets or automatic successor.
