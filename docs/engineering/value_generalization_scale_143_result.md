# Experiment 143: more data helps slightly; public features are much stronger

2026-09-08. The experiment completed with valid data, unchanged policies and a
successful independent audit. **The larger neural value head generalizes better,
but its gain is small. The simple public score/resource evaluator is substantially
more accurate.** The prespecified primary decision is
`positive_value_improvement_below_material_threshold`.

All 5,120 games completed naturally, producing 326,204 positions. There were no
cutoffs, exclusions, replacements or forced-pass adjudications. Both neural fits
preserved every policy-affecting 035 tensor exactly. No search, gameplay arena,
policy update, installation or promotion ran; champions remain 084 and 035.

## Protected prediction results

The same 512 untouched test games were used for every model. Each game receives
equal weight after averaging squared outcome error across its positions. Lower
MSE is better. Five test games were draws, making neutral-predictor MSE .990234
rather than 1.000000.

| Evaluator | Test MSE | Error reduction versus neutral |
|---|---:|---:|
| Neutral prediction | 0.990234 | — |
| Neural head, 256 training games | 0.988903 | 0.13% |
| Neural head, 4,096 training games | 0.973134 | 1.73% |
| Simple public score/resource evaluator | 0.835858 | 15.59% |

The primary large-minus-neutral difference was **-0.017100**, paired whole-game
CI95 **[-0.025474, -0.008568]**. This detects improvement, but the 1.73% point
reduction misses the prespecified 5% practical threshold.

| Paired MSE difference | Mean | CI95 |
|---|---:|---:|
| Large minus small neural head | -0.015769 | [-0.023038, -0.008500] |
| Large neural head minus simple evaluator | +0.137276 | [+0.116631, +0.157452] |
| Simple evaluator minus neutral | -0.154376 | [-0.177431, -0.130757] |

These secondary intervals are unadjusted diagnostics. They support a data-scale
benefit under this training/selection recipe and a clear predictive advantage
for the public-feature reference. They are not measurements of win rate or a
family-wide statistical claim.

The descriptive phase breakdown is also informative: the large neural head's
early/middle/late MSEs were .992981/.991572/.936634, while the simple evaluator's
were .979266/.909708/.635634. Most predictive improvement occurs later in the
game. No phase result selected a model or changed the experiment's decision.

## Training and selection

The large pool contained 4,096 games and 260,904 positions. Its nested small
pool contained 256 games and 16,300 positions. Shared validation used 512 games
and 32,704 positions; protected test used 512 games and 32,596 positions. Test
games were generated only after all model choices and development artifacts
were locked by hash.

Both heads used the same initialization, 33,025 trainable parameters, optimizer,
maximum 4,096-update budget and validation rule. Minibatches sampled games first,
then positions. The same rule allowed different actual stopping times:

| Neural fit | Selected update | Updates completed | Selected train MSE | Selected validation MSE |
|---|---:|---:|---:|---:|
| Small | 128 | 640 | 0.966028 | 0.989925 |
| Large | 1,408 | 1,920 | 0.970283 | 0.971189 |

The selected small checkpoint had seen all 256 games and 10,333 distinct
positions; the large checkpoint had seen all 4,096 games and 129,860 distinct
positions. The available data and maximum optimizer budget were controlled;
actual updates differ through validation selection. This estimates the effect
of the data-scale training recipe, not a pure comparison at identical selected
update counts. Neither head used a protected test outcome for selection.

The simple evaluator used 15 actor-minus-opponent public score/resource features
and a ridge coefficient selected from the fixed grid. Validation selected .001;
its validation MSE was .836588, close to its .835858 test MSE. No card IDs,
hidden cards, future information or outcome labels entered its input.

## Scientific interpretation and next direction

More independent games and controlled validation selection prevented the severe
generalization failure seen in 142, but produced only modest neural improvement.
The larger head's train, validation and test errors are close; merely memorizing
the small corpus is no longer a sufficient explanation for its weak accuracy.

Useful value information is clearly available in basic public state. This makes
the current frozen policy representation and its value-learning setup stronger
bottleneck candidates than a blanket inability to learn game values. It does
not prove that the representation discarded information: input representation,
function class, regularization and optimization differ in the reference model.
Nor does it rule out more data or an independently trainable value representation.

The next useful proposal is a **fresh gameplay test of search using the frozen
public-feature evaluator**, with raw 035 as a competence reference and a fixed
decision budget. It would test whether the predictive gain helps actual choices
before investing in another full self-play learning cycle. The 143 test set is
now consumed; do not tune models against it or promote a checkpoint based on
these offline errors. No next experiment is automatically authorized.

## Execution, reporting recovery and verification

The original collection/fitting/test pipeline ran for 302.438 seconds on eight
CPU workers with Torch one thread each. It saved all 512 test prediction files,
then hit a final-report JSON error on a NumPy Boolean scalar. The original failure
and incomplete report are preserved. A reporting-only adapter reproduced the
frozen evaluator, required exact equality with **every saved prediction tensor**,
and converted scalar types to identical Python values for JSON. It did not
generate games, fit or select models, change statistics, or edit frozen source.
Details: `docs/engineering/value_generalization_scale_143_serialization_note.md`.

The independent audit subsequently reconstructed all 5,120 complete games using
the original 035 scorer, including every behavior RNG choice, legal action,
state hash, observation, latent, public input and outcome label. It checked
selection histories, restored head tensors, unchanged policy, ridge normal
equations and validation losses, all test predictions, and all paired statistics.
All 272 current frozen dependencies and all 271 old 142 dependencies matched.
This uses the repository engine, not an independent implementation of game rules.

The full suite passed 986/986 before launch; both additional serialization
recovery tests passed. Original session 97865, recovery 75658 and verification
19680 have exited. `final_status.json` records completion; the preserved
`failure.json` describes the recovered reporting error, not an active failure.
Do not rerun the main pipeline.

## Artifacts

- Protocol: `docs/engineering/value_generalization_scale_143.md`.
- Result: `outputs/value_generalization_scale_143/result.json`, SHA256
  `2bf923c2a15dfa62a19aa1cf1ec8f2f1756f46e0ed6c4da4773586d0d47a5098`.
- Verification: `outputs/value_generalization_scale_143/verification.json`, SHA256
  `7984744ffcf8cb656afbeaee435b46bcc2fb2f278c1661e26c569307bd71134a`.
- Reporting recovery SHA256:
  `8badf267fd9c1cd04d5931f015e6318f191ef885a92cfe975fa716e880b681fd`.
- Selection lock SHA256:
  `484a9ad799ee046274319f355cb63ee0be25b93d38657df584cd42fb3cad2986`.
- Large checkpoint SHA256:
  `5a2086914e8a820c20a87f2795f4ce8387f11e5f4aff71bc8d061b8d7a3d0b77`.
- Simple evaluator SHA256:
  `a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c`.
