# Next after the verified learned128 result

2026-09-24. Scientific review and recommendation only. No experiment, training,
simulation, implementation, successor process or monitoring was started. Three
independent read-only reviews challenged the learning design, the evaluation
budget and competing research directions. Historical and frozen files remain
unchanged; this is a new strategy note.

## Recommendation

Test **one more learning generation, directly inside the 128-world planner**.
Train from the current three fixed 161 models using the newly improved planner's
saved decisions, with retained old examples and a matched extra-training control.
First test whether this improves actual planning strength. Only if that succeeds,
run the comparison that tests whether the new examples added value beyond more
practice on the old examples. Freeze this entire contingency before fitting.

Confidence is moderate that this is the highest-value next scientific question;
the outcome is uncertain. The recommendation is a bounded staged experiment,
not an open-ended instruction to keep training until something wins.

## What the latest result establishes

The final learned128 result is 263 wins, 3 draws and 136 losses in 402 games:
65.7960% score counting draws as half, paired 95% interval 60.9453--70.5224%.
Both frozen success criteria passed. All three fixed-fit point estimates were
positive; the pooled interval does not establish each fit separately. Every
actual game was checked, and the prescribed sampled continuation audit passed.
That audit is not exhaustive inner replay or an independently implemented rules
engine. See `outputs/learned_proposal_128/report.md` and `final_audit.json`.

Both players used 128 worlds, the same first three proposals and greedy035
continuations. Replacing the fourth random proposal with the 161 model's proposal
improved this controlled matchup. It did not require a larger simulation budget.
Reached-state decision timings were similar, but this is not a controlled
equal-wall-time comparison.

The learning lineage matters: the original 128-world teacher supplied the 161
training targets; a model trained on those targets now supplies proposals that
make that teacher stronger. We therefore have **one demonstrated
search -> learning -> stronger search cycle**. The unproved step is whether
repeating this process gives another improvement.

The result weakens the explanation that learned proposals only compensate for
the noisier 32-world search. It does not measure an interaction between sampling
budget and proposal quality, or show that their benefits add or multiply.
The learned-versus-original-ranked-fourth attribution was established at 32
worlds; it was not separately established at 128. These results do not establish
strength against independent strong opponents, humans, or optimal play.

## What the accumulated evidence narrows

| Evidence | Consequence for the next decision |
| --- | --- |
| 160: original128 beats original32, 65.5% score | More careful evaluation helped this fixed candidate set. This does not establish that 128 is optimal. |
| 161: 128-label training beats matched 32-label training, 52.54% | Better search can yield useful supervision. Label agreement alone did not rank playing strength correctly. |
| 162: refreshed data versus matched old-data training, 50.42% with interval spanning parity | New data and lower validation loss did not establish an advantage in the old fast-selector role. Another uncontrolled training pass is inadequate evidence. |
| Generalization panel after162: no demonstrated refresh advantage | The narrow fast-player improvement did not establish a broader benefit on that related local panel. |
| Learned proposals: 64.43% versus original32; 56.97% versus the original-ranked-fourth control | The learned ranking contributed useful candidates beyond that alternative at 32 worlds. |
| Latest learned128: 65.80% versus original128 | The proposal role also improves the stronger teacher at its existing budget. This is the relevant deployment role for the next learner. |

Earlier failures remain specific. Experiment134 strongly rejected its tested
expanded-root recipe; Experiment137 did not pass its chance-coverage mechanism
gate. Neither rules out planning in general. The tested AlphaZero, value, feature
and loss variants did not establish a dependable route under their local budgets;
they do not prove that broader versions cannot work. Experiment158's stronger
direct player did not show a planning benefit as the continuation controller,
but its audit was owner-stopped and that result remains provisional. Experiment159
did not resolve that mismatch sufficiently to justify making it the next change.

The best-supported positive hypothesis is now role-specific: learning can help
search by proposing moves worth examining. The most important competing
hypothesis is that this first benefit saturates: subsequent restricted-choice
labels might mainly reinforce existing preferences, or fail to improve rankings
of moves the teacher did not examine. The next experiment distinguishes useful
continued progress from that possibility for one fixed update recipe.

## Proposed learning design

- Use all **11,951 candidate-side learned128 decisions**, including losses,
  ties, private-reserve positions and small-support roots. These come from
  201 starting-seed pairs, not 11,951 independent games. The opposing player's
  11,951 roots are different reached states and are not matched old-teacher labels.
- Retain the original160 training/validation split used in161. Assign the new
  67 source pairs per fit to 53 training and 14 validation pairs by a prospective
  identity-based rule: 159 training and 42 validation pairs in total. Keep every
  root and both seats from a source pair together. Pool all three source fits;
  do not select the best teacher fit from its arena score.
- Warm-start the exact three selected161 checkpoints, updates563/488/563.
  Train a treatment and a control from each initializer: six fixed fits. Do not
  restart from035, select only the strongest fit, or change model architecture.
- Treatment receives the retained old128 examples plus the new128 examples.
  Use each combined training row once per epoch. In the control, replace each
  new-data slot with an old training example drawn independently of labels.
  Match optimizer updates, batch sizes and row exposures exactly. The new
  examples will constitute approximately half the treatment pool, rather than
  the 16.1% used in162.
- Retain the established four-pass, batch64, learning-rate1e-5, fresh-AdamW
  recipe and offered-canonical-group cross-entropy. Score lawful legal inputs;
  do not treat unoffered moves as bad moves. Keep value-head parameters frozen.
  Use the same held-out validation panel, pair-mean loss rule, checkpoint
  schedule and earliest-tie rule for both arms, with the initial checkpoint
  eligible. No checkpoint selection from arena results.

The targets are the teacher's finite-sample preferences within its offered
actions, not optimal-action certificates. The new corpus changes reached states,
offered support and supervision together. Matched old-data training controls for
extra practice; it does not isolate teacher-label quality on identical states.

## Proposed evaluation and stopping rules

Use the same proposal role as the latest successful study: only the fourth-slot
scorer changes. Retain the original first three proposals, 128 observer-legal
worlds, greedy035 continuations, terminal-WDL scoring and tie rules. Keep all
three fits, both seats and fresh evaluation namespaces. Evaluation states and
seeds must not enter fitting or checkpoint selection.

1. **Stage A: 402 games against the unchanged learned128 configuration.**
   Assign 67 fresh seed pairs to each fit, with both seats. Require observed
   score at least55% and the paired95% interval's lower bound above50%.
   This asks whether the new learning generation improves actual planner play.
2. **Stage B, only after a verified Stage A pass: 402 fresh games against the
   matched old-data training control.** Use a disjoint schedule, again all three
   fits and both seats. Require the paired95% lower bound above50%.
   This asks whether the new corpus helps beyond simply doing more training.

Freeze both schedules, criteria, analysis, verification and contingency before
training. Use the established within-fit seed-pair analysis. Do not extend an
inconclusive stage, substitute a different fit, or inspect intermediate win rates
to decide the sample size. Report Stage A whatever its outcome; report Stage B
as conditional on the first stage passing. Overall success requires both gates.
If all three selected treatment checkpoints are exactly the initial anchors,
record that the recipe selected no update and stop before an identical-player
arena. Do not prune individual unchanged fits or infer global behavioral identity
from agreement on a small sample of positions.

Use the established sampled-verification policy with every actual game checked,
deterministic per-root world samples and selected full stress roots, together
with frozen provenance and independent fitting reproduction. Verification should
be proportional to the new code and data path; avoid automatically returning to
exhaustive replay of every continuation. Resolve any integrity failure before
interpreting strength or allowing the contingent second stage.

If Stage A fails, stop this fixed update and investigate its retained diagnostics;
do not declare iterative learning impossible. If A passes but B fails, practical
improvement against the anchor is still evidence, but added value from the new
corpus remains unestablished. If both pass, that supports a second useful learning
generation and the whole feedback update, without proving indefinite improvement.

## Why evaluate at128 rather than32

Two 32-world comparisons would cost much less, roughly36--44 hours plus learning
and preparation. They would be legitimate tests of 32-world deployment. But a
failure there would not rule out a benefit at128, and a success would not establish
a stronger128 teacher. The current question is whether the strongest validated
teacher configuration can improve again. Direct128 evaluation avoids making a
cheaper proxy study a prerequisite to answering it.

The last full 402-game128 study took70h20m including its probe and sampled audit;
main collection was63.084h and main verification4.524h. Budget roughly **three
days for Stage A, up to six days if both stages run**, plus preparation and
fitting. Extraction and fitting are likely hours rather than days, but a fresh
operational probe must check this forecast. Different reached positions can
change runtime. The recommendation accepts the direct endpoint's cost instead
of claiming a cheap definitive test exists.

## Alternatives and limits

- A broader benchmark remains necessary before broad-strength or superhuman
  claims. The existing related local panel is not a ready strong independent
  anchor, and another tournament against it is less informative about continued
  learning than this controlled generation.
- A raw-ranked-fourth control at128 would refine mechanism attribution. The
  positive attribution at32 and direct improvement at128 are sufficient to make
  another learning generation the next question; a further attribution arena
  need not block it.
- Same-state old-teacher relabeling could reuse the first three candidate/world
  outcomes and evaluate only a missing original fourth proposal. At most
  11,447*128 = 1,465,216 extra continuations would be needed, less when fourth
  proposals coincide. A rough throughput extrapolation is8--12 hours including
  checks and overhead. This would support cleaner comparison of teacher/support
  changes on identical states. It can wait until the feedback package shows
  value; opponent logs cannot substitute for this work.
- Batching had a measured benefit but missed its frozen adoption gate:1.245
  collection speedup against a1.25 threshold, despite passing the combined-time
  threshold. That result remains a miss. A separate bounded128/eight-worker
  integration study could be worthwhile engineering, but is not a prerequisite
  or a substitute for testing continued learning.
- More candidates, adaptive rollout allocation, changed continuations and new
  value objectives remain plausible. Combining any of them with this generation
  would obscure whether the now-supported learning route can repeat.

Preparation, if later requested, should create a new isolated tool/output tree,
preserve every existing dirty and frozen artifact, bind the protocol before
fitting, and use hidden independent workers. There is no new authorization to
launch in this review and no recurring monitoring has been created.

## Primary local sources

- `outputs/learned_proposal_128/report.md`, `result.json`, `final_audit.json`
- `docs/engineering/learned_proposal_128.md`
- `docs/strategy/next_after_proposal_attribution.md`
- `docs/engineering/teacher_budget_161.md`; `outputs/teacher_budget_161/report.md`
- `docs/engineering/student_refresh_162.md`; `outputs/student_refresh_162/report.md`
- `outputs/student_generalization_panel/report.md`
- `docs/engineering/learned_proposal_screen_result.md`
- `docs/engineering/learned_proposal_arena_result.md`
- `docs/engineering/learned_proposal_attribution_result.md`
- `docs/engineering/fast_selector_transfer_157_result.md`
- `docs/strategy/independent_superhuman_review_after_134_137.md`
- `docs/engineering/batching_benchmark_result.md`
