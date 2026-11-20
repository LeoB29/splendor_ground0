# Preparation: distinguish game diversity from update scope

6 November 2026. **Design only. No models, fitting, new games or assessment have
been launched.** The user authorized moving directly to the recommended next
step: preparing a bounded learning-study design. The previous collector study is
closed with separately disclosed recovered final verification. Larger-data work
remains deferred.

## Recommendation and question

Run one small diagnostic on the **existing, accepted corrected-rules corpus**
before paying for another source-game collection. Compare more independent games
with fewer positions per game against fewer games with more positions, at equal
training-record and optimizer-update budgets. Cross that comparison with the two
already implemented training scopes: joint network training and policy-head-only
training.

The joint update scored 49.125% against its parent (paired 95% interval
44.50–53.75%); the policy-head-only update scored 50.375% (46.00–54.75%). These
different fresh assessments establish neither a useful gain nor that one update
scope is stronger. The saved postmortem found a small held-out policy-loss gain,
uncertain value changes and a nontrivial weight update. It did not diagnose data
scarcity, overfitting, value harm or gradient interference as the cause.

This diagnostic asks which explanation deserves the next investment. It tests
**training scope**, not network size. It cannot establish that collecting beyond
192 games will help or that an offline improvement produces stronger play.

## Four cells, two predetermined repetitions

| Training cell | Independent training games | Unique positions per game | Unique records | Trainable parameters |
|---|---:|---:|---:|---|
| Dense / joint | 64 | 48 | 3,072 | Shared network and both heads: 4,780,883 |
| Broad / joint | 192 | 16 | 3,072 | Shared network and both heads: 4,780,883 |
| Dense / policy only | 64 | 48 | 3,072 | Policy weight and bias only: 20,817 |
| Broad / policy only | 192 | 16 | 3,072 | Policy weight and bias only: 20,817 |

Use two prespecified sampling/order repetitions: **eight scientific fits total**.
Each starts independently from the pristine original EF8A checkpoint
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
Never initialize from an unpromoted successor. Architecture, float32 precision,
input representation, legal masks and target meanings stay fixed.

The accepted 192 training games contain 10,660 eligible roots: minimum 48,
median 56, maximum 66 per game. The proposed quotas therefore need no repeated
position within a selected dataset, replacement game or outcome-based exclusion.
The current collector benchmark's 64 engineering trajectories, old benchmark
games, qualification fixtures and historical/unlabelled corpora are excluded.

### Outcome-blind, nested selection

Use the fixed domain `splendorust_learning_diagnostic_20261106_v1`. Before any
scientific fitting, the implementation must materialize and freeze the complete
source-game identities, root identities/hashes, split assignments and batch
orders. The design below is the selection rule, not an executable freeze.

1. Authenticate the accepted corpus and the original 192 training game IDs. Sort
   those canonical IDs by SHA256 of the domain and game ID, using a specified
   canonical encoding. Repetition 0 uses the first 64 for its dense arm;
   repetition 1 uses the next 64. These two dense groups are disjoint. Both broad
   arms use all 192 training games. The remaining 64 are training games in the
   broad arms, not an extra held-out set.
2. Within each game, order eligible roots by original ply and divide that list
   at floor(n/3) and floor(2n/3). These are relative trajectory thirds, not labels
   inferred from scores or eventual winners. Each third has at least 16 roots.
3. For each repetition and game, rank roots within each third by a separate
   domain-separated hash of repetition, game ID and original root ID. The dense
   pool takes 16 per third: 48 unique roots. Construct this pool for every game.
4. The sparse selection is nested in that pool: take 6 roots from one third and
   5 from each other third. Rotate the six-root third by canonical game rank
   plus repetition modulo 3. Across 192 games this gives exactly 1,024 records
   from each third, matching the dense arm's 64 × 16 per third.
5. Use identical selected roots and batch order for the two training scopes
   within each breadth/repetition cell. No selection may read losses, winners,
   target confidence, gradients, held-out predictions or previous assessments.

A missing identity, changed count or infeasible quota is a preparation/qualification
error to resolve before freeze, not permission to substitute games or change
quotas after fitting. Repetitions vary the predetermined root sampling and batch
order; they do not create another 192 independent source games.

## Fixed fitting recipe and what changes relative to the old pilot

Keep the accepted optimizer and loss conventions: AdamW learning rate 1e-5,
weight decay 1e-4, betas 0.9/0.999, epsilon 1e-8, batch 32, gradient norm cap 1,
float32 and the qualified CPU fitting route. Use **668 full optimizer steps** in
each fit. Each pass has 96 batches; cycle through independently predetermined
permutations until the fixed final step. This is 21,376 record presentations,
approximately 6.96 passes over 3,072 unique records. It is **not** the old
two-epoch/10,660-record recipe and must not be described as its direct reproduction.
Conclusions apply to this fixed exposure budget. A null breadth effect would not
rule out diversity gains under a different number of updates or passes.

The objective remains masked policy cross-entropy plus mean squared error over
the two canonical signed value outputs, with the original equal loss weights.
Policy targets are genuine search visit distributions; value targets are the
official natural game outcomes. Keep all qualified legal Main channels in the
mask, including zero-target legal channels. All selected games have equal record
counts within an arm, so mean record loss also gives equal total weight to each
game. No hidden reweighting by game length or outcome is introduced.

For policy-only fits, every shared and value tensor remains byte-identical to
the parent, and value outputs must remain identical on the fixed evaluation
inputs. The value-loss term is constant with respect to trainable parameters.
Joint fits update the shared network and both heads. This contrast changes the
trainable parameter set; it does not isolate a causal gradient-conflict mechanism.

The primary checkpoint is always step 668. Save step 334 for a predeclared
diagnostic trajectory and step 667 with optimizer/RNG state for one exact final-step
reproduction. No checkpoint selection, adaptive early stopping, learning-rate
sweep, best-of-eight promotion or follow-on generation is allowed.

Order the fits prospectively: repetition 0 dense/joint, broad/joint, dense/policy,
broad/policy; repetition 1 reverses that order. Each is a fresh owned worker with
one active fitter. Timing is a cost measurement, not a competition between fits.

## Measurements and analysis

Evaluate the **fixed parent and fixed checkpoint models**, rather than comparing
online training averages computed under changing weights.

- At steps 0, 334 and 668, report mean policy cross-entropy and value MSE on the
  selected training records and on all 64 existing held-out games/3,557 roots.
  Report the original 32-game validation and 32-game offline sets separately as
  well as their prospectively defined equal-game pooled summary. All have already
  been opened; every result is exploratory, not fresh confirmation.
- Report per-game losses, parent-relative changes, teacher-top-action agreement,
  policy-argmax changes, signed-value shifts and parameter-group displacement.
  Count finite-gradient and clipping events. These are diagnostics, not strength
  surrogates or new success criteria for the historical pilots.
- On disposable copies of the saved checkpoints, use four predetermined training
  batches per fit/checkpoint for bounded gradient probes. For joint models,
  measure policy and value gradient norms and their shared-parameter alignment;
  for policy-only models, measure the trainable policy gradient. No probe performs
  an optimizer step or alters the scientific training trajectory. Record zero
  norms explicitly and do not invent a cosine value when it is undefined.
  The fixed inventory is 144 per-loss backward probes: 96 for the four joint
  fits and 48 for the four policy-only fits, each at three checkpoints and four
  batches. No additional probes are scheduled. A negative alignment is a signal to investigate, not proof that
  interference caused weaker play.

Primary comparisons are broad minus dense policy loss, separately for joint and
policy-only scope, and broad minus dense value loss for joint scope. Show both
repetitions, both held-out splits and their predeclared mean. Scope differences
and breadth-by-scope interactions are secondary. Policy-only value equality is
a correctness requirement, not an improvement endpoint.

Use 20,000 paired whole-game bootstrap draws with seed 20261106. Apply the same
drawn game indices to all cells/repetitions. Report nominal 95% intervals as
**exploratory uncertainty**, with no family-wide significance or promotion claim.
Do not treat positions, repeated predictions or the eight models as additional
independent held-out games. Report absolute loss changes; relative percentages
use the ratio of equal-game mean losses, not averages of unstable per-game ratios.

Complete all eight fits and the fixed analyses after valid qualification, even
if early losses are disappointing. Never change the plan after opening a cell's
results. A technically valid negative or mixed result ends this diagnostic.

## Qualification, reproduction and limits for a future execution

This document does not qualify a new entry point. Before execution, use separate
versioned source and an exact source/data/configuration freeze. Follow the
[corrected-rules policy](future_training_rules_20261021.md), including one
`verify_runtime_sources()` call at every fitter, evaluator and probe-worker
startup, even when the job does not construct an environment.

Bind ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and
turn contract `atomic_prereveal_completed_turn_v1` throughout data and checkpoint
metadata. Keep EF8A's historical warm-start origin distinct from the accepted
corrected source corpus. No default historical decoder or dataset loader is an
approved replacement for the qualified route.

Qualify the actual selected-subset loader and collator on all six corrected card
faces, noble/final-round cases, gold/returns, private reserves, canonical value
signs and bad/missing identity/hash rejection. Test exact quotas, uniqueness,
phase counts, sparse/dense nesting, split exclusion and invariance of selection
to permuted outcomes. A manifest declaration or helper self-check is insufficient.

Use one disposable engineering optimizer step per scope before the scientific
fits. Prove the intended trainable set, finite genuine updates, policy-only frozen
tensor/value preservation and save/load output parity. Reload pristine EF8A for
every scientific fit. Reproduce the final update of each fit from its saved
step-667 optimizer/RNG state and require the established exact CPU parameter/
state agreement. Maximum optimizer calls: 5,344 scientific + 2 engineering +
8 disposable reproductions = 5,354. No new games or search rollouts are needed.

Proposed actual ceilings: **12 hours overall, 8 hours model work, eight scientific
fits, 1 hour qualification, 1 hour per fit, 2 hours evaluation/probes and 4 hours
saved audit**, all clipped by the global clocks. One fitter at a time, at most
two model instances during parity/probe checks, 4 GiB RSS, 4 GiB new evidence and
a 1 GiB actual free-disk floor. Existing corpus bytes remain read-only inputs,
not duplicated. Fit stage forecasts are advisory; no available-RAM reserve,
stability wait or resource forecast may refuse admission or stop a valid run.

Rough work estimate: four joint and four policy-only fits at the historical
complete-stage rates would take about 1.7 hours. Those stages took 11m33.5s and
13m51.3s respectively, including saved verification after worker exit; the fitter
workers themselves took 7m46.2s and 7m11.3s. Allow roughly **2–4 hours for fitting,
qualification, evaluation and probes**, plus final audit/reporting. This is an
estimate, not an execution gate; the actual ceilings are deliberately wider.

Stop on invalid rules/data, nonfinite calculations, failed exact reproduction,
an actual resource/deadline violation or an ownership failure. Preserve every
receipt and checkpoint; no silent retry, replacement or clock reset. Final
acceptance requires independent source/evidence checks, actual process exits and
cleanup. A saved completion file alone is insufficient. No scheduled monitor is
created by this design.

## How the result changes the next recommendation

| Observed pattern | What it would support | What it would not establish |
|---|---|---|
| Broader coverage helps consistently across repetitions/splits | Prepare one fresh game-diversity study with a fixed recipe | That expanding beyond 192 games helps, or that the player is stronger |
| Joint/policy scope changes transfer substantially | Prioritize one controlled update-scope/objective question | That the network needs more parameters, or gradient conflict is causal |
| Training loss improves but held-out loss does not | Investigate generalization/regularization and fresh validation | A precise cause of overfitting or an automatic mandate for more data |
| Training loss barely changes despite the fixed updates | Investigate optimization and target fit with the recorded gradients | That the optimizer failed, based on clipping alone |
| Mixed or small effects with broad uncertainty | Report the limit of this small diagnostic and choose one next question | Equivalence, a winner among eight models, or permission to enlarge automatically |

No tiny offline gain triggers an automatic 400-game assessment. A future strength
study must freeze one recipe and its selection rule prospectively, use genuinely
fresh game-level evaluation, fix the checkpoint, qualify the player and then run
six excluded operational games plus 400 fresh games/200 swapped-seat pairs at a
matched search budget. The established useful-gain criteria stay unchanged:
score at least 55%, paired lower 95% bound above 50%, at least 50% from each seat,
and the existing official/conservative scoring and verification safeguards.

## Collection cost if a later study is justified

The measured CPU8 reference is 51.149 collection games/hour and 36.808 games/hour
including startup, warmups, selected replays and batch audits. GPU16's corresponding
rates are 55.371 and 38.927; it remains a qualified alternative, without meeting
the frozen consistency criterion. The following are CPU8 extrapolations using
the current complete recording format, not promises or admission gates.

| Future source allocation | Verified collection | Additional final audit estimate | Collection evidence estimate |
|---|---:|---:|---:|
| 192 training + 64 held out | 6.95 h | ~0.58 h | 15.75 GiB |
| 384 training + 64 held out | 12.17 h | ~1.02 h | 27.57 GiB |
| 512 training + 64 held out | 15.65 h | ~1.31 h | 35.45 GiB |
| 1,024 training + 64 held out | 29.56 h | ~2.47 h | 66.95 GiB |

The final-audit column scales the recovered 523.625-second audit of 64 mixed-route
games, including some fixed checks. It is a rough anchor; CPU-only auditing can
differ. Add source qualification, fitting, strength assessment and its evidence. The
original joint pilot's 400-game assessment took 7h30m, plus substantial verification;
this is a historical anchor, not a new guaranteed runtime. The current collector's
CPU evidence averaged 63.02 MiB per game. Fixed study/qualification files added
about 0.15 GiB. Around 26.3 GiB was free when preparation began: even 384 new training
games plus fresh held-out games exceed that before assessment and reserves. A
larger proposal must therefore identify available storage or separately qualify
lossless recording changes; this preparation deletes or compresses nothing.

Keeping 10,660 retained training positions while increasing game diversity can
hold fitting work roughly constant. It does not avoid completing and verifying
every teacher game. No larger collection, GPU tuning, external-opponent testing,
six-world work or new training job is launched by this design.

## Evidence

- Accepted joint assessment: `outputs/splendorust_selfplay_pilot_resume2_20261103_01/assessment/main_analysis.json`.
- Learning postmortem: `outputs/splendorust_selfplay_postmortem_20261104_01/report.json`.
- Accepted policy-only assessment: `outputs/splendorust_policy_head_report_20261105_01/summary.json`.
- Source counts and scope: `outputs/splendorust_policy_head_20261104_01/fit/fit_result.json`.
- Measured collection costs and recovered acceptance: `outputs/splendorust_gpu_collector_report_20261105_01/results.md` and `closeout.json`.
- Preparation evidence bindings and cost derivation: `outputs/splendorust_learning_design_20261106_01/evidence.json`.
