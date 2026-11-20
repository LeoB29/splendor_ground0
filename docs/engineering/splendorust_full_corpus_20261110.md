# Full-corpus matched value-target learning screen

10 November 2026. The user explicitly authorized stopping the 96-update study
and implementing/running the recommended full-corpus comparison. That study
was stopped during offline evaluation: both small fits were saved, but no live
games started. Its source, partial evidence and cancellation receipts remain
preserved. It is not a scientific negative result and will not be resumed.

This is a separately versioned, qualified and frozen study. Original EF8A
remains the reference. No new self-play collection, rental, external-opponent
testing, six-world work or scheduled chat monitor is authorized by this plan.

## Scientific question

Does one full pass through the existing corrected training corpus produce a
useful learning gain, and does a mixed value target help compared with training
on terminal outcomes alone? The previous small updates used 3,072 positions.
Both new arms use all 57,022 eligible training positions from 1,024 games.

There are four new fits: terminal-only and mixed-value, each with two paired
fixed seeds. Every fit begins at pristine historical EF8A. Within each seed,
the two arms have exactly the same examples, ordering, batches, policy targets,
optimizer, architecture, precision and training exposure. The value target is
the only difference between arms. Neither arm starts from a preceding fit.

One pass means 1,781 batches of 32 and a final genuine batch of 30: 1,782 updates
per fit. No row is dropped, duplicated or padded. Average the last batch's loss
over its actual 30 examples. All policy/shared/value parameters are jointly
trained using the same accepted loss coefficients, AdamW LR 1e-5, weight decay
1e-4 and gradient clip 1, in float32. No learning-rate sweep, early stopping,
checkpoint selection or best-seed selection. This larger exposure may still
cause deterioration; the study does not assume more training is beneficial.

For the canonical actor, terminal-only uses official outcome z in {-1,0,1}.
Mixed-value uses t=(z+0.837*q)/1.837, q=2*r-1, where r is the authenticated saved
512-search visit-weighted root win credit in [0,1]. Targets are [z,-z] or [t,-t].
Do not substitute the chosen action's Q, unweighted action mean or raw network
prediction. Policy targets remain the original genuine root-visit distribution.

The coefficient comes from upstream commit
1828507b8d2f0e02bc03d102b9c6d29a54fa8d57, strategy_train.py SHA256
06caf53b28a58fb74fe0ab69378cabfd01b15dcdd9485c8cbbb0853cb5947c94 and data.py SHA256
2465fc6bd44cfc1e0792ea82b64d0c7cf4f82ab479b940c1d930afd06dd0d5f8. Only the value
formula is reproduced; other upstream optimization/policy settings differ.

Save checkpoints 0, 1781 and 1782. Reproduce the final partial-batch update exactly
for each fit, including model, optimizer, RNG, outputs, gradients and losses.
Maximum optimizer calls: 7,128 scientific, four final-update reproductions,
and two discarded qualification updates (one terminal/full32 and one mixed/30).
Both qualification routes use genuine accepted engineering positions spanning
wins, losses and draws. Qualification updates never enter a scientific model.

## Corrected data and reusable authenticated cache

Source games and their original acceptance remain immutable. Freeze exact source
corpus/selection/metadata before building a new cache. Use the corrected loader,
explicit legal lists and full capture/transition validation, authenticating the
source game, move, official outcome, actor orientation and search value.

Build compact encoded array shards and lightweight row identities once in an
owned model-free stage. An independent saved-data stage compares every cached
input/policy/outcome/search-value row with its accepted source. Only then seal
cache acceptance. Each trainer/evaluator rechecks artifact hashes, schema,
row inventory and corrected provenance before using memory-mapped arrays.
Materialize batches, not tens of thousands of Python feature lists. Reusing
the accepted cache saves repeated decoding; it does not bypass source checks.
The freeze specifies the cache recipe and inputs; future cache hashes are
authenticated after creation rather than invented before they exist.

Every new worker, including cache builder/auditor and fit-only jobs, calls
verify_runtime_sources() exactly once at startup. Bind ruleset
splendor_base_qualified_rules_20261018_v1, catalogue SHA256
db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312 and contract
atomic_prereveal_completed_turn_v1 in source, cache, checkpoint and live metadata.
EF8A remains an explicitly historical warm start. Never relabel old targets.
Existing pre-reveal choices and internal forced-pass conventions are unchanged.

Exercise the actual loader, new cache/batch route and trainer on all six corrected
faces, non-purchase/multiple/final-round nobles, payments/returns and private
reserves; reject bad rules identity, hashes, actor sign, nonfinite values,
malformed captures and wrong denominator/partial-batch shape. Preserve accepted
entry-point, graph, numerical, legal-support and hidden-world checks.

## Prediction diagnostics

Commit all four final checkpoints before any assessment prediction. The existing
64 validation and 64 test games have already been examined: these are reused
diagnostics, not fresh independent confirmation. Evaluate the original and all
four new models on every eligible held-out position. A fixed outcome-blind
3,072-position training panel, common to all five models, supplies descriptive
training predictions; all 57,022 positions still contribute to fitting.

Compare terminal-outcome MSE, policy cross-entropy, search-value error, value
drift and raw policy changes. Mixed-target loss alone is not evidence of gain.
Compare the two arms within matched seed and game; use fixed whole-game paired
bootstrap uncertainty. No metric selects a seed/checkpoint or gates entry into
the prospectively fixed live comparisons. No new fitting or target adjustment
follows inspection of diagnostic results within this study.

## Fresh three-matchup strength screen

Freeze 1,200 main games and 36 excluded operational games before model work:

| Matchup | Main games | Operational games | Purpose |
|---|---:|---:|---|
| Mixed vs original EF8A | 400 | 12 | Strength of proposed training recipe |
| Terminal-only vs original EF8A | 400 | 12 | Strength from full-corpus training alone |
| Mixed vs terminal-only | 400 | 12 | Primary controlled value-target contrast |

Each matchup uses both training seeds, 200 games per seed in 100 swapped-seat
pairs. Direct mixed-versus-terminal games compare models with the same training
seed. Setups are disjoint between matchups, operational/main cohorts and known
historical reservations. Alternate ten-game blocks across matchups and seeds.
No adaptive allocation, extra games, discarded games or best-seed selection.

All players use matched 128 simulations, three worlds, cpuct 0.4, depth 64 and
unchanged corrected rules. Qualify each exact imported player and its profile;
the trained reference in mixed-versus-terminal receives the same graph, startup,
decision, privacy, trace and output checks as every other player.

The primary scientific contrast is the fixed two-seed mean mixed-versus-terminal
score with a paired 95% interval. The two original-parent comparisons are
secondary. Report their ordinary 95% intervals descriptively and use paired
97.5% intervals for each when claiming either improves on the parent (Bonferroni
family-wise 95% coverage). Use 50,000 fixed stratified paired bootstrap draws.

Retain the previous practical useful-gain criteria: official AND conservative
score >=55%, the relevant lower confidence bound >50%, pooled score in each
seat >=50%, plus the degenerate-interval safeguard. Parent claims use the
adjusted intervals. Distinguish a measured target-effect estimate from meeting
an adoption threshold. A direct target advantage without improvement over the
original is not grounds to promote a model. No model is promoted automatically.

This is a learning screen. Three 400-game cohorts do not provide the precision
of 1,200 games for a single contrast. Small gains, including roughly three score
points, may remain unresolved. Report effect sizes and uncertainty even if the
practical threshold is missed; never describe an inconclusive result as proof
that learning cannot help. Two fixed seeds do not establish arbitrary-seed
generalization, external-opponent strength or superhuman performance.

Actual move guards remain 9 seconds internal and 10 seconds full response. One
second and 700 ms are descriptive, not admission gates. Accept completed late
moves; classified hard-guard expiry loses the responsible player's game.
Infrastructure/correctness failure stops incomplete. Preserve every failure,
with no active-clock reset, failed-decision retry or replacement game.
Retain full saved game verification and at most 96 selected exact reproductions
per seed/matchup, at most 576 total, followed by independent acceptance.

## Bounded execution and preservation

One retained owner has an exclusive claim and original wall/monotonic clocks.
Overall cap 120h; model cutoff 96h. Stage ceilings, clipped to those originals:
qualification 4h; cache build plus independent acceptance 24h (12h per worker);
four fits/commitment 12h (6h per fit); prediction diagnostics 12h; all strength
qualification/operations/main games 64h; exact replays 16h; saved finalization
32h including a 12h independent-reader window. These maxima do not extend the
overall clock. Exact process identity, Windows jobs, actual exits and cleanup
are required. A completion file alone is not acceptance.

Actual aggregate worker RSS ceilings: 8 GiB cache/audit, 4 GiB fit/offline,
2 GiB strength inference. Two live players at a time. Actual new-evidence cap
96 GiB, with 12 GiB reserved for finalization; actual free-disk floor 5 GiB.
No available-RAM reserve, stability wait or forecast refusal. The advisory
40–55-hour forecast is not an admission gate. Monitor actual disk use; preserve
historical evidence. No silent deletion or new resource exception is implied.

Full model-free tests and three independent source-bound reviews precede freeze.
Run qualification, cache build/acceptance, all four fits and final-step replays,
commitment, diagnostic predictions, all player qualifications, operational games,
main games, selected replays and final independent readers automatically.
Technical corrections require separate tested/frozen versions; never edit a
running source or reset an active scientific allowance. User cancellation is
recorded separately from technical or scientific failure.

Final acceptance rechecks consumed source/cache identities, all expected stage
exits, each matchup's independent acceptance, paired statistics and adjusted
claims, preservation inventory and process cleanup. Produce one or two intuitive
charts summarizing the three matchups and the training diagnostics. Update
continuity. Keep EF8A as reference pending a separately considered promotion;
no further collection or study launches automatically after this screen.
