# Parallel tools prepared during Experiment 162

Implements `docs/strategy/parallel_work_during_162.md`. Separate modules only;
no frozen predecessor, checkpoint or running162 file is edited. Implementation
authorization does not queue a later experiment. All commands are explicit;
there is no watcher, automatic successor, training or promotion.

2026-09-17 execution authorization: the owner approved the recommendation in
`docs/strategy/next_after_162.md`, including the54-game timing/integrity probe
and the fixed5,400-game main panel if operationally reasonable. Add the planned
replay-minus-anchor secondary before freezing; it uses the same games. No extra
training, teacher-batching profile, extended arena or monitoring is authorized
by this panel launch.

## Generalization panel

Entry: `run_student_generalization_panel.py`. The `plan` command reports
configuration without loading players, reading interim162 outcomes or starting
games. `prepare` requires162's completed audited final status and binds all nine
selected models, fit audits, final evidence, opponent weights, source code,
analysis code and local runtime. Positive162 strength gates are not prerequisites.
All three selected fits from each arm remain in the comparison.

The panel contains anchor161, refresh162 and replay162, each with three fits,
against unchanged035, information-legal shallow search and loop-safe023. Exact
opponent configurations/weights follow051. A fresh opponent instance is created
for every game and replay; history is never shared across games.

100 initial-seed clusters x three arms x three fits x three opponents x both
seats =5,400 main games. Main environment seeds2000000000--2000000099;
timing-probe seed2000100000 (54 additional games, excluded from strength).
Proposal streams share a cluster across all comparisons and are disjoint from162.
Two-player base game,15-point target, current engine end-of-round and tiebreak
rules, lawful actor observations. Omniscient replay evidence is not model input.

Each game is fully replayed before acceptance, including independently rebuilt
student decisions and freshly instantiated opponent decisions. Engine and
opponent implementations are shared trusted dependencies, not independent
reimplementations. Repetition4, no-progress60 and turn400 cutoffs are invalid;
natural completion and the existing two-forced-pass adjudication are valid.
Failed and partial games are preserved. A duplicate stage/game is rejected;
no automatic repair, exclusion, resampling or extension.

Primary endpoint: refresh-minus-anchor score difference over the equally weighted
panel. Secondary contrasts: refresh-minus-replay (fresh-feedback attribution)
and replay-minus-anchor (old-data update's broader performance). These are
differences in scores against the shared panel, not direct matches between the
two compared players. Draws count0.5.50,000 paired
bootstrap samples with seed2000300000 resample100 whole initial-seed clusters,
keeping every opponent, fit and seat together. Report per-opponent differences
and descriptive95% intervals, arm scores, W/D/L, fit and seat breakdowns and
mean/p95 decision wall/CPU times. Subgroup intervals are not simultaneous claims.
There is no promotion threshold. Intervals condition on these fixed fits and
opponents. Local-panel results establish neither human superiority nor strength
at equal search time. Once used for development, this opponent panel is no
longer protected confirmation evidence.

Commands from the repository root:

```
python run_student_generalization_panel.py plan
python run_student_generalization_panel.py check --checks outputs/parallel_work_162_checks/checks_01
python run_student_generalization_panel.py prepare --checks outputs/parallel_work_162_checks/checks_01/checks.json
python run_student_generalization_panel.py launch-probe
python run_student_generalization_panel.py launch-main
```

`check` runs the new focused tests on one Torch thread and records exact source
identity and log hashes. `prepare` freezes the configuration only after162 has
finished. `launch-probe` and `launch-main` use hidden processes and eight hidden
workers with one Torch thread each. The probe's elapsed time includes full
replay; its extrapolation is provisional. Main requires a completed probe.
Neither stage is automatically launched by preparation or by the other stage.
Inspect the timing estimate before allocating the main run. Main writes a
report plus all detailed statistics. Existing outputs are never overwritten.

## Batched teacher prototype

`src/splendor_ai/alphazero/teacher_batching.py` implements CPU inference over
variable-length legal-action sets using padding/masks, and a scheduler advancing
independent simulated games. It preserves job ordering, supplied root actions,
actor-relative inputs, tie ordering, per-job state/counters and termination
records. It is not imported by any production teacher. Models with sampling,
loop fallback, training mode or a non-CPU device are rejected.

`scripts/analysis/check_teacher_batching.py` runs an explicit small equivalence
and timing probe after162 completes. It constructs fixed snapshots from new
games at turns0/20/40, compares batch sizes1/4/16 to original complete serial
trajectories, and saves every action trace/final state, mismatch index and timing.
It does not consume162 data. Output must be a new file beneath
`outputs/teacher_batching_checks`. Example:

```
python -m scripts.analysis.check_teacher_batching --output outputs/teacher_batching_checks/probe_01.json
```

Any move/trace difference disqualifies a claim of transparent equivalence; this
probe does not silently fall back, enable production batching or claim speedup.
Batch arithmetic can differ even when the selected moves agree. The small probe
cannot prove universal equivalence, and serial-first timing is not a reliable
performance benchmark. A later representative, order-balanced performance and
equivalence study is required before deployment. No GPU benefit is assumed.

## Decision map

The panel module exposes a pure function mapping completed audited162 gate
results to the four interpretations in the strategy note. It rejects unaudited
or inconsistent records and never changes the existing162 classification.

## Validation scope

Focused tests cover balanced/disjoint schedules, paired cluster statistics with
independent numerical expectations, missing/duplicate/invalid games, complete
short-game replays against every opponent from both seats, tampering, preserved
failures, parent readiness and artifact guards. Batching checks cover legal
padding, private-world invariance, first-maximum ties, forced root actions,
terminal/cutoff cases, per-trajectory counters and full trace equality on small
fixtures. These are correctness checks, not new research-strength results.
