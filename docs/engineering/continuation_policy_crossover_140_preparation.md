# Experiment 140 preparation

Recorded 2026-09-07. The owner authorized implementation and execution of the
fixed-move continuation-policy comparison following experiment 139. Preserve all
existing work. No training, promotion, arena or automatic successor is included.

Protocol: `docs/engineering/continuation_policy_crossover_140.md`, SHA256
`c71dd40e4756d40c787befd4ede98488d14e26d65ebc1b0820720068eeac5b8b`.

## Implementation and verification before timing

The isolated runner is
`src/splendor_ai/eval/run_continuation_policy_crossover.py`, with wrapper
`run_continuation_policy_crossover_140.py`. It forces original 084/086 choices
once, then uses stochastic 035, greedy 035 or exact native 084 for both seats.
The immediate opponent reply belongs to the assigned continuation policy. There
is no outer sampled opponent minimum, so positive weak-control reproduction is
required before attributing attenuation to stronger subsequent play.

Exact 084 is imported unchanged, including its native ordered reply-set check,
structural fallbacks, inner rollout horizon and seed formulas. The 139 reply-set
adapter is not applied. Inner execution is serial within each of ten outer CPU
workers, with one Torch thread each. Stochastic 035 calls the existing sampler.

The driver records complete starting worlds, semantic moves and state hashes,
terminal scores/card counts and per-seat resolver work. It rejects outer cutoffs
and illegal moves. Atomic trajectory files, frozen dependency identities and a
kernel process lock protect accounting and resumptions.

Initial focused checks passed 50/50, covering experiments 138/139, native resolver
and the new crossover. After adding an identical-root accounting check, the full
suite passed **939/939 in 257.13 seconds** with a fresh temporary directory.
Checks cover selection disjointness and nesting, source-action binding, seed
separation, exact stochastic sampling, native terminal/deadlock rules, cutoff
rejection, independent replay and tamper rejection, paired bootstrap and the
prespecified decision branches. No protected 140 returns were used in testing.

`scripts/analysis/verify_continuation_policy_crossover_140.py` independently
replays saved moves through the engine, verifies scoring, tiebreaks, telemetry,
world and input identities, and reconstructs all primary and secondary means
and bootstrap intervals without using the runner's driver or summary helpers.
Synthetic replay/tamper and bootstrap reconstruction checks passed in the suite.

The three original source/checkpoint hashes were verified. All 266 dependency
files frozen by 139 remain byte-identical. The new identity freezes 267 current
source/wrapper/test/verifier files. The intentionally dirty worktree remains.

## Outcome-blind runtime preflight

The ID-only selectors exclude all 120 games in 138 and 139. The maximum panel
contains 60 distinct games, ten per phase/seat cell, and six additional probe
games are disjoint from that entire panel. Smaller panels are nested balanced
prefixes. Selection never uses returns or action disagreement.

Six parallel probes started at approximately 20:38 Europe/Paris, parent Python
PID 55768, execution session 39478. Each forces its original default action in
one fresh world and finishes under each policy. The late seat-0 strong probe is
repeated from zero to check determinism. Only timing, integrity/work counters
and an opaque deterministic digest are retained; probe outcomes are discarded.

All 66 snapshot structures passed before the probe workers started. Runtime
will select the largest of 60/48/36/24 states under the frozen 129,600-second
serial projection. If no panel fits or integrity fails, stop before protected
evaluation without changing the protocol. Protected launch is pending preflight.

## Preflight passed and protected schedule frozen

All six probes completed naturally, with no inner hard caps or adjudications.
The from-zero late seat-0 repeat matched exactly. The primary six probes made
27,456 inner continuations; the exact repeat adds 384. All probe processes exited
successfully, stderr empty, before launch. No probe outcomes were retained.

The slower phase costs were 907.2161 seconds early, 223.1302 middle and 3.7646
late. Frozen projections were 56.71 serial hours for 60 states, 45.36 for 48,
**34.0233 for 36**, and 22.6822 for 24. The timing-only rule therefore selects
36 states, six per phase/seat cell. This is approximately 3.4 hours across ten
workers before overhead under the conservative projection, not a finish promise.

There are 22 distinct action pairs and 14 coincident pairs. Deduplication gives
**522 protected outer continuations**, 174 per policy, versus the 648 maximum
used by the conservative projection. No state was selected using disagreement.
Prospective normal CI95 half-widths for contrast SD .05/.10/.20 are respectively
.01633/.03267/.06533; statistical power remains uncertain.

- Manifest canonical digest:
  `b725ab75391ddf7f5b4df4cb4d68cfce63c6589544f04d05426bd0ad918c958e`.
- Manifest file SHA256:
  `abbdc0fb67b264d6fd999b90efbd2cb9aa24ff66ae390d5288c5d180296a52ac`.
- Preflight SHA256:
  `aa5529589e6b3fea26676090646a38d690f4ee7dcf45b5d997f23f7ce25792a4`.

All frozen inputs remained unchanged after preparation. Launch is authorized
by the owner's existing instruction; no additional approval is required. Complete
the selected panel once, preserving all evidence on any integrity failure.

Protected evaluation started at **20:55:22 Europe/Paris on 2026-09-07**,
parent Python PID 23804, execution session 30740. Ten workers were confirmed
active, with increasing CPU use and empty stderr. Follow the current status in
`AGENT_CURRENT.md`; do not start a duplicate execution.

Environment: `C:/Users/leoba/anaconda3/python.exe`, CPU, editable project install.
The repository `.venv` is not used. Output directory:
`outputs/continuation_policy_crossover_140/`.
