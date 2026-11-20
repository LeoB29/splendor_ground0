# Local player correctness and timing result — 18 October 2026

The unchanged four-candidate/128-world player completed the entire prescribed
120-second panel correctly across all three frozen models. It is the operational
candidate for a research benchmark at that allowance, subject to opponent
qualification. The eight-candidate/128-world player missed eight deadlines.
Neither exact configuration met the one-second deployment target on this setup.

This is a finite saved-position correctness and timing result. It provides no
new playing-strength evidence and no universal deadline guarantee. The
eight-candidate successor retains its stronger historical local comparison;
four candidates do not inherit that result.

## Execution and verification

All **98 scheduled attempts** ran: 18 predetermined core positions covering
three fits, both seats and three phases; five metadata-selected edge positions;
both players at one and 120 seconds; and six declared repetitions at 120 seconds.
The panel and order were frozen before production decisions. No position was
selected using its outcome or measured time. All required edge strata existed.

The numerical players, 128 worlds, proposal construction, frozen 035
continuations and anchors 161001/161002/161003 remained unchanged. There was no
partial-search decision, fallback, reduced work, selected best fit or faster
numerical backend.

| Player | Allowance | Correct timely completions | Timeouts | Other player failures | Entire prescribed panel |
|---|---:|---:|---:|---:|---|
| Four candidates × 128 worlds | 1 s | 0 / 23 | 23 | 0 | Missed |
| Eight candidates × 128 worlds | 1 s | 0 / 23 | 23 | 0 | Missed |
| Four candidates × 128 worlds | 120 s | 26 / 26 | 0 | 0 | Met |
| Eight candidates × 128 worlds | 120 s | 18 / 26 | 8 | 0 | Missed |

There were **44 complete verified decisions and 54 timeouts**. Every complete
decision matched its original scientific record: actions, proposals, choices,
world identities, traces and continuation outcomes. The independent analyzer
checked 5,632 worlds and 29,568 continuations. It found zero differing model
values even within the predeclared 1e-6 tolerance. Four-candidate expectations
were independently reconstructed from the original breadth record's preserved
four-prefix on the same worlds; these are counterfactual correctness fixtures,
not new four-candidate game results.

All **563 unique frozen inputs** passed independent SHA256 and byte-size checks,
including the 451 original dependencies. Preflight comprised 65 synthetic
adapter, clock, schedule, verification and external-catalogue checks. The separate
analysis program also passed seven focused arithmetic/integrity self-tests.

At 120 seconds, core completions by fit were:

| Fit | Four candidates | Eight candidates |
|---|---:|---:|
| 161001 | 6 / 6 | 4 / 6 |
| 161002 | 6 / 6 | 4 / 6 |
| 161003 | 6 / 6 | 5 / 6 |
| Five edge positions | 5 / 5 | 4 / 5 |
| Three repetitions | 3 / 3 | 1 / 3 |

All 23 actual hidden-state permutations produced identical lawful player inputs.
All three four-candidate repeat pairs completed with identical scientific
results. No eight-candidate pair had two completed decisions: one repetition
completed after its first attempt timed out, while the other two timed out on
both attempts. Their pairwise equality is therefore unestablished, not disproved.
The later successful repetition does not replace the earlier timeout.

## Observed timing and limits

| Player at 120 seconds | Completed sample | Mean | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Four candidates | 26 | 43.61 s | 40.54 s | 82.41 s | 87.28 s |
| Eight candidates | 18 | 54.16 s | 44.87 s | 107.99 s | 109.62 s |

These distributions include **only timely verified completions**, including
repetitions. In particular, the eight-candidate figures omit eight censored
timeouts and are not its overall completion-time distribution. The percentiles
use linear interpolation. No timeout is treated as an observed 120-second
completion, and this run establishes no sufficient larger allowance for eight
candidates.

The trusted supervisor timed from before input serialization through receipt of
the complete response, including scientific trace evidence. It used a high
resolution clock and terminated overdue process trees. Model initialization was
separate: median 2.686 seconds, maximum 3.152 seconds across 98 workers, below its
60-second limit. Execution used one numerical thread on logical CPU 0 of a
verified performance core, at normal priority, on an ordinary shared desktop.
Desktop load and full-evidence delivery limit extrapolation to other deployments.

All eight research-budget timeouts were early-game positions:

| Scheduled case (one-based) | Fit | Root | Kind |
|---:|---:|---|---|
| 48 | 161001 | 0204:010 | Core |
| 53 | 161001 | 0385:017 | Core |
| 60 | 161002 | 0250:000 | Core |
| 65 | 161002 | 0117:007 | Core |
| 72 | 161003 | 0292:002 | Core |
| 85 | 161003 | 0342:006 | Edge |
| 96 | 161002 | 0250:000 | Repeat |
| 98 | 161003 | 0292:002 | Repeat |

The complete run finished at 13:43:36 UTC (15:43:36 Paris), **3,588.38 seconds
(59 minutes 48 seconds) after the original launch**, inside the original four-hour
ceiling. Peak sampled worker RSS was 241.15 MiB against a 2 GiB limit. Sampled
request CPU totaled 2,814.14 seconds. Recovery evidence including the independent
analysis occupies 27,598,004 bytes (about 26.32 MiB), below the 256 MiB allowance.

The first launch stopped before initializing any player when the referee failed
its one-GiB available-memory check. Its failure and receipts remain in
`outputs/local_player_qualification_20261018_01/`. The explicit recovery moved
observation preparation to a process that exited before the lightweight timing
supervisor started. It retained the players, selection, clock, resource thresholds
and original four-hour deadline. There was no extra run window and no erased
failed decision. The recovered run had no resource or correctness failure.

## External opponent and next action

The pinned `roeey777/Splendor-AI` PPO at commit
`95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b` failed the catalogue prerequisite:
**84/90 card faces and 10/10 nobles match**. An independent comparison using the
live local engine confirmed the same six mismatches. See
[the catalogue qualification report](external_strength_catalog_qualification_2026-10-18.md)
for exact costs, source hashes and why an identifier translation cannot remove
the disagreement. This establishes disagreement, not which catalogue is official.

No external checkpoint was deserialized, no external inference ran, and the
**60-game competitiveness screen did not run**. The adapter fails closed. Even
after resolving the catalogue, lawful observations, feature/action equivalence,
action-subset and empty-mask behavior, safe checkpoint loading, and the enforced
one-second limit still require qualification.

The next gate is authoritative component adjudication. A bounded follow-up review
of the local CSV, publisher resources and official-hosted rules did not establish
all six disputed faces; the local CSV repeats the same legacy values. The
[publisher's resources](https://www.spacecowboys-games.com/game/splendor/) and
[official-hosted rules](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/Splendor-EN.pdf)
are starting evidence, not a completed component audit.

Resolve the intended catalogue before substantial one-second player redesign or
an external matchup. If changes are warranted, version the corrected rules
separately and preserve the historical engine, checkpoints and evidence. Existing
results remain comparisons under the frozen local rules; performance and transfer
under any corrected ruleset require fresh qualification.

After compatibility and remaining adapter gates pass, retain the planned cheap
60-game PPO-versus-raw-035 screen. Four candidates at 120 seconds are the explicit
operational option for a later research benchmark, with all three models retained;
that is a different objective from the one-second deployment target. The strongest
verified eight-candidate reference still lacks a qualified practical allowance.
No larger benchmark or optimization campaign launched. The larger independent-data
study remains deferred until both storage and investment justification are satisfied.

## Reproducibility records

- Protocol: `docs/engineering/local_player_qualification_20261018.md`.
- Recovery protocol: `docs/engineering/local_player_qualification_recovery_20261018.md`.
- Implementation: `tools/local_player_qualification_128/` and
  `tools/local_player_qualification_recovery_20261018.py`.
- Independent analyzer: `tools/analyze_local_player_qualification_20261018.py`.
- Initial preserved freeze: `5d097f17430d8a46637ff5799ce0d008187b395004c09c8ce74c793d81a461f0`.
- Recovery freeze: `fae25ac6a0b10f8f344f20f28bddb680e3bf0726c2bfd3434a2e8dedc6964115`.
- Completed output: `outputs/local_player_qualification_20261018_recovery_01/`;
  `selection.json`, `schedule.json`, `rows/`, worker receipts/logs,
  `independent_check.json`, `final_input_check.json`, `external_qualification.json`,
  `result.json`, `completion.json`, and `independent_analysis.json`.

The independent analysis binds its own source hash and the source result hash.
Full fit/player/clock/phase/seat/kind summaries and all censored outcomes are saved
there. Do not rerun or overwrite this completed qualification to improve its result.
