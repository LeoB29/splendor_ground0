# Experiment 156: the unchanged155 selector beats084

Completed 13 September 2026 at04:59:21 Paris, after41h13m total. All
prespecified statistical and integrity gates passed. The launcher exited0.
This result confirms the earlier provisional full-match totals.

## Result

400 fresh games from200 starting-seed pairs, with each player in each seat.
Candidate155 scored259 wins,3 draws and138 losses against exact serial084.
That is65.125% of match points (draws count as half), with a complete-pair
95% bootstrap interval of60.375%-69.75%. Opponent084 scored138 wins,3 draws
and259 losses, or34.875% of match points. Candidate mean final-score margin
was+1.3 points.

| Candidate seat | Wins | Draws | Losses | Score share | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 143 | 0 | 57 | 71.5% | +2.285 |
| 1 | 116 | 3 | 81 | 58.75% | +0.315 |

All gates pass: share at least55%, paired interval lower bound above50%,
positive mean margin, each seat at least50%, and full independent replay.
The point estimates differ by seat; this experiment did not test the
statistical significance or cause of that difference.

## What was tested

Exactly155's repeated32-world terminal-WDL selector, with its original
greedy035 action, ranked alternative and two random remaining canonical
proposals. Simulated continuations still use exact greedy035 for both sides.
The opponent is the unchanged serial084 player, including192 stochastic
continuations per completed search, structural fallbacks and loop handling.
No policy weights, search budgets or continuation opponents were tuned.

This establishes a direct gameplay improvement over the programme's strongest
previously confirmed research champion under the two systems' fixed budgets.
It shows that155's improvement extends beyond the035 opponent used inside
its imagined futures. It does not establish superhuman performance, general
dominance against other opponents, equal-compute superiority, a successful
training recipe or a self-improving learning loop.

## Complete verification and cost

All1,258 repository tests passed before freeze. Eight off-panel preflight
games received full replay and agreement checks against untouched084 on every
084 decision. The main audit independently reconstructed all400 games,
24,314 actual moves,12,157 candidate roots and12,157 champion roots.

Every one of3,619,264 simulated continuations was replayed, covering
116,823,399 simulated moves. This includes1,547,776 candidate paths and
2,071,488 champion paths. Every actual root received information-invariance
checks. World reconstruction, decisions and final statistics passed. The
auditor shares the frozen engine, scorer and stated canonical primitives;
this is not an independent implementation of Splendor's rules.

Post-completion checks confirmed unchanged tested identity and parent evidence,
the final/audit/result/log hashes, all408 main/preflight game and replay
certificate hashes, and all24,832 linked decision-record hashes. No experiment
process or recorded failure remained.

All400 outer games ended normally. Candidate paths had33 valid forced-PASS
deadlocks and otherwise ordinary endings.084 completed10,789 searches and
used1,368 structural fallbacks. Its112 loop-adjudicated paths were0.005407%
of its paths, below the0.1% limit; all paths resolved and there were zero
hard caps. Its known structural fallbacks remained part of the tested player.

The candidate changed the greedy035 action on6,755/12,157 turns (55.56%).
Mean measured decision wall time was20.08s for155 and27.29s for084;
p95 was37.86s and57.47s respectively. These are instrumented timings in this
match, not a controlled equal-time comparison. Mean game length was60.785
turns. Main collection took20.23h; full main replay took19.92h; total pipeline
time including preflight was41.22h.

## Frozen evidence and stop

- Binding: `dab0e5502a4eca5005c04c02d2095acc601a7cadee196c2173725d85bce69774`
- Result SHA256: `c468037bd8459957c455d8869c02502a4c2fee4aeb6b74f174839aeedb73fb8f`
- Full-audit SHA256: `20e079af9fcb851f36c6c2fa460912325a2fd0b75a4564d111f0c709dc4dd2ab`
- Final-status SHA256: `7ca4c20902e57b3ad491618889c75dfa767af3e1e424d0f96c9d5dac949329aa`
- Evidence folder: `outputs/rollout_selector_champion_156`
- Frozen protocol: `docs/engineering/rollout_selector_champion_156.md`

Experiment closed after the complete audit. No training, automatic promotion,
budget change or successor was launched. Preserve the frozen evidence and all
intentionally dirty work. Any next research step is a separate decision.
