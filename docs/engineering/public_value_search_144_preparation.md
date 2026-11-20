# Experiment 144 preparation

2026-09-08, before protected gameplay. Owner authorized the proposed gameplay
test. No prior source or model was modified. New inference adapter, arena runner,
independent verifier, protocol and 14 focused tests were added.

All 14 focused tests passed in 2.21 seconds. Full repository suite passed
1,002/1,002 in 286.77 seconds; session 25480 exited successfully. Log:
`outputs/public_value_search_144_tests.log`. Identity was rechecked unchanged
after preparation: 273 current dependencies and all 272 old 143 dependencies.

Identity-bound preflight passed; digest:
`8cccbc2e525e40b89870e3746a5a1f181dd415b8c4481f02cbc6aca5ec9a620c`.
Both off-schedule games completed naturally, 58 and 72 turns. All 130 actions,
65 raw-035 choices and 65 search roots passed audit; four prespecified searches
reproduced exactly using their saved simulation counts. Timing over 65 searches:
mean .402739 seconds, p95 .406422, maximum .418531. The protocol allows one
simulation plus assembly past the deadline. Projected arena wall time on eight
workers: 11.0 minutes, excluding startup, auditing and contention differences.

Preflight session 64223 exited successfully. Artifacts:
`outputs/public_value_search_144/preflight.json`,
`outputs/public_value_search_144/preflight/games/game_0000.json`,
`outputs/public_value_search_144/preflight/games/game_0001.json`.
These are engineering probes, excluded from all protected statistics. No
protected seeds or outcomes were used in design or preparation.
