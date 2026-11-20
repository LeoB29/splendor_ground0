# Experiment 145 preparation

2026-09-08, before protected outcomes. New isolated public/neutral adapter,
runner, protocol, verifier and tests preserve all prior frozen work.
Twenty-one focused tests passed in 2.42 seconds. The full repository suite
passed 1,023/1,023 in 229.95 seconds; execution session 21577 exited successfully.
Log: `outputs/public_value_ablation_145_tests.log`. Identity was rechecked:
274 current dependencies and all 273 prior frozen dependencies unchanged.

Preflight passed, execution session 65794 exited. Exact identity digest:
`f110daefbfe714954639c7294f24ba2dfdb5de062cab2f3746e56bea48916672`.
Two full seat-swapped probe games on a separate seed finished naturally in
62 and 68 turns. Every one of 130 roots/decisions and eight full first/last
search reproductions passed (four per value arm).

Public mean decision time .402914 seconds, p95 .404805; neutral .402622,
p95 .405321. Ratio 1.000726, passing the frozen [.95,1.05] match requirement.
Both arms passed the individual mean/p95 budget gates. The projected 800-game
arena time on eight workers is 43.69 minutes, excluding startup, contention
differences and the final independent audit. The estimate is not a cutoff.

Artifacts: `outputs/public_value_ablation_145/preflight.json`,
`outputs/public_value_ablation_145/preflight/games/game_0000.json` and
`game_0001.json`. No protected seeds or outcomes were used for preparation.
