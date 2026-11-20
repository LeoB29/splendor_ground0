# Experiment 148 preparation

2026-09-08. Owner authorized the two-stage proposal with "ok, please do it".
Frozen protocol: `docs/engineering/search_teacher_headroom_148.md`.

Implemented a separate runner, verifier, wrapper and focused tests. Reused
the unchanged public evaluator, strong-policy bridge, search, engine and replay.
No earlier source, model, protocol or result was changed. Existing dirty work
was preserved; only new 148 files and current strategy/handoff notes were edited.

- Focused tests: 26 passed in 2.28 seconds.
- Full repository suite: 1,096 passed in 299.12 seconds. Session 85439 exited
  successfully; log `outputs/search_teacher_headroom_148_full_tests.log`.
- Four full probes, both seats in each matchup, passed naturally. Preflight
  session 95254 exited successfully; log `outputs/search_teacher_headroom_148_preflight.log`.
- Reconstructed all 260 probe moves: 195 timed roots and 65 exact raw-035 moves.
  Reproduced 12 complete first/last searches exactly (8 compute, 4 competence).
- Compute probe mean seconds: long 1.604086, short .404798; ratio 3.962677.
  Both per-arm timing gates passed. Mean simulations 262.34 and 65.02.
- Competence probe mean seconds: long 1.603917, raw .003708; long timing passed.
- Projected parallel arena minutes: compute 54.45; conditional competence 43.58,
  excluding audit. These are feasibility estimates, not protected strength results.

Preflight binding:
`1abb1e379a589bda16017ae1b3e5c840e05ae6e79ed94d9ba3b5b6805499cd34`.
All 280 current dependencies and the explicit 280 prior 147 dependencies matched;
prior inputs and the anchored 144/145/147 manifest files also matched. Both
stages' schedules, configurations and gates are included in this binding.

Protected execution uses `run_search_teacher_headroom_148.py run`, eight CPU
workers, one Torch thread each. It saves first-stage results only after all
400 games, launches a separate independent audit, and permits the predeclared
raw-035 panel only after an audited primary pass. It stops on failed/inconclusive
primary strength or any integrity failure. No training, replacement, promotion
or automatic successor. Follow AGENT_CURRENT and final_status.json for execution
state; do not duplicate or restart an existing manifest.
