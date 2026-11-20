# AlphaZero adaptive corrective-candidate replication 098

Status: frozen before any 098 gameplay evidence.

## Purpose and multiplicity

Experiment 097 selected `visit_argmax_aux` offline and produced a promising 19-13 searched development result, but failed its prospectively frozen seat gate: 75% from seat 0 and 43.75% from seat 1. Therefore 097 did not promote the candidate.

098 is an explicitly adaptive, second-chance replication prompted by that inspected result. It is not a continuation that retroactively waives the 097 gate. To reduce the risk of promoting noise after adaptive follow-up, 098 uses a single exact candidate, no tuning, 100 entirely new paired seeds, a paired-bootstrap gate, and retained per-seat gates. No 098 game may be used to change these rules.

## Frozen inputs

- incumbent generation 2: `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256 `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- exact 097 `visit_argmax_aux` candidate: `outputs/alphazero_decisive_objective_097/candidates/visit_argmax_aux/generation_003.pt`, SHA-256 `59b17517ae0f0e1f996f05bca7c66acaeecd710862066f22c2eec99da983bdfa`
- sealed 097 result: `outputs/alphazero_decisive_objective_097/result.json`, SHA-256 `d8d8de1521065389bdfa003de0bd09f77b9371587ae2e821e941894c1796d5cf`

The runner must verify that the sealed result selected `visit_argmax_aux`, ended with `searched_development_failed_keep_generation_2`, recorded a 19-13 development arena, and kept the exact generation-2 incumbent installed.

## Frozen evaluation

- 100 paired seeds / 200 searched games
- seed range starts at `1,690,000`, disjoint from 097
- both candidate seats for every seed
- 256 simulations, `c_puct=2.25`, real-information mode, no root noise
- maximum 200 turns
- paired bootstrap: 50,000 resamples, seed `980098`

The exact candidate is confirmed only if every condition holds:

- overall score share at least 55%;
- paired-bootstrap 95% lower bound strictly above 50%;
- positive mean score margin;
- candidate score share at least 50% from seat 0;
- candidate score share at least 50% from seat 1;
- all 200 games terminate normally.

On failure, retain generation 2. On pass, install the exact 097 checkpoint as generation 3. Only after a pass, run a 200-game raw-policy diagnostic on the identical seeds; it is descriptive and cannot veto the searched confirmation. No new training, self-play, search selection, 035 arena, or human arena is permitted.

Interpretation on failure is frozen qualitatively: if the overall gates pass but a seat gate fails again, the next action is a seat-asymmetry forensic rather than promotion or objective retuning. If an overall gate fails, abandon this auxiliary-objective recipe.
