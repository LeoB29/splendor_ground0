# Architecture Notes

## Near-Term Architecture

The initial implementation is split into five layers.

1. `engine`
- source of truth for rules and transitions

2. `encoding`
- converts canonical state into legal observations and fixed action masks

3. `bots`
- baseline agents that consume engine/encoding outputs

4. `training`
- backend selection, models, optimization, self-play orchestration

5. `eval`
- tournaments, benchmark suites, and rating calculations

## Model Direction

The current intended learning path is:

1. heuristic/search data generation
2. supervised policy/value warm start
3. self-play RL
4. optional search-augmented inference under a `1s/move` budget

This repository does not assume a final network architecture yet. The environment and encoders are being designed so feedforward or recurrent policy/value models can both be supported later.

## Backend Direction

The project will be written around PyTorch with centralized device selection:

- preferred: CUDA
- acceptable fallback: CPU
- compatibility path: DirectML

Training code should request a logical backend policy from a single module rather than instantiating device logic ad hoc across the codebase.
