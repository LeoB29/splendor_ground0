# Frozen Score-Delta Rolling-Horizon Gate (`041`)

Frozen before implementing this gate or generating any `041` game. This is a
no-training, no-tuning confirmation of the exact score-delta control already
implemented and exercised in `040`.

## Question

Did the `040` score-delta Branching Mutation Rolling Horizon (BMRH) control beat
event-value only because the published event weights were poor, or is it a
genuinely stronger gameplay operator than raw legal policy `035`?

## Candidate contract

Freeze the `040` score-delta planner literally:

- legal-observation `035` prior and simulated opponent policy;
- two root-player decisions per genome;
- `(1+1)` branching mutation with root-covering initialization;
- top four distinct canonical-successor actor candidates;
- 32 sequence evaluations over two observer-relative determinizations per
  non-forced planning decision;
- seeds `40_000_000 + environment_seed` for mutation schedules and
  `41_000_000 + environment_seed` for determinizations;
- root-relative prestige-score change, with terminal win/loss override;
- CPU inference, 400-turn cap, repetition limit 4, and no-progress limit 60.

Do not change the evaluator, horizon, budget, candidate cap, opponent model,
tie-breaking, checkpoint, or seed derivation. `040` outcomes may justify this
gate but may not select any parameter.

## Direct parent gate

Run 100 fresh paired seeds / 200 games against raw deterministic legal `035` at
seeds `403000--403099`, reversing seats within each seed. Require all of:

- exactly 200 completed games and zero stalls, timeouts, or other non-completed
  termination;
- exact `32 x non-forced planning decisions` candidate sequence budget;
- candidate paired score share at least `55%`;
- paired-seed bootstrap CI95 score-share lower bound strictly above `50%`;
- positive mean paired final-score margin.

Any failure stops the experiment, keeps `035`, and closes rolling-horizon
planning as the current improvement branch. Do not repair or tune after the
result.

## Conditional broad gates

Only a complete direct pass unlocks three fresh 50-pair / 100-game blocks:

- gate-`030` information-legal shallow search, seeds `403200--403249`;
- historical legal scorer `016`, seeds `403300--403349`;
- loop-safe clean canonical entity policy `023`, seeds `403400--403449`.

Each block independently requires zero non-completed games, exact candidate
sequence budget, score share at least `55%`, CI95 lower bound above `50%`, and
positive paired margin. Every block must pass.

A complete pass establishes the score-delta BMRH operator as a stronger search
teacher and authorizes a separately preregistered search-every-move corpus and
distillation loop. It does not promote a neural checkpoint by itself. CPU is
appropriate; GPU use is not authorized for this gate.
