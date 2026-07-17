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

## Result

Formally rejected on 2026-07-18 at the direct integrity gate. Pre-evidence
verification passed all 401 tests, and the implementation was checkpointed at
`7e9765a` before any `041` game.

All 100 paired seeds / 200 games were generated. The score-delta planner showed
clear directional strength against raw `035`:

- 117-83-0;
- paired score share `58.5%`, CI95 `[52.0%, 65.0%]`;
- mean paired final-score margin `+0.91`, CI95 `[+0.16, +1.64]`;
- 6,364 planning decisions, 203,648 sequence evaluations, and exact
  `32 x planning decisions` accounting.

The score-share, confidence-bound, margin, expected-count, and exact-budget
gates all passed. The sole failed gate was zero non-completed games. Seed
`403078`, candidate seat 0, ended after 30 turns through the match runner's
`forced_pass_deadlock` adjudication at score 2-0; the candidate was adjudicated
the winner. This was not marked as a stall or timeout, but the frozen protocol
explicitly rejected every non-`completed` termination.

Therefore do not run the conditional shallow/`016`/`023` blocks, do not call
this a promoted teacher, and keep `035` champion. The result is operationally
frustrating but unambiguous under the preregistration: rolling-horizon closes as
the current improvement branch without a post-result repair. Scope the evidence
carefully—the planner passed every statistical strength gate, and was rejected
only for one forced-pass terminal-contract event.

Evidence is under `outputs/score_delta_rolling_horizon_041`.
`direct_vs_035.json` SHA256 is
`3ece75beb2ad19d2b5eeadaa7321b684d8e7ac1ec1de388b34a79ec3af968465`;
`gate_final.json` SHA256 is
`c90daf97251c0621e16416e783a2ff3cdf3a966e66c9a394043059171b398423`.
