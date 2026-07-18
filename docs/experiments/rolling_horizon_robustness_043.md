# Final Score-Delta Rolling-Horizon Robustness Gate (`043`)

Frozen before implementing the runner or generating any `043` game. This is a
large, fresh estimate motivated by the conflict between `041` and `042`; it is
not a rescue, parameter change, or continuation of either evidence block.

## Question and interpretation

The unchanged score-delta BMRH candidate scored 58.5% on `041` and 48.0% on
the terminal-correct `042`. Both blocks used 100 paired seeds, and `042`
confirmed that the discrepancy was not an integrity failure. A larger sample is
therefore useful to distinguish robust strength from finite-seed variance.

Do not pool `041` or `042` into the `043` promotion decision. They remain
historical motivation only. Run all five direct blocks without inspecting or
acting on interim outcomes. No early stopping, seed replacement, parameter
tuning, or threshold change is allowed.

## Frozen candidate and match contract

Use literally the `040`/`041`/`042` score-delta BMRH gameplay path:

- raw legal `035` is both the candidate prior and simulated opponent;
- actor horizon 2 and at most four distinct canonical successors;
- 32 sequence evaluations over two observer-relative determinizations;
- unchanged branching mutation, score-delta evaluator, terminal override,
  checkpoint, tie-breaking, and seed derivations;
- CPU, max 400 turns, repetition limit 4, no-progress limit 60.

Every seed is played twice with candidate seats 0 and 1. Valid terminal reasons
are exactly `completed` and clean `forced_pass_deadlock`; preserve the engine's
existing score/fewest-purchased-cards adjudication. Every timeout, stall,
cutoff, unknown reason, or non-exact candidate budget fails integrity.

## Fresh direct robustness estimate

Run five non-contiguous, disjoint blocks of 100 paired seeds / 200 games:

- block A: `431000--431099`;
- block B: `443000--443099`;
- block C: `457000--457099`;
- block D: `469000--469099`;
- block E: `487000--487099`.

The complete direct estimate therefore contains 500 paired seeds / 1,000
games. The blocks are equally weighted, and confidence intervals bootstrap the
paired seed as the sampling unit with 20,000 fixed-seed samples.

Every block must independently pass game-count, terminal-integrity, timeout,
stall, and exact-budget checks. Direct strength passes only if all of these
preregistered conditions hold:

- aggregate score share at least 55%;
- aggregate paired-bootstrap CI95 lower bound strictly above 50%;
- positive aggregate mean paired final-score margin;
- strictly more than 50% score share in at least four of the five blocks.

Run all five blocks even if an earlier block would make a pass unlikely. The
block consistency rule is descriptive protection against another isolated seed
band; it does not require each 100-pair block to be individually significant.

If direct fails, reject this BMRH formulation as a robust improvement, retain
raw `035`, do not tune it, and do not generate broad games or teacher data.

## Conditional broad gates

Only a complete direct pass unlocks three fresh 50-pair / 100-game blocks:

- information-legal gate-`030` shallow search: `510000--510049`;
- historical scorer `016`: `520000--520049`;
- loop-safe clean canonical policy `023`: `530000--530049`.

Each broad block independently requires exactly 100 valid games, zero timeout
or stall, exact candidate budget, score share at least 55%, paired-bootstrap
CI95 lower bound strictly above 50%, and positive mean paired score margin. All
three must pass.

A complete pass authorizes only a separately preregistered search-every-move
teacher corpus. It does not promote a neural checkpoint. GPU use is not
authorized for `043`.
