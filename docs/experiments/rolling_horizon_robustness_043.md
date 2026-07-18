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

## Result (2026-07-18)

Reject the candidate as a robust improvement and close this rolling-horizon
formulation. Protocol commit `feb7ae1` preceded implementation commit
`2212606`; all 409 pre-evidence tests passed.

All five blocks and all 500 paired seeds / 1,000 games ran in 4,922.37 seconds.
Every game ended through ordinary `completed`; there were zero forced-pass
deadlocks, invalid terminations, timeouts, or stalls. The candidate made 31,954
planning decisions and exactly 1,022,528 sequence evaluations, with 3,025,656
simulated actions. Every per-block and aggregate integrity/budget gate passed.

The aggregate result was 505-494-1: score share 50.55%, paired-bootstrap CI95
[47.75%, 53.40%]. Mean paired final-score margin was +0.211, CI95
[-0.141, +0.561]. The positive point margin passed its gate, but score share
was far below 55% and its lower confidence bound was below 50%.

The five block shares were:

- A: 94-106-0, 47.00%, margin -0.170;
- B: 105-95-0, 52.50%, margin +0.365;
- C: 98-101-1, 49.25%, margin +0.170;
- D: 100-100-0, 50.00%, margin +0.395;
- E: 108-92-0, 54.00%, margin +0.295.

Only B and E were strictly above 50%, so the four-of-five consistency gate
also failed. Candidate share was 53.40% from seat 0 and 47.70% from seat 1;
paired seating correctly cancelled that start-role asymmetry. The sole draw was
a normal 15-15 completed game at seed `457041`, candidate seat 1.

The conditional broad blocks were correctly not generated. As historical
context only, not promotion evidence, `041` + `042` + `043` total 718-681-1
(51.32%) across 1,400 games. The large fresh `043` estimate resolves the earlier
conflict: this operator is near parity, perhaps slightly positive, but not a
robust 55% search teacher. Do not add games, tune it, or reopen promotion.
Raw legal `035` remains champion.

Evidence hashes:

- block A: `a811de4d0d3a2374bbb324dbadb8e38252992df4a31a638e082ced0c5ebf5f9d`;
- block B: `29cb64348dc8f71bcdc2b27c500e1f29c96ff07930008cf1af140c284cc74748`;
- block C: `4e59c18856e51aeb2c9859eaf01b81280637b548d451151bb35598765fb81d6c`;
- block D: `9e1e8fb34e26d99153c5f5c7156928b2ff8789f93616622c424f900fccb0ea9b`;
- block E: `3118b5b29e4c1f1bd46a6804148b9b8f2ec0b1d2b4f806b1d74b96bd6529a54e`;
- direct aggregate: `a18f8d5e494b2422e23dcbaf8a778d261fc856f30df959392d054d6bd8308e8a`;
- final gate: `aeb56fa318b43ad9913a3cef728a3d281c12991c0793d9b0e5bd397b7b4a43f2`.
