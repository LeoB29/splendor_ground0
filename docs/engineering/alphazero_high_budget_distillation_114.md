# AlphaZero High-Budget Distillation Pilot 114

Date frozen: 2026-08-26

## Purpose

Experiment 113 established that exact generation 11 at 1,024 simulations is
materially stronger than the same network at 256 simulations. Experiment 114
asks the next causal question: can one same-architecture generation absorb the
stronger teacher so that strength improves at the ordinary 256-simulation
budget?

This is one bounded generation, not an automatic loop. It preserves the exact
network architecture and proven training recipe. The only upstream change is
that fresh self-play policy targets come from 1,024-simulation search.

## Frozen inputs

- completed 113 result SHA-256:
  `aea9e5a69fa91669ff3ee05d66d9d912b4d5920c90ca2944f5de7003dab18f8d`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify all hashes, the exact completed 113 pass and arena, generation-11
lineage, and retained installation before evidence.

## Stage 1: fresh high-budget self-play

Generate exactly 256 clean exact-gen11 self-play games in four concurrent
one-thread shards of 64 games. Original seed starts are `2090000`, `2100000`,
`2110000`, and `2120000`. Search uses 1,024 simulations, `c_puct=2.25`, real
mode, root-noise fraction 0.25, and search seed `1114000`; established
temperature and termination settings remain unchanged.

Every game is independently resumable and hashed. Abnormal originals are
excluded outcome-blind and replaced from the immediately following shard seed
range, up to 16 attempts per shard. Each shard seals exactly 64 clean games and
a streaming contract audit before the combined manifest is accepted.

## Stage 2: whole-game partition and training

Within each clean shard, the first 56 selected games enter the 224-game training
pool and the last eight enter a common untouched 32-game holdout. Stream the
four sealed shards into disjoint JSONL files and bind their hashes and exact seed
lists before training.

Starting from exact gen11, train one same-architecture generation-12 candidate
on only the 224-game pool. Use the proven parent-safe configuration: up to 12
epochs, batch 256, AdamW learning rate and weight decay `1e-4`, policy weight
1.0, value weight 0.5, deterministic whole-game internal validation fraction
0.2, patience 3, minimum improvement 0.001, and epoch zero eligible.

Evaluate exact parent and selected candidate once on every row of the untouched
32-game holdout. Gameplay opens only if a nonzero epoch was selected, holdout
policy cross-entropy and weighted objective both strictly improve, value MSE
regresses by at most 0.025, and all metrics are finite. Report policy drift and
target-top agreement. Failure seals without gameplay.

## Stage 3: fresh gameplay, only after offline qualification

Primary compression test: candidate and exact gen11 both use 256 simulations,
`c_puct=2.25`, real mode, and zero root noise on seeds `2130000..2130199`, both
seats, for 400 games in four concurrent 50-seed shards.

Diagnostic high-budget test: candidate and exact gen11 both use 1,024
simulations under the same settings on seeds `2140000..2140099`, both seats,
for 200 games in four concurrent 25-seed shards. Run it after the primary so
the machine never exceeds four arena workers.

Every shard has a pre-game protocol/checkpoint/search/schedule identity and
exact-prefix resume. Each combined panel reports W-D-L, share, margin, seats,
terminations, and a 50,000-sample paired bootstrap.

## Decisions

The primary establishes successful compression only if score share is at least
55%, margin is positive, both seats are at least 50%, every game is normal, and
paired CI95 is strictly above 50%. The high-budget diagnostic uses the
established practical checks but is not a separate promotion claim.

- Primary and diagnostic pass: teacher absorbed at both budgets; review a
  short high-budget continuation block.
- Primary passes, diagnostic fails: useful compression with unresolved
  high-budget interaction; replicate before continuing.
- Primary fails, diagnostic passes: improvement still requires expensive
  search; review representation capacity.
- Both fail: stronger targets were not converted into gameplay strength;
  proceed to a larger representation/capacity pilot.
- Offline failure: no gameplay; proceed to capacity/optimization diagnosis.

No result authorizes automatic continuation, `035`, installation, or promotion.
Exact generation 2 remains installed. Expected wall time is roughly 10--14
hours with four workers during compute-heavy stages.
