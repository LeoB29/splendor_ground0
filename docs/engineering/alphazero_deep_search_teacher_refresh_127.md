# AlphaZero Deep-Search Teacher-Refresh Continuation 127

Date frozen: 2026-08-28

## Purpose

Generation-11 search at 1,024 simulations materially beat the same network at
256 simulations in experiment 113. Experiment 114 produced a valid
generation-12 student with improved untouched offline losses, but one-generation
gameplay was flat and experiment 115 did not confirm an expensive-search gain.
Experiments 116--126 closed simple capacity and start-role explanations.

Earlier AlphaZero work established that adjacent generations can look flat
while a three-generation endpoint compounds. This experiment therefore tests
one bounded three-generation deep-search lineage from generation 11 to
generation 14. Generation 12 is the already finalized experiment-114 student;
only generations 13 and 14 require new work.

## Frozen inputs

- experiment-114 result SHA-256:
  `dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`;
- experiment-115 result SHA-256:
  `02d4d1d2557c57c6778b92950fb4c06d02e4112dc83d45d20791f638df0db993`;
- experiment-126 result SHA-256:
  `f2043a8b86925081dca058b0643c1906ce9ac69fcda48c181364713e0b71136e`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- exact generation-12 experiment-114 checkpoint SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- experiment-114 generation-11 teacher replay shard hashes, in shard order:
  `f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b`,
  `c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60`,
  `00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8`,
  and `7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e`.

Verify exact hashes, checkpoint generations/lineage, 114's valid parent-safe
training, 115's finalized flat replication, 126's closed start-role branch,
256 clean games in the starting replay, and unchanged installation.

## Two teacher-refresh generations

For parent generations 12 and 13, generate exactly 256 clean self-play games in
four concurrent one-thread shards of 64. Search uses 1,024 simulations,
`c_puct=2.25`, real mode, root noise 0.25, maximum 200 turns, and a fixed search
seed `1127000 + parent_generation`.

Shard seed starts are:

- generation-12 actor: 2,300,000; 2,310,000; 2,320,000; 2,330,000;
- generation-13 actor: 2,400,000; 2,410,000; 2,420,000; 2,430,000.

Each shard first attempts its 64 scheduled games, then at most 16 fixed
replacement seeds. Retain the first 64 normally completed games outcome-blind.
Seal every single game, shard, replay hash, seed order, checkpoint identity,
search configuration, and sanitation result for exact resume.

## Rolling replay and training

Generation 13 trains on all 256 sealed experiment-114 generation-11-actor
games plus all 256 new generation-12-actor games. Generation 14 trains on all
256 generation-12-actor games plus all 256 generation-13-actor games. Whole
games remain intact in deterministic train/validation splits.

Use the exact experiment-114 parent-safe training recipe: ordinary policy CE
plus 0.5 value MSE, AdamW, 12 epochs maximum, batch size 256, learning rate
`1e-4`, weight decay `1e-4`, validation fraction 0.2, early-stopping patience 3,
minimum delta 0.001, bucket batches, and epoch-zero parent eligibility. Training
seeds are 1,127,013 and 1,127,014 respectively.

Training validity, finite metrics, exact parent/child generation, replay hashes,
and checkpoint round-trip are mandatory. Selected epoch and adjacent offline
changes are diagnostic only. A valid epoch-zero selection does not stop or
rescue the frozen endpoint block.

## Powered endpoint

After valid generation 14 exists, run searched generation 14 versus searched
generation 11 at 256 simulations, `c_puct=2.25`, real mode, zero root noise,
and maximum 200 turns. Use 200 paired seeds / 400 games, seeds
2,500,000--2,500,199, both seats, four exact shards, and bootstrap seed
1,127,300 with 50,000 game-pair samples.

The endpoint passes only if all games complete normally and all of:

- score share at least 55%;
- positive mean score margin;
- score share at least 50% in each seat;
- paired-bootstrap CI95 lower bound strictly above 50%.

If it passes, decide
`deep_search_refresh_endpoint_compounded_review_035_benchmark_no_install`.
If it fails, decide
`deep_search_refresh_endpoint_failed_close_current_distillation_recipe_no_install`.
An integrity failure decides
`deep_search_refresh_integrity_failed_no_strength_conclusion`.

No adjacent-generation arena controls continuation. No raw or 1,024-simulation
endpoint can rescue the operative 256-simulation result. A pass authorizes only
review and a separately frozen `035` benchmark. Nothing installs, promotes,
changes the retained generation-2 checkpoint, or continues automatically.

## Launch guard

Default invocation performs input, starting-replay, lineage, seed-schedule,
training-contract, one-game smoke, resume-path, and output-path preflight only.
It must not write durable self-play or train. The long run requires explicit
`--launch`. The owner's 2026-08-28 request to start this experiment authorizes
launch only after preflight and focused tests pass.
