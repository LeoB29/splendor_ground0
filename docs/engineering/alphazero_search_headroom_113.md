# AlphaZero Search-Teacher Headroom 113

Date frozen: 2026-08-25

## Purpose

Experiments 110--112 closed fresh-data dose and separate-value-path hypotheses
under the current 256-simulation teacher. Before rebuilding the deliberately
compact mechanism-validation network, experiment 113 asks whether the exact
generation-11 network can create a materially stronger teacher simply through
deeper search.

This is a direct compute intervention. Candidate and baseline use the exact same
checkpoint, policy, value, encoding, and search formula. Only simulations per
move differ: 1,024 for the candidate and 256 for the baseline.

## Frozen inputs

- completed 112 result SHA-256:
  `bb42a38826990f94ecedd75b6f05649ac7461183a56bc1c6871502c33ed13251`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Before evidence, verify all hashes, the complete 112 contract and negative
decision, generation-11 lineage, and the retained installation.

## Frozen arena

Use fresh game seeds `2080000..2080199`, both candidate seats, for 400 games.
Both agents use exact generation 11, `c_puct=2.25`, real mode, zero root noise,
and operative search seed `2 * game_seed + seat`. Candidate search uses 1,024
simulations; baseline search uses 256. No `c_puct` retuning is allowed: this
isolates the marginal value of deeper search under the current proven operator.

Run four concurrent one-thread shards of 50 paired seeds each. Each shard writes
an identity sidecar before its first game, binds both search configurations and
the exact schedule, and resumes only by an exact completed-game prefix. Combine
only all 400 unique scheduled rows and report W-D-L, score share, margin, seats,
terminations, and a 50,000-sample paired-seed bootstrap.

## Primary decision

Deeper search establishes useful teacher headroom only if all checks pass:

- score share at least 55%;
- positive mean margin;
- score share at least 50% from each candidate seat;
- every game terminates normally;
- paired-bootstrap CI95 lower bound strictly above 50%.

If it passes, authorize review of a bounded high-budget self-play/distillation
pilot. If it fails, do not spend four times the self-play compute; proceed to a
larger representation/capacity pilot. A failure tests deeper search at the
current fixed operator, not every possible search retuning.

Estimated wall time is 6--8 hours with four parallel shards. This experiment
performs no training or self-play, runs no `035` benchmark, and cannot install
or promote a checkpoint. Exact generation 2 remains installed.
