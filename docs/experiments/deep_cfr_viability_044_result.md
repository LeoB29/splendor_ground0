# Reduced Splendor CFR Viability Result (`044`)

Experiment `044` is a formal fail on two of thirteen frozen gates. It does not
authorize full-game neural Deep CFR, GPU rental, or checkpoint promotion. The
result is nevertheless positive evidence that the information-state and CFR
machinery are basically sound, because every privacy, legality, determinism,
regret, coverage, and root-policy gate passed.

## Execution and integrity

- Protocol checkpoint: `c506cba`.
- Implementation checkpoint: `6406c10`.
- Immutable-graph lookup optimization: `e072cd9`. The first invocation was
  stopped by the shell after ten minutes before producing any evidence because
  it repeatedly rebuilt large information keys. The optimization changed no
  game, iteration count, solver update, seed, metric, or gate; the evidence run
  restarted from zero.
- Pre-evidence verification: 415 tests passed in 157.41 seconds.
- The complete evidence run finished normally in 120.9 seconds.
- Exhaustive reduced game: 985 nodes, 750 terminal paths, 206 information sets,
  and exactly four player actions on every path.
- All four privacy checks passed. Hidden deck order is masked, and a blind
  reserve identity is remembered by its owner but absent from the opponent's
  information state and the public history.
- Exact and external deterministic reruns produced identical payload hashes.
- Result JSON SHA-256:
  `6b7edee724b98cc149d8bbc1826a75fa7592e246a7d85ea384e703734fbaf632`.

## Frozen metrics

Exact CFR passed all its gates after 10,000 iterations:

- player 0 normalized positive regret: `0.0000292944`, down from `0.00292944`
  at iteration 100;
- player 1 normalized positive regret: `0.00000373672`, down from
  `0.000373672`;
- zero illegal actions and zero unstable information-set action sets.

External-sampling MCCFR passed its convergence, integrity, coverage, and root
agreement gates after 50,000 iterations:

- normalized positive regret: `0.000137200` for player 0 and `0.00000775862`
  for player 1;
- exact-information-set coverage: `96.1165%`;
- root-strategy total variation from exact CFR: `0.000211297`;
- zero illegal actions and zero unstable action sets.

It failed the two deeper policy-shape replication gates over the 30 shared
information sets with at least 100 sampled observations:

- mean total variation: `0.1503728`, required at most `0.15`;
- 90th-percentile total variation: `0.6666667`, required at most `0.30`.

## Interpretation and next decision

The preregistered gate remains failed even though the mean miss is small,
because the p90 miss is large. A descriptive post-result inspection found that
several of the largest discrepancies compare an exactly uniform reference
strategy with a concentrated sampled strategy, while both solvers have tiny
regret, nearly identical root play, and near-zero average self-play value. This
is consistent with strategically interchangeable actions or non-unique
equilibria at later information sets, but `044` did not preregister an
action-value-equivalence or exploitability test, so it cannot establish that
interpretation retroactively.

The next justified experiment is therefore a small, separately preregistered
diagnostic on this same reduced game: measure exact best-response
exploitability and reach/value-weighted action equivalence for the discrepant
information sets. If sampled CFR is strategically equivalent despite raw
policy TV, that diagnostic can justify a full-game neural Deep CFR pilot under
a better correctness contract. If it is exploitable, repair the sampled CFR
averaging/update implementation and rerun a fresh fixed replication. Do not
rent a GPU before this distinction is resolved.

The BGA replay idea remains available only if an authorized, reproducible way
to obtain the records becomes available. It may later provide a warm start or
external validation set, but no BGA data was collected or used here.

## Why full-game self-play has not started (simple version)

We have not entered the real self-play loop yet because a bot can only improve
by playing itself if its learning procedure can reliably tell which new
choices are better. Our earlier value, search, and trajectory targets did not
pass fresh gameplay tests, so training on them risked teaching the bot its own
mistakes. CFR is a more direct self-play route, and it now works on the small
test game, but its sampled version still disagrees with the exact reference in
some later positions. We need to show that those disagreements are harmless—or
fix them—before spending GPU time on full Splendor self-play.
