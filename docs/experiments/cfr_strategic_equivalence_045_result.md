# CFR Strategic-Equivalence Result (`045`)

Experiment `045` formally failed two of ten frozen gates. The high raw-policy
TV discrepancies from `044` were harmless, but exact best response exposed a
different and much larger defect in rare information sets of the external
solver's stored average policy. Full-game neural Deep CFR and GPU rental remain
locked.

## Integrity

- Protocol checkpoint: `d038a47`.
- Implementation checkpoint: `6ac8b4e`.
- Pre-evidence verification: 419 tests passed in 167.22 seconds.
- The evidence run completed normally in 70.2 seconds.
- Reconstructed policy payload hashes exactly matched immutable `044`:
  - exact: `b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab`;
  - external: `3c6644db39998800690a2adfa6efc8b64c7f69cd55be99d678ca2ea7ebadbf78`.
- The complete strategic analysis was deterministic. Every best response chose
  one legal semantic action consistently across every hidden history in each
  information set. Finiteness, nonnegative deviation, and zero-sum checks all
  passed.
- Result JSON SHA-256:
  `528e0b0582d2c39c752e88d0cfb5c82188fba510fa0caba8dd915b281a691701`.

## Frozen results

Exact CFR was essentially unexploitable:

- profile value to player 0: `-0.000154458`;
- player-0 deviation incentive: `0.000233741`;
- player-1 deviation incentive: `0.000121118`;
- NashConv: `0.000354859`;
- exploitability: `0.000177429`, passing the `0.005` gate.

The external sampled average policy was highly exploitable:

- profile value to player 0: `0.0000563907`;
- best-response value for player 0: `0.558514`;
- best-response value for player 1: `0.0281131`;
- NashConv: `0.586628`;
- exploitability: `0.293314`, failing the `0.020` gate and the maximum `0.015`
  external-minus-exact gap.

This failure was invisible to ordinary cross-play. External player 0 versus
exact player 1 differed from exact self-play by only `0.0000712884`; exact
player 0 versus external player 1 differed by `0.000522537`. Thus the sampled
policy behaves normally against equilibrium-like play while a deliberate best
response can steer into its weaknesses.

All seven preregistered high-TV information sets were resolved. Their total
conservative reach-weighted positive loss was `0.0000171397` for player 0 and
zero for player 1, far below the `0.002` limits. Therefore the raw-TV failures
that motivated `045` really were strategically harmless; they simply were not
the sampled average policy's important defect.

## Bounded post-result localization

A no-gate-change reconstruction localized the defect:

- external stored average policy: exploitability `0.293314`;
- filling its eight missing information sets from exact CFR: unchanged at
  `0.293314`;
- replacing only external average strategies with fewer than 100 observations
  by exact strategies: exploitability `0.000401395`;
- analogous replacement below 1,000 observations: `0.000248775`;
- final external regret-matched policy, with no averaging: measured
  exploitability `0.0` on this reduced game.

The regret updates converged; neither missing coverage nor the seven high-TV
states caused the failure. The stored average strategy was unreliable at rare
but adversarially reachable information sets. Self-play and cross-play almost
never visit those states, whereas exact best response intentionally does.

## Decision

Do not abandon CFR, but do not scale the current external-average implementation.
The next experiment must separately preregister a tabular average-policy repair
and replication. It should implement the established reach/sampling-correct
MCCFR averaging or the strategy-memory formulation used by Deep CFR, validate
against the exact reference over multiple fixed sampling seeds, and require
low exploitability globally and within rare-observation bands. Merely adding
iterations, substituting the current policy, loosening `045`, or renting a GPU
is not justified.

Only a corrected tabular sampled average that passes exploitability should
authorize the full-game neural Deep CFR feasibility pilot. The Deep CFR pilot
must then treat strategy-memory quality as a first-class gate rather than infer
strength from regret or self-play value alone. Raw legal `035` remains champion.
BGA/expert data remains a separate future authorized-data possibility and was
not used here.

Primary references for the next repair are Lanctot et al., *Monte Carlo
Sampling for Regret Minimization in Extensive Games* (NeurIPS 2009), Brown et
al., *Deep Counterfactual Regret Minimization* (ICML 2019), and the reference
OpenSpiel CFR implementation's reach-weighted cumulative-policy contract.
