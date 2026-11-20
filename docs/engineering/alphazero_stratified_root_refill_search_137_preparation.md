# AlphaZero Stratified Root-Refill Search `137` Preparation

Date: 2026-09-04  
Status: prepared and preflight-passed; not launched

## Frozen identity

- Protocol:
  `docs/engineering/alphazero_stratified_root_refill_search_137.md`
- Protocol SHA-256:
  `a81c68e6b9921671ad72c77f6587295214f3f0229fb7f31a8cacc16125e9a54a`
- Generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`
- Protected state-manifest SHA-256:
  `a81ea0293d56974993a1ca2bbde099d283203acdc0b6ef4ad0416665518988c6`
- Preparation preflight SHA-256:
  `d37952294abf9f9949ef912d8187159c98d09f130487e16d70d85ccd320e8df5`

## Prepared work

- Added a default no-op transition-sampling hook to production
  `AlphaZeroSearch`. Existing search behavior and random-number consumption are
  unchanged unless an experimental subclass overrides the hook.
- Implemented ordinary root-refill tracking and action-conditioned stratified
  root refill. Each exact root visible-card action receives its own deterministic
  sequence of shuffled possible-card cycles.
- Stratification occurs only after PUCT selects the root action. It cannot
  reveal the future refill to action selection. Deeper transitions remain
  ordinary sampled search.
- Implemented the complete guarded panel runner, exact resume identities,
  reference pooling, phase breakdowns, source-game-cluster bootstrap, coverage
  and runtime accounting, and frozen decisions.
- Added focused tests covering determinism, hidden-order invariance, legality,
  action-specific balancing, no change to non-refill/deeper transitions,
  production search regression, and exact visit-budget accounting.

## Protected states

The outcome-blind selector reconstructed all 256 experiment-114 source games.
The no-blind-reserve condition is restrictive: only nine games remain eligible
into the middle/late cells. The final exact assignment therefore uses:

- 36 states;
- six states in each phase/root-player cell;
- 21 distinct source games;
- at most three states from any source game;
- game-clustered statistical resampling;
- no blind reserves for either player;
- at least one visible refill action with at least eight possible replacement
  identities in every state.

The state selector used no policy, visit, Q, value target, score, or winner.
The manifest passed an identical repeat preflight.

## Launch workload

An explicit future `--launch` will run:

- two ordinary 256-simulation searches per state;
- two stratified 256-simulation searches per state;
- two ordinary 1,024-simulation references per state;
- 216 root searches and 110,592 total simulations.

No protected search has run. No model was trained, no game was played, no
checkpoint was changed, and experiment `134` was neither read nor modified.

## Verification

The combined experiment-090 production search regression plus experiments
136/137 focused band passes 22/22. The preparation-only preflight passed twice
with the same protected state-manifest hash.

Runner:
`run_alphazero_stratified_root_refill_search_137.py`

Omitting `--launch` is preparation-only. Do not launch until the owner asks.
