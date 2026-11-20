# AlphaZero Pivot `090`: Engineering Result

Date: 2026-08-09

## Decision

`090` successfully builds and executes the programme's first complete,
from-scratch iterative policy/value self-play loop:

`N -> PUCT(N) -> visit/outcome replay -> N+1 -> PUCT(N+1)`.

The result is a **mechanical and architectural pass**, but **not a strength or
promotion pass**. Generation 2 learned both policy and value targets and beat
generation 1 in the small search arena. Generation 3 then regressed against
generation 2, and the final raw network scored only 1-7 against `035`.
Accordingly, neither generation 3 nor the new architecture is a champion
candidate.

The next step should not simply spend more compute on the unchanged loop.
First add candidate acceptance/rollback and diagnose the value-head and search
improvement failure seen at generation 3. Then run a larger `091` loop only
after the raw-network and search-improvement gates pass again.

## Frozen contract and boundaries

- Engineering contract:
  `docs/engineering/alphazero_pivot_090.md`.
- Contract SHA-256:
  `60014db6b16936738562acf45ea1501ba8872106004ca2918329bddf8e7b89d8`.
- This is an engineering sandbox, not preregistered promotion evidence.
- Unknown future public cards are chance outcomes. `RESERVE_DECK` remains
  legal and its card identity remains private to the reserving player.
- `035` is used only as an optional external arena baseline. It is not a
  search prior, continuation policy, encoder, target, distillation teacher, or
  initialization source.
- Owner-waived `087` continued independently and was not read for outcomes or
  altered by `090`.

## What was implemented

- A compact structured policy/value network with shared entity/action
  encoders, one logit per canonical legal action, and a scalar value head.
- A clean shared-tree PUCT search over all canonical legal successors, neural
  leaf evaluation, alternating-player value backup, root Dirichlet noise, and
  temperature sampling.
- Perfect-information debug operation plus observer-relative determinization
  and information-legal node keys for normal operation.
- Durable encoded replay containing root visits and terminal WDL targets,
  without full game state or hidden deck order.
- Whole-game train/validation splits, policy cross-entropy plus value MSE,
  checkpoint lineage, per-game progress, paired development arenas, and a
  multi-generation command-line runner.
- Focused contracts for action coverage, exact simulation accounting, backup
  sign, determinism, hidden-information invariance/privacy, replay round-trip,
  loss/reload behavior, end-to-end generation reload, and isolation from
  `035` in the learning/search path.

## Executed dose

The target pilot completed in 2.409 hours with an empty stderr log.

| Self-play generation | Games | Simulations per move | Positions | Mean turns | Normal completions |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 32 | 2,450 | 76.56 | 32/32 |
| 1 | 64 | 64 | 4,336 | 67.75 | 64/64 |
| 2 | 64 | 64 | 4,208 | 65.75 | 64/64 |
| **Total** | **160** |  | **10,994** |  | **160/160** |

The realized game and simulation schedules exactly equal the frozen target
schedules `(32, 64, 64)` and `(32, 64, 64)`.

## Learning evidence

Validation deltas below are child minus parent, so negative is better.

| Child | Replay rows | Total-loss delta | Policy delta | Value delta | Target-top agreement, parent -> child |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,450 | -0.3710 | +0.0003 | -0.3713 | 0.147 -> 0.128 |
| 2 | 6,786 | -0.1131 | -0.0527 | -0.0604 | 0.182 -> 0.266 |
| 3 | 10,994 | **+0.0965** | -0.0226 | **+0.1191** | 0.427 -> 0.441 |

Generation 1 mostly learned the value signal; its policy movement was tiny and
did not improve target-top agreement. Generation 2 is the clearest positive
result: both heads improved on held-out games and policy target agreement rose
materially. Generation 3 improved policy loss and agreement slightly, but its
value head generalized worse and made total held-out loss worse than its
parent. The saved generation-3 checkpoint is the best of its four epochs, so
this is not merely the last epoch being selected accidentally.

## Arena evidence

Every cell is candidate wins-draws-losses over eight paired-seat games.

| Child | Parent search vs parent raw | Child raw vs parent raw | Child search vs parent search | Child raw vs `035` |
|---:|---:|---:|---:|---:|
| 1 vs 0 | 7-0-1 | 7-0-1 | 8-0-0 | 0-0-8 |
| 2 vs 1 | 8-0-0 | 8-0-0 | 5-0-3 | 0-0-8 |
| 3 vs 2 | 4-0-4 | 3-1-4 | 2-0-6 | 1-0-7 |

Six generation-0 arena games ended at the repetition cutoff: two in parent
search versus parent raw and four in child raw versus parent raw. All other 90
arena games completed normally. The generation-0 raw-policy comparisons are
therefore especially weak evidence.

The small arenas show that the loop can initially transfer search behavior
into the next network, but not that improvement is yet stable. By generation
2, search no longer beat its own raw parent (4-4), and generation 3 lost both
the raw and searched parent comparisons. Its single win over `035` is useful
as a first nonzero result, but a 1-7 score is still decisive evidence of large
remaining weakness.

## Search and action diagnostics

| Self-play generation | Search changes raw argmax | Visit/prior JSD | Search minus raw value | `RESERVE_DECK` selected |
|---:|---:|---:|---:|---:|
| 0 | 84.49% | 0.0928 | +0.0550 | 29 / 2,450 positions (1.18%) |
| 1 | 78.14% | 0.1062 | +0.1592 | 257 / 4,336 positions (5.93%) |
| 2 | 47.62% | 0.1042 | +0.1255 | 225 / 4,208 positions (5.35%) |

Search remains active, but its diminishing argmax-change rate is ambiguous:
it may indicate policy uptake, loss of useful correction, or both. The final
arena favors the latter as at least part of the explanation.

Blind deck reservation became much more common after generation 0. This does
not show that strong Splendor play values it; these networks are immature and
weak. It is a diagnostic to retain. The action was neither prohibited nor
artificially penalized, consistent with the programme decision to leave the
door open while expecting strong play to use it only marginally.

## Integrity and verification

- Focused `090` suite: 11/11 passed.
- Full repository suite: 681/681 passed in 333.83 seconds while `087`
  continued in the background.
- All 10,994 replay rows parsed successfully.
- Every policy target, raw network prior, and noisy search prior normalized.
- Every root visit vector summed to its configured simulation count.
- All action-aligned vectors had matching lengths; all WDL targets were in
  `{-1, 0, +1}`.
- No top-level full-state or deck-order field appeared in replay.
- All recorded replay, parent, child, and final checkpoint SHA-256 identities
  matched their files.
- Final checkpoint:
  `outputs/alphazero_pivot_090/pilot/generation_003.pt`.
- Final checkpoint SHA-256:
  `44d5c6a68359adbb4957e0f032d3d32cdaba52857aa35582ac2525775631f23d`.
- Result:
  `outputs/alphazero_pivot_090/pilot/result.json`.
- Result SHA-256:
  `41a1d89f68e9c15b47679bae376e58528bf505799cc91dc984adec70e9e90621`.

Three outcome-free engineering restarts were archived and excluded from the
target replay: one added prior-divergence metrics, one separated sampled moves
from visit argmax, and one established that one Torch thread was faster for
this small inference workload. The retained target run uses one Torch thread.

## Recommended `091`

Before scaling self-play, implement and test:

1. A true candidate-acceptance gate: a child that fails held-out loss and
   paired parent arenas must not automatically become the next generator.
2. Value-head stabilization and diagnosis: learning-curve/seed checks,
   early-stopping behavior, replay recency weighting or a bounded replay
   window, and calibration by game phase.
3. A search-improvement gate requiring search to beat the raw network before
   its visits are treated as a successful improvement target.
4. Runtime work that preserves semantics: streaming/bucketed replay batches,
   checkpoint reuse in arenas, and parallel self-play. The target run showed
   replay training, not resolver mechanics, as the immediate bottleneck.

Only after those gates pass should `091` spend the larger compute budget on
additional generations. `035` should remain an external thermometer, not a
teacher or architectural dependency.
