# AlphaZero Corrective Loop `091`: Result

Date: 2026-08-09  
Status: complete engineering result; no strength or promotion claim

## Decision

`091` completed at its first candidate acceptance gate with the frozen decision:

`rollback_candidate_failed_parent_arenas`.

Both stability candidates improved held-out policy and value metrics. The
offline-selected `value_quarter` child then:

- lost raw-policy play against the generation-2 incumbent, 6-1-9;
- won search play against the same incumbent, 11-0-5.

The frozen protocol required both arenas to pass. It therefore correctly kept
the installed checkpoint at `090` generation 2, opened no search-configuration
discovery, generated no fresh replay, and performed no `035` arena.

This is not evidence that the candidate is uninteresting. It is evidence that
the raw-policy and searched-agent criteria disagree, and that the acceptance
definition must be resolved prospectively rather than waived after seeing the
score.

## Frozen identity

- Protocol: `docs/engineering/alphazero_corrective_091.md`.
- Protocol SHA-256:
  `9313105b1578aff95ee322ee2ee6ccae2ac3d78b378d1d14deb9761080fac307`.
- Result: `outputs/alphazero_corrective_091/result.json`.
- Result SHA-256:
  `b3dc44707e8d5948c3f18da5799bfc27c6bd99411e0e9b799ec84cbaeaa3335b`.
- Runtime: 20.01 minutes.
- Runner stderr: zero bytes.

All three locked input hashes reproduced:

- incumbent generation-2 checkpoint:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- generation-1 replay:
  `500b9e6a68e8e2332767c03eaecede595a63ee30b38b52a5efd001f6619d925a`;
- generation-2 replay:
  `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`.

## Offline screen

Both candidates used the same 8,544 replay positions, stable whole-game split,
and generation-2 parent. Negative deltas are improvements.

| Candidate | Epochs | Policy delta | Value delta | Weighted objective delta | Target-top agreement | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| `balanced_half` | 3, early stop | -0.048043 | -0.028902 | -0.062494 | 0.486047 -> 0.492442 | yes |
| `value_quarter` | 3, early stop | -0.048697 | -0.032852 | -0.056910 | 0.486047 -> 0.490698 | yes |

The frozen selector chose `value_quarter` by lowest child weighted objective.
That rule has a conceptual defect: objective levels with different value-loss
weights are not directly comparable. This was noticed only after the metrics
opened and was not changed. In this run it did not appear to alter the selected
model: `value_quarter` also had slightly lower unweighted child policy loss
(2.443120 versus 2.443774) and value loss (1.055709 versus 1.059659), so it
Pareto-dominated `balanced_half` on those two held-out losses.

The selected child's validation-slice deltas were:

| Slice | Rows | Policy delta | Value delta |
|---|---:|---:|---:|
| turns `<20` | 520 | about -0.04 | about -0.05 |
| turns `20--49` | 780 | about -0.05 | about -0.05 |
| turns `>=50` | 420 | about -0.05 | about +0.01 |
| replay generation 1 | 680 | about -0.01 | about +0.02 |
| replay generation 2 | 1,040 | about -0.07 | about -0.07 |

The recency/stability changes fixed the aggregate generation-3 value regression
seen in `090`, but mild value degradation remains concentrated in late games
and the older replay distribution.

## Parent acceptance arenas

Both arenas used eight paired seeds / 16 games and every game completed
normally. All 32 `(seed, candidate seat)` identities were unique and complete.

| Arena | W-D-L | Score share | Mean margin | Gate |
|---|---:|---:|---:|---:|
| child raw vs incumbent raw | 6-1-9 | 40.625% | -1.8125 | fail |
| child search vs incumbent search | 11-0-5 | 68.750% | +1.6250 | pass |

The searched result is promising development evidence: the same child that is
weaker under greedy raw action selection is materially better when its policy
and value heads guide equal-dose PUCT. It is still only a 16-game engineering
arena and was not a preregistered promotion panel.

The rollback is nevertheless mandatory. The installed checkpoint remains:

`outputs/alphazero_pivot_090/pilot/generation_002.pt`

with verified SHA-256:

`60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

## Implementation and verification

`091` added:

- weighted policy/value objectives while preserving unweighted diagnostics;
- early stopping and restoration of the best held-out objective;
- stable whole-game validation membership;
- bounded two-generation replay windows;
- phase- and replay-generation validation slices;
- explicit offline, raw-parent, searched-parent, search-discovery, and
  search-confirmation gates with rollback;
- shape-bucketed replay batches;
- immutable inference reuse across arena games with fresh search trees.

Verification:

- focused `090` + `091` suites: 17/17 passed before and after the run;
- complete pre-evidence repository suite: 687/687 passed in 426.18 seconds;
- all input, candidate, installed-checkpoint, and result hashes checked;
- no new self-play replay exists because the mandatory gate failed;
- no `035` metric was opened;
- no champion changed.

The process used roughly 1.4--1.5 GB during training and about 0.5 GB in the
arena, below the roughly 2 GB-plus final-fit footprint observed in `090`.
Because the replay dose also differed, this is an operational observation, not
a controlled speed or memory benchmark. The remaining dominant cost was
search-vs-search gameplay. Arena results are still written only after a whole
paired block; per-game durable arena progress should be added before longer
panels.

## Recommended next step: `092`

Do not retroactively accept the `091` child. Freeze a new protocol that resolves
the criterion conflict explicitly:

1. Treat searched-agent performance as the primary acceptance measure because
   the deployed AlphaZero-style agent searches; retain raw policy as a mechanism
   diagnostic rather than an automatic veto.
2. Compare both already-frozen `091` candidates on new discovery seeds, select
   without comparing differently weighted objective levels, and confirm only
   the selected searched candidate on a larger fresh paired panel.
3. Require held-out policy/value non-regression, phase diagnostics, normal
   terminations, positive searched margin, and a confidence-aware searched
   parent threshold before installing anything.
4. Only an accepted and independently confirmed searched child may enter the
   search-improvement/configuration gate and generate the next self-play corpus.
5. Add per-game arena progress before running that panel.

This preserves the main lesson of `091`: value stabilization appears to have
worked, and searched strength may have improved, but the evidence is not yet
clean enough to install the child or resume an unrestricted self-play loop.
