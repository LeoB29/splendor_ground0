# AlphaZero search-teacher health audit 100

Status: frozen before any 100 gameplay evidence.

## Purpose

099 completed generation-2 self-play and trained generation 3, but stopped after generation-3 self-play contained 255 normal games and one `forced_pass_deadlock`. The abnormal seed was `1,710,004`; it ended after two forced passes at 0-0 and received an adjudicated winner, so its ±1 rows are not acceptable training evidence.

The first fixed diagnostics were also concerning but small: raw generation 3 beat raw generation 2 by 9-7, searched generation 3 lost 6-10 to searched generation 2, and generation-3 search tied its own raw policy 8-8. 100 determines whether the search teacher still produces a material gameplay improvement and audits the observed reserve/deadlock feedback before any continuation or redesign.

## Frozen inputs

- exact incumbent generation 2: `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256 `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- exact 099 generation-3 child: `outputs/alphazero_scaled_loop_099/checkpoints/generation_003.pt`, SHA-256 `4db34392dbf79cc91e9cce404dd6ba09400d19055d68bf61d3c4ad8ef9127c75`
- 099 generation-2 replay: SHA-256 `855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56`
- 099 generation-3 replay: SHA-256 `73afe6f16f784a1040dd5f215f15708a498b9c13582aa9969edb7766015a514e`
- sealed 099 result: SHA-256 `37534c4d057d4dd8aef9e0d9973e8b9adf427130303aa322465de7d4701c1693`

The runner must verify the stopped 099 decision, exact generation-3 child identity, one and only one abnormal generation-3 game at seed `1,710,004`, and unchanged installed generation 2.

## Offline audit

The two replay files are processed sequentially and released between generations. No new replay is generated.

For each generation report:

- game, position, termination, turn, and selected-action-family counts;
- raw-policy top, visit-argmax, and sampled-selection rates by action family;
- raw network-prior mass and search-visit mass by action family;
- search promotion/demotion counts for each family when visit and raw argmax differ;
- mean root Q for visited actions by family;
- raw-network-value and search-root-value MAE, MSE, bias, sign accuracy, and mean prediction against terminal outcome;
- the same value metrics by early/middle/late phase, visit-argmax action family, and whether search changes the raw top action.

The report must highlight generation-3 minus generation-2 changes for `RESERVE_DECK`, overall search disagreement, visit/prior total variation, and value calibration. It must also seal a turn-level trace of the abnormal game including legal family counts, selected/raw/visit families, selected visit probability, prior, Q, and the assigned target. The audit is descriptive and cannot itself authorize continuation.

## Frozen gameplay panel

- exact generation-3 search versus exact generation-3 raw policy;
- 50 new paired seeds / 100 games, starting at `1,790,000`;
- both seats for every seed;
- search uses 256 simulations, `c_puct=2.25`, real mode, no root noise;
- maximum 200 turns.

Search-teacher health passes only if score share is at least 55%, mean margin is positive, score share is at least 50% from each seat, and all 100 games complete normally. There is deliberately no bootstrap promotion gate: this is a mechanism decision, not a strength promotion.

On a pass, the next separately frozen step may replace every abnormal replay game using a predetermined continuation-seed policy and resume the remaining two sandbox generations. On failure, 099 is not resumed; the next work is redesign of the search/value training interface. 100 never modifies replay, trains a checkpoint, resumes 099, or installs a model.
