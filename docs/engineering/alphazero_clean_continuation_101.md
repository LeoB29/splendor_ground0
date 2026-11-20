# AlphaZero clean rolling-loop continuation 101

Status: frozen before replacement generation, training, or 101 diagnostics.

## Purpose

100 confirmed that exact generation-3 search materially improves its raw network (66-2-32, 67%, +2.25 margin, 67% from both seats). 101 therefore completes the intended 099 lineage while preventing the single arbitrary deadlock adjudication from entering training.

101 is a sanitation and continuation experiment, not a retroactive claim that 099 completed. It does not alter the generation-3 checkpoint, standard objective, replay dose, search operator, training configuration, or diagnostic schedule.

## Frozen inputs

- exact installed generation 2: `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256 `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- exact generation-3 child: `outputs/alphazero_scaled_loop_099/checkpoints/generation_003.pt`, SHA-256 `4db34392dbf79cc91e9cce404dd6ba09400d19055d68bf61d3c4ad8ef9127c75`
- clean 099 generation-2 replay: SHA-256 `855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56`
- contaminated 099 generation-3 replay: SHA-256 `73afe6f16f784a1040dd5f215f15708a498b9c13582aa9969edb7766015a514e`
- sealed 099 result: SHA-256 `37534c4d057d4dd8aef9e0d9973e8b9adf427130303aa322465de7d4701c1693`
- sealed 100 result: SHA-256 `58bbaaee4079392537b8a4294d1a780811b711bac0f9edac9e8ac772b8161cc2`

The runner must verify the exact 099 stopped decision and deadlock seed `1,710,004`, the exact 100 healthy decision and 66-2-32 panel, and unchanged installed generation 2.

## Outcome-blind sanitation policy

For every parent-generation corpus, retain only games whose termination reason is exactly `completed`. Never train on an adjudicated abnormal game.

If fewer than 256 clean games remain, generate one-game replacement attempts using consecutive seeds beginning immediately after the original 256-seed schedule. The first replacement seed for generation 3 is `1,710,256`; for generation 4 it is `1,720,256`. Use the same parent checkpoint and self-play search configuration as the original corpus. Retain normally completed attempts and discard abnormal attempts without regard to winner, score, length, action content, or any other outcome. Stop after 16 attempts if 256 clean games have not been obtained.

Write a new sanitized replay; never overwrite the source replay. The sanitized corpus must contain exactly 256 unique games, only `completed` terminations, contiguous turn rows per game, non-null outcomes, the exact parent/checkpoint generation, and exactly 256 simulations per position. Record every excluded and replacement seed and the sanitized SHA-256.

## Remaining lineage

1. Sanitize generation-3 replay, then train generation 4 from exact generation 3 using the replay window `[clean generation 2, sanitized generation 3]`.
2. Run the frozen generation-4 diagnostics.
3. Generate 256 generation-4 self-play games starting at `1,720,000`, sanitize using the same policy, then train generation 5 from exact generation 4 using `[sanitized generation 3, sanitized generation 4]`.
4. Run the frozen generation-5 diagnostics and stop for trajectory review.

Generation-4 self-play uses real PUCT, 256 simulations, `c_puct=2.25`, root noise fraction `0.25`, search seed `990101`, and maximum 200 turns. Generation-3 replacement attempts use search seed `990100`.

Generation-4 and generation-5 training use the original 099 settings: 12 epochs maximum, batch 256, AdamW learning rate `1e-4`, weight decay `1e-4`, whole-game validation fraction `0.2`, policy weight `1.0`, value weight `0.5`, patience 3, minimum improvement `0.001`, bucketed batches, and seeds `990100` / `990101`. Only the ordinary AlphaZero objective is allowed.

Engineering validity remains the only automatic advancement condition: exact checkpoint lineage, nonempty validation, at least one epoch, finite metrics, and a valid sanitized corpus. Gameplay strength does not gate the automated continuation.

## Fixed diagnostics and outcome

For generation 4 and generation 5, reuse the automated fixed 099 panels at 8 paired seeds each:

- raw child versus raw parent, seed start `1,780,000`;
- searched child versus searched parent, seed start `1,780,100`;
- child search versus child raw, seed start `1,780,200`.

Search uses 256 simulations, `c_puct=2.25`, real mode, and no noise. Reusing seeds permits longitudinal comparison; no choice is made between generations.

101 never installs a sandbox checkpoint. Exact generation 2 remains the programme incumbent. Successful completion seals exact generation 5 and the full generation-3/4/5 trajectory for review. A later confirmation requires a separate protocol.
