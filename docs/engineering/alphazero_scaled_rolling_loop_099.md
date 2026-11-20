# AlphaZero scaled rolling sandbox loop 099

Status: frozen before any 099 self-play or gameplay evidence.

## Purpose

098 tested the exact 097 corrective candidate over 100 new paired seeds and finished 107-2-91: 54.0% score share, positive margin, 54.5%/53.5% by seat, but paired-bootstrap CI95 `[46.75%, 61.25%]`. It failed the frozen overall and confidence gates, so generation 2 remains installed and the decisive visit-argmax auxiliary recipe is abandoned.

099 stops asking each single child for promotion-grade evidence. It tests the central AlphaZero mechanism at useful scale: whether three consecutive `search -> self-play -> ordinary policy/value training -> next search` generations form a positive trajectory. It is an engineering sandbox, not a promotion experiment.

## Frozen inputs

- exact generation-2 parent: `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256 `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- bootstrap prior replay for the first rolling window: `outputs/alphazero_confirmed_operator_successor_095/selfplay/generation_002_high_dose.jsonl`, SHA-256 `c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47`
- sealed 098 result: `outputs/alphazero_adaptive_replication_098/result.json`, SHA-256 `79bbb9f1730d1167767205ec55001daf44af04b9ed1e58fefa3a533ac73d4772`

The runner must verify that 098 ended `replication_overall_failed_keep_generation_2`, kept exact generation 2 installed, and recorded the exact 107-2-91 searched result. The 097 auxiliary checkpoint and loss are forbidden inputs.

## Automated lineage

The isolated lineage begins at generation 2 and produces generations 3, 4, and 5. Each iteration automatically advances its valid trained child as the next sandbox parent. This does not install the child as programme incumbent and does not claim that it is stronger.

For parent generations 2, 3, and 4 respectively:

- generate 256 fresh self-play games;
- use seed starts `1,700,000`, `1,710,000`, and `1,720,000`;
- use real-information PUCT with 256 simulations, `c_puct=2.25`, root noise fraction `0.25`, maximum 200 turns;
- require exactly 256 normally completed games, at least one position, and exactly 256 simulations per replay position;
- train the child from the exact parent checkpoint with the standard visit-distribution cross-entropy plus terminal outcome value loss; no corrective policy weighting or auxiliary target is allowed.

Replay windows are frozen as:

- generation 3: bootstrap 095 high-dose replay + fresh generation-2 replay;
- generation 4: fresh generation-2 + fresh generation-3 replay;
- generation 5: fresh generation-3 + fresh generation-4 replay.

This is a two-corpus rolling window. Self-play rows are released from memory before the trainer loads the window.

## Training

Every child uses batch size 256, AdamW learning rate `1e-4`, weight decay `1e-4`, 12 epochs maximum, whole-game validation fraction `0.2`, policy weight `1.0`, value weight `0.5`, early-stopping patience 3, minimum validation-objective improvement `0.001`, and bucketed batches. Training seeds are `990099`, `990100`, and `990101`.

Automatic advancement is guarded only by engineering validity, not gameplay strength: the checkpoint generation must increment exactly once; all parent/child aggregate and epoch metrics must be finite; validation must be nonempty; and at least one epoch must complete. A failure stops the sandbox and retains the programme incumbent.

## Fixed diagnostics

After each child is trained, run three descriptive panels, each with the same fixed 8 paired seeds across generations:

- raw child versus raw parent: seed start `1,780,000`;
- searched child versus searched parent: seed start `1,780,100`;
- child search versus its own raw policy: seed start `1,780,200`.

Searched diagnostics use the confirmed evaluation operator: 256 simulations, `c_puct=2.25`, real mode, no root noise. Reusing the diagnostic seeds makes the automated longitudinal comparisons less noisy; no mid-run choice depends on their results. The panels have no pass/fail gate.

## Outcome

099 always leaves the programme incumbent at exact generation 2. On successful completion it seals generation 5 and reports the three-generation trajectory for review. It does not automatically run a large confirmation, a raw 035 arena, a research-champion arena, or a human arena.

If the trajectory is coherently positive, the exact final checkpoint may later enter a separately frozen independent confirmation. If it is flat or negative, the next step is a network/training architecture and capacity redesign rather than another objective coefficient or confirmation panel.
