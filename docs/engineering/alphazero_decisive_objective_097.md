# AlphaZero decisive-objective screen 097

Status: frozen before candidate evidence.

## Question

Can a narrowly corrective policy objective convert the confirmed generation-2 search operator into a successor that transfers under the same searched operator, without generating another self-play corpus?

## Frozen inputs

- generation-2 parent: `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256 `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- retained generation-2 replay: `outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl`, SHA-256 `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`
- fresh 095 replay: `outputs/alphazero_confirmed_operator_successor_095/selfplay/generation_002_high_dose.jsonl`, SHA-256 `c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47`
- sealed 095 result: SHA-256 `6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655`
- sealed 096 result: SHA-256 `11cc6d1fcc72a6744fa87646dd31ddac89a505c0d86a6e1396287a1dd48259e6`

No new self-play is permitted in 097. The rejected 095 checkpoint is historical evidence only and is not an initialization or selector candidate.

## Fair offline screen

Three candidates start independently from the exact generation-2 parent and use the same replay rows, whole-game train/validation split, ordering seed, optimizer, and stopping rule:

1. `standard_control`: ordinary visit-distribution cross-entropy.
2. `divergence_weighted`: the ordinary per-position cross-entropy is weighted by `1 + 2 * TV(visit distribution, stored root network prior)`, normalized by total batch weight.
3. `visit_argmax_aux`: ordinary cross-entropy plus `0.5` times cross-entropy on the visit-count argmax, only where that argmax differs from the stored raw-policy argmax.

Common training configuration: 6 epochs maximum, batch size 256, AdamW learning rate `1e-4`, weight decay `1e-4`, validation fraction `0.2`, seed `970097`, policy weight `1.0`, value weight `0.5`, patience 2, minimum improvement `0.002`, bucketed batches. Early stopping uses each run's own objective; cross-candidate selection never compares those differently scaled objectives.

## Offline eligibility and selection

A corrective candidate is eligible only if all conditions hold on the common validation split:

- ordinary overall policy loss is no worse than the parent;
- overall value loss is no worse than the parent;
- overall policy-target top agreement is no worse than the parent;
- value-loss degradation is at most `0.025` in every nonempty early/middle/late phase;
- on decisive search-change rows, target-top agreement exceeds the retrained standard control by at least `0.02`;
- on those rows, probability assigned to the visit argmax exceeds the standard control by at least `0.01`.

If neither corrective candidate is eligible, stop and retain generation 2. If both are eligible, select lexicographically by: higher decisive target-top agreement, higher decisive visit-argmax probability, lower ordinary decisive policy loss, then the frozen order `divergence_weighted`, `visit_argmax_aux`. Only one candidate may reach gameplay.

## Searched gameplay gates

The frozen operator is 256 simulations, `c_puct=2.25`, real mode, no root noise.

Development uses 16 paired seeds (32 games), starting at `1,680,000`. It passes only with score share at least `56.25%`, positive mean margin, score share at least `50%` from both seats, and all games normally completed.

If development passes, independent confirmation uses 50 paired seeds (100 games), starting at `1,681,000`. It passes only with score share at least `55%`, paired-bootstrap 95% lower bound strictly above `50%` (50,000 resamples, seed `970097`), positive mean margin, at least `50%` from both seats, and all games normally completed.

Only a confirmation pass installs the selected checkpoint. A raw-policy arena on the confirmation seeds may run afterward as a non-vetoing diagnostic. There is no 035 or human arena and no broader champion claim.

