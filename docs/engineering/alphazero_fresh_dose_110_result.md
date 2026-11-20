# AlphaZero Fresh Gen11 Dose 110 Result

Completed: 2026-08-21 23:43 Europe/Paris

## Result

Experiment 110 completed normally in 37,277.66 seconds. All four deterministic
self-play shards sealed 256 clean games, for 1,024 games and 63,792 positions.
Two abnormal original games were excluded outcome-blind and replaced; stderr
was empty.

Both parent-safe candidates selected epoch 1 and passed the common untouched
holdout gate. The low-dose objective improved by 0.011274 versus exact gen11.
The high-dose objective improved by 0.003264: policy cross-entropy improved by
0.010684, but value MSE worsened by 0.014840. The high-dose top policy action
changed on 7.69% of holdout rows.

Gameplay did not establish strength:

- raw low dose versus raw gen11: 103-2-95, 52%, margin +0.44;
- raw high dose versus raw gen11: 101-0-99, 50.5%, margin +0.02;
- searched low dose versus searched gen11: 92-0-108, 46%, margin -0.735;
- searched high dose versus searched gen11: 100-2-98, 50.5%, margin +0.22.

The searched high-minus-low paired difference was +4.5 percentage points with
CI95 [-5,+14], so the apparent difference was unresolved and high dose remained
flat against gen11. The high-dose primary missed the frozen 55% practical gate.

Decision:
`high_dose_fresh_selfplay_failed_internal_gate_review_capacity_no_install`.
No checkpoint was installed or promoted; exact gen2 remained installed.

## Interpretation

Increasing the fresh training dose from 256 to 896 games did not convert small
offline target improvements into gameplay strength. The high-dose value update
generalized worse even while policy loss improved, and later epochs increasingly
overfit value. This makes value representation, shared policy/value interference,
and fixed-teacher absorption the next hypotheses to distinguish before changing
the whole architecture.

Raw result SHA-256:
`df35d3893f4c184ee24906b884aef33542c954380bf03c0b45f14a6d829f34c2`.

