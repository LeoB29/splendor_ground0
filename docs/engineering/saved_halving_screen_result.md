# Saved-data sampling screen: small positive conditional gain

Completed 2026-09-17 at 22:00:14 Paris. The owner-authorized screen, focused tests,
frozen analysis and independent reconstruction all completed without a failure.
No new simulation, inference, training or gameplay was performed. No further
experiment was launched and monitoring remains off.

## Result

| Measure | Result |
|---|---:|
| Uniform32 mean held-out terminal score | 57.2592% |
| Successive-halving mean held-out terminal score | 57.3680% |
| Primary adaptive-minus-uniform difference | +0.108805 percentage points |
| Paired 95% interval | [+0.065072, +0.153532] percentage points |
| Frozen classification | positive_development_screen |
| Root-partitions choosing different moves | 24.6224% |
| Uniform choice discarded by adaptive first stage | 10.8229% |

These are continuation scores for selected moves, not full-game win rates
between two planners. Both methods spent the same continuation count. For four
actions this is 128: uniform32 each, versus16 each followed by32 extra for the
two survivors. The 23 smaller-support positions use the unchanged uniform rule.

All 600 roots from 200 source pairs and all64 fixed partitions were retained.
Support counts were1/13/9/577 for one/two/three/four candidates. Each partition
judged its locked decisions on80 worlds excluded from either decision. The
50,000-resample interval retained whole source pairs.

Descriptive effects were positive in every phase: early +0.1362pp, middle
+0.1070pp and late +0.0832pp; seat0 +0.1358pp, seat1 +0.0829pp. These are not
separately established subgroup effects and do not authorize selective use or
new phase-dependent rules. Four-support positions averaged +0.1131pp.

## Interpretation

Concentrating part of the budget on two survivors improved average conditional
decision quality under this exact saved-data test. The interval excludes zero,
but the effect is small. Frequent choice changes do not imply large gains: the
net effect averages helpful and harmful substitutions.

This is useful development evidence for an efficiency mechanism. It does not
establish stronger full-game play, equal wall-time strength, preserved128-world
strength, an improved learning loop or human superiority. It also does not show
that early elimination is generally safe. The states and outcomes were reused
from160; repeating partitions does not create independent experiments or erase
prior use of the corpus.

The frozen rule permits proposing a fresh equal-budget gameplay comparison, but
does not require or authorize one automatically. The small effect should inform
that cost decision. Retain the positive result without expanding this analysis,
tuning the schedule or selecting a favorable subgroup. Learned proposals remain
a distinct candidate for the next substantive mechanism study.

## Verification and evidence

- All21 focused contract tests passed in0.38s. Preparation checked and froze
  1,422 input identities and five analysis/protocol/recommendation files.
- Arithmetic analysis took32.10s. The separately implemented verifier
  reconstructed all38,400 paired root-partitions, checked9,674,752 decision-cell
  accesses, all held-out totals, 200 pair endpoints and the50,000 bootstrap draws.
- Both paths verified the source chain through400 games,400 original audit
  receipts and600 selected roots. Original dependencies and all existing dirty
  work were preserved. No failure record was created.
- The source's prescribed inner-trajectory audit covered two worlds per root.
  This study checked saved evidence and did not perform new trajectory replays.

Output: `outputs/saved_halving_screen`. Read `final_status.json` and
`verification.json` for final completion; `analysis_complete.json` is the
historical record written before independent verification. Its pending flag and
the preliminary wording of the sealed report are intentionally not rewritten.

Binding:
`d398c6bb294d064a0a64b93421f1d6f3798fbb059d92820bc3efe8bf36cf038c`.

Result SHA256:
`8ee09c9199b06ed05e5b0d36471ae741146f961f4053f54f87616ee2241344dc`.

Verification SHA256:
`dd9ad01e14d46851836bb70471c40f132b6e29b26e81041c0545e54a9b93b6ba`.

The frozen analysis files, selected inputs and output evidence remain unchanged.
