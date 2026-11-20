# Saved-data equal-budget sampling screen

Owner authorized implementation and execution on 2026-09-17. Scope is the one
screen recommended in `../strategy/next_after_generalization_panel.md`, including
focused checks, frozen identities, arithmetic-only analysis and independent
reconstruction. No new model inference, gameplay, simulation, training, profiling,
successor, promotion or monitoring is included.

## Fixed scientific contract

- Preserve the 600 previously identity-selected roots from 200 source pairs in
  Experiment 160. Three phases per pair; one existing hash-assigned source seat.
- Four-action uniform baseline: 32 worlds per action. Adaptive: 16 worlds per
  action, retain two, then 32 additional worlds per survivor. Choose using the
  survivor's combined 48 outcomes. Both spend exactly 128 continuations.
- Both elimination and final ties prefer the original canonical candidate index.
  Final ties do not prefer the earlier leading survivor. Keep action identities,
  representations, terminal WDL, greedy035 futures and world law unchanged.
- For support sizes one, two and three, use uniform32 in both arms. Include these
  positions with zero paired difference, and report support frequencies.
- Exactly 64 permutations of all 128 world indices, using NumPy default_rng
  seeds 2026091700 through 2026091763. Uniform uses positions0:32; adaptive uses
  0:16 and16:48. Evaluate both locked choices on positions48:128 only.
- Average all partitions within each root, then the three roots within each pair,
  then the 200 pairs equally. Score is (WDL+1)/2, hence win1/draw0.5/loss0.
- Primary is adaptive-minus-uniform held-out score, in percentage points.
  50,000 whole-pair bootstrap resamples, seed2026091764, 1,000-resample chunks,
  NumPy linear 2.5/97.5% quantiles. Phases, seats, support and interventions are
  descriptive; no subgroup selects a new rule or advancement path.
- A primary interval lower bound above zero is `positive_development_screen`;
  otherwise `no_positive_development_evidence`. A negative/inconclusive result
  cannot trigger extra data, a threshold sweep, phase rescue or automatic retry.
  A positive result only supports proposing fresh gameplay after considering
  effect size and cost. Neither result promotes a planner.

These are reused source states/outcomes. Within each partition the evaluation
worlds were not used for selection, but a world can have another role in another
partition. Repetition reduces split randomness without creating independent
studies. The 200 source pairs are the bootstrap units. This is not a fresh
confirmation, a comparison of full-game policies, or proof of equal wall time.

## Implementation and evidence

New standalone script `scripts/analysis/saved_halving_screen.py` imports only
standard Python and NumPy. Its separate command stages are:

1. `check`: run only the focused hand-checkable tests, retaining command output,
   return code, runtime and hashes of all new files. No experiment data analysis.
2. `freeze`: verify the completed160 source chain and outcome-blind source list;
   persist selected identities and all permutations; bind those files, checks,
   protocol, recommendation, code, verifier, environment and source inputs.
3. `run`: verify the frozen identity, record a write-once start marker, consume
   only allocated cells to choose moves, then read separate evaluation worlds.
   Write every cell access, choice, held-out total, root/pair endpoint and report.
4. Run `scripts/analysis/verify_saved_halving_screen.py` separately. It does not
   import production analysis code; it reconstructs original root selection,
   choices, access lists, scores, all pair means and bootstrap statistics.
5. `finalize`: recheck the frozen inputs and completed outputs and bind the
   independent receipt in `final_status.json`. It never edits the sealed report.

Output directory: `outputs/saved_halving_screen`. All evidence files use exclusive
creation. Failures are preserved; no automatic resume or overwrite. Logs can be
read on request; no background service is required for this bounded analysis.

The saved source result hash is
`2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5`,
binding `5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3`.
The selected root-list hash is
`3c762d22dfcd412239c1f485826fe6ef72e802a5b7f340d2732fa1e305763755`.
Source result binds the audit lock, which binds the collection lock; verify all
400 games/audit receipts and the 600 selected root links. Bind and verify the
source manifest dependencies. No source file, frozen src file, old test or
existing research record is edited.

The independent reconstruction verifies original game/turn/actor identities,
candidate order, complete128-world schedule, WDL perspective and source full128
choice consistency. These are artifact checks, not new simulated trajectory
replays. Experiment160's prescribed inner audit replayed two worlds per root;
do not upgrade that claim to all worlds in this study.

## Focused validation

Hand fixtures cover initial and final ties, final ranking including both sampling
stages, exact budget/accesses, untouched short supports, changes confined to
unread loser cells, held-out changes unable to alter decisions, permuted world
vectors, malformed partitions/scores, pair aggregation, bootstrap units and
constant known effects, independent implementation agreement, write-once output,
modified frozen inputs and refusal to run unfrozen. These checks target research
validity; a full unrelated repository test rerun is unnecessary for an isolated
arithmetic script that imports no existing engine/model modules.

Report counts and uncertainty once the entire screen and independent verifier
complete. Preserve fixed seeds/rules if a mechanical issue is discovered; keep
failure evidence and document any necessary correction before rerunning anything.
