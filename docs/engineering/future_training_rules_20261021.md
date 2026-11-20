# Corrected rules required for future training

21 October 2026. Owner instruction: make sure future training uses corrected
rules, record remaining fixes, and leave the current experiment unchanged.

This is an additive future-work requirement. It does not migrate frozen code,
relabel old data, authorize training, or alter the running matched benchmark.
No general training entry point is yet migrated or qualified. New collection
and training must not launch through an old entry point until the checks below
are integrated and demonstrated for the chosen pipeline.

## Required identity and implementation

- Ruleset: `splendor_base_qualified_rules_20261018_v1`.
- Catalogue SHA256: `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
- Current turn contract: `atomic_prereveal_completed_turn_v1`.
- Reviewed environment: `tools.external_strength_screen_20261018.rules.CorrectedEnv`.
- Additive factory, strict decoder, transition/data guards:
  `tools/future_training_rules_20261021.py`.

The guard authenticates eight reviewed rule-source files, constructs the
corrected environment, rejects unlabelled/legacy snapshot shapes and incompatible
data metadata, checks data hashes, and provides independent transition and
official terminal-target validation. It does not infer that an entire collector
or trainer is correct from its manifest. A declared provenance tag is not proof
of how an old file was generated.

`python -B -m tools.future_training_rules_20261021 self-check` checks only these
helpers. It explicitly reports `training_pipeline_qualified: false`. It loads
no learned model, launches no experiment, and is not a production admission.

## What must be checked before the next job

Call `verify_runtime_sources()` once at startup in every new collector, trainer
and evaluator worker, including fit-only jobs that never create an environment.
Save that receipt and bind source/data hashes and the ruleset in the next frozen
protocol, corpus metadata and saved checkpoint metadata. Per-row decoding does
not rehash source files. Cover the actual runtime chain:

1. Source-game collection and any imported starting positions.
2. Dataset reconstruction, including every snapshot decoder and afterstate cache.
3. Legal-action enumeration and the explicit legal lists supplied to encoders.
4. Hidden-world reconstruction: all 90 card objects must retain corrected faces.
5. Teacher decisions, continuation policies and every simulated transition.
6. Replay/audits, terminal targets and fresh evaluation.

Use the selected collector AND actual training dataset loader on bounded
counterexamples: each of the six corrected faces; a noble awarded after a
non-purchase; multiple eligible nobles; noble-triggered final-round termination;
gold payment/returns; private reserve reconstruction. Verify the corrected
costs/legal support in the encoded training batch, not just the outer game.
Confirm old/missing identity, mismatched hashes and historical reconstruction
are rejected. Keep independent transition checks and source identities in the
qualification receipt. No training optimization is needed for this check.

Known traps: `EntityActionReplayDataset` and other historical datasets construct
`SplendorEnv` and use the old `deserialize_state_snapshot`; they can silently
restore historical card faces even when supplied IDs from corrected games.
Some encoders default to historical legality unless explicitly supplied the
corrected legal list. Follow the explicit binding pattern in the corrected
planner; never globally rebind historical modules during an active experiment.

## Historical data and initialization

Do not mix historical or unlabelled targets into a purported corrected-rules
corpus. Adding tags or substituting card prices cannot correct old choices,
continuations, outcomes or labels. Regenerate corrected evidence, or design and
label any historical-data comparison separately with explicit authorization.
Historical reproduction remains possible without editing its frozen sources.

An old checkpoint may be an explicitly identified warm start: record its hash
and historical training origin separately. That does not certify its original
data as corrected or its strength as transferred. No full retraining is being
launched by this policy.

The earlier larger-data cost proposal's reuse of old corpora is historical
planning, not an approved corrected-rules execution plan. A new experiment must
specify and price its corrected baseline/data rather than silently inherit it.

## Remaining fixes and interpretation

- **Sequential reveal/return/noble choices:** both players precommit auxiliary
  choices before blind-reserve or market-refill information appears. Their final
  moves are legal, but available strategies are restricted. After the current
  benchmark is accepted, inspect affected decisions and prioritize a bounded,
  separate player version. A fixed helper used consistently in actual play and
  sampled worlds can avoid a neural-network redesign. Keep one total turn clock
  and reveal only information lawful at each phase. This is not an automatic
  prerequisite for another corrected-rules learning experiment.
- **Forced pass/deadlock:** no authoritative rule for a literally blocked turn
  was found. Current local rollouts score two forced passes by current score;
  the actual matched referee does not award that winner. New protocols must
  explicitly disclose internal search conventions. Do not treat cutoff/deadlock
  results as official completed-game labels; the helper rejects those targets.
- **Default-engine migration:** old defaults remain for historical evidence.
  A future controlled migration should make accidental historical routing hard,
  with old reproduction explicitly isolated. No active source is changed now.

Audit evidence: `outputs/rules_audit_20261021_01/audit_summary.json` and its linked
component, referee, action-set and adapter checks. All 90 cards and 10 nobles
matched fresh licensed BGA data; ordinary corrected referee tests passed.
