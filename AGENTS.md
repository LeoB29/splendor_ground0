# Research continuity and future training

Read `AGENT_CURRENT.md` first, then the relevant section of `AGENT_HANDOFF.md`.
The worktree is intentionally dirty. Preserve existing work and frozen evidence.

## Required rules for new collection and training

All future collection, fitting and fresh evaluation must use the corrected
two-player rules described in `docs/engineering/future_training_rules_20261021.md`.
The historical default `SplendorEnv`, old snapshot decoders and old training
dataset loaders are NOT approved routes for new corrected-rules training.

Use the additive helpers in `tools/future_training_rules_20261021.py` and bind
the corrected ruleset/catalogue in new source, corpus and checkpoint metadata.
Call `verify_runtime_sources()` once at startup in every new collector, trainer
and evaluator worker, including fit-only jobs that do not create an environment.
Before a new job starts, its chosen entry point must integrate the checks and
pass the end-to-end counterexamples listed in the policy. A helper self-check,
manifest declaration or outer corrected environment alone is not sufficient.
There is currently no generally migrated training entry point.

Never relabel historical targets as corrected data. Old checkpoints may be used
as explicitly identified warm starts. Historical reproduction is a separate,
clearly labelled task; it must not silently become a new corrected-rules study.

## Running and historical experiments

Do not modify active/frozen players, source files, settings, data, termination
rules or evidence as part of future migration. The current matched benchmark
continues unchanged. New rules/interface work belongs in separate versioned
files. Read the active freeze inventory before editing shared files.

The pre-reveal token-return/noble restriction and forced-pass/deadlock conventions
are documented follow-ups, not authority to change the running experiment or
launch training. The corrected catalogue/noble requirement is mandatory for new
training; a sequential turn interface is not an automatic prerequisite.
