# Experiment 139 preparation

Prepared on 2026-09-07 under the owner's explicit authorization to execute the
proposed propose/check/test experiment. Scientific protocol was frozen on
2026-09-06, before selection or new continuation outcomes.

## Frozen identities and workload

- Protocol: `docs/engineering/independent_confirmation_root_policy_139.md`.
  SHA256 `f07956464427aeb86456116a89ca1859e12a87d89ad1923c8980c9e80f58fb57`.
- State manifest file SHA256:
  `3acddd3c887786ba853c2d08568111a311127f1d448f85aca459de52bc9c6bdb`.
- Canonical JSON state-manifest digest used in launch identity:
  `435b23bb9bb288ee69ce4a3b73f8b78b0e1d5c8acab1b814fc06d75818915b03`.
- Preflight file SHA256:
  `698403fd4d670090490821a088ac5cdc99832726f93092c113b3a460e522e17e`.
- Source 086 selection/decisions and legal 035 checkpoint hashes are unchanged
  from 138 and verified by the runner. All src Python files, the new wrapper,
  tests and independent verifier are included in dependency identity.

The manifest contains 60 positions from 60 distinct source games, ten per
phase/seat cell, with no source-game overlap with the 138 panel. All original
proposal/default action bindings are preserved. Selection uses identifiers only.
All 1,440 structural blocks passed root information, inventory, hidden-deck
permutation, legal-reply and accounting checks. Original-world branches reproduce
the source builder. No actual reply-membership mismatch or state exclusion occurred.

Exact protected workload is 117,504 continuations out of the maximum 138,240,
with identical actions sharing their evaluation. Eight off-panel runtime probes
on source sample 0 took 1.3910632 seconds in total. The conservative serial
projection is 48,075.14 seconds (13.35 CPU-hours), below the 36-hour preparation
ceiling. This is a conservative serial projection, not a promised elapsed runtime.
Use ten CPU workers, one Torch thread each. Structural/probe preparation took
24.44 seconds. Only timing and integrity were retained from the eight probes.

## Verification and preparation corrections

The final focused 138/139 suite passes 26/26 in 1.70 seconds. Final full-suite
verification passed 921/921 in 199.55 seconds before protected launch. Earlier full runs passed
919/919 and 920/920 before the two preparation corrections described below.
The independent verifier passed a synthetic grouped-minimum reconstruction and
rejected a deliberately corrupted final score, without simulating a game.

Two preflight-only issues were fixed before evidence:

1. JSON saves deck-count keys as strings; deserialization restores integer keys.
   The new snapshot guard now compares canonical JSON rather than Python key
   types. No serialized game information is ignored or changed.
2. The inherited 138 branch guard compared ordered reply tuples. Sample 252's
   test block 4 contained identical four-reply sets in different ranking order.
   The 139 adapter calls the unchanged 138 branch constructor per world, then
   checks semantic membership across worlds. It preserves every reply, its local
   ordering, transition and seed; the same information-set minimum still applies.
   Changed membership remains an integrity failure. No 138 source was edited.

Both failed preparation manifests and explanatory records are preserved in
`outputs/independent_confirmation_root_policy_139/preparation_attempt_01/` and
`preparation_attempt_02/`. Their selected positions and original action choices
match the final manifest exactly. The protocol, selector, seeds, screen and
scientific thresholds were never changed. Neither failed attempt ran any probe
or protected continuation.

The structural diagnostic also found different 035 scores under cosmetic
permutations collapsed by the canonical information key. That limitation of the
frozen scorer is recorded, not repaired by this experiment. The incident did not
change the actual four-reply membership anywhere in the full prepared schedule.

## Execution contract

```powershell
& 'C:\Users\leoba\anaconda3\python.exe' -u run_independent_confirmation_root_policy_139.py --launch
```

Launch verifies the preparation/dependency identities and acquires a kernel lock.
All 60 confirmation records must finish before the all-state choice manifest is
written. Only then can test-stage continuations begin. Test workers receive the
fixed original action pair without confirmation outcomes. Atomic checkpoints
and choice hashes prevent a resume from changing the screening policy.

Report only progress, time and integrity until all test positions complete.
Then run `scripts/analysis/verify_independent_confirmation_139.py` to reconstruct
every return, grouped value, screening decision, policy contrast and primary
bootstrap interval independently. No model training, full-game arena, promotion,
installation, or automatic successor is included. Exact 084/035 and the dirty
worktree are preserved.
