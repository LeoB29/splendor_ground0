# Experiment 138 preparation record

Prepared 2026-09-06 following the owner's explicit request to execute the
experiment recommended in the independent 134/137 review.

The frozen protocol is
`docs/engineering/independent_world_root_validation_138.md`, SHA256
`30a080f364760a5c5885ff4a55d1a49bfbdde6aabaf9147b764a40f5cf573a78`.

Verification completed:

- Focused new-audit/085/086/grouped-resolver tests: 29 passed in 2.37 seconds.
- Full repository suite on the final implementation: 907 passed in 293.25 seconds.
- Structural preflight: all 60 states, ten per phase/seat cell and one per
  source game; exact original branch/reply reproduction, legal fresh worlds,
  hidden-order invariance, conserved observer information and card inventory,
  fixed action binding, and snapshot round trips passed.
- Eight off-panel runtime probes completed. Total probe wall time was 2.2506
  seconds; no probe outcome was serialized. Conservative maximum-work projection,
  including a factor of two allowance, was 25,926.77 CPU seconds (7.20 hours),
  below the frozen 12-hour preflight ceiling. This is a small-probe estimate,
  not a guarantee about ten-worker wall time.
- Actual prepared primary work is 39,552 continuations, below the 46,080 maximum,
  because identical fixed actions share their evaluation. No state was excluded
  for its outcomes or for whether the two choices agreed. No protected
  continuation was run during preparation.

Protected state-manifest file SHA256:
`5bcc900a2a03c0aa3fde95c036819164617cde2f65a0e450f8502e6371dbdee9`.
Its canonical-JSON identity digest, stored as `manifest_sha256` in preflight:
`1daffab9ddd6e05a354414c6b9b07f3d00e826b494c48f6dbf8112465b6cab05`.
Preflight file SHA256:
`bb3f9c4201592e113a00e78dd11ac91a6486ded33de006e64d699e7f30ed3d8f`.

The manifest also freezes every source Python dependency, the runner and tests,
the two 086 input artifacts, and the exact 035 checkpoint. The runner rechecks
these on launch/resume and before final reporting; a kernel lock prevents
concurrent copies. Every completed state must match its prepared world hashes
and exact continuation budget before being saved.

Authorized launch command (ten CPU workers, one Torch thread each):

```powershell
& 'C:\Users\leoba\anaconda3\python.exe' run_independent_world_root_validation_138.py --launch
```

Outputs: `outputs/independent_world_root_validation_138/`. The runner saves
atomic per-state results and can resume the same command after interruption.
Keep outcomes sealed until all 60 states finish. No training, promotion,
installation, new source games, full-game arena, or successor is authorized.
