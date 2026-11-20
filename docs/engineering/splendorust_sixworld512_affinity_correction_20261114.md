# Six-world512 launch-order correction

14 November 2026. Attempt01 stopped before any model service process was
created: the controller selected CPU2 before the unchanged trusted clock checked
whether CPU0 was available. All failure records and the original freeze remain
immutable. No model decision, game or outcome exists in that attempt.

Attempt02 uses a separate copied package. It restores the controller's original
allowed CPU mask only while constructing a service, then returns the controller
to CPU2 before decisions. The trusted clock still pins the actual model to CPU0.
The same procedure must work for a later fresh replay service and after a genuine
timeout. Tests reproduce the failed admission and check this lifecycle.

Scientific calculations and player code are retained byte-for-byte. Changes are
limited to this transport startup correction, version/output identifiers,
original-clock ownership and explicit startup-correction reporting. The schedule, initial snapshots,
players, rules, 512 simulations, three/six worlds, confidence calibration,
400-game interims, teacher/player thresholds, deadlines and verification checks
are unchanged. Reuse the frozen prospective sample: no games were played.

Run model-free tests, three independent reviews and a detached lifecycle fixture
before a new execution freeze. Reuse passed statistical calibration as its code
and mathematical configuration are unchanged. The new freeze inventories all
attempt01 files and verifies every original input. No old file is edited.

Retain attempt01's original monotonic and UTC global deadlines and 216-hour model
cutoff; no added time budget. Actual-model qualification, six operational games,
main blocks and final verification remain automatically gated. No failed move is
retried and no active move clock is reset. This is a technical startup correction,
not a scientific change or promotion. Training and larger-data work stay deferred.
