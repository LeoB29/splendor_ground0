# Recovery implementation review before production

20 October 2026. This reviews the separately authorized recovery described in
`live_game_operational_recovery_20261020.md`. Production admission still requires
the frozen preflight suite, original-source preservation, exact saved-position
reconstruction and stable RAM admission. This review is not a completed-game
result or a playing-strength assessment.

The primary agent reviewed the recovery parent, worker path boundary, independent
analyzer and completion helper. The implementation and analyzer authors reviewed
each other's schemas and source mappings. A third reviewer checked the resource
envelope, combined stress selection, clock separation and completion helper.
The exact reviewed code/test/protocol hashes are retained in
`outputs/live_game_operational_recovery_20261020_preparation/reviewed_inputs.json`.

The implementation preserves the five completed games and both timeout
forfeits. It independently reconstructs the partial game from its original seed
and contiguous accepted actions, carries the reconstructed environment forward,
and admits one request at the previously uninitialized ply 15. Synthetic tests
reject changed seeds, gaps, altered observers, modified receipts, initialized
requests, overwritten imports and repeated completed decisions. A continuation
fixture compares restored and uninterrupted environment states.

Worker syntax-tree comparisons allow only the new common-module import and
authenticated audit-only source-trace path. The numerical decision function,
profiles, contexts, sampling functions and proof contracts remain unchanged.
Local trace writing cannot use the original evidence tree. Audit reads of an
original trace require containment under its game-worker tree and an exact
frozen artifact hash/size. Traversal and outside-path rejection are tested.
Windows does not permit the test user to create a symlink, so that one fixture
is explicitly skipped; resolved-path containment and the other path cases run.

The independent analyzer preserves separate original and recovery worker,
receipt, timing and process inventories. It authenticates the original full
validation evidence, exact copy aliases, complete original snapshot and the
unstarted failure. It independently joins the partial game's original prefix
to its new suffix and checks every actual transition and ending internally.
The original sample/stress selection is applied once across the combined
eighteen-game stage, not independently per attempt.

Review found and resolved two integration issues before freezing: the recovery
failure list needed to bind the original progress and runner-exit receipts;
worker admission and protected reserves needed to use the smaller wall-clock
and monotonic remainder. The converted monotonic worker deadline uses a timestamp
captured before the resource check, avoiding addition of guard overhead.
The stable-RAM admission guard uses the same smaller remainder. Targeted
forward-wall-clock tests cover these cases.

The completion helper binds its source, the scientific freeze, the prospective
completion plan, the new runner identity and both clock origins. It runs one
frozen analyzer only after complete successful collection/replay. It preserves
holds, handles only the bounded incomplete-JSON delivery window, bounds logs
and owned process resources, authenticates cleanup identities, and validates
final artifact hashes before acceptance. The report explicitly retains the
two earlier timing failures and distinguishes verification from deadline
reliability. No extra games or automatic retry are enabled.

Targeted synthetic suites cover recovery/state/path invariants, RAM admission,
the independent analyzer, and completion orchestration. Their combined actual
result is recorded by preparation in `preflight_tests.json`; it must be
successful before the freeze is emitted. Tests run on CPU 2 without player
inference or production requests. The initial RAM-test helper naming failure
and its source snapshot remain in the preparation directory.

All earlier failed evidence and original frozen files are retained. Game
outcomes stay closed, the larger-data study stays deferred, and the independent
analyzer's shared model/feature/rules trust boundaries remain explicit.
