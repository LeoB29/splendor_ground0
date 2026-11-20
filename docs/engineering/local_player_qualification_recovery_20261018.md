# Explicit supervisor memory recovery

The first qualification execution stopped before starting any player worker:
available physical memory fell below the frozen one-GiB prelaunch reserve.
Its launch, failed case, completion and external catalogue failure are retained
under `outputs/local_player_qualification_20261018_01/`. No model initialization,
timed decision or trajectory from that execution is counted as a success.

The observer conversion imported Torch into the parent and retained all original
root records. The recovery prepares and seals lawful observations in a separate
short-lived process, which exits before timing starts. Its supervisor imports no
Torch during player timing and reads one original record per case. This changes
supervisor memory use only. Frozen local players, models, original records,
selected positions, 98-case schedule, clock, priority, affinity, 128-world work,
expected-result verifier and resource thresholds are unchanged.

The frozen numerical code remains untouched. New recovery code and prepared
observer payloads are separately bound before launch. Original true states and
expected decisions still stay with the referee. Hidden variants have identical
lawful observation hashes and remain separate predetermined repetition cases.
All original 65 synthetic checks remain bound and unchanged; review verifies the
memory-only recovery and its explicit prohibition on early Torch import.

The four-hour ceiling is measured from the FIRST launch, not reset for recovery.
One shared OS execution lease still permits only one supervisor. This is one
explicit recovery of an infrastructure failure before any player ran, not a
silent retry of a failed or timed-out player. Any new failure is preserved and
stops at the same applicable boundary. The original failure is linked in the
recovery result and remains part of the report. The external screen remains
blocked by the independently confirmed six-card catalogue mismatch.
