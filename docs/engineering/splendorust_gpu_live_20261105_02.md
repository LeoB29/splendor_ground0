# Separate Windows status-reader repair — 5 November 2026

This is a separately frozen technical correction within the user-authorized GPU
integration task. Preserve all attempt01 source and evidence. No player, weights,
rules, search settings, samples, numerical gates or deadlines are changed.

Attempt01 passed all35 historical CPU cases,840 GPU packing cases and80 concurrent
service calls. During CPU entrypoint qualification, Windows denied atomic replacement
of active_decisions.json during a sharing conflict. Later writer-only stress also
reproduced the error, so our own reader is not established as the sole cause; the
additional external/system sharing participant or filesystem behavior is unidentified.
The error occurred while clearing the completed reference operation's status. The
worker failed closed, controller observed exit1, and cleanup succeeded. No game began.
This is an orchestration failure, not a GPU numerical or playing-strength result.

Actual model-free OS tests found that adding sharing flags alone does not suffice
on this host. A ReplaceFileW alternative passed a single held-handle case but failed
concurrent stress, including a publisher error1175. Preserve both rejected technical
implementations and test receipts; neither was launched with models or games.

The final correction coordinates status access with one Windows named mutex per
canonical absolute status path. Readers hold it across open/read/close; the writer
holds it across in-place open/write/flush of at most64KiB, using its already
completed temporary bytes. It then releases the reader mutex before fsync on the
retained handle and closes the handle; there are no further content writes. The
single publisher's existing ownership lock remains held through return. Any fsync
failure propagates, and final deadline accounting includes the complete publication.
This avoids making readers wait behind slow disk durability flushing; a failed
250ms contention stress of the earlier fsync-inside-lock version is preserved.
No rename or deletion is used for publication. The fixed
temporary remains as bounded latest-status evidence and is overwritten on the next
publication. A mutex with rename still failed writer-only stress and is also preserved
as a rejected implementation. Readers cannot observe our writer's partial content;
any write/read/parse error or abandoned ownership fails closed.

The lock is separate from search/inference and has a250ms acquisition bound. No
stale cached status, polling omission, publication retry or clock reset is permitted.
Read handles also permit read/write/delete sharing; consistency relies on coordination.
Apply only to mutable active-decision and stage status. All original9/10s and study
clocks keep running through I/O; final whole-operation expiry is also checked after
publication. Model-free tests reproduce the old sharing conflict, verify coordinated
held-reader behavior, bound lock waits and stress1000publications both writer-only
and with two separate reader processes in the workspace where failures reproduced.
The full prior90 tests run against the separate copied version.

Everything else follows docs/engineering/splendorust_gpu_live_20261105.md, whose
frozen bytes are retained. Run fresh full qualification before any of the originally
specified eight CPU plus eight GPU games; accept none of the partial attempt01
qualification as a pass. There is no failed gameplay decision or game to retry.
The previously interrupted reference qualification invocation is repeated and
explicitly disclosed as engineering recovery; it was not a delivered response.
Preserve the failed run and its original execution receipts with hash bindings.

The original total clock remains11:40:27 Paris to13:40:27 Paris on5November2026.
Use the original wall and monotonic origins from attempt01; this correction does
not add time. The worker also retains its original100-minute deadline (approximately
13:20:27 Paris), followed by the original total deadline. All actual RSS/disk/output bounds and no-forecast/no-RAM-reserve
policy remain. No automatic third attempt is included. Require independent review,
separate freeze, actual exit/cleanup and independent saved-evidence acceptance.
