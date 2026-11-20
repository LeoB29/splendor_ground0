"""Prepared technical correction for the practical study's idle phase handoff.

This module has no launch or retry entrypoint. A separately approved controller
may call ``advance_idle_worker(..., common=its_own_common)``. Frozen controllers
and their evidence are never rebound or modified by this module.
"""
import math
import time


def live_worker_identity(child, *, common):
    """Bind the live owner's separate identity to its full durable READY record.

    StudyWorker.initialization is the unaugmented response receipt. The saved
    initialization.json additionally contains identity, command and host fields.
    Consequently, saved readers correctly use saved['identity']; live code must
    use child.identity instead of child.initialization['identity'].
    """
    c = common
    identity = child.identity
    c.require(type(identity) is dict and set(identity) == {'pid', 'created'} and
              type(identity['pid']) is int and identity['pid'] > 0 and
              type(identity['created']) in (int, float) and
              math.isfinite(identity['created']) and identity['created'] > 0,
              'Exact live worker PID and process creation identity')
    c.require(identity['pid'] == child.process.pid == child._ps.pid and
              identity['created'] == child._ps.create_time(),
              'Live owning process identity has not changed')
    saved = c.read(child.output_dir / 'initialization.json')
    c.require(saved['identity'] == identity and
              saved['result']['pid'] == identity['pid'],
              'Live identity matches durable initialization and READY PID')
    c.require(child.initialization['status'] == 'ok' and
              child.initialization['result']['pid'] == identity['pid'] and
              all(key in saved and saved[key] == value
                  for key, value in child.initialization.items()),
              'Unaugmented live READY receipt matches the durable receipt')
    return dict(identity)


def advance_idle_worker(child, envelope, *, common):
    """Preserve the existing idle-only, original-clock handoff with valid identity.

    Validate and exclusively publish the receipt before changing the idle phase
    deadline, so an identity or publication failure leaves that deadline intact.
    No active request clock, allowance, player configuration or scientific input
    is changed. This helper does not authorize any new attempt or recovery.
    """
    c = common
    c.require(not child._closed and child._counter == 3 and
              child.process.poll() is None,
              'Only a live idle fully qualified worker may enter evaluation')
    c.require(not child._failure and not child._lock.locked(),
              'No active or failed decision clock is moved')
    old = child.absolute_deadline
    c.require(old <= envelope.start + c.COLLECTION_CUTOFF_SECONDS and
              envelope.deadline <= envelope.start + c.COLLECTION_CUTOFF_SECONDS,
              'Both phase limits belong to the original model cutoff')
    identity = live_worker_identity(child, common=c)
    receipt = dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time(),
        perf_counter_ns=time.perf_counter_ns(), variant=child.output_dir.name,
        worker_pid=identity['pid'], worker_created=identity['created'],
        completed_startup_requests=3, next_request_id='request-00000004',
        old_absolute_deadline=old, new_absolute_deadline=envelope.deadline,
        original_monotonic_origin=envelope.start, original_unix_origin=envelope.wall_start,
        active_request=False, request_allowance_unchanged=10.)
    artifact = c.save(c.OUT / 'worker_phase_transitions' /
                      f'{child.output_dir.name}.json', receipt)
    child.absolute_deadline = envelope.deadline
    return artifact
