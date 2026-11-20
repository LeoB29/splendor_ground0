"""Bounded aggregate timing around the unchanged eager or graph callables.

The inherited bindings wrap the currently installed tensor-forward function,
including the qualified graph replacement. No tensor/module hooks are added.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from tools.splendorust_component_timing_20261025.child_profile import (
    ChildProfiler as _OriginalProfiler, InstrumentedBindings, LABELS, TOTAL_FIELDS,
)

SCHEMA = 'splendorust_graph_timing_child_v1'
MAX_PROFILE_BYTES = 8192
MAX_BATCH_PROFILES = 64
MAX_BATCH_BYTES = 512 * 1024
REQUEST_ID = re.compile(r'request-[0-9]{8}\Z')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class ChildProfiler(_OriginalProfiler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_inference_end_ns = None
        self.last_inference_wall_ns = None
        self.last_inference_cpu_ns = None

    def end(self, frame, failed=False):
        # Reconstruct the inherited end sample without an extra clock call.
        before = tuple(self.totals['inference']) if frame[0] == 'inference' else None
        super().end(frame, failed)
        if before is not None:
            row = self.totals['inference']
            self.last_inference_wall_ns = row[1] - before[1]
            self.last_inference_cpu_ns = row[2] - before[2]
            self.last_inference_end_ns = frame[1] + self.last_inference_wall_ns

    def snapshot(self, *, adapter=None, callbacks_started=None):
        value = super().snapshot()
        value['schema'] = SCHEMA
        value.pop('before_emit_ns')
        value.pop('post_handle_gap_includes_profile_assembly_serialization_transport_reader_and_scheduling')
        row = self.totals['inference']
        # SearchOpponent.exchange replaces its list before native work. Merely
        # inspecting an unchanged list after decoder rejection would report
        # stale progress from the previous request. Observe identity only;
        # never clear or otherwise mutate the scientific adapter's state.
        current = None if adapter is None else getattr(adapter, 'callbacks', None)
        exchange_observed = current is not None and current is not callbacks_started
        callbacks = current if exchange_observed else ()
        value['partial_counters'] = dict(
            inference_calls_returned=row[0] - row[4], inference_calls_failed=row[4],
            last_inference_end_ns=self.last_inference_end_ns,
            last_inference_wall_ns=self.last_inference_wall_ns,
            last_inference_cpu_ns=self.last_inference_cpu_ns,
            last_exchange_callback_hash_count=len(callbacks),
            last_exchange_callback_rows=sum(item['batch'] for item in callbacks),
            last_exchange_observed_in_handle=exchange_observed,
            callback_scope='last_native_select_exchange_only_not_all_decision_phases',
            inference_scope='all_tensor_inference_calls_in_this_handle; returned does not imply deadline acceptance')
        return value


def validate_payload_size(value, *, batch=False):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    maximum = MAX_BATCH_BYTES if batch else MAX_PROFILE_BYTES
    if len(encoded) > maximum:
        raise ValueError('Child timing evidence exceeds bounded size')
    return encoded


def publish_profile(folder, request_id, value, *, batch=False):
    """Publish after success emission; preserve unfinished temporary on I/O error.

    Windows rename is atomic and refuses an existing destination. Never replace
    a prior request's sidecar, including on an error path.
    """
    if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError('Timing sidecar request identity')
    folder = Path(folder)
    target = folder / (request_id + '_child_profile.json')
    temporary = folder / (request_id + '_child_profile.pending')
    encoded = validate_payload_size(value, batch=batch) + b'\n'
    if target.exists():
        raise FileExistsError(str(target))
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    # The supported runtime is Windows, whose os.rename does not overwrite.
    # A preexisting destination is an error on every supported study host.
    os.rename(temporary, target)
    return target
