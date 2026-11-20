"""Boundary evidence helpers; no inference, native, hashing or CPU hooks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re

SCHEMA = 'splendorust_boundary_timing_child_v1'
MAX_PROFILE_BYTES = 8192
MAX_BATCH_PROFILES = 64
MAX_BATCH_BYTES = 512 * 1024
REQUEST_ID = re.compile(r'request-[0-9]{8}\Z')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


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
