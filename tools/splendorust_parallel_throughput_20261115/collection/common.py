"""Versioned evidence context; never rebind an accepted public module."""
from pathlib import Path
import os
from .. import common as root
from . import storage

ROOT, OUT = root.ROOT, root.OUT/'collection'
PLAN, CLAIM = root.PLAN, root.OUT/'claim.json'
INTERNAL_SECONDS, OUTER_SECONDS = 60., 75.
EXECUTION_POLICY_ID = 'throughput_fixed512_internal60_outer75_v1'
TOTAL_SECONDS, MODEL_SECONDS = 6*3600, 4.5*3600
MAX_RSS, MAX_OUTPUT, MIN_DISK = 2*1024**3, 10*1024**3, 5*1024**3
require, canonical, digest = root.require, root.canonical, root.digest
artifact, sha, resolve, verify = root.artifact, root.sha, root.resolve, root.verify
now = root.now


def read(path):
    return storage.parse_bytes(Path(path).read_bytes())


def save(path, value):
    path = root.normalized(path)
    require(path.is_relative_to(root.normalized(OUT)), 'Only separate pilot collection evidence')
    raw = (canonical(value)+'\n').encode('utf-8')
    payload = storage.encode_bytes(raw) if path.parent.name in storage.COMPRESSED_DIRECTORIES else raw
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return artifact(path)


def atomic_json(path, value):
    require(Path(path).name in ('stage.json', 'progress.json'), 'Explicit mutable status only')
    return root.status(path, value)
