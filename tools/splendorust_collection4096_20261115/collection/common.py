"""Explicit new evidence context; only immutable large JSON records compress."""
from pathlib import Path
import os
from .. import common as root
from . import storage

ROOT, OUT = root.ROOT, root.OUT/'collection'
PLAN, CLAIM = OUT/'plan.json', root.OUT/'claim.json'
INTERNAL_SECONDS, OUTER_SECONDS = 60., 75.
EXECUTION_POLICY_ID = 'throughput_fixed512_internal60_outer75_v1'
TOTAL_SECONDS, MODEL_SECONDS = root.TOTAL_SECONDS, root.MODEL_SECONDS
MAX_RSS, MAX_OUTPUT, MIN_DISK = root.COLLECTION_RSS, root.MAX_OUTPUT, root.MIN_DISK
require, now, canonical, digest = root.require, root.now, root.canonical, root.digest
artifact, sha, resolve, verify = root.artifact, root.sha, root.resolve, root.verify
nested_artifacts = root.nested_artifacts


def read(path):
    return storage.parse_bytes(Path(path).read_bytes())


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()), 'Only new collection evidence')
    raw = (canonical(value)+'\n').encode('utf-8')
    payload = storage.encode_bytes(raw) if path.parent.name in storage.COMPRESSED_DIRECTORIES else raw
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if root.WRITE_OBSERVER is not None:
        root.WRITE_OBSERVER(path, len(payload))
    return artifact(path)


def atomic_json(path, value):
    require(Path(path).name in ('stage.json', 'progress.json'), 'Explicit mutable status')
    return root.status(path, value)
