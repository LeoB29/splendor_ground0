"""Additive identity and bounded execution helpers; no model import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import PersistentWorker
from tools.external_strength_screen_20261018_02.common import ProcessTreeWorker

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/move_evaluation_64_20261023_01'
PREP = ROOT/'outputs/move_evaluation_64_20261023_preparation_01'
OLD_PREP = ROOT/'outputs/move_evaluation_study_20261023_preparation_01'
BUILD = ROOT/'outputs/move_evaluation_20261023_build_01'
PROTOCOL = ROOT/'docs/engineering/move_evaluation_64_study_20261023.md'
PROFILE = ROOT/'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PROFILE_SHA = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
GiB = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': file_hash(p), 'bytes': p.stat().st_size}


def verify(items):
    for item in items:
        require(artifact(ROOT/item['path']) == item, 'Changed frozen input: '+item['path'])


def save(path, value):
    p = Path(path)
    require(p.resolve().is_relative_to(OUT.resolve()), 'Output outside study directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    return artifact(p)


def selector_payload(row):
    return {k: row[k] for k in ('observer', 'fit', 'candidates', 'selection_schedule')}


class StudyWorker(ProcessTreeWorker):
    """Reuse the qualified clock/tree sampler with one new authenticated native."""
    def __init__(self, *args, native_artifact=None, **kwargs):
        self._tree_cpu = {}
        self._tree_identities = set()
        self._tree_roles = {}
        self._role_owners = {}
        self._verified_executables = {}
        self._console_startup_affinity = {}
        self._initializing = True
        self._python_executable = Path(sys.executable).resolve()
        self._native_executable = (ROOT/native_artifact['path']).resolve() if native_artifact else None
        self._expected_executables = {self._python_executable: file_hash(self._python_executable)}
        if native_artifact:
            verify([native_artifact])
            self._expected_executables[self._native_executable] = native_artifact['sha256']
        self._console_executable = (Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve() if os.name == 'nt' else None
        if self._console_executable:
            self._expected_executables[self._console_executable] = file_hash(self._console_executable)
        try:
            PersistentWorker.__init__(self, *args, **kwargs)
        finally:
            self._initializing = False
