"""Versioned profile and unchanged trusted clock for the 120-second study."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys

from tools.external_strength_screen_20261018_02.common import (
    ProcessTreeWorker as FrozenProcessTreeWorker,
    serialize_action, deserialize_action, legal_digest, verify,
)
from tools.local_player_qualification_128.common import (
    ROOT, MAX_SECONDS, MAX_BYTES, MIN_FREE, MAX_WORKER_RSS, ResourceHold,
    now, read, canonical, digest, file_hash, artifact, save_new, memory,
    directory_bytes, boundary, execution_lease,
)
from tools.local_player_qualification_128.clock import PersistentWorker

OUTPUT = ROOT / 'outputs/splendorust_120_qualification_20261018_01'
BASE_PROFILE_PATH = 'outputs/external_strength_screen_20261018_02/opponent_profile.json'
BASE_PROFILE_SHA256 = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
BASE_PROFILE_BYTES = 15889
PROFILE_SCHEMA = 'splendorust_120_fixed_work_profile_v1'
OPPONENT_ID = 'splendorust'
EXTERNAL_SECONDS = 120
INTERNAL_SECONDS = 108
INITIALIZATION_SECONDS = 60
QUOTA_LADDER = (256, 1024, 4096, 6400)
RULESET_ID = 'splendor_base_qualified_rules_20261018_v1'
CATALOGUE_SHA256 = 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
PROFILE_CHANGES = frozenset(('schema', 'profile_id', 'iterations', 'internal_budget_seconds'))
PROFILE_ADDITIONS = frozenset(('external_budget_seconds', 'base_profile'))


def base_profile():
    path = ROOT / BASE_PROFILE_PATH
    if path.stat().st_size != BASE_PROFILE_BYTES or file_hash(path) != BASE_PROFILE_SHA256:
        raise ValueError('Frozen one-second baseline profile changed')
    return read(path)


def profile_id(iterations):
    if type(iterations) is not int or iterations not in QUOTA_LADDER:
        raise ValueError('Quota outside the declared qualification ladder')
    return f'entity_onehot_puct{iterations}_canonical_atomic_prereveal_cpu_120s_v1'


def make_profile(iterations):
    """Construct metadata only; never initialize or execute a model."""
    value = deepcopy(base_profile())
    value.update(schema=PROFILE_SCHEMA, profile_id=profile_id(iterations), iterations=iterations,
                 internal_budget_seconds=INTERNAL_SECONDS, external_budget_seconds=EXTERNAL_SECONDS,
                 base_profile={'path': BASE_PROFILE_PATH, 'sha256': BASE_PROFILE_SHA256,
                               'bytes': BASE_PROFILE_BYTES})
    return value


def validate_profile(value):
    """Only quota/deadlines and their explicit versioned identity may change."""
    if not isinstance(value, dict):
        raise ValueError('Opponent profile must be an object')
    baseline = base_profile()
    if set(value) != set(baseline) | PROFILE_ADDITIONS:
        raise ValueError('Unexpected versioned profile fields')
    quota = value['iterations']
    if (value['schema'] != PROFILE_SCHEMA or value['profile_id'] != profile_id(quota) or
            type(value['internal_budget_seconds']) is not int or value['internal_budget_seconds'] != INTERNAL_SECONDS or
            type(value['external_budget_seconds']) is not int or value['external_budget_seconds'] != EXTERNAL_SECONDS or
            value['base_profile'] != {'path': BASE_PROFILE_PATH, 'sha256': BASE_PROFILE_SHA256,
                                      'bytes': BASE_PROFILE_BYTES}):
        raise ValueError('Invalid versioned profile identity or deadlines')
    for field in set(baseline) - PROFILE_CHANGES:
        # Canonical JSON also distinguishes bool from int and int from float.
        if canonical(value[field]) != canonical(baseline[field]):
            raise ValueError('Changed frozen opponent setting: ' + field)
    return value


def load_profile(path, expected_sha256=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Opponent profile must be in the workspace')
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError('Opponent profile hash changed')
    value = validate_profile(read(path))
    verify(value['artifacts'])
    return value


class ProcessTreeWorker(FrozenProcessTreeWorker):
    """Same sampler/receipt/termination code; only the profile loader is new."""
    def __init__(self, *args, **kwargs):
        self._tree_cpu = {}
        self._tree_identities = set()
        self._tree_roles = {}
        self._role_owners = {}
        self._verified_executables = {}
        self._console_startup_affinity = {}
        self._initializing = True
        self._python_executable = Path(sys.executable).resolve()
        environment = kwargs.get('env') or {}
        profile = load_profile(environment['SPLENDOR_OPPONENT_PROFILE'],
                               environment['SPLENDOR_OPPONENT_PROFILE_SHA256'])
        self._native_executable = (ROOT / profile['native_binary']).resolve()
        native_items = [v for v in profile['artifacts']
                        if (ROOT / v['path']).resolve() == self._native_executable]
        if len(native_items) != 1:
            raise ValueError('Native executable must have exactly one authenticated artifact')
        self._expected_executables = {
            self._python_executable: file_hash(self._python_executable),
            self._native_executable: native_items[0]['sha256'],
        }
        self._console_executable = ((Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve()
                                    if os.name == 'nt' else None)
        if self._console_executable is not None:
            self._expected_executables[self._console_executable] = file_hash(self._console_executable)
        try:
            # Bypass only FrozenProcessTreeWorker.__init__'s 0.9-second loader.
            # Its sampler and every PersistentWorker clock/kill path are inherited.
            PersistentWorker.__init__(self, *args, **kwargs)
        finally:
            self._initializing = False
