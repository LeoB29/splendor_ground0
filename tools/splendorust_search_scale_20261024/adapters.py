"""Two authenticated PUCT quotas sharing immutable CPU weights, never tree state.

Imports do not load a model. Each native select constructs a fresh native agent;
only the challenger iteration quota differs from the established PUCT16 player.
"""
from __future__ import annotations

import atexit
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import (
    PIN, CHECKPOINT_SHA256, Opponent, file_hash,
)

PROFILE_PATH = ROOT / 'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PROFILE_SHA256 = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
SEARCH_BINARY_SHA256 = 'b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c'
REFERENCE_PROFILE_ID = 'entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1'
INTERNAL_SECONDS = 0.900
MODEL_ARRANGEMENT = 'one_shared_immutable_entity_model_two_native_transports'
ARCHITECTURE_WARMUP = 'one_batch_of_zero_31x48_tokens_by_frozen_Opponent_constructor'
STATE_PERSISTENCE = 'immutable_model_and_architecture_only_fresh_native_agent_per_phase'


def require(condition, message):
    if not condition:
        raise ValueError('SEARCH_SCALE_ADAPTER: ' + message)


def metadata_identity(quota):
    from . import common as c
    require(type(quota) is int and quota in (32, 64), 'challenger quota must be32or64')
    return {
        'schema': 'splendorust_search_scale_worker_identity_v1',
        'model_arrangement': MODEL_ARRANGEMENT, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'search_simulations': quota, 'reference_simulations': 16,
        'search_profile_id': c.profile_id(quota), 'search_profile_sha256': c.profile_sha256(quota),
        'reference_profile_id': c.profile_id(16), 'reference_profile_sha256': c.profile_sha256(16),
        'reference_wrapper_id': c.wrapper_id(16), 'search_wrapper_id': c.wrapper_id(quota),
        'search_native_sha256': SEARCH_BINARY_SHA256, 'reference_native_sha256': SEARCH_BINARY_SHA256,
        'architecture_warmup': ARCHITECTURE_WARMUP, 'internal_seconds': INTERNAL_SECONDS,
        'state_persistence': STATE_PERSISTENCE,
    }


def validate_ready_metadata(metadata, *, expected_sources, worker_pid, process_tree, expected_quota=None, challenger_quota=None):
    """Bind both quotas to the parent's plan and trusted live process ownership."""
    from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
    if challenger_quota is not None:
        require(expected_quota is None or expected_quota == challenger_quota, 'conflicting expected quotas')
        expected_quota = challenger_quota
    expected = {**metadata_identity(expected_quota), 'pid': worker_pid, 'threads': 1,
                'sources': expected_sources, 'ruleset_id': RULESET_ID,
                'catalogue_sha256': CATALOGUE_SHA256, 'turn_contract': TURN_CONTRACT}
    live = {}
    for row in process_tree:
        role = row.get('role')
        if role in ('python', 'search', 'reference'):
            require(role not in live, 'duplicate live compute role')
            live[role] = row['pid']
    require(set(live) == {'python', 'search', 'reference'} and live['python'] == worker_pid,
            'three exact live compute roles required at readiness')
    expected['native_pids'] = {role: live[role] for role in ('search', 'reference')}
    require(type(metadata) is dict and
            json.dumps(metadata, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), 'ready metadata differs from frozen identity')
    return {'status': 'passed', 'live_roles': live, 'metadata_sha256': digest(metadata)}


class _Deadline:
    """Fail closed at all Python boundaries; parent enforces transport time."""
    request_deadline = None
    active_deadline = None
    admission_guard = None

    def boundary(self):
        deadline = self.active_deadline
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: 0.900-second allowance exceeded')

    def _begin(self):
        if self.admission_guard is not None:
            self.admission_guard()
        self.active_deadline = (self.request_deadline if self.request_deadline is not None
                                else time.perf_counter() + INTERNAL_SECONDS)
        self.boundary()

    def _finish(self):
        self.active_deadline = None


class SearchOpponent(_Deadline, Opponent):
    """Unchanged native PUCT; quota and identity are supplied by the frozen profile."""
    def __init__(self, profile_document, *, shared_from=None):
        self.capture = False
        self.callbacks = []
        if shared_from is None:
            super().__init__(profile_document)
            return
        # The second transport shares only immutable host weights/constants.
        # Its executable is a byte-identical copy to make live roles unambiguous.
        self.profile = profile_document
        require(self.profile['opponent_id'] == 'splendorust' and
                self.profile['deterministic_fixed_work'], 'unsupported search profile')
        for item in self.profile['artifacts']:
            path = ROOT / item['path']
            require(path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'],
                    'opponent input identity: ' + str(path))
        self.model, self.reference = shared_from.model, shared_from.reference
        self.proc = subprocess.Popen([str(ROOT / self.profile['native_binary'])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1, cwd=ROOT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        atexit.register(self.close)
        try:
            ready = json.loads(self.proc.stdout.readline())
            require(ready == {'kind': 'ready', 'bridge': 'observer_only_fixed_work_v1',
                             'source_pin': PIN}, 'native bridge identity')
        except BaseException:
            self.close()
            raise
        self.deadline = None
        self.inference_calls = self.inference_rows = 0

    def _infer(self, tokens, batch):
        self.boundary()
        outputs = Opponent._infer(self, tokens, batch)
        self.boundary()
        # Count actual host callbacks independently of native leaf_calls:
        # late-game heuristic leaves deliberately do not invoke this method.
        input_sha, output_sha = digest(tokens), digest(outputs)
        item = {'callback_sha256': digest([input_sha, output_sha, batch]), 'batch': batch}
        if self.capture:
            item.update(input_sha256=input_sha, output_sha256=output_sha)
            # The first callback is the root encoder counterexample. Recording
            # every hypothetical input would add no required audit evidence.
            if not self.callbacks:
                item.update(tokens=copy.deepcopy(tokens), outputs=copy.deepcopy(outputs))
                if batch == 1:
                    item['output83'] = copy.deepcopy(outputs[0])
        self.callbacks.append(item)
        self.boundary()
        return outputs

    def exchange(self, request):
        self.boundary()
        self.callbacks = []
        result = Opponent.exchange(self, request)
        self.boundary()
        result = {**result,
                  'neural_callback_count': len(self.callbacks),
                  'neural_callback_rows': sum(callback['batch'] for callback in self.callbacks),
                  'neural_callbacks': [callback['callback_sha256'] for callback in self.callbacks]}
        if self.capture:
            result = {**result, 'inference': [
                {key: copy.deepcopy(value) for key, value in callback.items() if key != 'callback_sha256'}
                for callback in self.callbacks]}
        return result

    def choose(self, observer, *, diagnostic_capture=False):
        self._begin()
        self.capture = bool(diagnostic_capture)
        try:
            action, trace = Opponent.choose(self, observer)
            self.boundary()
            return action, trace
        except TimeoutError as error:
            # The original adapter has its own 0.9s checkpoints. Preserve that
            # failure while giving the supervisor one unambiguous category.
            raise TimeoutError('ONE_SECOND_INTERNAL_DEADLINE: ' + str(error)) from error
        finally:
            self.capture = False
            self._finish()


    def reference_choose(self, observer, *, diagnostic_capture=True):
        """Run original unbound choose for a separate reference16 reproduction."""
        require(self.profile['iterations'] == 16, 'reference reproduction is PUCT16 only')
        self._begin()
        self.capture = bool(diagnostic_capture)
        try:
            result = Opponent.choose(self, observer)
            self.boundary()
            return result
        finally:
            self.capture = False
            self._finish()


def initialize_adapters(output_dir, quota):
    """Authenticate both byte-identical transports and initialize exactly one model."""
    from . import common as c
    identity = metadata_identity(quota)
    require(file_hash(PROFILE_PATH) == PROFILE_SHA256, 'original reference profile identity')
    profiles = {'reference': c.profile_for(16), 'search': c.profile_for(quota)}
    for arm, count in (('reference', 16), ('search', quota)):
        profile = profiles[arm]
        require(profile['profile_id'] == c.profile_id(count) and profile['iterations'] == count,
                'scientific profile quota')
        require(profile['internal_budget_seconds'] == INTERNAL_SECONDS, 'internal search limit')
        require(file_hash(ROOT / profile['native_binary']) == SEARCH_BINARY_SHA256,
                'native executable identity')
    require(profiles['reference']['native_binary'] != profiles['search']['native_binary'],
            'separate executable paths required for exact process roles')
    reference = SearchOpponent(profiles['reference'])
    try:
        search = SearchOpponent(profiles['search'], shared_from=reference)
        require(search.model is reference.model and search.reference is reference.reference,
                'shared immutable architecture identity')
        require(not reference.model.training, 'evaluation mode')
    except BaseException:
        reference.close()
        if 'search' in locals():
            search.close()
        raise
    identity['native_pids'] = {'search': search.proc.pid, 'reference': reference.proc.pid}
    return {'search': search, 'reference': reference}, identity
