"""Separate timed wrappers around the two authenticated scientific policies.

Both native transports are persistent; their frozen bridges construct a fresh
native agent for each select/greedy operation. Only the immutable CPU model and
architecture constants are shared. This module never loads a model on import.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import (
    CHECKPOINT_SHA256, Opponent, file_hash,
)
from tools.move_evaluation_20261023.greedy_adapter import (
    BUILD, GreedyOpponent, NativeOnly, PROFILE_ID as GREEDY_PROFILE_ID,
)

PROFILE_PATH = ROOT / 'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PROFILE_SHA256 = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
SEARCH_BINARY_SHA256 = 'b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c'
GREEDY_BINARY_SHA256 = '843ef76a157537dfe93bcb215dedd0113995abf2127098b2847bceee8dc352c4'
SEARCH_PROFILE_ID = 'entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1'
GREEDY_WRAPPER_ID = 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_1s_20261024_v2'
SEARCH_WRAPPER_ID = 'entity_onehot_puct16_canonical_atomic_prereveal_cpu_1s_20261024_v2'
INTERNAL_SECONDS = 0.900
MODEL_ARRANGEMENT = 'one_shared_immutable_entity_model_two_native_transports'
ARCHITECTURE_WARMUP = 'one_batch_of_zero_31x48_tokens_by_frozen_Opponent_constructor'
STATE_PERSISTENCE = 'immutable_model_and_architecture_only_fresh_native_agent_per_phase'


def require(condition, message):
    if not condition:
        raise ValueError('ONE_SECOND_ADAPTER: ' + message)


def validate_ready_metadata(metadata, *, expected_sources, worker_pid, process_tree):
    """Validate initialized identities against trusted live process ownership."""
    from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
    expected = {
        'schema': 'splendorust_one_second_worker_identity_v1',
        'model_arrangement': MODEL_ARRANGEMENT, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'search_profile_id': SEARCH_PROFILE_ID, 'search_profile_sha256': PROFILE_SHA256,
        'greedy_profile_id': GREEDY_PROFILE_ID, 'greedy_wrapper_id': GREEDY_WRAPPER_ID,
        'search_wrapper_id': SEARCH_WRAPPER_ID,
        'search_native_sha256': SEARCH_BINARY_SHA256, 'greedy_native_sha256': GREEDY_BINARY_SHA256,
        'architecture_warmup': ARCHITECTURE_WARMUP, 'internal_seconds': INTERNAL_SECONDS,
        'state_persistence': STATE_PERSISTENCE, 'pid': worker_pid, 'threads': 1,
        'sources': expected_sources, 'ruleset_id': RULESET_ID,
        'catalogue_sha256': CATALOGUE_SHA256, 'turn_contract': TURN_CONTRACT,
    }
    live = {}
    for row in process_tree:
        role = row.get('role')
        if role in ('python', 'search', 'greedy'):
            require(role not in live, 'duplicate live compute role')
            live[role] = row['pid']
    require(set(live) == {'python', 'search', 'greedy'} and live['python'] == worker_pid,
            'three exact live compute roles required at readiness')
    expected['native_pids'] = {role: live[role] for role in ('search', 'greedy')}
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
    """PUCT16 unchanged; the extra guard includes decoding done by the worker."""
    def __init__(self, profile_document):
        self.capture = False
        self.callbacks = []
        super().__init__(profile_document)

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


class SharedGreedyOpponent(_Deadline, GreedyOpponent):
    """Use exact GreedyOpponent.choose and exact frozen host arithmetic."""
    def __init__(self, output_dir, search):
        self.model, self.reference = search.model, search.reference
        NativeOnly.__init__(self, output_dir, infer=self._infer, boundary=self.boundary)

    def _infer(self, tokens, batch):
        self.boundary()
        outputs = Opponent._infer(self, tokens, batch)
        self.boundary()
        return outputs

    def choose(self, observer, *, diagnostic_capture=False):
        self._begin()
        try:
            # The frozen operation includes logits only in diagnostic mode.
            # Keep their arithmetic and hash fields verbatim, then remove the
            # unnecessary entity tokens from ordinary game evidence.
            action, trace = GreedyOpponent.choose(self, observer, diagnostic_capture=True)
            if not diagnostic_capture:
                for phase in trace['phases']:
                    for callback in phase['inference']:
                        callback.pop('tokens', None)
            self.boundary()
            return action, trace
        finally:
            self._finish()

    def reference_choose(self, observer, *, diagnostic_capture=True):
        """Qualification invokes the original operation, bypassing our wrapper."""
        self._begin()
        try:
            result = GreedyOpponent.choose(self, observer, diagnostic_capture=diagnostic_capture)
            self.boundary()
            return result
        finally:
            self._finish()


def initialize_adapters(output_dir):
    """Authenticate both transports and share one immutable model per worker."""
    require(file_hash(PROFILE_PATH) == PROFILE_SHA256, 'search profile identity')
    profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    require(profile['profile_id'] == SEARCH_PROFILE_ID and profile['iterations'] == 16,
            'PUCT16 scientific profile')
    require(profile['internal_budget_seconds'] == INTERNAL_SECONDS, 'internal search limit')
    require(file_hash(ROOT / profile['native_binary']) == SEARCH_BINARY_SHA256,
            'search native executable identity')
    build = json.loads(BUILD.read_text(encoding='utf-8'))
    require(build['executable']['sha256'] == GREEDY_BINARY_SHA256, 'greedy build identity')
    require(file_hash(ROOT / build['executable']['path']) == GREEDY_BINARY_SHA256,
            'greedy native executable identity')
    search = SearchOpponent(profile)
    try:
        greedy = SharedGreedyOpponent(Path(output_dir) / 'greedy_transport', search)
    except BaseException:
        search.close()
        raise
    require(search.model is greedy.model and search.reference is greedy.reference,
            'shared immutable architecture identity')
    require(not search.model.training, 'evaluation mode')
    metadata = {
        'schema': 'splendorust_one_second_worker_identity_v1',
        'model_arrangement': MODEL_ARRANGEMENT,
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'search_profile_id': SEARCH_PROFILE_ID, 'search_profile_sha256': PROFILE_SHA256,
        'greedy_profile_id': GREEDY_PROFILE_ID, 'greedy_wrapper_id': GREEDY_WRAPPER_ID,
        'search_wrapper_id': SEARCH_WRAPPER_ID,
        'search_native_sha256': SEARCH_BINARY_SHA256,
        'greedy_native_sha256': GREEDY_BINARY_SHA256,
        'native_pids': {'search': search.proc.pid, 'greedy': greedy.proc.pid},
        'architecture_warmup': ARCHITECTURE_WARMUP,
        'internal_seconds': INTERNAL_SECONDS,
        'state_persistence': STATE_PERSISTENCE,
    }
    return {'search': search, 'greedy': greedy}, metadata
