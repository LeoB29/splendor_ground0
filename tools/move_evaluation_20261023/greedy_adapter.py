"""Versioned, observer-only Entity greedy policy; no imports load a model.

Construct only inside the frozen job's initialization clock. The surrounding
worker owns startup rule authentication, the whole-decision clock and resources.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from tools.external_strength_screen_20261018.common import ROOT, digest, serialize_action
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.external_strength_screen_20261018_02.splendorust_adapter import (
    PIN, CHECKPOINT_SHA256, Opponent, load_model,
)

PROFILE_ID = 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1'
BRIDGE_ID = 'observer_only_entity_greedy_v1'
BUILD = ROOT / 'outputs/move_evaluation_20261023_build_01/build_receipt.json'
MODEL_DEFINITION = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust/research/architecture_pivots/models.py'
MODEL_DEFINITION_SHA256 = '0a554cd2a7b21b76a00add1157ab10b9cbdadd94b2b0e82097da2d9c06054417'
CHECKPOINT = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust_model.pt'


def require(condition, message):
    if not condition:
        raise ValueError('GREEDY_ADAPTER: ' + message)


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical_index(action):
    """Independently expressed canonical67 vocabulary, used for validation."""
    kind = action['kind']
    if kind == 'Take':
        values = action['tokens']
        require(len(values) == 5 and all(type(x) is int and 0 <= x <= 2 for x in values), 'take shape')
        if 2 in values:
            require(sum(values) == 2, 'double take shape')
            return 32 + values.index(2)
        return sum(1 << i for i, value in enumerate(values) if value)
    offsets = {'ReserveVisible': (37, 12), 'ReserveDeck': (49, 3),
               'BuyVisible': (52, 12), 'BuyReserved': (64, 3)}
    require(kind in offsets, 'not a canonical Main action')
    offset, count = offsets[kind]
    index = action['tier' if kind == 'ReserveDeck' else 'index']
    require(type(index) is int and 0 <= index < count, 'canonical action index')
    return offset + index


def first_max(logits, actions):
    require(len(logits) == 67 and all(type(x) in (int, float) and math.isfinite(x) for x in logits), 'policy logits')
    require(bool(actions), 'empty support')
    return max(range(len(actions)), key=lambda i: logits[canonical_index(actions[i])])


class NativeOnly:
    """Transport usable for model-free contract/reference fixtures.

    No callback means any attempt to perform inference fails closed.
    """
    def __init__(self, outdir, *, build_manifest=None, infer=None, boundary=lambda: None):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        receipt = json.loads(Path(build_manifest or BUILD).read_text(encoding='utf-8'))
        require(receipt['schema'] == 'move_evaluation_greedy_build_v1' and receipt['status'] == 'compiled', 'build identity')
        require(receipt['source_pin'] == PIN, 'source pin')
        for item in receipt['runtime_inputs']:
            path = ROOT / item['path']
            require(path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], 'runtime source: ' + str(path))
        exe = ROOT / receipt['executable']['path']
        require(exe.stat().st_size == receipt['executable']['bytes'] and file_hash(exe) == receipt['executable']['sha256'], 'native binary')
        self.boundary = boundary
        self.infer = infer
        self.callbacks = []
        self.stderr = (self.outdir / 'native_stderr.log').open('xb')
        self.proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, encoding='utf-8', bufsize=1, cwd=ROOT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        atexit.register(self.close)
        try:
            ready = json.loads(self.proc.stdout.readline())
            require(ready == {'kind': 'ready', 'bridge': BRIDGE_ID, 'source_pin': PIN}, 'ready identity')
        except BaseException:
            self.close()
            raise

    def close(self):
        proc = getattr(self, 'proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        stderr = getattr(self, 'stderr', None)
        if stderr is not None:
            stderr.close()

    def exchange(self, request, *, capture=False):
        self.boundary()
        self.callbacks = []
        self.proc.stdin.write(json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n')
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            require(bool(line), 'native child exited: ' + str(self.proc.poll()))
            result = json.loads(line)
            self.boundary()
            require(isinstance(result, dict), 'nonobject native response')
            if result.get('kind') == 'prerequisite_error':
                raise ValueError('GREEDY_ADAPTER: ' + str(result.get('detail')))
            if result.get('kind') == 'result':
                return result
            require(result.get('kind') == 'infer' and self.infer is not None, 'unexpected inference/response')
            require(result.get('batch') == 1 and len(result.get('tokens', [])) == 31 * 48, 'batch-one input required')
            tokens = result['tokens']
            require(all(type(x) in (int, float) and math.isfinite(x) for x in tokens), 'nonfinite tokens')
            outputs = self.infer(tokens, 1)
            require(len(outputs) == 1 and len(outputs[0]) == 83 and
                    all(type(x) in (int, float) and math.isfinite(x) for x in outputs[0]), 'nonfinite outputs')
            item = {'input_sha256': digest(tokens), 'output_sha256': digest(outputs[0])}
            if capture:
                item.update(tokens=tokens, output83=outputs[0])
            self.callbacks.append(item)
            self.proc.stdin.write(json.dumps({'kind': 'inference_result', 'outputs': outputs},
                                             separators=(',', ':'), allow_nan=False) + '\n')
            self.proc.stdin.flush()

    def inspect(self, observer):
        return self.exchange({'op': 'contract', 'observation': bridge.encode_observation(observer),
                              'legal': bridge.main_actions(observer), 'seed': int(digest(observer)[:16], 16)})

    def native_reference(self, observer, policy81):
        require(len(policy81) == 81 and all(type(x) in (int, float) and math.isfinite(x) for x in policy81), 'policy81')
        return self.exchange({'op': 'reference', 'observation': bridge.encode_observation(observer),
                              'legal': bridge.main_actions(observer), 'seed': int(digest(observer)[:16], 16),
                              'policy81': policy81})


class GreedyOpponent(NativeOnly):
    # Exactly the already qualified float32 forward arithmetic; no constructor
    # inheritance, warm inference, history or mutable search state.
    _infer = Opponent._infer

    def __init__(self, outdir, *, build_manifest=None, boundary=lambda: None):
        import torch
        require(file_hash(MODEL_DEFINITION) == MODEL_DEFINITION_SHA256, 'model definitions')
        require(file_hash(CHECKPOINT) == CHECKPOINT_SHA256, 'checkpoint')
        torch.set_num_threads(1)
        self.model, self.reference = load_model(MODEL_DEFINITION, CHECKPOINT)
        super().__init__(outdir, build_manifest=build_manifest, infer=self._infer, boundary=boundary)

    def choose(self, observer, *, diagnostic_capture=False):
        require(set(observer) == {'schema', 'ruleset_id', 'catalogue_sha256', 'observation'}, 'observer envelope allowlist')
        seed = int(digest(observer)[:16], 16)
        phases = []

        def select(native_observation, legal):
            result = self.exchange({'op': 'greedy', 'observation': native_observation,
                                    'legal': legal, 'seed': seed}, capture=diagnostic_capture)
            required = {'kind', 'action', 'branch', 'native_legal_count', 'filtered', 'canonical_indices',
                        'selected_index', 'canonical_logits', 'simulations', 'leaf_calls'}
            require(set(result) == required, 'result schema')
            filtered = result['filtered']
            require(type(result['native_legal_count']) is int and result['native_legal_count'] == len(legal), 'legal count')
            require(isinstance(filtered, list) and filtered and all(a in legal for a in filtered), 'filtered support')
            indices = [legal.index(a) for a in filtered]
            require(indices == sorted(set(indices)), 'filter must preserve unique order')
            selected = result['selected_index']
            require(type(selected) is int and 0 <= selected < len(filtered) and result['action'] == filtered[selected], 'selection coupling')
            main = native_observation['phase']['kind'] == 'Main'
            expected = ('auxiliary_heuristic' if not main else 'single_choice' if len(filtered) == 1
                        else 'late_heuristic' if native_observation['turns'] >= 124 else 'neural_greedy')
            require(result['branch'] == expected, 'policy branch')
            neural = expected == 'neural_greedy'
            require(type(result['simulations']) is int and result['simulations'] == 0 and
                    type(result['leaf_calls']) is int and result['leaf_calls'] == int(neural) and
                    len(self.callbacks) == int(neural), 'greedy work count')
            require(result['canonical_indices'] == ([canonical_index(a) for a in filtered] if main else []), 'canonical index map')
            if neural:
                require(first_max(result['canonical_logits'], filtered) == selected, 'first maximum')
            else:
                require(result['canonical_logits'] is None, 'heuristic received logits')
            phase = {'phase': native_observation['phase']['kind'], 'observation_sha256': digest(native_observation),
                     'legal_sha256': digest(legal), 'branch': expected, 'native_legal_count': len(legal),
                     'retained_indices': indices, 'canonical_indices': result['canonical_indices'],
                     'selected_index': selected, 'action': result['action'],
                     'policy_logits_sha256': digest(result['canonical_logits']) if neural else None,
                     'inference': self.callbacks}
            if diagnostic_capture:
                phase['canonical_logits'] = result['canonical_logits']
            phases.append(phase)
            return result['action']

        native = bridge.encode_observation(observer)
        legal_main = bridge.main_actions(observer)
        forced = legal_main == [{'kind': 'Pass'}]
        selected = legal_main[0] if forced else select(native, legal_main)
        action = bridge.resolve_compound(observer, selected, select)
        state = rules.state_from_observation(observer)
        require(action in rules.legal_actions(state), 'atomic action is illegal')
        trace = {'schema': 'entity_greedy_decision_v1', 'profile_id': PROFILE_ID,
                 'observer_sha256': digest(observer), 'observer_seed': seed,
                 'action_sha256': digest(serialize_action(action)), 'forced_pass': forced,
                 'phases': phases, 'inference_calls': sum(len(p['inference']) for p in phases),
                 'simulations': 0}
        self.boundary()
        return action, trace
