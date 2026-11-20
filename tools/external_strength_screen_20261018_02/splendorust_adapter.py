"""Pinned Entity PUCT with an observer-only native transport and CPU host inference.

The search, public encoder, auxiliary heuristics and model definition are upstream
code. The atomic action connection and fixed work profile are explicitly local.
"""
from __future__ import annotations

import ast
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.external_strength_screen_20261018.common import ROOT, digest

PIN = '1828507b8d2f0e02bc03d102b9c6d29a54fa8d57'
CHECKPOINT_SHA256 = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_model(source_path, checkpoint_path):
    """Evaluate only the authenticated model definitions, without upstream imports."""
    import torch
    from torch import nn
    names = {'normalized', 'entities', 'entities_fast', 'EntityTransformer'}
    tree = ast.parse(Path(source_path).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Pinned model definitions missing')
    namespace = {'torch': torch, 'nn': nn, '_ENTITY_CACHE': {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), namespace)
    if file_hash(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError('Published checkpoint identity mismatch')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if payload['kind'] != 'entity':
        raise ValueError('Unsupported checkpoint architecture')
    model = namespace['EntityTransformer']()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    if sum(p.numel() for p in model.parameters()) != 4780883 or model.has_history:
        raise ValueError('Unexpected champion architecture')
    return model, namespace


class Opponent:
    def __init__(self, profile_document):
        import torch
        self.profile = profile_document
        if self.profile['opponent_id'] != 'splendorust' or not self.profile['deterministic_fixed_work']:
            raise ValueError('Unsupported opponent profile')
        for item in self.profile['artifacts']:
            path = ROOT / item['path']
            if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
                raise ValueError('Opponent input identity mismatch: ' + str(path))
        torch.set_num_threads(1)
        self.model, self.reference = load_model(ROOT / self.profile['model_definition'],
                                               ROOT / self.profile['checkpoint'])
        self.proc = subprocess.Popen([str(ROOT / self.profile['native_binary'])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1, cwd=ROOT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        atexit.register(self.close)
        ready = json.loads(self.proc.stdout.readline())
        if ready != {'kind': 'ready', 'bridge': 'observer_only_fixed_work_v1', 'source_pin': PIN}:
            self.close()
            raise ValueError('Native bridge identity mismatch')
        self.deadline = None
        self.inference_calls = 0
        self.inference_rows = 0
        # Warm only the fixed architecture, with no game position or result.
        with torch.inference_mode():
            self._infer([0.0] * (31 * 48), 1)

    def close(self):
        proc = getattr(self, 'proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def _infer(self, tokens, batch):
        import torch
        if batch not in (1, 2, 8) or len(tokens) != batch * 31 * 48:
            raise ValueError('Malformed public entity tokens')
        with torch.inference_mode():
            x = torch.tensor(tokens, dtype=torch.float32).reshape(batch, 31, 48)
            if not bool(torch.isfinite(x).all()):
                raise ValueError('Nonfinite model input')
            h = self.model.norm(self.model.blocks(self.model.project(x) + self.model.identity)[:, 0])
            out = torch.cat((self.model.policy(h), self.model.value(h).tanh()), dim=1)
            if not bool(torch.isfinite(out).all()):
                raise ValueError('Nonfinite model output')
            return out.tolist()

    def exchange(self, request):
        self.proc.stdin.write(json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n')
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('Native observer bridge exited: ' + str(self.proc.poll()))
            result = json.loads(line)
            if result['kind'] == 'prerequisite_error':
                raise ValueError('ADAPTER_PREREQUISITE: ' + result['detail'])
            if result['kind'] == 'result':
                return result
            if result['kind'] != 'infer':
                raise ValueError('Unexpected native bridge response')
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                raise TimeoutError('Internal fixed-work deadline exceeded')
            outputs = self._infer(result['tokens'], result['batch'])
            self.inference_calls += 1
            self.inference_rows += result['batch']
            self.proc.stdin.write(json.dumps({'kind': 'inference_result', 'outputs': outputs},
                                             separators=(',', ':'), allow_nan=False) + '\n')
            self.proc.stdin.flush()

    def choose(self, observer):
        from .bridge import encode_observation, main_actions, resolve_compound
        self.deadline = time.perf_counter() + self.profile['internal_budget_seconds']
        self.inference_calls = self.inference_rows = 0
        seed = int(digest(observer)[:16], 16)
        phases = []

        def select(obs, choices):
            result = self.exchange({'op': 'select', 'observation': obs, 'legal': choices,
                                    'seed': seed, 'iterations': self.profile['iterations']})
            phases.append({k: v for k, v in result.items() if k != 'kind'})
            return result['action']

        try:
            observation = encode_observation(observer)
            legal = main_actions(observer)
            selected = legal[0] if legal == [{'kind': 'Pass'}] else select(observation, legal)
            action = resolve_compound(observer, selected, select)
            if time.perf_counter() >= self.deadline:
                raise TimeoutError('Internal fixed-work deadline exceeded')
            return action, {'opponent_id': 'splendorust', 'profile_id': self.profile['profile_id'],
                            'observer_seed': seed, 'phases': phases,
                            'inference_calls': self.inference_calls, 'inference_rows': self.inference_rows,
                            'forced_pass': selected['kind'] == 'Pass'}
        except (ValueError, KeyError, AssertionError, IndexError) as error:
            raise ValueError('ADAPTER_PREREQUISITE: ' + str(error)) from error
        finally:
            self.deadline = None
