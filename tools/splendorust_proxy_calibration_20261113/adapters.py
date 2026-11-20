"""Separate saved-position adapter; unchanged EF8A, native search and CPU graph.

Only quotas, observer-only seed domains and the research clock differ. No
historical module/class is rebound. Importing this module runs no learned model.
The owning evaluator calls verify_runtime_sources once before initialize.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent, file_hash, CHECKPOINT_SHA256
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024.worker import decode_observer, _save, _forensic_recorder
from tools.splendorust_decision_quality_20261027_v2 import adapters as inherited, native_build
from tools.splendorust_inference_opt_20261025 import graph

INTERNAL_SECONDS = 119.0
COMPLETE_SECONDS = 120.0
DOMAIN = 'splendorust_proxy_calibration_20261113_v1'
QUOTAS = (64, 128, 256, 512, 2048)
CONTROL_QUOTAS = QUOTAS[:-1]
SEED_LABELS = ('A', 'B', 'R', 'S')
SCHEMA = 'splendorust_proxy_calibration_search_v1'
NATIVE_SHA256 = '15919d0fac12ed8b3fd695272b06d1159289ce3ab10ede509071b23fcd45f18c'
CONTROL_NATIVE_SHA256 = 'b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c'


def require(value, message):
    if not value:
        raise ValueError('PROXY_ADAPTER: ' + message)


def profile_id(quota, *, control=False):
    require(type(quota) is int and quota in (CONTROL_QUOTAS if control else (16, *QUOTAS)),
            'supported fixed quota')
    return f'proxy_{"control" if control else "research"}_puct{quota}_20261113_v1'


def wrapper_id(quota, *, control=False):
    return profile_id(quota, control=control) + '_120s'


def profile(quota, *, control=False):
    result = deepcopy(original.profile_for(16))
    identity = profile_id(quota, control=control)
    require(result['world_pool'] == 3 and result['depth'] == 64 and result['cpuct'] == .4,
            'unchanged native search parameters')
    if not control:
        receipt_path = native_build.BUILD / 'build_receipt.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        require(receipt['status'] == 'compiled_not_model_qualified' and
                receipt['copied_core_agents_unchanged'] is True, 'additive native build identity')
        native = receipt['natives']['reference' if quota == 16 else 'search']
        require(native['sha256'] == NATIVE_SHA256, 'pinned telemetry binary')
        result.update(native_binary=native['path'], native_binary_sha256=native['sha256'])
        result['artifacts'] += [native, native_build.artifact(receipt_path)]
    else:
        native = next(item for item in result['artifacts'] if item['path'] == result['native_binary'])
        require(native['sha256'] == CONTROL_NATIVE_SHA256, 'original control binary')
        result['native_binary_sha256'] = native['sha256']
    result.update(profile_id=identity, wrapper_id=wrapper_id(quota, control=control),
        iterations=quota, internal_budget_seconds=INTERNAL_SECONDS,
        complete_budget_seconds=COMPLETE_SECONDS, purpose='saved_position_proxy_calibration_only',
        seed_family='A canonical; B/R/S domain-separated lawful observer hashes',
        telemetry='disabled original binary' if control else 'existing SearchTargets after unchanged selection',
        deployment_reference_unchanged=True, ruleset_id=guard.RULESET_ID,
        catalogue_sha256=guard.CATALOGUE_SHA256, turn_contract=guard.TURN_CONTRACT,
        origin='generated_under_declared_rules', checkpoint_sha256=CHECKPOINT_SHA256)
    return result


def observer_seed(observer, label):
    require(type(label) is str and label in SEED_LABELS, 'fixed observer-only seed family')
    if label == 'A':
        return int(digest(observer)[:16], 16)
    encoded = json.dumps(observer, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256((DOMAIN + '|' + label + '|' + encoded).encode()).hexdigest()[:16], 16)


class ResearchOpponent(inherited.ResearchOpponent):
    """Reuse accepted callback capture; selection remains an instance-local copy."""
    decision_guard = None
    control = False

    def boundary(self):
        if self.active_deadline is not None and time.perf_counter() >= self.active_deadline:
            raise TimeoutError('PROXY_INTERNAL_DEADLINE: 119.000-second allowance exceeded')

    def choose(self, observer, *, diagnostic_capture=True, seed_label='A', telemetry=True, quota=None):
        if self.admission_guard is not None:
            self.admission_guard()
        quota = self.profile['iterations'] if quota is None else quota
        identity = profile_id(quota, control=self.control)
        require(type(telemetry) is bool and (not self.control or not telemetry), 'explicit telemetry route')
        self.active_deadline = self.request_deadline if self.request_deadline is not None else time.perf_counter() + INTERNAL_SECONDS
        self.boundary()
        self.capture = bool(diagnostic_capture)
        self.telemetry_enabled = telemetry
        self.deadline = self.active_deadline
        self.inference_calls = self.inference_rows = 0
        seed = observer_seed(observer, seed_label)
        phases = []
        self.last_research_phases = []

        def select(obs, choices):
            result = self.exchange({'op': 'select', 'observation': obs, 'legal': choices,
                                    'seed': seed, 'iterations': quota})
            phases.append({k: v for k, v in result.items() if k != 'kind'})
            self.last_research_phases.append(dict(phase_index=len(phases)-1,
                phase_kind=obs['phase']['kind'], telemetry=deepcopy(self.last_exchange_telemetry),
                callback_bits=deepcopy(self.callback_bits)))
            return result['action']

        scope = nullcontext() if self.decision_guard is None else self.decision_guard()
        try:
            with scope:
                observation = bridge.encode_observation(observer)
                legal = bridge.main_actions(observer)
                selected = legal[0] if legal == [{'kind': 'Pass'}] else select(observation, legal)
                action = bridge.resolve_compound(observer, selected, select)
                self.boundary()
                return action, dict(opponent_id='splendorust', profile_id=identity, observer_seed=seed,
                    phases=phases, inference_calls=self.inference_calls, inference_rows=self.inference_rows,
                    forced_pass=selected['kind'] == 'Pass')
        except (ValueError, KeyError, AssertionError, IndexError) as error:
            raise ValueError('ADAPTER_PREREQUISITE: ' + str(error)) from error
        finally:
            self.deadline = self.active_deadline = None
            self.capture = False

    def reference_choose(self, observer, *, diagnostic_capture=True):
        require(self.profile['iterations'] == 16, 'qualification-only reference16')
        return self.choose(observer, diagnostic_capture=diagnostic_capture, seed_label='A', telemetry=True, quota=16)


def initialize(folder, profiles, runtime_sources, boundary=lambda: None, save_json=_save,
               *, control_profiles=None, profile_artifacts=None, control_profile_artifacts=None,
               decision_guard=None):
    """Load once after the evaluator's single runtime guard call; no admission heuristics."""
    require(runtime_sources == guard.SOURCE_HASHES, 'outer worker runtime-source receipt')
    require(set(profiles) == {str(q) for q in (16, *QUOTAS)}, 'six frozen research profiles')
    controls = control_profiles or {str(q): profile(q, control=True) for q in CONTROL_QUOTAS}
    require(set(controls) == {str(q) for q in CONTROL_QUOTAS}, 'four frozen control profiles')
    require(type(profile_artifacts) is dict and set(profile_artifacts) == set(profiles),
            'frozen research profile file artifacts required')
    require(type(control_profile_artifacts) is dict and set(control_profile_artifacts) == set(controls),
            'frozen control profile file artifacts required')
    for documents, artifacts in ((profiles, profile_artifacts), (controls, control_profile_artifacts)):
        for q, document in documents.items():
            item = artifacts[q]
            path = ROOT / item['path']
            require(native_build.artifact(path) == item and
                    json.loads(path.read_text(encoding='utf-8')) == document,
                    'profile file artifact binding')
    for q, document in profiles.items():
        require(document == profile(int(q)), 'research profile differs from authenticated construction')
    for q, document in controls.items():
        require(document == profile(int(q), control=True), 'control profile differs from authenticated construction')
    folder = Path(folder)
    require(folder.resolve().is_relative_to((ROOT / 'outputs').resolve()), 'worker evidence folder')
    folder.mkdir(parents=True, exist_ok=True)
    boundary()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    instances = {}
    try:
        instances['reference'] = ResearchOpponent(profiles['16'])
        instances['search'] = ResearchOpponent(profiles['64'], shared_from=instances['reference'])
        instances['control'] = ResearchOpponent(controls['64'], shared_from=instances['reference'])
        instances['control'].control = True
        forward = graph.build_graph(instances['search'].model, folder, save_json)
        for instance in instances.values():
            instance.forward = forward
            instance.admission_guard = boundary
            instance.decision_guard = decision_guard
        require(len({id(x.model) for x in instances.values()}) == 1 and
                not instances['search'].model.training, 'one immutable EF8A model')
        context = dict(adapters=instances, profiles=deepcopy(profiles), control_profiles=deepcopy(controls),
            profile_artifacts=deepcopy(profile_artifacts), control_profile_artifacts=deepcopy(control_profile_artifacts),
            folder=folder, boundary=boundary, save_json=save_json, runtime_sources=runtime_sources,
            request_started=None, diagnostic_capture=False, qualification_active=False,
            graph_parity_passed=False, graph_parity_attempted=False, frozen_forward=forward,
            inference_binding=SimpleNamespace(original=Opponent._infer))
        save_json(folder / 'proxy_runtime_identity.json', dict(schema=SCHEMA,
            sources=runtime_sources, checkpoint_sha256=CHECKPOINT_SHA256,
            graph_identity=forward.identity, profile_digests={q: digest(p) for q, p in profiles.items()},
            control_profile_digests={q: digest(p) for q, p in controls.items()},
            profile_artifacts=profile_artifacts, control_profile_artifacts=control_profile_artifacts,
            native_pids={name: adapter.proc.pid for name, adapter in instances.items()},
            ruleset_id=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
            turn_contract=guard.TURN_CONTRACT, internal_seconds=INTERNAL_SECONDS,
            complete_seconds=COMPLETE_SECONDS, parent_owns_complete_response_clock=True))
        boundary()
        return context
    except BaseException:
        for instance in instances.values():
            instance.close()
        raise


def ordinary(context, request):
    """Strict actual corrected entry point; caller charges transport/arrival to 120s."""
    require(type(request) is dict and set(request) == {'player', 'observer', 'quota', 'seed_label', 'telemetry'},
            'ordinary request allowlist')
    player, quota = request['player'], request['quota']
    require(type(player) is str and player in context['adapters'], 'player allowlist')
    require(type(quota) is int and quota in ((16,) if player == 'reference' else
            (CONTROL_QUOTAS if player == 'control' else QUOTAS)), 'player fixed quota')
    require(player != 'reference' or context.get('qualification_active'), 'reference16 qualification only')
    require(player != 'control' or context.get('qualification_active'), 'original native control qualification only')
    observer, label, telemetry = request['observer'], request['seed_label'], request['telemetry']
    seed = observer_seed(observer, label)
    require(type(telemetry) is bool, 'telemetry bool')
    state = decode_observer(observer)
    require(context.get('qualification_active') or state.turn_index < 124, 'exclude native late heuristic from proxy panel')
    legal = rules.CorrectedEnv().legal_actions(state)
    require(bool(legal), 'complete legal actions nonempty')
    adapter = context['adapters'][player]
    started = context.get('request_started')
    if started is None:
        started = time.perf_counter()
    deadline = started + INTERNAL_SECONDS
    adapter.request_deadline = deadline
    evidence = context.get('evidence_callback', lambda value: None)
    try:
        action, trace = adapter.choose(observer, diagnostic_capture=True, seed_label=label,
                                       telemetry=telemetry, quota=quota)
        document = (context['control_profiles'] if player == 'control' else context['profiles'])[str(quota)]
        document_artifact = (context['control_profile_artifacts'] if player == 'control' else context['profile_artifacts'])[str(quota)]
        research = dict(schema=SCHEMA, quota=quota, seed_label=label, seed=seed,
            telemetry_enabled=telemetry, internal_seconds=INTERNAL_SECONDS,
            native_binary_sha256=document['native_binary_sha256'], phases=deepcopy(adapter.last_research_phases))
        evidence(dict(event='complete_result_before_worker_validation', arm=player, observer=observer,
                      action=rules.serialize_action(action), diagnostics=trace, research=research))
        require(action in legal, 'completed decision outside corrected legal support')
        result = dict(player=player, action=rules.serialize_action(action), diagnostics=trace,
            profile_id=document['profile_id'], profile_sha256=document_artifact['sha256'],
            wrapper_id=document['wrapper_id'], simulations_quota=quota, observer_sha256=digest(observer),
            legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]), legal_count=len(legal),
            ruleset=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
            turn_contract=guard.TURN_CONTRACT, research=research)
        json.dumps(result, separators=(',', ':'), allow_nan=False)
        require(trace['profile_id'] == document['profile_id'], 'scientific profile identity')
        if time.perf_counter() >= deadline:
            raise TimeoutError('PROXY_INTERNAL_DEADLINE: complete response exceeded 119 seconds')
        return result
    except BaseException as error:
        evidence(dict(event='worker_research_decision_failed', arm=player, observer=observer,
            error=repr(error), last_phase_callbacks=getattr(adapter, 'callbacks', None),
            last_research_phases=deepcopy(adapter.last_research_phases)))
        raise
    finally:
        adapter.request_deadline = None


def handle(context, request):
    if type(request) is dict and request.get('operation') == 'graph_parity':
        from tools.splendorust_inference_opt_20261025.worker import graph_parity
        return graph_parity(context, request)
    require(context['graph_parity_passed'], 'eleven exact graph parity cases required first')
    if request == {'operation': 'entrypoint_checks'}:
        from .qualification import run_entrypoint_checks
        require(not context['qualification_active'], 'no nested entrypoint qualification')
        outer_started = context['request_started']
        context['request_started'] = None
        context['qualification_active'] = True
        record, stream = _forensic_recorder(context['folder'] / 'entrypoint_forensics.jsonl')
        context['evidence_callback'] = record
        try:
            return run_entrypoint_checks(context, lambda payload: handle(context, payload),
                boundary=context['boundary'], evidence_callback=record)
        except BaseException as error:
            record(dict(event='entrypoint_failed', error=repr(error)))
            raise
        finally:
            stream.close()
            context.pop('evidence_callback', None)
            context['qualification_active'] = False
            context['request_started'] = outer_started
    return ordinary(context, request)


def close_context(context):
    for adapter in context.get('adapters', {}).values():
        adapter.close()
