"""Separate teacher-variability evaluator; no production player is changed.

The owning actual worker calls verify_runtime_sources once before initialize.
Legacy mode follows the original shared random stream. Split mode is an explicit
diagnostic intervention separating root worlds from neural-input completions.
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
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent, CHECKPOINT_SHA256
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024.worker import decode_observer, _save, _forensic_recorder
from tools.splendorust_decision_quality_20261027_v2 import adapters as inherited
from tools.splendorust_decision_quality_20261027_v2 import native_build as control_build
from tools.splendorust_inference_opt_20261025 import graph
from . import native_build

INTERNAL_SECONDS = 119.0
COMPLETE_SECONDS = 120.0
DOMAIN = 'splendorust_teacher_variability_20261113_v1'
QUOTAS = (512, 2048)
ALL_QUOTAS = (16, 64, *QUOTAS)
POOLS = (3, 6)
MODES = ('legacy', 'split')
SEED_LABELS = ('A', 'B', 'C', 'D')
SCHEMA = 'splendorust_teacher_variability_search_v1'
NATIVE_SHA256 = '38940717f81f0aee4342237a706f4bfc1c4395d04a6dca5c12145bcd1553d7bd'
CONTROL_NATIVE_SHA256 = '15919d0fac12ed8b3fd695272b06d1159289ce3ab10ede509071b23fcd45f18c'


def require(value, message):
    if not value:
        raise ValueError('VARIABILITY_ADAPTER: ' + message)


def profile_key(quota, pool=3, mode='legacy'):
    require(type(quota) is int and quota in ALL_QUOTAS, 'fixed quota')
    require(type(pool) is int and pool in POOLS, 'root pool3or6')
    require(type(mode) is str and mode in MODES, 'sampling mode')
    return f'{mode}_p{pool}_q{quota}'


def profile_id(quota, pool=3, mode='legacy', *, control=False):
    return f'teacher_variability_{"control" if control else "research"}_{profile_key(quota,pool,mode)}_20261113_v1'


def wrapper_id(quota, pool=3, mode='legacy', *, control=False):
    return profile_id(quota, pool, mode, control=control) + '_120s'


def profile(quota, pool=3, mode='legacy', *, control=False):
    key = profile_key(quota, pool, mode)
    require(not control or (pool == 3 and mode == 'legacy'), 'original control is legacy pool3')
    result = deepcopy(original.profile_for(16))
    require(result['world_pool'] == 3 and result['depth'] == 64 and result['cpuct'] == .4,
            'unchanged starting search parameters')
    if control:
        receipt_path = control_build.BUILD/'build_receipt.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        native = receipt['natives']['search']
        require(native['sha256'] == CONTROL_NATIVE_SHA256 and receipt['copied_core_agents_unchanged'],
                'original telemetry control identity')
    else:
        receipt_path = native_build.BUILD/'build_receipt.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        require(receipt['status'] == 'compiled_not_model_qualified' and receipt['copied_core_unchanged']
                and receipt['copied_encoder_unchanged'] and receipt['legacy_rng_draw_order_unchanged_by_construction'],
                'separate diagnostic native identity')
        native = receipt['native']
        require(native['sha256'] == NATIVE_SHA256, 'qualified additive release identity')
    require(native_build.artifact(ROOT/native['path']) == native, 'release binary content binding')
    result.update(native_binary=native['path'], native_binary_sha256=native['sha256'])
    result['artifacts'] += [native, native_build.artifact(receipt_path)]
    result.update(profile_id=profile_id(quota, pool, mode, control=control),
        wrapper_id=wrapper_id(quota, pool, mode, control=control), profile_key=key,
        iterations=quota, world_pool=pool, sampling_mode=mode,
        internal_budget_seconds=INTERNAL_SECONDS, complete_budget_seconds=COMPLETE_SECONDS,
        purpose='saved_position_teacher_variability_diagnostic_only',
        seed_family='A/B/C/D observer-derived; split root-world and neural-encoding domains',
        telemetry='existing SearchTargets plus full hypothetical root worlds and RNG boundary identity' if not control else 'original SearchTargets',
        deployment_reference_unchanged=True, ruleset_id=guard.RULESET_ID,
        catalogue_sha256=guard.CATALOGUE_SHA256, turn_contract=guard.TURN_CONTRACT,
        origin='generated_under_declared_rules', checkpoint_sha256=CHECKPOINT_SHA256)
    return result


def observer_seed(observer, label, stream='encoding'):
    require(type(label) is str and label in SEED_LABELS, 'four fixed observer-only seed labels')
    require(stream in ('shared', 'world', 'encoding'), 'seed stream domain')
    if stream == 'shared' and label == 'A':
        return int(digest(observer)[:16], 16)
    encoded = json.dumps(observer, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256((DOMAIN+'|'+stream+'|'+label+'|'+encoded).encode()).hexdigest()[:16], 16)


def seed_pair(observer, mode, world_seed_label, encoding_seed_label):
    require(type(mode) is str and mode in MODES, 'sampling mode')
    if mode == 'legacy':
        require(world_seed_label == encoding_seed_label, 'legacy has one shared seed label')
        seed = observer_seed(observer, encoding_seed_label, 'shared')
        return seed, seed
    return (observer_seed(observer, world_seed_label, 'world'),
            observer_seed(observer, encoding_seed_label, 'encoding'))


def enrich_sampling(packet):
    """SHA256 full hypothetical states; retain original strings for independent audit."""
    require(type(packet) is dict and set(packet) == {'schema','mode','world_pool','world_seed','encoding_seed',
        'root_world_debug','encoding_rng_before_worlds','encoding_rng_after_worlds'}, 'native sampling fields')
    require(packet['schema'] == 'teacher_variability_sampling_v1', 'native sampling schema')
    worlds = packet['root_world_debug']
    require(type(worlds) is list and len(worlds) in (0, packet['world_pool'])
            and all(type(x) is str and x.startswith('GameState {') for x in worlds), 'full sampled world identity')
    result = deepcopy(packet)
    result['root_world_sha256'] = [hashlib.sha256(x.encode('utf-8')).hexdigest() for x in worlds]
    result['root_worlds_sha256'] = digest(worlds)
    result['root_world_prefix3_sha256'] = digest(worlds[:3])
    return result


class ResearchOpponent(inherited.ResearchOpponent):
    decision_guard = None
    control = False
    qualification_mode = 'legacy'
    qualification_pool = 3

    def boundary(self):
        if self.active_deadline is not None and time.perf_counter() >= self.active_deadline:
            raise TimeoutError('VARIABILITY_INTERNAL_DEADLINE: 119-second allowance exceeded')

    def exchange(self, request):
        self.last_sampling = None
        result = super().exchange(request)
        packet = result.pop('sampling', None)
        if request.get('op') == 'select' and not self.control:
            self.last_sampling = enrich_sampling(packet)
        elif self.control:
            require(packet is None, 'control native must remain original')
        return result

    def choose(self, observer, *, diagnostic_capture=True, mode='legacy', pool=3,
               world_seed_label='A', encoding_seed_label='A', telemetry=True, quota=None):
        if self.admission_guard is not None:
            self.admission_guard()
        quota = self.profile['iterations'] if quota is None else quota
        identity = profile_id(quota, pool, mode, control=self.control)
        require(not self.control or (mode == 'legacy' and pool == 3), 'control configuration unchanged')
        require(type(telemetry) is bool, 'explicit telemetry bool')
        world_seed, encoding_seed = seed_pair(observer, mode, world_seed_label, encoding_seed_label)
        self.active_deadline = self.request_deadline if self.request_deadline is not None else time.perf_counter()+INTERNAL_SECONDS
        self.boundary()
        self.capture = bool(diagnostic_capture)
        self.telemetry_enabled = telemetry
        self.deadline = self.active_deadline
        self.inference_calls = self.inference_rows = 0
        phases = []
        self.last_research_phases = []

        def select(obs, choices):
            result = self.exchange(dict(op='select', observation=obs, legal=choices,
                seed=encoding_seed, iterations=quota, sampling_mode=mode, world_pool=pool, world_seed=world_seed))
            sampling = deepcopy(self.last_sampling)
            if not self.control:
                require(sampling['mode'] == mode and sampling['world_pool'] == pool
                        and sampling['world_seed'] == world_seed and sampling['encoding_seed'] == encoding_seed,
                        'native returned request sampling identity')
                if mode == 'split':
                    require(sampling['encoding_rng_before_worlds'] == sampling['encoding_rng_after_worlds'],
                            'split root pool must not consume neural encoding stream')
            phases.append({k:v for k,v in result.items() if k != 'kind'})
            self.last_research_phases.append(dict(phase_index=len(phases)-1, phase_kind=obs['phase']['kind'],
                telemetry=deepcopy(self.last_exchange_telemetry), callback_bits=deepcopy(self.callback_bits),
                sampling=sampling))
            return result['action']

        scope = nullcontext() if self.decision_guard is None else self.decision_guard()
        try:
            with scope:
                observation = bridge.encode_observation(observer)
                legal = bridge.main_actions(observer)
                selected = legal[0] if legal == [{'kind':'Pass'}] else select(observation, legal)
                action = bridge.resolve_compound(observer, selected, select)
                self.boundary()
                return action, dict(opponent_id='splendorust', profile_id=identity, observer_seed=encoding_seed,
                    phases=phases, inference_calls=self.inference_calls, inference_rows=self.inference_rows,
                    forced_pass=selected['kind']=='Pass')
        except (ValueError, KeyError, AssertionError, IndexError) as error:
            raise ValueError('ADAPTER_PREREQUISITE: '+str(error)) from error
        finally:
            self.deadline = self.active_deadline = None
            self.capture = False

    def reference_choose(self, observer, *, diagnostic_capture=True):
        return self.choose(observer, diagnostic_capture=diagnostic_capture,
            mode=self.qualification_mode, pool=self.qualification_pool, quota=16)


def initialize(folder, profiles, runtime_sources, boundary=lambda:None, save_json=_save,
               *, control_profiles=None, profile_artifacts=None, control_profile_artifacts=None,
               decision_guard=None):
    require(runtime_sources == guard.SOURCE_HASHES, 'worker runtime-source receipt required')
    keys = {profile_key(q,p,m) for q in ALL_QUOTAS for p in POOLS for m in MODES}
    control_keys = {profile_key(q) for q in ALL_QUOTAS}
    require(set(profiles) == keys and type(control_profiles) is dict and set(control_profiles) == control_keys,
            'all sixteen research and four original control profiles')
    for documents, artifacts, control in ((profiles,profile_artifacts,False), (control_profiles,control_profile_artifacts,True)):
        require(type(artifacts) is dict and set(artifacts) == set(documents), 'frozen profile artifact allowlist')
        for key, document in documents.items():
            item = artifacts[key]
            require(native_build.artifact(ROOT/item['path']) == item
                    and json.loads((ROOT/item['path']).read_text(encoding='utf-8')) == document,
                    'frozen profile artifact binding')
            require(key == document['profile_key'] and document == profile(document['iterations'],
                    document['world_pool'], document['sampling_mode'], control=control), 'authenticated profile construction')
    folder = Path(folder)
    require(folder.resolve().is_relative_to((ROOT/'outputs').resolve()), 'worker evidence output folder')
    folder.mkdir(parents=True, exist_ok=True)
    boundary()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    instances = {}
    try:
        instances['reference'] = ResearchOpponent(profiles[profile_key(16)])
        instances['search'] = ResearchOpponent(profiles[profile_key(512)], shared_from=instances['reference'])
        instances['control'] = ResearchOpponent(control_profiles[profile_key(16)], shared_from=instances['reference'])
        instances['control'].control = True
        forward = graph.build_graph(instances['search'].model, folder, save_json)
        for instance in instances.values():
            instance.forward = forward
            instance.admission_guard = boundary
            instance.decision_guard = decision_guard
        require(len({id(x.model) for x in instances.values()}) == 1 and not instances['search'].model.training,
                'one unchanged EF8A model')
        context = dict(adapters=instances, profiles=deepcopy(profiles), control_profiles=deepcopy(control_profiles),
            profile_artifacts=deepcopy(profile_artifacts), control_profile_artifacts=deepcopy(control_profile_artifacts),
            folder=folder, boundary=boundary, save_json=save_json, runtime_sources=runtime_sources,
            request_started=None, diagnostic_capture=False, qualification_active=False,
            graph_parity_passed=False, graph_parity_attempted=False, frozen_forward=forward,
            inference_binding=SimpleNamespace(original=Opponent._infer))
        save_json(folder/'variability_runtime_identity.json', dict(schema=SCHEMA, sources=runtime_sources,
            checkpoint_sha256=CHECKPOINT_SHA256, graph_identity=forward.identity,
            profile_digests={k:digest(v) for k,v in profiles.items()},
            control_profile_digests={k:digest(v) for k,v in control_profiles.items()},
            profile_artifacts=profile_artifacts, control_profile_artifacts=control_profile_artifacts,
            native_pids={k:v.proc.pid for k,v in instances.items()}, ruleset_id=guard.RULESET_ID,
            catalogue_sha256=guard.CATALOGUE_SHA256, turn_contract=guard.TURN_CONTRACT,
            internal_seconds=INTERNAL_SECONDS, complete_seconds=COMPLETE_SECONDS,
            parent_owns_complete_response_clock=True))
        boundary()
        return context
    except BaseException:
        for instance in instances.values():
            instance.close()
        raise


def ordinary(context, request):
    require(type(request) is dict and set(request) == {'player','observer','quota','pool','mode',
        'world_seed_label','encoding_seed_label','telemetry'}, 'ordinary request allowlist')
    player, quota, pool, mode = (request[k] for k in ('player','quota','pool','mode'))
    require(type(player) is str and player in context['adapters'], 'player allowlist')
    key = profile_key(quota, pool, mode)
    require(context.get('qualification_active') or (player == 'search' and quota in QUOTAS),
            'qualification-only reference/control/quota16')
    require(player != 'reference' or quota == 16, 'reference fixed16')
    require(player != 'control' or (mode == 'legacy' and pool == 3), 'legacy control only')
    observer, world_label, encoding_label, telemetry = (request[k] for k in
        ('observer','world_seed_label','encoding_seed_label','telemetry'))
    world_seed, encoding_seed = seed_pair(observer, mode, world_label, encoding_label)
    require(type(telemetry) is bool, 'telemetry bool')
    state = decode_observer(observer)
    require(context.get('qualification_active') or state.turn_index < 124, 'panel excludes native late heuristic')
    legal = rules.CorrectedEnv().legal_actions(state)
    require(bool(legal), 'complete legal actions nonempty')
    adapter = context['adapters'][player]
    started = context.get('request_started')
    if started is None:
        started = time.perf_counter()
    deadline = started+INTERNAL_SECONDS
    adapter.request_deadline = deadline
    evidence = context.get('evidence_callback', lambda value:None)
    try:
        action, trace = adapter.choose(observer, diagnostic_capture=True, mode=mode, pool=pool,
            world_seed_label=world_label, encoding_seed_label=encoding_label, telemetry=telemetry, quota=quota)
        document = (context['control_profiles'] if player == 'control' else context['profiles'])[key]
        item = (context['control_profile_artifacts'] if player == 'control' else context['profile_artifacts'])[key]
        research = dict(schema=SCHEMA, quota=quota, mode=mode, pool=pool,
            world_seed_label=world_label, encoding_seed_label=encoding_label,
            world_seed=world_seed, encoding_seed=encoding_seed, telemetry_enabled=telemetry,
            internal_seconds=INTERNAL_SECONDS, native_binary_sha256=document['native_binary_sha256'],
            phases=deepcopy(adapter.last_research_phases))
        evidence(dict(event='complete_result_before_worker_validation', arm=player, observer=observer,
            action=rules.serialize_action(action), diagnostics=trace, research=research))
        require(action in legal, 'completed decision outside corrected legal support')
        result = dict(player=player, action=rules.serialize_action(action), diagnostics=trace,
            profile_id=document['profile_id'], profile_sha256=item['sha256'], wrapper_id=document['wrapper_id'],
            simulations_quota=quota, observer_sha256=digest(observer),
            legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]), legal_count=len(legal),
            ruleset=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
            turn_contract=guard.TURN_CONTRACT, research=research)
        json.dumps(result, separators=(',',':'), allow_nan=False)
        require(trace['profile_id'] == document['profile_id'], 'scientific profile identity')
        if time.perf_counter() >= deadline:
            raise TimeoutError('VARIABILITY_INTERNAL_DEADLINE: complete result exceeded119seconds')
        return result
    except BaseException as error:
        evidence(dict(event='worker_research_decision_failed', arm=player, observer=observer,
            error=repr(error), last_phase_callbacks=getattr(adapter,'callbacks',None),
            last_research_phases=deepcopy(getattr(adapter,'last_research_phases',[]))))
        raise
    finally:
        adapter.request_deadline = None


def handle(context, request):
    if type(request) is dict and request.get('operation') == 'graph_parity':
        from tools.splendorust_inference_opt_20261025.worker import graph_parity
        return graph_parity(context, request)
    require(context['graph_parity_passed'], 'eleven exact graph parity cases required first')
    if type(request) is dict and request.get('operation') == 'entrypoint_checks':
        from .qualification import run_entrypoint_checks
        require(set(request) == {'operation','mode','pool'}, 'explicit entrypoint mode/pool')
        profile_key(16, request['pool'], request['mode'])
        require(not context['qualification_active'], 'no nested entrypoint qualification')
        outer_started = context['request_started']
        context['request_started'] = None
        context['qualification_active'] = True
        for adapter in context['adapters'].values():
            adapter.qualification_mode, adapter.qualification_pool = request['mode'], request['pool']
        record, stream = _forensic_recorder(context['folder']/f'entrypoint_{request["mode"]}_p{request["pool"]}_forensics.jsonl')
        context['evidence_callback'] = record
        try:
            return run_entrypoint_checks(context, lambda payload:handle(context,payload),
                mode=request['mode'], pool=request['pool'], boundary=context['boundary'], evidence_callback=record)
        except BaseException as error:
            record(dict(event='entrypoint_failed',error=repr(error)))
            raise
        finally:
            stream.close()
            context.pop('evidence_callback',None)
            context['qualification_active'] = False
            context['request_started'] = outer_started
            for adapter in context['adapters'].values():
                adapter.qualification_mode, adapter.qualification_pool = 'legacy', 3
    return ordinary(context, request)


def close_context(context):
    for adapter in context.get('adapters',{}).values():
        adapter.close()
