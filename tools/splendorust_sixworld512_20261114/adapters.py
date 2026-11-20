"""Separate live entrypoint for the unchanged EF8A pool3/pool6 search.

Original native, graph, seed mapping, compound-turn helpers and profile identities
are reused read-only. Only the saved-panel admission restriction is removed.
The new service/manifest identifies this live use; old frozen profiles are never
modified or represented as new scientific implementations.
"""
from copy import deepcopy
import base64
import gzip
import hashlib
import json
from pathlib import Path
import time
import zlib

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.splendorust_search_scale_20261024.worker import decode_observer, _forensic_recorder
from tools.splendorust_teacher_variability_20261113 import adapters as inherited

INTERNAL_SECONDS=inherited.INTERNAL_SECONDS
COMPLETE_SECONDS=inherited.COMPLETE_SECONDS
SCHEMA=inherited.SCHEMA
NATIVE_SHA256=inherited.NATIVE_SHA256
CHECKPOINT_SHA256=inherited.CHECKPOINT_SHA256
profile_key=inherited.profile_key
profile_id=inherited.profile_id
wrapper_id=inherited.wrapper_id
seed_pair=inherited.seed_pair
require=inherited.require
MAX_RAW_BYTES=16*1024**2
MAX_COMPRESSED_BYTES=16*1024**2
PACK_SCHEMA='sixworld512_lossless_gzip_json_v1'
LIVE_SCHEMA='sixworld512_legacy_live_v1'
INTERNAL_DEADLINE_MARKER='VARIABILITY_INTERNAL_DEADLINE'


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate packed JSON key');result[key]=value
    return result


def pack_result(value):
    raw=_canonical(value).encode('utf-8')
    require(0<len(raw)<=MAX_RAW_BYTES,'Bounded full result before compression')
    compressed=gzip.compress(raw,compresslevel=1,mtime=0)
    require(0<len(compressed)<=MAX_COMPRESSED_BYTES,'Bounded packed result')
    return dict(schema=PACK_SCHEMA,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(compressed),compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        data_b64=base64.b64encode(compressed).decode('ascii'))


def unpack_result(value):
    require(type(value) is dict and set(value)=={'schema','raw_bytes','raw_sha256','compressed_bytes',
        'compressed_sha256','data_b64'} and value['schema']==PACK_SCHEMA,'Exact packed result envelope')
    for key,maximum in (('raw_bytes',MAX_RAW_BYTES),('compressed_bytes',MAX_COMPRESSED_BYTES)):
        require(type(value[key]) is int and 0<value[key]<=maximum,'Strict bounded packed size')
    require(type(value['data_b64']) is str and len(value['data_b64'])<=4*((MAX_COMPRESSED_BYTES+2)//3),
        'Bounded base64 encoding')
    compressed=base64.b64decode(value['data_b64'],validate=True)
    require(len(compressed)==value['compressed_bytes'] and hashlib.sha256(compressed).hexdigest()==value['compressed_sha256'],
        'Packed bytes/hash')
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    raw=decoder.decompress(compressed,MAX_RAW_BYTES+1)
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
        len(raw)==value['raw_bytes'] and len(raw)<=MAX_RAW_BYTES and hashlib.sha256(raw).hexdigest()==value['raw_sha256'],
        'Complete single gzip member with exact expanded bytes/hash')
    result=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite packed JSON')))
    require(_canonical(result).encode('utf-8')==raw,'Canonical complete lossless JSON')
    return result


def initialize(folder, profiles, runtime_sources, boundary=lambda:None, save_json=None, **kwargs):
    require(save_json is not None,'Explicit new-output writer')
    context=inherited.initialize(folder,profiles,runtime_sources,boundary=boundary,save_json=save_json,**kwargs)
    context['qualification_enabled']=False
    save_json(Path(folder)/'live_adapter_identity.json',dict(schema=LIVE_SCHEMA,
        inherited_implementation='tools.splendorust_teacher_variability_20261113.adapters',
        live_entrypoint=__name__+'.handle',ordinary_quota=512,pools=[3,6],mode='legacy',
        world_seed_label='A',encoding_seed_label='A',late_native_heuristic_unchanged=True,
        forced_pass_unchanged=True,atomic_prereveal_helpers_unchanged=True,
        checkpoint_sha256=CHECKPOINT_SHA256,native_sha256=NATIVE_SHA256,
        corrected_sources=runtime_sources,profile_identities_preserved=True,
        internal_seconds=INTERNAL_SECONDS,complete_seconds=COMPLETE_SECONDS))
    return context


def ordinary(context, request):
    require(type(request) is dict and set(request) == {'player','observer','quota','pool','mode',
        'world_seed_label','encoding_seed_label','telemetry'}, 'ordinary request allowlist')
    player, quota, pool, mode = (request[k] for k in ('player','quota','pool','mode'))
    require(type(player) is str and player in context['adapters'], 'player allowlist')
    key = profile_key(quota, pool, mode)
    require(context.get('qualification_active') or (player == 'search' and quota == 512),
            'live requests use search512; other requests require explicit qualification')
    require(player != 'reference' or quota == 16, 'reference fixed16')
    require(player != 'control' or (mode == 'legacy' and pool == 3), 'legacy control only')
    observer, world_label, encoding_label, telemetry = (request[k] for k in
        ('observer','world_seed_label','encoding_seed_label','telemetry'))
    world_seed, encoding_seed = seed_pair(observer, mode, world_label, encoding_label)
    require(type(telemetry) is bool, 'telemetry bool')
    state = decode_observer(observer)
    require(mode == 'legacy' and world_label == encoding_label == 'A', 'Frozen shared canonical seed and legacy mode')
    require(pool in (3,6) and quota in (16,64,512), 'Bounded live/qualification search settings')
    require(context.get('qualification_active') or telemetry is True, 'Live telemetry required')
    # Live play retains the original >=124 native heuristic and forced pass.
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
    if type(request) is dict and request.get('operation')=='graph_parity':
        from tools.splendorust_inference_opt_20261025.worker import graph_parity
        return graph_parity(context,request)
    require(context['graph_parity_passed'],'Eleven exact graph fixtures required before any decision')
    if type(request) is dict and request.get('operation')=='qualification_decision':
        require(set(request)=={'operation','request'} and context.get('qualification_enabled') is True
            and not context.get('qualification_active'),'Explicit finite qualification service route')
        context['qualification_active']=True
        try:return ordinary(context,request['request'])
        finally:context['qualification_active']=False
    if type(request) is dict and request.get('operation')=='entrypoint_checks':
        from tools.splendorust_teacher_variability_20261113.qualification import run_entrypoint_checks
        require(set(request)=={'operation','mode','pool'} and request['mode']=='legacy' and request['pool'] in (3,6),
            'Explicit corrected legacy pool route')
        require(context.get('qualification_enabled') is True and not context.get('qualification_active'),
            'Entrypoint suite only in qualification service')
        mode,pool=request['mode'],request['pool'];outer_started=context['request_started']
        context['request_started']=None;context['qualification_active']=True
        for adapter in context['adapters'].values():
            adapter.qualification_mode,adapter.qualification_pool=mode,pool
        record,stream=_forensic_recorder(context['folder']/f'entrypoint_{mode}_p{pool}_forensics.jsonl')
        previous_callback=context.get('evidence_callback');context['evidence_callback']=record
        try:
            return run_entrypoint_checks(context,lambda payload:handle(context,payload),mode=mode,pool=pool,
                boundary=context['boundary'],evidence_callback=record)
        except BaseException as error:
            record(dict(event='entrypoint_failed',error=repr(error)));raise
        finally:
            stream.close();context['qualification_active']=False;context['request_started']=outer_started
            if previous_callback is None:context.pop('evidence_callback',None)
            else:context['evidence_callback']=previous_callback
            for adapter in context['adapters'].values():
                adapter.qualification_mode,adapter.qualification_pool='legacy',3
    return ordinary(context,request)


def close_context(context):
    return inherited.close_context(context)
