"""Output-bound facade over the immutable qualified live player.

Scientific schema, profile bytes, class objects, callbacks, payloads and clocks
are inherited exactly. Only READY evidence points to this separate timing job.
"""
from copy import deepcopy
import json
from pathlib import Path

from tools.splendorust_pool6_live_20261027 import adapters as immutable
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import file_hash

PRIOR=ROOT/'outputs/splendorust_pool6_live_20261027_qualification_01'
PRIOR_FREEZE_SHA='247b143d1f5256048b336231fb3ba3c544f74ceb7580e7087a1e69b967d480a0'
SCHEMA=immutable.SCHEMA
INTERNAL_SECONDS=immutable.INTERNAL_SECONDS
PROFILE_KEYS=immutable.PROFILE_KEYS
QUOTAS=immutable.QUOTAS
WORLD_POOLS=immutable.WORLD_POOLS
SEED_LABELS=immutable.SEED_LABELS
require=immutable.require
pool_for=immutable.pool_for
profile_id=immutable.profile_id
wrapper_id=immutable.wrapper_id
observer_seed=immutable.observer_seed
LiveOpponent=immutable.LiveOpponent
initialize_adapters=immutable.initialize_adapters
ordinary_payload=immutable.ordinary_payload


def profile_descriptor(quota,world_pool=None):
    from . import common as c
    pool=pool_for(quota) if world_pool is None else world_pool
    profile_id(quota,pool)
    require(file_hash(PRIOR/'freeze.json')==PRIOR_FREEZE_SHA,'Immutable prior live freeze')
    item=c.read(PRIOR/'freeze.json')['research_profiles'][f'{quota}-{pool}']
    c.verify([item])
    require((ROOT/item['path']).resolve().is_relative_to((PRIOR/'profiles').resolve()),
            'Immutable live profile location')
    return deepcopy(item)


def profile_path(quota,world_pool=None):
    return ROOT/profile_descriptor(quota,world_pool)['path']


def profile_sha256(quota,world_pool=None):
    return profile_descriptor(quota,world_pool)['sha256']


def profile(quota,world_pool=None):
    from . import common as c
    return c.read(profile_path(quota,world_pool))

def validate_ready(metadata,child,freeze):
    from . import common as c
    from tools.future_training_rules_20261021 import SOURCE_HASHES,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    roles={}
    for row in child._sample()['process_tree']:
        if row.get('role') in ('python','search','reference'):
            require(row['role'] not in roles,'Duplicate compute role');roles[row['role']]=row['pid']
    require(set(roles)=={'python','search','reference'} and roles['python']==child.process.pid,'Exact process roles')
    expected=dict(schema=SCHEMA,route='research',phase=c.PHASE,pid=child.process.pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,
        native_pids={key:roles[key] for key in ('reference','search')},
        research_profiles={q:item['sha256'] for q,item in freeze['research_profiles'].items()},
        native_binary_sha256=freeze['research_native_sha256'],request_quotas=[64],qualification_reference_quota=16,
        world_pools=[6],seed_labels=['A'],qualification_reference_world_pool=3,
        telemetry_enabled=False,live_turns_unrestricted=True,
        internal_seconds=INTERNAL_SECONDS,freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        sources=SOURCE_HASHES,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)
    require(c.canonical(metadata)==c.canonical(expected),'Live READY metadata mismatch')
    require(c.read(child.output_dir/'player_identity.json')==metadata,'Persisted live identity mismatch')
    return dict(status='passed',metadata_sha256=digest(metadata),native_roles=roles)

def verify_saved_ready(folder,freeze):
    from . import common as c
    from types import SimpleNamespace
    folder=Path(folder);receipt=c.read(folder/'initialization.json')
    require(receipt['status']=='ok' and receipt['result']['pid']==receipt['identity']['pid'],'Saved READY identity')
    proxy=SimpleNamespace(output_dir=folder,process=SimpleNamespace(pid=receipt['identity']['pid']),
        _sample=lambda:receipt['telemetry'])
    result=validate_ready(receipt['result'],proxy,freeze)
    identity=c.read(folder/'boundary_timing_identity.json')
    require(identity['backend']=='research_graph' and identity['quota']==64 and identity['variable_request_quota'] is True
        and identity['variable_world_pool'] is False and identity['metadata']==receipt['result'] and
        identity['additive_research_response'] is True,'Saved live boundary identity')
    return result

def graph_parity(child,envelope,freeze,*,request_fn,collect_fn):
    from . import common as c
    from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity
    item=freeze['graph_parity_plan'];c.verify([item]);fixtures=c.read(ROOT/item['path'])['fixtures']
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(60))
    c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'graph_parity_profiles.json',profiles)
    require(receipt['status']=='ok','Graph parity failed; no retry')
    require('profile_capture_error' not in profiles and 'placement_error' not in profiles,'Graph boundary evidence failure')
    validate_graph_parity(receipt['result'],fixtures)
    return receipt

def actual_entrypoint(child,envelope,freeze=None,*,request_fn,collect_fn):
    from . import common as c
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(300))
    c.save(child.output_dir/'entrypoint_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'entrypoint_profiles.json',profiles)
    require(receipt['status']=='ok','Actual entrypoint failed; no retry')
    require('profile_capture_error' not in profiles and 'placement_error' not in profiles,'Entrypoint boundary evidence failure')
    require(receipt['result']['status'] in ('entrypoint_counterexamples_passed','research_entrypoint_counterexamples_passed') and
        receipt['result']['counts']['complete_decisions']==52,'Original52 suite coverage')
    return receipt
