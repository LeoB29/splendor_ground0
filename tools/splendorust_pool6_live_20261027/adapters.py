"""Fixed pool6-A live wrapper around the accepted sampling arithmetic.

The immutable sampling adapter supplies every inference, native exchange and
compound-action implementation. Only the live identity and fixed configuration
are new; no old class, module global, search or random stream is modified.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import file_hash
from tools.splendorust_sampling_stability_20261027 import adapters as accepted
from tools.splendorust_search_scale_20261024 import common as original

PRIOR=ROOT/'outputs/splendorust_sampling_stability_20261027_01'
PRIOR_FREEZE_SHA='cc550ef7f48597840ed5e41166b317574ef78d8ad88d3ae11b6ef5baca5b1eaa'
SCHEMA='splendorust_pool6_live_v1'
INTERNAL_SECONDS=.9
PROFILE_KEYS=('16-3','64-6')
QUOTAS=(64,)
WORLD_POOLS=(6,)
SEED_LABELS=('A',)

def require(value,message):
    if not value:raise ValueError('POOL6_LIVE: '+message)

def pool_for(quota):
    require(type(quota) is int and quota in (16,64),'Only qualification16 or live64')
    return 3 if quota==16 else 6

def profile_id(quota,world_pool=None):
    pool=pool_for(quota) if world_pool is None else world_pool
    require(type(pool) is int and pool==pool_for(quota),'Fixed role/world-pool configuration')
    return f'live_puct{quota}_pool{pool}_20261027_v1'

def wrapper_id(quota,world_pool=None):return profile_id(quota,world_pool)+'_1s'

def profile_path(quota,world_pool=None):
    from . import common as c
    pool=pool_for(quota) if world_pool is None else world_pool
    profile_id(quota,pool)
    return c.QUAL_OUT/'profiles'/f'live{quota}_pool{pool}.json'

def profile_sha256(quota,world_pool=None):return file_hash(profile_path(quota,world_pool))

def profile(quota,world_pool=None):
    """Bind the accepted exact model/native profile; no numerical execution."""
    from . import common as c
    pool=pool_for(quota) if world_pool is None else world_pool
    identity=profile_id(quota,pool)
    require(file_hash(PRIOR/'freeze.json')==PRIOR_FREEZE_SHA,'Accepted sampling freeze identity')
    old=json.loads((PRIOR/'freeze.json').read_text(encoding='utf-8'))
    item=old['research_profiles'][f'{quota}-{pool}'];c.verify([item])
    result=deepcopy(c.read(ROOT/item['path']))
    require(result['iterations']==quota and result['world_pool']==pool and
        result['internal_budget_seconds']==INTERNAL_SECONDS,'Accepted sampling configuration')
    result.update(profile_id=identity,wrapper_id=wrapper_id(quota,pool),
        purpose='live_pool6_A_fixed_work_candidate' if quota==64 else 'qualification_only_pool3_reference',
        seed_family='A only: first16 hex digits of canonical lawful-observer SHA256',
        telemetry='OFF; callback inputs, outputs and bit hashes retained',
        source_sampling_profile_sha256=item['sha256'],
        original_late_game_and_compound_behavior_preserved=True)
    result['artifacts'] += [item]
    return result

def observer_seed(observer,label='A'):
    require(type(label) is str and label=='A','Live seed family is fixedA')
    return accepted.observer_seed(observer,'A')

class LiveOpponent(accepted.ResearchOpponent):
    """Instance-local fixed configuration; all numerical methods inherited."""
    def __init__(self,profile_document,*,shared_from=None):
        profile_id(profile_document['iterations'],profile_document['world_pool'])
        super().__init__(profile_document,shared_from=shared_from)
        self.telemetry_enabled=False

    def choose(self,observer,*,diagnostic_capture=True):
        quota=self.profile['iterations'];pool=self.profile['world_pool']
        profile_id(quota,pool)
        action,trace=super().choose(observer,diagnostic_capture=True,seed_label='A',
            telemetry=False,quota=quota,world_pool=pool)
        trace['profile_id']=profile_id(quota,pool)
        return action,trace

    def reference_choose(self,observer,*,diagnostic_capture=True):
        require(self.profile['iterations']==16 and self.profile['world_pool']==3,'Qualification-only reference')
        return self.choose(observer,diagnostic_capture=True)

def initialize_adapters(folder,profiles):
    require(set(profiles)==set(PROFILE_KEYS),'Two fixed live/reference profiles')
    reference=LiveOpponent(profiles['16-3'])
    try:
        search=LiveOpponent(profiles['64-6'],shared_from=reference)
        require(reference.model is search.model and not reference.model.training,'Shared immutable model')
        return {'reference':reference,'search':search}
    except BaseException:
        reference.close()
        if 'search' in locals():search.close()
        raise

def ordinary_payload(spec,observer):
    require(spec.get('route') in ('frozen','research'),'Known live route')
    return dict(player='search',observer=observer)

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
