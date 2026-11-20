"""One exact graph/native/evidence path for fixed64 and128 with three worlds."""
from copy import deepcopy
from pathlib import Path
import time
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent,file_hash
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent, SEARCH_BINARY_SHA256
from tools.splendorust_pool6_strengthfirst_20261028.clock_policy import INTERNAL_SECONDS,SECONDS_PER_MOVE,boundary,require

SCHEMA='splendorust_search128_strength_v1'
EXECUTION_SCHEMA='splendorust_search128_strength_execution_v1'
QUOTAS=(64,128)
WORLD_POOLS=(3,)
SEED_LABELS=('A',)
BASELINE_PROFILE_SHAS={16:'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424',64:'2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9'}

def route_quota(route):
    require(route in ('frozen','research'),'Known quota route')
    return 64 if route=='frozen' else 128

def pool_for(quota):
    require(type(quota) is int and quota in (16,64,128),'Supported genuine quota')
    return 3

def profile_id(quota,world_pool=3,*,route=None):
    pool_for(quota);require(type(world_pool) is int and world_pool==3,'Three worlds fixed')
    if route is not None:require(quota in (16,route_quota(route)),'Route quota fixed')
    return f'strength_search_puct{quota}_pool3_20261028_v1'

def wrapper_id(quota,world_pool=3,*,route=None):return profile_id(quota,world_pool,route=route)+'_10s'
def profile_path(quota,world_pool=3,*,route=None):
    from . import common as c
    profile_id(quota,world_pool,route=route)
    return ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles'/f'puct{quota}.json'
def profile_sha256(quota,world_pool=3,*,route=None):return file_hash(profile_path(quota,world_pool,route=route))
def baseline_profile_path(quota):return profile_path(quota,route='frozen')

def original_profile_descriptor(route,quota):
    from . import common as c
    profile_id(quota,route=route)
    original_quota=16 if quota==16 else 64
    path=original.profile_path(original_quota)
    require(file_hash(path)==BASELINE_PROFILE_SHAS[original_quota],'Pinned original profile')
    return c.artifact(path)

def profile(quota,world_pool=3,*,route=None):
    from . import common as c
    identity=profile_id(quota,world_pool,route=route)
    origin=original_profile_descriptor('frozen' if quota in (16,64) else 'research',quota)
    value=deepcopy(c.read(ROOT/origin['path']))
    require(value['world_pool']==3 and value['depth']==64 and value['internal_budget_seconds']==.9,'Original scientific settings')
    require(file_hash(ROOT/value['native_binary'])==SEARCH_BINARY_SHA256,'Exact immutable original native binary')
    value.update(profile_id=identity,wrapper_id=identity+'_10s',iterations=quota,internal_budget_seconds=INTERNAL_SECONDS,
        execution_policy=dict(schema=EXECUTION_SCHEMA,seconds_per_move=SECONDS_PER_MOVE,internal_seconds=INTERNAL_SECONDS,
            original_profile=origin,changed_scientific_fields=[] if quota!=128 else ['iterations'],
            numerical_implementation_unchanged=True,full_callback_capture_for_both_arms=True,additional_search_when_time_remains=False))
    return value

def baseline_profile(quota):return profile(quota,route='frozen')
def observer_seed(observer,label='A'):
    require(label=='A','Canonical observer seed only')
    return int(digest(observer)[:16],16)

class BaselineOpponent(SearchOpponent):
    """Original SearchOpponent operations; instance-local graph and new guard only."""
    def __init__(self,profile_document,*,shared_from=None):
        self.forward=None
        super().__init__(profile_document,shared_from=shared_from)
    def boundary(self):boundary(self.active_deadline)
    def _begin(self):
        if self.admission_guard is not None:self.admission_guard()
        self.active_deadline=self.request_deadline if self.request_deadline is not None else time.perf_counter()+INTERNAL_SECONDS
        self.boundary()
    def _infer(self,tokens,batch):
        # Identical original body/hashes; graph dispatch is instance-local.
        self.boundary()
        outputs=Opponent._infer(self,tokens,batch) if self.forward is None else self.forward.infer(self,tokens,batch)
        self.boundary()
        input_sha,output_sha=digest(tokens),digest(outputs)
        item={'callback_sha256':digest([input_sha,output_sha,batch]),'batch':batch}
        if self.capture:
            item.update(input_sha256=input_sha,output_sha256=output_sha)
            if not self.callbacks:
                item.update(tokens=deepcopy(tokens),outputs=deepcopy(outputs))
                if batch==1:item['output83']=deepcopy(outputs[0])
        self.callbacks.append(item);self.boundary()
        return outputs
    def choose(self,observer,*,diagnostic_capture=False):
        self._begin();self.capture=bool(diagnostic_capture)
        try:
            action,trace=Opponent.choose(self,observer);self.boundary();return action,trace
        except TimeoutError as error:
            raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: 9s guard; '+str(error)) from error
        finally:self.capture=False;self._finish()
    def direct_choose(self,observer,*,diagnostic_capture=True):
        """Direct original native/host path at this profile's actual quota."""
        self._begin();self.capture=bool(diagnostic_capture)
        try:
            action,trace=Opponent.choose(self,observer);self.boundary();return action,trace
        except TimeoutError as error:
            raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: 9s direct parity guard; '+str(error)) from error
        finally:self.capture=False;self._finish()

    def reference_choose(self,observer,*,diagnostic_capture=True):
        require(self.profile['iterations']==16,'Qualification reference16 only')
        return self.choose(observer,diagnostic_capture=diagnostic_capture)

def initialize_adapters(folder,profiles,*,route):
    quota=route_quota(route)
    require(set(profiles)=={'16',str(quota)},'Explicit shared profile inventory')
    reference=BaselineOpponent(profiles['16'])
    try:
        search=BaselineOpponent(profiles[str(quota)],shared_from=reference)
        require(reference.model is search.model and not reference.model.training,'Shared immutable model')
        return {'reference':reference,'search':search}
    except BaseException:
        reference.close()
        if 'search' in locals():search.close()
        raise

def profile_identity(freeze,route,quota=None,*,boundary=lambda:None):
    from . import common as c
    quota=route_quota(route) if quota is None else quota
    profile_id(quota,route=route)
    items=freeze['baseline_profiles' if route=='frozen' else 'research_profiles']
    item=items[str(quota)];boundary();c.verify([item]);value=c.read(ROOT/item['path'])
    require(value==profile(quota,route=route),'Full explicit profile identity')
    return dict(route=route,quota=quota,world_pool=3,profile_id=value['profile_id'],profile_sha256=item['sha256'],
        wrapper_id=value['wrapper_id'],native_binary_sha256=SEARCH_BINARY_SHA256,internal_seconds=9.,seconds_per_move=10.)

def validate_execution_identity(result,observer,route,*,identity):
    from tools.splendorust_search_scale_20261024.audit import RESULT_FIELDS
    q=identity['quota'];profile_id(q,route=route)
    require(identity['route']==route and identity['world_pool']==3 and identity['internal_seconds']==9. and
        identity['seconds_per_move']==10. and identity['native_binary_sha256']==SEARCH_BINARY_SHA256,'Bound execution identity')
    require(type(result) is dict and set(result)==RESULT_FIELDS and type(result['simulations_quota']) is int and
        result['simulations_quota']==q and result['profile_id']==identity['profile_id']==profile_id(q) and
        result['profile_sha256']==identity['profile_sha256'] and result['wrapper_id']==identity['wrapper_id']==wrapper_id(q) and
        result['diagnostics']['profile_id']==profile_id(q),'Raw exact quota/profile identity')
    require(result['observer_sha256']==digest(observer),'Lawful observer binding')
    return True

def validate_result(result,observer,route,*,identity):
    from . import scientific
    validate_execution_identity(result,observer,route,identity=identity)
    scientific.validate_result(result,observer,result['player'],expected_quota=identity['quota'])
    require(all('inference' in p for p in result['diagnostics']['phases']),'Same full capture for both arms')
    require(all(not p['neural_callback_count'] or
        set(p['inference'][0])=={'input_sha256','output_sha256','batch','tokens','outputs','output83'}
        for p in result['diagnostics']['phases']),'Full first callback arrays required symmetrically')
    return True

def ordinary_payload(spec,observer):return dict(player='search',observer=observer)

def ready_metadata(route,freeze,*,pid,native_pids,sources,freeze_sha256):
    from . import common as c
    from tools.future_training_rules_20261021 import RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    key='baseline_profiles' if route=='frozen' else 'research_profiles'
    return dict(schema=EXECUTION_SCHEMA,route=route,phase=c.PHASE,pid=pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,native_pids=native_pids,
        execution_profiles={q:item['sha256'] for q,item in freeze[key].items()},
        request_quotas=[route_quota(route)],qualification_reference_quota=16,world_pools=[3],
        seed_labels=['A'],qualification_reference_world_pool=3,telemetry_enabled=False,live_turns_unrestricted=True,
        seconds_per_move=SECONDS_PER_MOVE,internal_seconds=INTERNAL_SECONDS,freeze_sha256=freeze_sha256,
        sources=sources,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)

def validate_ready(metadata,child,freeze,*,route=None):
    from . import common as c
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    route=metadata['route'] if route is None else route;roles={}
    for row in child._sample()['process_tree']:
        if row.get('role') in ('python','search','reference'):
            require(row['role'] not in roles,'Duplicate compute role');roles[row['role']]=row['pid']
    require(set(roles)=={'python','search','reference'} and roles['python']==child.process.pid,'Exact compute roles')
    expected=ready_metadata(route,freeze,pid=child.process.pid,native_pids={k:roles[k] for k in ('reference','search')},
        sources=SOURCE_HASHES,freeze_sha256=c.file_hash(c.OUT/'freeze.json'))
    require(metadata==expected and c.read(child.output_dir/'player_identity.json')==metadata,'New execution READY identity')
    return dict(status='passed',route=route,metadata_sha256=digest(metadata),native_roles=roles)

def verify_saved_ready(folder,freeze,*,route=None):
    from . import common as c
    from types import SimpleNamespace
    folder=Path(folder);receipt=c.read(folder/'initialization.json');metadata=receipt['result']
    route=metadata['route'] if route is None else route
    require(receipt['status']=='ok' and metadata['pid']==receipt['identity']['pid'],'Saved READY receipt')
    proof=validate_ready(metadata,SimpleNamespace(output_dir=folder,process=SimpleNamespace(pid=metadata['pid']),
        _sample=lambda:receipt['telemetry']),freeze,route=route)
    identity=c.read(folder/'boundary_timing_identity.json')
    require(identity['backend']=='qualified_graph64' and
        identity['quota']==route_quota(route) and identity['metadata']==metadata and identity['execution_policy_changed'] is True,
        'Saved explicit timing-policy identity');return proof

def graph_parity(child,envelope,freeze,*,request_fn,collect_fn):
    from . import common as c
    from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity
    item=freeze['graph_parity_plan'];c.verify([item]);fixtures=c.read(ROOT/item['path'])['fixtures']
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(60));c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'graph_parity_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Graph startup failure')
    validate_graph_parity(receipt['result'],fixtures);return receipt

def actual_entrypoint(child,envelope,freeze=None,*,request_fn,collect_fn):
    from . import common as c
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(300));c.save(child.output_dir/'entrypoint_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'entrypoint_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Entrypoint startup failure')
    require(receipt['result']['counts']['complete_decisions']==(67 if child.route=='research' else 52),'Original52 plus declared15 candidate direct checks');return receipt

