"""Unchanged numerical policies with separate explicit9s/10s execution identities."""
from copy import deepcopy
from pathlib import Path
import time
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent,file_hash
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent
from tools.splendorust_pool6_live_20261027 import adapters as live
from .clock_policy import INTERNAL_SECONDS,SECONDS_PER_MOVE,boundary,require

SCHEMA='splendorust_pool6_strength_live_v1'
EXECUTION_SCHEMA='splendorust_pool6_strength_execution_v1'
PROFILE_KEYS=('16-3','64-6')
QUOTAS=(64,)
WORLD_POOLS=(6,)
SEED_LABELS=('A',)
PRIOR=ROOT/'outputs/splendorust_pool6_live_20261027_qualification_01'
PRIOR_FREEZE_SHA='247b143d1f5256048b336231fb3ba3c544f74ceb7580e7087a1e69b967d480a0'
BASELINE_PROFILE_SHAS={16:'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424',
    64:'2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9'}

def pool_for(quota):
    require(type(quota) is int and quota in (16,64),'Only qualification16 or play64')
    return 3 if quota==16 else 6

def profile_id(quota,world_pool=None,*,route='research'):
    require(route in ('frozen','research'),'Known numerical route')
    pool=(3 if route=='frozen' else pool_for(quota)) if world_pool is None else world_pool
    require(type(quota) is int and quota in (16,64) and type(pool) is int and
        pool==(3 if route=='frozen' else pool_for(quota)),'Fixed route/quota/world pool')
    return f"strength_{'baseline' if route=='frozen' else 'candidate'}_puct{quota}_pool{pool}_20261028_v1"

def wrapper_id(quota,world_pool=None,*,route='research'):return profile_id(quota,world_pool,route=route)+'_10s'
def profile_path(quota,world_pool=None,*,route='research'):
    from . import common as c
    return c.QUAL_OUT/'profiles'/(profile_id(quota,world_pool,route=route)+'.json')
def baseline_profile_path(quota):return profile_path(quota,3,route='frozen')
def profile_sha256(quota,world_pool=None,*,route='research'):return file_hash(profile_path(quota,world_pool,route=route))

def original_profile_descriptor(route,quota):
    from . import common as c
    profile_id(quota,route=route)
    if route=='frozen':
        path=original.profile_path(quota)
        require(file_hash(path)==BASELINE_PROFILE_SHAS[quota],'Pinned original baseline profile')
        return c.artifact(path)
    require(file_hash(PRIOR/'freeze.json')==PRIOR_FREEZE_SHA,'Pinned original live freeze')
    item=c.read(PRIOR/'freeze.json')['research_profiles'][f'{quota}-{pool_for(quota)}'];c.verify([item])
    require((ROOT/item['path']).resolve().is_relative_to((PRIOR/'profiles').resolve()),'Original live profile location')
    return deepcopy(item)

def profile(quota,world_pool=None,*,route='research'):
    from . import common as c
    identity=profile_id(quota,world_pool,route=route);item=original_profile_descriptor(route,quota)
    value=deepcopy(c.read(ROOT/item['path']))
    require(value['iterations']==quota and value['internal_budget_seconds']==.9,'Original work/clock provenance')
    value.update(profile_id=identity,wrapper_id=identity+'_10s',internal_budget_seconds=INTERNAL_SECONDS,
        execution_policy=dict(schema=EXECUTION_SCHEMA,route=route,seconds_per_move=SECONDS_PER_MOVE,
            internal_seconds=INTERNAL_SECONDS,original_profile=deepcopy(item),
            scientific_settings_unchanged=True,additional_search_when_time_remains=False))
    return value

def baseline_profile(quota):return profile(quota,3,route='frozen')
def observer_seed(observer,label='A'):return live.observer_seed(observer,label)

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
    def reference_choose(self,observer,*,diagnostic_capture=True):
        require(self.profile['iterations']==16,'Qualification reference16 only')
        return self.choose(observer,diagnostic_capture=diagnostic_capture)

class LiveOpponent(live.LiveOpponent):
    """Original pool6 arithmetic/capture with explicit9s direct-call fallback."""
    def boundary(self):boundary(self.active_deadline)
    def choose(self,observer,*,diagnostic_capture=True):
        prior=self.request_deadline
        if prior is None:self.request_deadline=time.perf_counter()+INTERNAL_SECONDS
        try:
            action,trace=super().choose(observer,diagnostic_capture=diagnostic_capture)
            trace['profile_id']=profile_id(self.profile['iterations'],self.profile['world_pool'])
            return action,trace
        except TimeoutError as error:
            raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: 9s guard; '+str(error)) from error
        finally:self.request_deadline=prior

def initialize_adapters(folder,profiles,*,route='research'):
    require(set(profiles)==(set(PROFILE_KEYS) if route=='research' else {'16','64'}),'Explicit route profile inventory')
    cls=LiveOpponent if route=='research' else BaselineOpponent
    key=lambda q:f'{q}-{pool_for(q)}' if route=='research' else str(q)
    reference=cls(profiles[key(16)])
    try:
        search=cls(profiles[key(64)],shared_from=reference)
        require(reference.model is search.model and not reference.model.training,'Shared immutable model')
        return {'reference':reference,'search':search}
    except BaseException:
        reference.close()
        if 'search' in locals():search.close()
        raise

def profile_identity(freeze,route,quota=64,*,boundary=lambda:None):
    from . import common as c
    pool=3 if route=='frozen' else pool_for(quota)
    items=freeze['baseline_profiles' if route=='frozen' else 'research_profiles']
    item=items[str(quota) if route=='frozen' else f'{quota}-{pool}'];boundary();c.verify([item])
    value=c.read(ROOT/item['path']);require(value==profile(quota,pool,route=route),'Full execution profile identity')
    old=original_profile_descriptor(route,quota);previous=c.read(ROOT/old['path'])
    native=value['native_binary_sha256'] if 'native_binary_sha256' in value else file_hash(ROOT/value['native_binary'])
    return dict(route=route,quota=quota,world_pool=pool,profile_id=value['profile_id'],profile_sha256=item['sha256'],
        wrapper_id=value['wrapper_id'],native_binary_sha256=native,original_profile_id=previous['profile_id'],
        original_profile_sha256=old['sha256'],original_wrapper_id=previous.get('wrapper_id',original.wrapper_id(quota)),
        internal_seconds=INTERNAL_SECONDS,seconds_per_move=SECONDS_PER_MOVE)

def validate_execution_identity(result,observer,route,*,identity):
    from tools.splendorust_search_scale_20261024 import audit as old_audit
    require(identity['route']==route and identity['internal_seconds']==INTERNAL_SECONDS and identity['seconds_per_move']==SECONDS_PER_MOVE,
        'Externally bound execution policy')
    q=identity['quota'];expected=old_audit.RESULT_FIELDS|({'research'} if route=='research' else set())
    require(type(result) is dict and set(result)==expected and type(result['simulations_quota']) is int and
        result['simulations_quota']==q and result['profile_id']==identity['profile_id'] and
        result['profile_sha256']==identity['profile_sha256'] and result['wrapper_id']==identity['wrapper_id'] and
        result['diagnostics']['profile_id']==identity['profile_id'],'Raw execution identity before science projection')
    require(result['observer_sha256']==digest(observer),'Execution lawful-observer binding')
    if route=='research':
        annotation=result['research']
        require(annotation['schema']==SCHEMA and annotation['internal_seconds']==INTERNAL_SECONDS and
            annotation['native_binary_sha256']==identity['native_binary_sha256'] and
            annotation['world_pool']==identity['world_pool'] and annotation['quota']==q and
            annotation['seed_label']=='A' and annotation['seed']==observer_seed(observer) and
            annotation['telemetry_enabled'] is False,'Raw candidate execution annotation')
    return True

def original_projection(result,observer,route,*,identity):
    """Authenticate raw identity first; project a copy for science, never a clock."""
    validate_execution_identity(result,observer,route,identity=identity)
    value=deepcopy(result);value.pop('research',None);q=identity['quota']
    value.update(profile_id=original.profile_id(q),profile_sha256=original.profile_sha256(q),wrapper_id=original.wrapper_id(q))
    value['diagnostics']['profile_id']=original.profile_id(q);return value

def validate_result(result,observer,route,*,identity):
    from tools.splendorust_search_scale_20261024 import audit as science
    value=original_projection(result,observer,route,identity=identity)
    checked=science.validate_result(value,observer,result['player'],expected_quota=identity['quota'])
    if route=='research':
        from .qualification import validate_raw_qualification_annotations
        validate_raw_qualification_annotations(result,observer,dict(research_native_sha256=identity['native_binary_sha256']))
    return checked

def ordinary_payload(spec,observer):return live.ordinary_payload(spec,observer)

def ready_metadata(route,freeze,*,pid,native_pids,sources,freeze_sha256):
    from . import common as c
    from tools.future_training_rules_20261021 import RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    key='baseline_profiles' if route=='frozen' else 'research_profiles'
    return dict(schema=EXECUTION_SCHEMA,route=route,phase=c.PHASE,pid=pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,native_pids=native_pids,
        execution_profiles={q:item['sha256'] for q,item in freeze[key].items()},
        request_quotas=[64],qualification_reference_quota=16,world_pools=[3 if route=='frozen' else 6],
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
    require(identity['backend']==('qualified_graph64' if route=='frozen' else 'research_graph') and
        identity['quota']==64 and identity['metadata']==metadata and identity['execution_policy_changed'] is True,
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
    require(receipt['result']['counts']['complete_decisions']==52,'Original52 complete scientific checks');return receipt
