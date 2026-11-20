"""Same128 search around the parent or one fixed learned checkpoint."""
from copy import deepcopy
from pathlib import Path
import time
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent,file_hash
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent, SEARCH_BINARY_SHA256
from tools.splendorust_pool6_strengthfirst_20261028.clock_policy import INTERNAL_SECONDS,SECONDS_PER_MOVE,boundary,require

SCHEMA='splendorust_full_corpus_strength_assessment_v1'
EXECUTION_SCHEMA='splendorust_full_corpus_execution_v1'
QUOTAS=(128,)
WORLD_POOLS=(3,)
SEED_LABELS=('A',)
ROUTES=('control','candidate')
COEFFICIENTS={'control':.4,'candidate':.4}
ORIGINAL_PROFILE_SHAS={16:'14aabe7a0354bd38736903a141ae1798f7af4b0852f8036822204d545e48f637',128:'048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb'}
ORIGINAL_PROFILE_ROOT=ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles'
def route_quota(route):
    require(route in ROUTES,'Known checkpoint route');return 128

def route_cpuct(route):route_quota(route);return .4

def pool_for(quota):require(type(quota) is int and quota in (16,128),'Genuine16/128 quota');return 3

def profile_id(quota,world_pool=3,*,route=None):
    from . import common as c
    pool_for(quota);route='control' if route is None else route;route_quota(route)
    require(world_pool==3,'Three worlds fixed')
    return (f'strength_search_puct{quota}_pool3_20261028_v1' if c.is_parent_route(route) else
        f'full_corpus_{c.MATCHUP}_r{c.REPETITION}_{route}_puct{quota}_pool3_20261110_v1')

def route_from_profile_id(identity,quota):
    matches=[r for r in ROUTES if profile_id(quota,route=r)==identity]
    require(len(matches)==1,'Unambiguous checkpoint profile identity');return matches[0]

def wrapper_id(quota,world_pool=3,*,route=None):return profile_id(quota,world_pool,route=route)+'_10s'

def profile_path(quota,world_pool=3,*,route=None):
    from . import common as c
    route='control' if route is None else route;profile_id(quota,world_pool,route=route)
    return ORIGINAL_PROFILE_ROOT/f'puct{quota}.json' if c.is_parent_route(route) else c.OUT/'profiles'/f'{route}_puct{quota}.json'

def profile_sha256(quota,world_pool=3,*,route=None):return file_hash(profile_path(quota,world_pool,route=route))
def baseline_profile_path(quota):return profile_path(quota,route='control')

def native_descriptor(route,quota=None):
    from . import common as c
    quota=128 if quota is None else quota;profile_id(quota,route=route)
    source=ORIGINAL_PROFILE_ROOT/f'puct{quota}.json'
    require(file_hash(source)==ORIGINAL_PROFILE_SHAS[quota],'Original native profile preserved')
    path=ROOT/c.read(source)['native_binary'];require(file_hash(path)==SEARCH_BINARY_SHA256,'Original native binary')
    return c.artifact(path)

def native_artifacts(route):return [native_descriptor(route,16),native_descriptor(route,128)]
def native_roles(route):return {i['path']:r for i,r in zip(native_artifacts(route),('reference','search'),strict=True)}

def profile_descriptors(freeze,route):
    route_quota(route);items=freeze['profiles'][route]
    require(set(items)=={'16','128'},'Explicit checkpoint-specific16/128 profiles');return items

def candidate_profile(quota,manifest_descriptor,reference,*,route='candidate'):
    from . import common as c
    from . import checkpoint
    c.verify([manifest_descriptor]);manifest=c.read(ROOT/manifest_descriptor['path'])
    require(manifest.get('schema')==checkpoint.SCHEMA and
        type(manifest.get('repetition')) is int and manifest['repetition']==c.REPETITION,
        'Explicit derived checkpoint schema and exact repetition')
    require(route in ROUTES and not c.is_parent_route(route) and manifest['route']==route and manifest['matchup']==c.MATCHUP, 'Explicit learned profile route')
    value=deepcopy(original_profile(quota));identity=profile_id(quota,route=route)
    value.update(profile_id=identity,wrapper_id=identity+'_10s',checkpoint=manifest['checkpoint']['path'])
    value['artifacts']=[i for i in value['artifacts'] if i['path']!=original_profile(quota)['checkpoint']]
    for item in (manifest_descriptor,manifest['checkpoint'],manifest['tensor_manifest'],reference):
        if item not in value['artifacts']:value['artifacts'].append(item)
    value['execution_policy']=dict(schema=EXECUTION_SCHEMA,original_profile=c.artifact(ORIGINAL_PROFILE_ROOT/f'puct{quota}.json'),
        changed_scientific_fields=['checkpoint'],internal_seconds=9.,seconds_per_move=10.,
        additional_search_when_time_remains=False,numerical_implementation_unchanged=True,
        full_callback_capture_for_both_arms=True,native_reference_export_qualification_only=True)
    value['learned_checkpoint_manifest']=manifest_descriptor
    value['native_reference_model']=reference
    return value

def original_profile(quota):
    from . import common as c
    path=ORIGINAL_PROFILE_ROOT/f'puct{quota}.json'
    require(file_hash(path)==ORIGINAL_PROFILE_SHAS[quota],'Immutable original profile')
    value=c.read(path)
    require(value['iterations']==quota and value['world_pool']==3 and value['cpuct']==.4 and value['depth']==64 and value['internal_budget_seconds']==9.,'Matched original search settings')
    return value


def profile(quota,world_pool=3,*,route=None):
    from . import common as c
    route='control' if route is None else route;profile_id(quota,world_pool,route=route)
    if c.is_parent_route(route):return original_profile(quota)
    freeze=c.read(c.OUT/'freeze.json')
    return candidate_profile(quota,freeze['trained_manifests'][route],freeze['checkpoint_bindings'][route]['native_reference'],route=route)


def prepare_profiles(manifests,bindings):
    from . import common as c
    profiles={}
    require(set(manifests)=={r for r in ROUTES if not c.is_parent_route(r)},'Every trained route declared')
    for route in ROUTES:
        profiles[route]={}
        for quota in (16,128):
            path=profile_path(quota,route=route)
            if not c.is_parent_route(route):c.save(path,candidate_profile(quota,manifests[route],bindings[route]['native_reference'],route=route))
            profiles[route][str(quota)]=c.artifact(path)
    return dict(profiles=profiles,native_artifacts_by_route={r:native_artifacts(r) for r in ROUTES},native_roles_by_route={r:native_roles(r) for r in ROUTES})


def baseline_profile(quota):return original_profile(quota)


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

def initialize_adapters(folder,profiles,*,route,freeze):
    from . import checkpoint as training
    from . import common as c
    from types import SimpleNamespace
    if c.is_parent_route(route):model=training.load_parent()
    else:model,_=training.load_candidate(freeze['trained_manifests'][route])
    holder=SimpleNamespace(model=model,reference={})
    reference=BaselineOpponent(profiles['16'],shared_from=holder)
    try:
        search=BaselineOpponent(profiles['128'],shared_from=reference)
        for instance in (reference,search):
            instance.native_reference_bin=c.ROOT/freeze['checkpoint_bindings'][route]['native_reference']['path']
        require(reference.model is search.model and not reference.model.training,'Shared frozen checkpoint')
        return {'reference':reference,'search':search}
    except BaseException:
        reference.close()
        if 'search' in locals():search.close()
        raise


def profile_identity(freeze,route,quota=None,*,boundary=lambda:None):
    from . import common as c
    quota=route_quota(route) if quota is None else quota
    profile_id(quota,route=route);item=profile_descriptors(freeze,route)[str(quota)]
    boundary();c.verify([item]);value=c.read(ROOT/item['path'])
    expected=(original_profile(quota) if c.is_parent_route(route) else candidate_profile(quota,freeze['trained_manifests'][route],freeze['checkpoint_bindings'][route]['native_reference'],route=route))
    require(value==expected,'Full explicit profile identity')
    return dict(route=route,quota=quota,world_pool=3,cpuct=.4 if quota==16 else route_cpuct(route),
        profile_id=value['profile_id'],profile_sha256=item['sha256'],wrapper_id=value['wrapper_id'],
        native_binary_sha256=native_descriptor(route,quota)['sha256'],internal_seconds=9.,seconds_per_move=10.)

def validate_execution_identity(result,observer,route,*,identity):
    from tools.splendorust_search_scale_20261024.audit import RESULT_FIELDS
    q=identity['quota'];profile_id(q,route=route)
    require(identity['route']==route and identity['world_pool']==3 and identity['internal_seconds']==9. and
        identity['seconds_per_move']==10. and identity['native_binary_sha256']==native_descriptor(route,q)['sha256'] and
        identity['cpuct']==(.4 if q==16 else route_cpuct(route)),'Bound execution identity')
    require(type(result) is dict and set(result)==RESULT_FIELDS and type(result['simulations_quota']) is int and
        result['simulations_quota']==q and result['profile_id']==identity['profile_id']==profile_id(q,route=route) and
        result['profile_sha256']==identity['profile_sha256'] and result['wrapper_id']==identity['wrapper_id']==wrapper_id(q,route=route) and
        result['diagnostics']['profile_id']==profile_id(q,route=route),'Raw exact quota/profile identity')
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

def ready_metadata(route,freeze,*,pid,native_pids,sources,freeze_sha256,graph_identity_sha256):
    from . import common as c
    from tools.future_training_rules_20261021 import RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    descriptors=profile_descriptors(freeze,route)
    return dict(schema=EXECUTION_SCHEMA,route=route,phase=c.PHASE,repetition=c.REPETITION,matchup=c.MATCHUP,trained_arm=c.ROUTE_ARMS[route],pid=pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=freeze['checkpoint_bindings'][route]['checkpoint']['sha256'],
        graph_identity_sha256=graph_identity_sha256,native_pids=native_pids,
        execution_profiles={q:item['sha256'] for q,item in descriptors.items()},
        request_quotas=[route_quota(route)],qualification_reference_quota=16,world_pools=[3],
        seed_labels=['A'],cpuct=route_cpuct(route),qualification_reference_cpuct=.4,qualification_reference_world_pool=3,telemetry_enabled=False,live_turns_unrestricted=True,
        seconds_per_move=SECONDS_PER_MOVE,internal_seconds=INTERNAL_SECONDS,freeze_sha256=freeze_sha256,
        sources=sources,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)

def validate_ready(metadata,child,freeze,*,route=None):
    from . import common as c
    from .worker import lifecycle_route
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    route=metadata['route'] if route is None else route;roles={}
    require(lifecycle_route(child.output_dir,freeze,route=route)==route,'Frozen lifecycle/route readiness')
    for row in child._sample()['process_tree']:
        if row.get('role') in ('python','search','reference'):
            require(row['role'] not in roles,'Duplicate compute role');roles[row['role']]=row['pid']
    require(set(roles)=={'python','search','reference'} and roles['python']==child.process.pid,'Exact compute roles')
    expected=ready_metadata(route,freeze,pid=child.process.pid,native_pids={k:roles[k] for k in ('reference','search')},
        sources=SOURCE_HASHES,freeze_sha256=c.file_hash(c.OUT/'freeze.json'),graph_identity_sha256=c.digest(c.read(child.output_dir/'graph_identity.json')))
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
    from .worker import verify_graph_admission
    require(child._counter==0,'Graph parity is the sole first request')
    item=freeze['graph_parity_plan'];c.verify([item]);fixtures=c.read(ROOT/item['path'])['fixtures']
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(60));c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'graph_parity_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Graph startup failure')
    verify_graph_admission(child.output_dir,freeze,receipt['result'],boundary=envelope.guard);return receipt

def actual_entrypoint(child,envelope,freeze=None,*,request_fn,collect_fn):
    from . import common as c
    require(child._counter==1,'Actual entrypoint is the sole second request')
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(300));c.save(child.output_dir/'entrypoint_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'entrypoint_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Entrypoint startup failure')
    require(receipt['result']['counts']['complete_decisions']==67,'Original52 plus15 declared same-quota direct checks');return receipt



def verify_native_configuration_receipts(folder,route,*,boundary=lambda:None):
    """Original binary has no configured-coefficient side channel on either route."""
    route_quota(route);boundary();folder=Path(folder)
    require(not (folder/'native_configuration_receipts.jsonl').exists() and
        not (folder/'native_configuration_contract.json').exists(),'Original native protocol unchanged')
    return dict(status='passed',route=route,configured_native=False,phase_receipts=0,
        native_binary_sha256=native_descriptor(route)['sha256'])
