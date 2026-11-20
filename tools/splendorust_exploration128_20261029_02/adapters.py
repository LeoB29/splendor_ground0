"""Explicit128 exploration routes; immutable original16 and128 controls."""
from copy import deepcopy
from pathlib import Path
import time
import struct,json,os
from tools.external_strength_screen_20261018.common import ROOT,digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent,file_hash
from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent, SEARCH_BINARY_SHA256
from tools.splendorust_pool6_strengthfirst_20261028.clock_policy import INTERNAL_SECONDS,SECONDS_PER_MOVE,boundary,require

SCHEMA='splendorust_exploration128_v1'
EXECUTION_SCHEMA='splendorust_exploration128_execution_v1'
QUOTAS=(128,)
WORLD_POOLS=(3,)
SEED_LABELS=('A',)
ROUTES=('frozen','low','high','parity')
COEFFICIENTS={'frozen':.4,'low':.2,'high':.8,'parity':.4}
ORIGINAL_PROFILE_SHAS={16:'14aabe7a0354bd38736903a141ae1798f7af4b0852f8036822204d545e48f637',128:'048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb'}
NATIVE_SCHEMA='exploration128_native_configuration_v1'
# Scientific artifacts remain the byte-identical first-attempt players.
SCIENTIFIC_OUT=ROOT/'outputs/splendorust_exploration128_20261029_01'
NATIVE_BUILD_RECEIPT_SHA256='161f0d59e0081982746baab0af00200edd32faf9f2025e9b84c56611c7c276f4'
CONFIGURED_PROFILE_SHAS={'low':'12396c90dce47535ade2870775617ccef80a3126d4720db362e274c62263f14f',
    'high':'428626abcae68fe40ab189df16a1c61e9991a7da208aedf8e6ac198d63368140',
    'parity':'b2a544b7c1fec7a5e8ffa29892597ea8b9123dfcc4940d8f6a28f0fccc0f2295'}

def route_quota(route):
    require(route in ROUTES,'Known exploration route');return 128

def route_cpuct(route):
    route_quota(route);return COEFFICIENTS[route]

def pool_for(quota):
    require(type(quota) is int and quota in (16,128),'Genuine16 or128 quota');return 3

def profile_id(quota,world_pool=3,*,route='frozen'):
    pool_for(quota);route_quota(route);require(type(world_pool) is int and world_pool==3,'Three worlds fixed')
    return (f'strength_search_puct{quota}_pool3_20261028_v1' if quota==16 or route=='frozen' else
            f'exploration_search_puct128_pool3_cpuct{int(route_cpuct(route)*10):02d}_20261029_v1')

def route_from_profile_id(identity,quota):
    matches=[route for route in ROUTES if profile_id(quota,route=route)==identity]
    require(bool(matches),'Declared exact profile identifier');return matches[0]


def wrapper_id(quota,world_pool=3,*,route='frozen'):return profile_id(quota,world_pool,route=route)+'_10s'

def profile_path(quota,world_pool=3,*,route='frozen'):
    from . import common as c
    profile_id(quota,world_pool,route=route)
    return (ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles'/f'puct{quota}.json'
            if quota==16 or route=='frozen' else SCIENTIFIC_OUT/'profiles'/f'{route}_128.json')

def profile_sha256(quota,world_pool=3,*,route='frozen'):return file_hash(profile_path(quota,world_pool,route=route))
def baseline_profile_path(quota):return profile_path(quota,route='frozen')

def native_build_receipt():
    from . import common as c
    path=SCIENTIFIC_OUT/'native_build/build_receipt.json'
    require(file_hash(path)==NATIVE_BUILD_RECEIPT_SHA256,'Immutable first-attempt native build receipt')
    value=c.read(path)
    require(value['status']=='compiled_and_model_free_tests_passed','Qualified native build receipt')
    c.verify([value['executable']]);return value

def native_descriptor(route,quota=128):
    from . import common as c
    profile_id(quota,route=route)
    if quota==16 or route=='frozen':
        value=c.read(profile_path(quota)); path=ROOT/value['native_binary']
        require(file_hash(path)==SEARCH_BINARY_SHA256,'Original immutable binary');return c.artifact(path)
    return native_build_receipt()['executable']

def native_artifacts(route):return [native_descriptor(route,16),native_descriptor(route,128)]
def native_roles(route):return {item['path']:role for item,role in zip(native_artifacts(route),('reference','search'),strict=True)}

def profile_descriptors(freeze,route):
    route_quota(route); items=freeze['profiles'][route]
    require(set(items)=={'16','128'},'Explicit reference16 and search128');return items

def profile(quota,world_pool=3,*,route='frozen'):
    from . import common as c
    identity=profile_id(quota,world_pool,route=route)
    origin_path=ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles'/f'puct{quota}.json'
    require(file_hash(origin_path)==ORIGINAL_PROFILE_SHAS[quota],'Pinned original scientific profile')
    value=deepcopy(c.read(origin_path))
    if quota==16 or route=='frozen':return value
    native=native_descriptor(route,quota);old_path=value['native_binary']
    value['artifacts']=[native if item['path']==old_path else item for item in value['artifacts']]
    value['artifacts']+=[c.artifact(SCIENTIFIC_OUT/'native_build/build_receipt.json'),c.artifact(ROOT/'tools/splendorust_exploration128_20261029/native_bridge.rs')]
    value.update(profile_id=identity,wrapper_id=identity+'_10s',native_binary=native['path'],cpuct=route_cpuct(route),
        execution_policy=dict(schema=EXECUTION_SCHEMA,seconds_per_move=10.,internal_seconds=9.,
            original_profile=c.artifact(origin_path),changed_scientific_fields=['cpuct'] if route!='parity' else [],
            configuration_route=route,native_configuration_schema=NATIVE_SCHEMA,full_callback_capture_for_both_arms=True,
            additional_search_when_time_remains=False))
    require(file_hash(profile_path(quota,route=route))==CONFIGURED_PROFILE_SHAS[route] and
        c.read(profile_path(quota,route=route))==value,'Exact preserved configured profile bytes/content')
    return value

def prepare_profiles():
    from . import common as c
    result={}
    for route in ROUTES:
        result[route]={}
        for quota in (16,128):
            path=profile_path(quota,route=route);value=profile(quota,route=route)
            require(path.is_file() and c.read(path)==value,'Read-only preserved profile identity')
            result[route][str(quota)]=c.artifact(path)
    return dict(profiles=result,native_artifacts_by_route={route:native_artifacts(route) for route in ROUTES},
        native_roles_by_route={route:native_roles(route) for route in ROUTES},native_build_receipt=c.artifact(SCIENTIFIC_OUT/'native_build/build_receipt.json'))

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

class ConfiguredOpponent(BaselineOpponent):
    """Only select requests gain the frozen coefficient; preserve native receipt."""
    def __init__(self,profile_document,*,shared_from,folder,route):
        self.route=route;self.attestation_folder=Path(folder);self.attestation_sequence=0;self.attestation_previous=None
        require(route in ('low','high','parity') and profile_document['iterations']==128 and
                profile_document['cpuct']==route_cpuct(route),'Frozen configured128 profile')
        super().__init__(profile_document,shared_from=shared_from)
        contract=super().exchange({'op':'exploration_contract'})
        expected={'kind':'result','schema':NATIVE_SCHEMA,'allowed_cpuct':[.2,.4,.8],'allowed_iterations':[128],
                  'selection_field':'NeuralAgent.cpuct','core_search_unchanged':True}
        # SearchOpponent adds empty callback receipt fields even for administrative calls.
        require({k:v for k,v in contract.items() if k not in ('neural_callback_count','neural_callback_rows','neural_callbacks')}==expected,
                'Actual configured native contract')
        from . import common as c
        c.save(self.attestation_folder/'native_configuration_contract.json',contract)
    def exchange(self,request):
        if request.get('op')!='select':return super().exchange(request)
        require('cpuct' not in request and request['iterations']==128,'Only128 frozen parameter injection')
        configured=dict(request,cpuct=route_cpuct(self.route))
        result=super().exchange(configured)
        require('native_configuration' in result,'Actual native configuration receipt missing')
        attestation=result['native_configuration'];validate_native_configuration(attestation,self.route)
        event={'sequence':self.attestation_sequence,'previous_sha256':self.attestation_previous,
               'route':self.route,'request_sha256':digest(configured),'observation_sha256':digest(request['observation']),
               'seed':request['seed'],'iterations':128,'cpuct':configured['cpuct'],'legal_sha256':digest(request['legal']),
               'native_result':{k:v for k,v in result.items() if k not in ('inference','neural_callbacks','neural_callback_rows','neural_callback_count')}}
        encoded=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        path=self.attestation_folder/'native_configuration_receipts.jsonl'
        require((path.stat().st_size if path.exists() else 0)+len(encoded.encode())<=32*1024**2,'Native attestation bound')
        with path.open('a',encoding='utf-8',newline='\n') as stream:stream.write(encoded);stream.flush()
        self.attestation_previous=digest(event);self.attestation_sequence+=1
        return {k:v for k,v in result.items() if k!='native_configuration'}

def validate_native_configuration(value,route):
    expected={'schema':NATIVE_SCHEMA,'cpuct':route_cpuct(route),'cpuct_bits':struct.unpack('<Q',struct.pack('<d',route_cpuct(route)))[0],
              'world_pool':3,'chance_universes':3,'dynamic_fpu':True,'fpu_reduction':.02965,'uniform_prior':0.,'inference_batch_size':1}
    require(type(value) is dict and digest(value)==digest(expected),'Actual native search configuration differs')
    return True

def initialize_adapters(folder,profiles,*,route):
    quota=route_quota(route)
    require(set(profiles)=={'16',str(quota)},'Explicit shared profile inventory')
    reference=BaselineOpponent(profiles['16'])
    try:
        search=(BaselineOpponent(profiles[str(quota)],shared_from=reference) if route=='frozen' else
                ConfiguredOpponent(profiles[str(quota)],shared_from=reference,folder=folder,route=route))
        require(reference.model is search.model and not reference.model.training,'Shared immutable model')
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
    require(value==profile(quota,route=route),'Full explicit profile identity')
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

def ready_metadata(route,freeze,*,pid,native_pids,sources,freeze_sha256):
    from . import common as c
    from tools.future_training_rules_20261021 import RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    descriptors=profile_descriptors(freeze,route)
    return dict(schema=EXECUTION_SCHEMA,route=route,phase=c.PHASE,pid=pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,native_pids=native_pids,
        execution_profiles={q:item['sha256'] for q,item in descriptors.items()},
        request_quotas=[route_quota(route)],qualification_reference_quota=16,world_pools=[3],
        seed_labels=['A'],cpuct=route_cpuct(route),qualification_reference_cpuct=.4,qualification_reference_world_pool=3,telemetry_enabled=False,live_turns_unrestricted=True,
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
    require(receipt['result']['counts']['complete_decisions']==67,'Original52 plus15 declared128 direct checks');return receipt



def _configuration_expected(request,result,route):
    configured=dict(request,cpuct=route_cpuct(route))
    fields=('kind','action','simulations','leaf_calls','native_legal_count','heuristic_phase')
    native={key:result[key] for key in fields if key in result};native.setdefault('kind','result')
    require(set(native)==set(fields),'Full native phase result')
    return dict(route=route,request_sha256=digest(configured),observation_sha256=digest(request['observation']),
        seed=request['seed'],iterations=128,cpuct=route_cpuct(route),legal_sha256=digest(request['legal']),native_result=native)

def _phase_configuration_expected(observer,trace,route):
    from tools.external_strength_screen_20261018_02 import bridge
    phases=iter(trace['phases']);rows=[];seed=int(digest(observer)[:16],16)
    def select(observation,legal):
        phase=next(phases)
        request=dict(op='select',observation=observation,legal=legal,seed=seed,iterations=128)
        rows.append(_configuration_expected(request,phase,route));return phase['action']
    legal=bridge.main_actions(observer)
    selected=legal[0] if legal==[{'kind':'Pass'}] else select(bridge.encode_observation(observer),legal)
    bridge.resolve_compound(observer,selected,select)
    require(next(phases,None) is None,'Exact phase attestation inventory');return rows

def verify_native_configuration_receipts(folder,route,*,boundary=lambda:None):
    """Fresh readback binds native agent fields to each successful scientific phase."""
    from . import common as c
    folder=Path(folder);path=folder/'native_configuration_receipts.jsonl'
    if route=='frozen':
        require(not path.exists() and not (folder/'native_configuration_contract.json').exists(),'Original native path unchanged')
        return dict(status='passed',route=route,configured_native=False,phase_receipts=0)
    contract=c.read(folder/'native_configuration_contract.json')
    require(contract==dict(kind='result',schema=NATIVE_SCHEMA,allowed_cpuct=[.2,.4,.8],allowed_iterations=[128],
        selection_field='NeuralAgent.cpuct',core_search_unchanged=True,neural_callback_count=0,neural_callback_rows=0,neural_callbacks=[]),
        'Exact native initialization contract')
    expected=[]
    forensic=folder/'entrypoint_forensics.jsonl'
    if forensic.exists():
        with forensic.open(encoding='utf-8') as stream:
            for line in stream:
                boundary();event=json.loads(line)['event']
                if event.get('event')!='result' or event.get('arm')!='search':continue
                op=event.get('operation')
                if op=='ordinary_worker_request':expected.extend(_phase_configuration_expected(event['request']['observer'],event['result']['diagnostics'],route))
                elif op=='direct_same_quota_choose':expected.extend(_phase_configuration_expected(event['request']['observer'],event['result']['diagnostics'],route))
                elif op=='exchange' and event['request']['native_request'].get('op')=='select':
                    expected.append(_configuration_expected(event['request']['native_request'],event['result'],route))
    failed=False
    for payload_path in sorted((folder/'inputs').glob('*.json')):
        boundary();payload=c.read(payload_path);clock=c.read(folder/(payload_path.stem+'.json'))
        require(not failed,'No requests after failed response')
        if clock['status']=='ok':expected.extend(_phase_configuration_expected(payload['observer'],clock['result']['diagnostics'],route))
        else:failed=True
    rows=[];previous=None
    if path.exists():
        require(path.stat().st_size<=32*1024**2,'Native configuration file bound')
        with path.open(encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                boundary();require(len(line)<=65536,'Bounded native configuration line');event=json.loads(line)
                require(event['sequence']==index and event['previous_sha256']==previous,'Native configuration hash chain')
                require(set(event)=={'sequence','previous_sha256','route','request_sha256','observation_sha256','seed','iterations','cpuct','legal_sha256','native_result'},'Exact native configuration event')
                require(event['route']==route and event['iterations']==128 and event['cpuct']==route_cpuct(route),'Native request coefficient binding')
                validate_native_configuration(event['native_result']['native_configuration'],route)
                value={key:value for key,value in event.items() if key not in ('sequence','previous_sha256')}
                value['native_result']={key:value for key,value in value['native_result'].items() if key!='native_configuration'}
                if index<len(expected):require(digest(value)==digest(expected[index]),'Actual native phase request/result differs from scientific evidence')
                rows.append(event);previous=digest(event)
    require(len(rows)>=len(expected) and (len(rows)==len(expected) or failed and len(rows)-len(expected)<=4),
        'Every completed native phase bound; only final failed-turn suffix allowed')
    return dict(status='passed',route=route,configured_native=True,phase_receipts=len(rows),bound_successful_phases=len(expected),
        preserved_failed_phase_suffix=len(rows)-len(expected),terminal_sha256=previous,
        artifact=c.artifact(path) if path.exists() else None,contract=c.artifact(folder/'native_configuration_contract.json'))
