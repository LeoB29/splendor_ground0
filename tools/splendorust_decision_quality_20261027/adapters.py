"""Additive research adapter: frozen arithmetic/search, explicit quota and seed.

No imported class or module is rebound. Graph inference is selected on these
new instances, and the original bridge/core/auxiliary helpers remain untouched.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import struct
import time

from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent, file_hash
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_search_scale_20261024 import common as original
from . import native_build

INTERNAL_SECONDS = 29.0
DOMAIN = 'decision_quality_saved_panel_20261027_v1'
QUOTAS = (64, 256, 1024)
SCHEMA = 'splendorust_saved_decision_research_v1'

def require(value, message):
    if not value:
        raise ValueError('DECISION_RESEARCH: '+message)

def profile_id(quota):
    require(type(quota) is int and quota in (16, *QUOTAS), 'Unsupported research quota')
    return f'research_puct{quota}_saved_diagnostic_20261027_v1'

def wrapper_id(quota):
    return profile_id(quota)+'_30s'

def profile_path(quota):
    from . import common as c
    profile_id(quota)
    return c.OUT/'profiles'/f'research{quota}.json'

def profile_sha256(quota):
    return file_hash(profile_path(quota))

def profile(quota):
    """Finite model-free profile construction after the additive build."""
    identity = profile_id(quota)
    result = copy.deepcopy(original.profile_for(16))
    receipt = json.loads((native_build.BUILD/'build_receipt.json').read_text(encoding='utf-8'))
    require(receipt['status']=='compiled_not_model_qualified', 'Additive build receipt')
    native = receipt['natives']['reference' if quota == 16 else 'search']
    result.update(profile_id=identity, wrapper_id=wrapper_id(quota), iterations=quota,
        internal_budget_seconds=INTERNAL_SECONDS, native_binary=native['path'],
        native_binary_sha256=native['sha256'], purpose='saved_position_research_only',
        seed_family='A/B/C derived only from canonical lawful observation',
        telemetry='existing native SearchTargets after unchanged selection',
        deployment_reference_unchanged=True)
    result['artifacts'] += [native, native_build.artifact(native_build.BUILD/'build_receipt.json')]
    return result

def observer_seed(observer, label):
    require(type(label) is str and label in ('A','B','C'), 'Seed label allowlist')
    if label == 'A':
        return int(digest(observer)[:16],16)
    encoded = json.dumps(observer,sort_keys=True,separators=(',',':'),allow_nan=False)
    return int(hashlib.sha256((DOMAIN+'|'+label+'|'+encoded).encode()).hexdigest()[:16],16)

def _bits(values):
    return hashlib.sha256(struct.pack('<'+'f'*len(values),*values)).hexdigest()

class ResearchOpponent(Opponent):
    """Original native transport with instance-local exact graph dispatch."""
    def __init__(self, profile_document, *, shared_from=None):
        self.capture = False
        self.callbacks = []
        self.callback_bits = []
        self.forward = None
        self.active_deadline = self.request_deadline = None
        self.admission_guard = None
        self.telemetry_enabled = True
        self.last_exchange_telemetry = None
        self.last_research_phases = []
        if shared_from is None:
            super().__init__(profile_document)
        else:
            # Same authenticated model/transport setup as the frozen second
            # adapter, without modifying its class or global numerical route.
            import atexit, subprocess, sys
            self.profile = profile_document
            require(self.profile['opponent_id']=='splendorust' and self.profile['deterministic_fixed_work'], 'Profile kind')
            for item in self.profile['artifacts']:
                path=ROOT/item['path']
                require(path.stat().st_size==item['bytes'] and file_hash(path)==item['sha256'], 'Profile input identity')
            self.model,self.reference=shared_from.model,shared_from.reference
            self.proc=subprocess.Popen([str(ROOT/self.profile['native_binary'])],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=sys.stderr,text=True,encoding='utf-8',bufsize=1,cwd=ROOT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            atexit.register(self.close)
            try:
                ready=json.loads(self.proc.stdout.readline())
                require(ready=={'kind':'ready','bridge':'observer_only_fixed_work_v1',
                    'source_pin':'1828507b8d2f0e02bc03d102b9c6d29a54fa8d57'},'Native bridge identity')
            except BaseException:
                self.close();raise
            self.deadline=None
            self.inference_calls=self.inference_rows=0

    def boundary(self):
        if self.active_deadline is not None and time.perf_counter() >= self.active_deadline:
            raise TimeoutError('RESEARCH_INTERNAL_DEADLINE: 29.000-second allowance exceeded')

    def _infer(self, tokens, batch):
        self.boundary()
        outputs = Opponent._infer(self,tokens,batch) if self.forward is None else self.forward.infer(self,tokens,batch)
        self.boundary()
        input_sha,output_sha=digest(tokens),digest(outputs)
        item={'callback_sha256':digest([input_sha,output_sha,batch]),'batch':batch}
        if self.capture:
            item.update(input_sha256=input_sha,output_sha256=output_sha)
            if not self.callbacks:
                item.update(tokens=copy.deepcopy(tokens),outputs=copy.deepcopy(outputs))
                if batch==1: item['output83']=copy.deepcopy(outputs[0])
        self.callbacks.append(item)
        self.callback_bits.append(dict(input_float32_sha256=_bits(tokens),
            output_float32_sha256=_bits([x for row in outputs for x in row])))
        self.boundary()
        return outputs

    def exchange(self, request):
        self.boundary()
        self.callbacks=[];self.callback_bits=[];self.last_exchange_telemetry=None
        if request.get('op')=='select':
            request=dict(request,capture_search_targets=self.telemetry_enabled)
        result=Opponent.exchange(self,request)
        self.boundary()
        self.last_exchange_telemetry=result.pop('search_targets',None)
        result={**result,'neural_callback_count':len(self.callbacks),
            'neural_callback_rows':sum(x['batch'] for x in self.callbacks),
            'neural_callbacks':[x['callback_sha256'] for x in self.callbacks]}
        if self.capture:
            result['inference']=[{k:copy.deepcopy(v) for k,v in x.items() if k!='callback_sha256'} for x in self.callbacks]
        return result

    def choose(self, observer, *, diagnostic_capture=True, seed_label='A', telemetry=True, quota=None):
        if self.admission_guard is not None: self.admission_guard()
        self.active_deadline=self.request_deadline if self.request_deadline is not None else time.perf_counter()+INTERNAL_SECONDS
        self.boundary()
        self.capture=bool(diagnostic_capture)
        require(type(telemetry) is bool,'Explicit telemetry flag')
        self.telemetry_enabled=telemetry
        quota=self.profile['iterations'] if quota is None else quota
        require(type(quota) is int and quota in (16,*QUOTAS),'Research simulation quota')
        self.deadline=self.active_deadline
        self.inference_calls=self.inference_rows=0
        seed=observer_seed(observer,seed_label)
        phases=[];self.last_research_phases=[]
        def select(obs, choices):
            result=self.exchange({'op':'select','observation':obs,'legal':choices,'seed':seed,'iterations':quota})
            phases.append({k:v for k,v in result.items() if k!='kind'})
            self.last_research_phases.append(dict(phase_index=len(phases)-1,phase_kind=obs['phase']['kind'],
                telemetry=copy.deepcopy(self.last_exchange_telemetry),callback_bits=copy.deepcopy(self.callback_bits)))
            return result['action']
        try:
            observation=bridge.encode_observation(observer)
            legal=bridge.main_actions(observer)
            selected=legal[0] if legal==[{'kind':'Pass'}] else select(observation,legal)
            action=bridge.resolve_compound(observer,selected,select)
            self.boundary()
            return action,dict(opponent_id='splendorust',profile_id=profile_id(quota),observer_seed=seed,
                phases=phases,inference_calls=self.inference_calls,inference_rows=self.inference_rows,
                forced_pass=selected['kind']=='Pass')
        except (ValueError,KeyError,AssertionError,IndexError) as error:
            raise ValueError('ADAPTER_PREREQUISITE: '+str(error)) from error
        finally:
            self.deadline=self.active_deadline=None;self.capture=False

    def reference_choose(self, observer, *, diagnostic_capture=True):
        require(self.profile['iterations']==16,'Direct reference reproduction quota')
        return self.choose(observer,diagnostic_capture=diagnostic_capture,seed_label='A',telemetry=True,quota=16)

def initialize_adapters(folder, profiles):
    require(set(profiles)=={'16','64','256','1024'},'Four frozen research profiles')
    reference=ResearchOpponent(profiles['16'])
    try:
        search=ResearchOpponent(profiles['64'],shared_from=reference)
        require(reference.model is search.model and not reference.model.training,'Shared immutable model')
        return {'reference':reference,'search':search}
    except BaseException:
        reference.close()
        if 'search' in locals(): search.close()
        raise

def ordinary_payload(spec,observer):
    from .qualification import ordinary_payload as make
    return make(spec,observer)

def validate_ready(metadata,child,freeze):
    """Authenticate variable-quota research metadata against parent-owned roles."""
    from . import common as c
    from tools.future_training_rules_20261021 import SOURCE_HASHES,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
    from tools.splendorust_inference_opt_20261025 import graph
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    tree=child._sample()['process_tree']
    roles={}
    for row in tree:
        if row.get('role') in ('python','search','reference'):
            require(row['role'] not in roles,'Duplicate research compute role')
            roles[row['role']]=row['pid']
    require(set(roles)=={'python','search','reference'} and roles['python']==child.process.pid,'Exact research process roles')
    expected=dict(schema=SCHEMA,route='research',pid=child.process.pid,threads=1,
        inference_backend=graph.BACKEND,checkpoint_sha256=CHECKPOINT_SHA256,
        native_pids={key:roles[key] for key in ('reference','search')},
        research_profiles={q:item['sha256'] for q,item in freeze['research_profiles'].items()},
        native_binary_sha256=freeze['research_native_sha256'],request_quotas=list(QUOTAS),qualification_reference_quota=16,
        internal_seconds=INTERNAL_SECONDS,freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        sources=SOURCE_HASHES,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT)
    require(c.canonical(metadata)==c.canonical(expected),'Research READY metadata mismatch')
    require(c.read(child.output_dir/'player_identity.json')==metadata,'Persisted research identity mismatch')
    return dict(status='passed',metadata_sha256=digest(metadata),native_roles=roles)

def verify_saved_ready(folder,freeze):
    from . import common as c
    from types import SimpleNamespace
    folder=Path(folder);receipt=c.read(folder/'initialization.json')
    require(receipt['status']=='ok' and receipt['result']['pid']==receipt['identity']['pid'],
            'Saved research initialization identity')
    proxy=SimpleNamespace(output_dir=folder,process=SimpleNamespace(pid=receipt['identity']['pid']),
        _sample=lambda:receipt['telemetry'])
    checked=validate_ready(receipt['result'],proxy,freeze)
    identity=c.read(folder/'boundary_timing_identity.json')
    require(identity['backend']=='research_graph' and identity['quota']==64 and identity['variable_request_quota'] is True
        and identity['metadata']==receipt['result'] and identity['additive_research_response'] is True,
        'Saved research observer identity')
    return checked

def graph_parity(child,envelope,freeze,*,request_fn,collect_fn):
    """One unchanged11-case eager/graph proof for each initialized tree."""
    from . import common as c
    from tools.splendorust_boundary_timing_20261025.runner import validate_graph_parity
    item=freeze['graph_parity_plan'];c.verify([item]);fixtures=c.read(c.ROOT/item['path'])['fixtures']
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(60))
    c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope)
    c.save(child.output_dir/'graph_parity_profiles.json',profiles)
    require(receipt['status']=='ok','Graph parity request failed; no retry')
    require('profile_capture_error' not in profiles and 'placement_error' not in profiles,'Graph parity boundary evidence failure')
    validate_graph_parity(receipt['result'],fixtures)
    return receipt

def actual_entrypoint(child,envelope,freeze=None,*,request_fn,collect_fn):
    from . import common as c
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(300))
    c.save(child.output_dir/'entrypoint_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope)
    c.save(child.output_dir/'entrypoint_profiles.json',profiles)
    require(receipt['status']=='ok','Actual entrypoint failed; no retry')
    require('profile_capture_error' not in profiles and 'placement_error' not in profiles,'Entrypoint boundary evidence failure')
    status=receipt['result']['status']
    require(status in ('entrypoint_counterexamples_passed','research_entrypoint_counterexamples_passed') and
        receipt['result']['counts']['complete_decisions']==52,'Original52 entrypoint coverage/status')
    return receipt
