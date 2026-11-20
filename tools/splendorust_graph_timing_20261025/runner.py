"""Bounded component timing, with overhead admission before the saved diagnostic."""
import argparse
import hashlib
import struct
import math
import re
from importlib.metadata import version
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import psutil
from . import common as c, qualification as plan_rules, analyze
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology, _decode
from tools.splendorust_search_scale_20261024 import audit, qualification as frozen_q
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_search_scale_20261024.compression import verify_log
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules
MODES=('baseline','profiled')
def bytes_used():
    return sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())

class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = 0., False, 0
        self.evidence = None
        self.resident = {}
        self.peak_combined_rss = 0

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def note_write(self, path, size):
        if not path.resolve().is_relative_to((c.OUT/'workers').resolve()):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        self.evidence['workers'][folder] = None

    def own_worker(self, child):
        proc = psutil.Process(child.process.pid)
        identity = (proc.pid, proc.create_time())
        c.require(identity not in self.resident, 'Duplicate resident worker identity')
        self.resident[identity] = proc

    def close_worker(self, folder, child=None):
        if child is not None:
            identities = [identity for identity in self.resident if identity[0] == child.process.pid]
            for identity in identities:
                self.resident.pop(identity)
        self.evidence['workers'][folder] = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())

    def aggregate_worker_rss(self, exclude_pid=None, additional_rss=0):
        seen, total = set(), additional_rss
        for identity, root in self.resident.items():
            if identity[0] == exclude_pid:
                continue
            try:
                c.require(root.create_time() == identity[1], 'Resident root PID identity changed')
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                raise ValueError('Resident worker disappeared before owned cleanup')
            for process in processes:
                try:
                    child_identity = (process.pid, process.create_time())
                    if child_identity not in seen:
                        total += process.memory_info().rss
                        seen.add(child_identity)
                except psutil.NoSuchProcess:
                    # Helpers may exit between enumeration and observation; the
                    # owning StudyWorker independently checks required roles.
                    continue
        self.peak_combined_rss = max(self.peak_combined_rss, total)
        c.require(total <= 2*c.GiB, 'Combined resident workers/helpers RSS exceeded 2GiB')
        return total

    def guard_worker_sample(self, child, telemetry):
        """Include a new tree before its constructor has returned readiness."""
        self.aggregate_worker_rss(exclude_pid=child.process.pid, additional_rss=telemetry['rss_bytes'])

    def accounted_bytes(self):
        if self.evidence is None:
            return bytes_used()
        total = self.evidence['baseline'] + self.evidence['parent_bytes']
        for folder, closed in self.evidence['workers'].items():
            total += closed if closed is not None else sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
        for name, initial in self.log_baseline.items():
            path = c.OUT/name
            total += max(0, (path.stat().st_size if path.exists() else 0)-initial)
        return total + 65536

    def begin(self, stage):
        mono, wall = time.monotonic(), time.time()
        self.stage = stage
        cap = c.STAGES[stage]
        cutoff = c.JOB_SECONDS if stage == 'final' else c.COLLECTION_CUTOFF_SECONDS
        self.deadline = min(mono+cap, self.start+cutoff, self.end)
        self.wall_deadline = min(wall+cap, self.wall_start+cutoff, self.wall_start+c.JOB_SECONDS)
        self.guard()
        c.save(c.OUT/'stages'/f'{stage}_start.json', dict(time=c.now(), monotonic=mono, unix_time=wall,
            monotonic_deadline=self.deadline, unix_deadline=self.wall_deadline, cap_seconds=cap))

    def remaining(self):
        return min(self.deadline-time.monotonic(), self.wall_deadline-time.time(),
                   self.end-time.monotonic(), self.wall_start+c.JOB_SECONDS-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(self.remaining() > 0, 'Stage/global real-clock deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = c.MAX_EVIDENCE_BYTES if self.stage == 'final' else c.PREFINAL_EVIDENCE_BYTES
        c.require(self.accounted_bytes() < cap, 'Study evidence ceiling')
        self.aggregate_worker_rss()

    def timeout(self, maximum):
        self.guard()
        return min(maximum, self.remaining())

    def admit(self):
        self.guard()
        begun = time.monotonic()
        available = psutil.virtual_memory().available
        stable = begun if available >= 2*c.GiB else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-begun < 600, 'Cumulative RAM wait exceeded ten minutes')
            if stable is not None and (self.admitted or current-stable >= 60):
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.require(self.waited < 600, 'Cumulative RAM admission bound')
        if not fast:
            c.save(c.OUT/'memory_admissions'/f'{self.admissions:06d}.json', dict(time=c.now(), wait_seconds=elapsed,
                cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum,
                stable_seconds=60 if self.admissions == 1 else 0))

    def finish(self):
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))

def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    c.require(processor_topology() == freeze['topology'], 'Processor topology changed')
    for item in freeze['torch_runtime']:
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'], 'Pinned Torch runtime changed')
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(runtime_contract() == freeze['compression_runtime'], 'Compression runtime changed')
    for item in freeze['external_runtime']:
        if envelope:
            envelope.guard()
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'],
                  'Pinned sampler runtime changed')
    c.require(snapshot_api_modules() == freeze['sampler_api_modules'],
              'Loaded snapshot API modules or actual function owners changed')
    c.require(version('numpy')==freeze['numpy_version'] and version('torch')==freeze['torch_package_version'],
              'Numerical runtime package changed')

def nested_artifacts(value):
    """Only explicit artifact triples, never an unbounded output-tree expansion."""
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):
            yield {key:value[key] for key in ('path','sha256','bytes')}
        for item in value.values():
            yield from nested_artifacts(item)
    elif isinstance(value,list):
        for item in value:
            yield from nested_artifacts(item)


BACKEND = 'torchscript_frozen_exact_forward_v1'

def graph_parity(child,envelope,freeze):
    """Every graph instance proves exact outputs before any scientific decision."""
    item=freeze['graph_parity_plan']
    c.verify([item])
    plan=c.read(c.ROOT/item['path'])
    fixtures=plan['fixtures']
    c.require(len(fixtures)==11, 'Fixed graph parity inventory')
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=child.request(payload,envelope.timeout(60))
    path=child.output_dir/'graph_parity_clock.json'
    c.save(path,receipt)
    collect_profiles(child,receipt,envelope,startup=True)
    envelope.guard()
    c.require(receipt['status']=='ok', 'Graph output parity failed before decisions; no fallback/retry')
    validate_graph_parity(receipt['result'],fixtures)
    c.require(c.read(child.output_dir/f"{receipt['request_id']}.json")==receipt,
              'Graph parity raw receipt changed')
    verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)

def validate_graph_parity(result,fixtures):
    c.require(type(result) is dict and result.get('schema')=='splendorust_inference_graph_parity_v1' and
        result.get('status')=='passed' and result.get('passed') is True and
        result.get('fixtures_count')==len(fixtures)==11 and result.get('fixture_payload_sha256')==c.digest(fixtures),
        'Exact graph parity gate failed')
    rows=result['rows']
    c.require(len(rows)==11 and len({row['fixture_id'] for row in rows})==11, 'Graph parity row inventory')
    for row,fixture in zip(rows,fixtures,strict=True):
        c.require(row['fixture_id']==fixture['fixture_id'] and row['batch']==fixture['batch'] and
            row['input_sha256']==c.digest(fixture['tokens']) and
            row['exact_float_equal'] is True and row['float32_bits_equal'] is True and
            row['eager_output_sha256']==row['candidate_output_sha256'] and
            row['eager_float32_sha256']==row['candidate_float32_sha256'],
            'Graph parity output/bit/fixture equality failed')

def exact_output_hashes(outputs,batch):
    """Independent scalar/output-byte verification; no Torch or NumPy import."""
    c.require(type(outputs) is list and len(outputs)==batch,'Graph output batch inventory')
    flat=[]
    for row in outputs:
        c.require(type(row) is list and len(row)==83 and
            all(type(value) is float and math.isfinite(value) for value in row), 'Graph output float32 inventory')
        flat.extend(row)
    bits=struct.pack('<'+'f'*len(flat),*flat)
    c.require(tuple(flat)==struct.unpack('<'+'f'*len(flat),bits),'Graph output is not exact float32')
    return c.digest(outputs),hashlib.sha256(bits).hexdigest()

def verify_graph_evidence(folder,freeze,*,boundary=lambda:None):
    from tools.external_strength_screen_20261018_02.splendorust_adapter import CHECKPOINT_SHA256
    folder=Path(folder)
    boundary()
    plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
    fixtures=plan['fixtures']
    clock=c.read(folder/'graph_parity_clock.json')
    c.require(clock['status']=='ok' and clock['receipt_elapsed_seconds']<=clock['deadline_seconds']<=60. and
        c.read(folder/f"{clock['request_id']}.json")==clock,'Graph parity trusted receipt changed')
    result=clock['result']
    validate_graph_parity(result,fixtures)
    identity=c.read(folder/'graph_identity.json')
    execution=c.read(folder/'inference_execution_identity.json')
    init=c.read(folder/'initialization.json')
    c.require(identity['schema']=='splendorust_inference_graph_identity_v1' and
        identity['backend']==execution['backend']==BACKEND and
        result['graph_identity_sha256']==execution['graph_identity_sha256']==c.digest(identity) and
        execution['metadata']==init['result'],'Graph identity/readiness/parity binding changed')
    c.require(identity['checkpoint_sha256']==CHECKPOINT_SHA256 and
        identity['weights_before_sha256']==identity['weights_after_sha256'] and
        identity['parameter_count']==4780883 and identity['dtype']=='torch.float32' and identity['device']=='cpu' and
        identity['threads']==identity['interop_threads']==1 and
        identity['supported_batches']==[1,2,8] and identity['input_shape']==['batch',31,48] and
        identity['output_shape']==['batch',83] and identity['serialized_graph_saved'] is False and
        identity['warmup_is_parity_proof'] is False, 'Exact graph settings or checkpoint changed')
    c.require(c.digest(identity)==freeze['unchanged_graph_backend_identity']['graph_identity_sha256'] and
        identity['weights_before_sha256']==freeze['unchanged_graph_backend_identity']['weights_sha256'],
        'Previously verified graph or weight identity changed')
    c.require(execution['metadata']['search_simulations']==64,'Graph32 is not part of this qualification')
    expected_options=dict(trace_batch=1,trace_check_trace=False,trace_strict=True,
        freeze_optimize_numerics=False,warm_batches=[1,2,8],
        trace_and_warm_inputs='architecture_zero_float32_only',automatic_fallback=False,alternative_candidates=0)
    c.require(identity['options']==expected_options,'Graph optimization settings changed')
    c.require(identity['numpy_version']==freeze['numpy_version'] and
        identity['torch_version']==freeze['torch_runtime_version'],'Graph numerical runtime version changed')
    c.require(identity['graph_ir_artifact']=='graph_ir.txt' and
        0<identity['graph_ir_bytes']<=256*1024 and (folder/'graph_ir.txt').stat().st_size==identity['graph_ir_bytes'] and
        c.file_hash(folder/'graph_ir.txt')==identity['graph_ir_sha256'],'Saved graph source changed')
    evidence=c.read(folder/'graph_parity.json')
    c.require(evidence['schema']=='splendorust_inference_graph_parity_v1' and
        evidence['summary']==result and evidence['fixtures']==fixtures and len(evidence['evidence'])==11,
        'Full graph parity evidence changed')
    for fixture,row,observed in zip(fixtures,result['rows'],evidence['evidence'],strict=True):
        boundary()
        c.require(observed['fixture_id']==fixture['fixture_id'] and observed['batch']==fixture['batch'] and
            observed['comparison']==row and observed['eager_outputs']==observed['candidate_outputs'],
            'Full exact graph output evidence differs')
        provenance=plan['provenance_by_fixture'][fixture['fixture_id']]
        if 'historical_outputs' in provenance:
            c.require(observed['eager_outputs']==provenance['historical_outputs'] and
                row['eager_output_sha256']==provenance['historical_output_sha256'],
                'Recorded source-root output differs from unchanged eager model')
        for key in ('eager','candidate'):
            output_hash,bits_hash=exact_output_hashes(observed[key+'_outputs'],fixture['batch'])
            c.require(output_hash==row[key+'_output_sha256'] and bits_hash==row[key+'_float32_sha256'],
                      'Independent graph output/bit hash differs')
    return dict(status='passed',worker=folder.name,fixtures=11,graph_identity_sha256=c.digest(identity),
                weights_sha256=identity['weights_before_sha256'])


def predecessor_acceptance():
    """Accept scheduled verification, preserving the incomplete optional audit."""
    names=('freeze.json','completion.json','execution.json','analysis.json','independent_statistics.json',
           'all_game_replay.json','all_graph_parity.json','all_worker_log_integrity.json',
           'repeated_decision_summary.json','independent_postrun_audit.json')
    saved={name:c.read(c.PRIOR/name) for name in names}
    finish,outer=saved['completion.json'],saved['execution.json']
    c.require(c.file_hash(c.PRIOR/'freeze.json')==c.PRIOR_FREEZE_SHA and
        finish['freeze_sha256']==outer['freeze_sha256']==c.PRIOR_FREEZE_SHA,'Exact completed benchmark freeze required')
    c.require(finish['status']=='completed_verified' and finish['failure'] is None and
        outer['status']=='completed' and outer['exit_code']==0 and not outer['cleanup_errors'] and
        outer['charged_job_seconds']<saved['freeze.json']['job_seconds'],'Scheduled benchmark verification/cleanup missing')
    c.require(saved['all_game_replay.json']['status']=='passed' and saved['all_game_replay.json']['games']==206 and
        saved['all_graph_parity.json']['status']=='passed' and saved['all_graph_parity.json']['workers']==231 and
        saved['all_worker_log_integrity.json']['status']=='passed' and saved['all_worker_log_integrity.json']['workers']==462 and
        saved['repeated_decision_summary.json']['status']=='passed' and saved['repeated_decision_summary.json']['requests']==48,
        'Scheduled benchmark verification inventory changed')
    c.require(saved['independent_statistics.json']['status']=='passed' and
        saved['analysis.json']['status']=='complete_verified_fixed_sample' and
        saved['analysis.json']['evidence_gates_passed'] is True and saved['analysis.json']['games']==200 and
        saved['analysis.json']['pairs']==100 and saved['analysis.json']['challenger_mode']=='candidate' and
        saved['analysis.json']['challenger_quota']==64 and saved['analysis.json']['opponent_mode']=='baseline' and
        saved['analysis.json']['opponent_quota']==32,'Scheduled statistical verification or player identities changed')
    # This extra audit did not complete. Preserve its status; do not promote it
    # to a prerequisite, silently repeat its full seal scan, or claim it passed.
    extra=saved['independent_postrun_audit.json']
    c.require(extra['status']=='incomplete_bounded_extra_audit' and extra['active_freeze_sha256']==c.PRIOR_FREEZE_SHA,
              'Preserved optional audit status changed')
    post_names=('freeze.json','completion.json','execution.json','finalization.json',
                'case_control_panel.json','independent_analysis_review.json','source_semantics_review.json')
    post={name:c.read(c.POSTMORTEM/name) for name in post_names}
    c.require(post['completion.json']['status']=='completed_saved_evidence_analysis' and
        post['completion.json']['benchmark_freeze_sha256']==c.PRIOR_FREEZE_SHA and
        post['completion.json']['model_calls']==0 and post['completion.json']['new_games']==0 and
        post['execution.json']['status']=='completed' and post['execution.json']['exit_code']==0 and
        post['finalization.json']['status']=='completed_saved_evidence_postmortem' and
        post['finalization.json']['benchmark_unchanged'] is True and post['finalization.json']['panel_records']==104 and
        post['independent_analysis_review.json']['status']=='passed_scoped_review',
        'Completed saved-evidence postmortem required')
    qualified=saved['freeze.json']['qualification_acceptance']
    c.require(qualified['status']=='passed' and qualified['qualified_players']=={
        'challenger':{'mode':'candidate','quota':64},'opponent':{'mode':'baseline','quota':32}},
        'Original qualified graph64/eager32 players changed')
    return dict(status='passed',benchmark_freeze_sha256=c.PRIOR_FREEZE_SHA,
        benchmark_evidence=[c.artifact(c.PRIOR/name) for name in names],
        postmortem_evidence=[c.artifact(c.POSTMORTEM/name) for name in post_names],
        qualified_player_acceptance=qualified,optional_extra_audit_status=extra['status'],
        no_full_benchmark_audit_rerun=True)


def validate_resource_forecast(value):
    expected=dict(workers=12,graph_workers=6,entrypoint_decisions=208,overhead_requests=48,diagnostic_requests=416)
    c.require(value['status']=='passed' and value['model_calls']==0 and value['fresh_games']==0 and
        value['inventories']==expected,'Complete model-free cost forecast inventory required')
    def finite(number):
        return type(number) in (int,float) and math.isfinite(number) and number>=0
    stages=value['projected_stage_seconds']
    c.require(set(stages)==set(c.STAGES) and all(finite(stages[k]) and stages[k]<v for k,v in c.STAGES.items()),
              'Forecast must fit every frozen stage')
    total=value['projected_total_seconds']; pre=value['projected_prefinal_bytes']; final=value['projected_final_bytes']
    c.require(finite(total) and sum(stages.values())<=total<c.JOB_SECONDS and
        sum(stages[k] for k in stages if k!='final')<c.COLLECTION_CUTOFF_SECONDS,
        'Forecast must fit global and collection windows')
    c.require(finite(pre) and finite(final) and pre<c.PREFINAL_EVIDENCE_BYTES and
        final<c.FINAL_RESERVE_BYTES and pre+final<c.MAX_EVIDENCE_BYTES,'Forecast exceeds frozen evidence allowance')
    return value


def preparation_dependencies(prior,accepted,plan,parity,resource,paths=()):
    inherited={item['path']:item for item in prior['inputs']}
    items={path:item for path,item in inherited.items() if not path.startswith('outputs/')}
    def bind(item):
        c.require(not Path(item['path']).is_absolute(),'Canonical repository-relative dependency required')
        c.require(item['path'] not in inherited or inherited[item['path']]==item,'Preserved input changed')
        c.require(item['path'] not in items or items[item['path']]==item,'Conflicting frozen dependency')
        items[item['path']]=item
    for document in (accepted,plan,parity,resource,frozen_q.build_plan(32),frozen_q.build_plan(64),
                     c.read(frozen_q.PARITY_RECEIPT)):
        for item in nested_artifacts(document): bind(item)
    bind(c.artifact(frozen_q.PARITY_RECEIPT))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in prior['native_artifacts']: bind(item)
    for path in paths:
        if '__pycache__' not in path.parts: bind(c.artifact(path))
    return sorted(items.values(),key=lambda item:item['path'])


def prepare():
    accepted=predecessor_acceptance()
    c.require(not (c.OUT/'freeze.json').exists(),'No replacement freeze')
    for name in ('prelaunch_tests.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name)
        c.require(receipt['status']=='passed','Prelaunch checks/review required')
        c.verify(receipt['source_inventory'])
    resource=validate_resource_forecast(c.read(c.OUT/'preparation_resource_estimate.json'))
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'Launch requires7GiB free disk')
    sources=verify_runtime_sources(); prior=c.read(c.PRIOR/'freeze.json')
    plan=plan_rules.build_plan(); parity=c.read(c.QUALIFIED/'graph_parity_plan.json')
    c.save(c.OUT/'qualification_plan.json',plan)
    c.save(c.OUT/'graph_parity_plan.json',parity)
    c.save(c.OUT/'baseline_acceptance.json',accepted)
    paths=set((c.ROOT/'tools/splendorust_graph_timing_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_graph_timing*.py'))
    paths.update(path for path in c.OUT.rglob('*') if path.is_file())
    paths.add(c.PROTOCOL)
    inputs=preparation_dependencies(prior,accepted,plan,parity,resource,paths)
    copied=('python','python_executable_sha256','torch_runtime','external_runtime','sampler_api_modules',
            'compression_runtime','topology','native_artifacts','numpy_version','torch_package_version',
            'torch_runtime_version','unchanged_graph_backend_identity')
    freeze=dict(schema='splendorust_current_graph_timing_execution_v1',time=c.now(),inputs=inputs,
        **{key:prior[key] for key in copied},baseline_acceptance=accepted,source_guard=sources,
        graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0,seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_is_diagnostic_only=True,overhead_requests=48,diagnostic_requests_maximum=416,
        actual_entrypoint_workers=4,actual_entrypoint_decisions=208,maximum_workers=12,maximum_graph_workers=6,
        modes=list(MODES),quotas=[64,32],max_rss_bytes=2*c.GiB,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,fresh_games=0,strength_games=0,larger_data_deferred=True,
        scientific_backends={'64':'tools.splendorust_inference_opt_20261025.worker',
                             '32':'tools.splendorust_search_scale_20261024.worker'},
        profile_child_module='tools.splendorust_graph_timing_20261025.worker',sidecar_wait_maximum_seconds=1.)
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical(dict(prepared=True,model_calls=0,inputs=len(inputs),freeze_sha256=c.file_hash(c.OUT/'freeze.json'))),flush=True)


def make_worker(name,envelope,freeze,*,mode,quota):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    c.require(mode in MODES and type(quota) is int and quota in (64,32) and 'torch' not in sys.modules,
              'Fixed mode/quota and model-free supervisor required')
    envelope.admit(); folder=c.OUT/'workers'/name; envelope.register_worker(folder)
    cls=c.BaselineStudyWorker if mode=='baseline' else c.ProfiledStudyWorker
    original='tools.splendorust_inference_opt_20261025.worker' if quota==64 else 'tools.splendorust_search_scale_20261024.worker'
    module=original if mode=='baseline' else 'tools.splendorust_graph_timing_20261025.worker'
    child=cls(module,folder,native_artifacts=freeze['native_artifacts'],aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'STUDY_COMPONENT_PROFILE':'1' if mode=='profiled' else '0',
             'CUDA_VISIBLE_DEVICES':'','RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child); child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample(); metadata=deepcopy(child.initialization['result'])
        if quota==64: c.require(metadata.pop('inference_backend')==BACKEND,'Original graph64 backend required')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,worker_pid=child.process.pid,
            process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota=mode,quota
        c.require(not child._failure,'Worker readiness supervision failed')
        if mode=='profiled':
            verify_instrumentation_identity(folder,freeze,quota,child.initialization['result'])
        if quota==64: graph_parity(child,envelope,freeze)
        envelope.guard(); return child
    except BaseException:
        child.close(); raise


def verify_instrumentation_identity(folder,freeze,quota,metadata):
    identity=c.read(Path(folder)/'component_timing_identity.json')
    c.require(identity['schema']=='splendorust_graph_timing_child_v1' and identity['quota']==quota and
        identity['backend']==('qualified_graph64' if quota==64 else 'unchanged_eager32') and
        identity['metadata']==metadata and identity['scientific_response_fields_unchanged'] is True and
        identity['clock']=='perf_counter_ns' and identity['graph_identity_sha256']==(
            freeze['unchanged_graph_backend_identity']['graph_identity_sha256'] if quota==64 else None),
        'Current-backend instrumentation identity changed')
    return identity


def collect_profiles(child,receipt,envelope,*,startup=False):
    """Out-of-clock evidence wait only; never restart/retry a player request."""
    if child.runtime_mode=='baseline': return dict(parent_profile=None,parent_profile_file=None,child_profile=None,child_profile_file=None)
    profile=deepcopy(child.last_parent_profile)
    c.require(profile['request_id']==receipt['request_id'],'Parent profile request identity')
    parent=child.output_dir/'parent_profiles'/f"{receipt['request_id']}.json"
    c.save(parent,profile)
    path=child.output_dir/f"{receipt['request_id']}_child_profile.json"
    begun=time.monotonic(); wait_end=begun+min(1.,max(0.,envelope.remaining()))
    while not path.exists() and not getattr(child,'_closed',False) and time.monotonic()<wait_end:
        envelope.guard(); time.sleep(min(.005,max(0.,wait_end-time.monotonic())))
    result=dict(parent_profile=profile,parent_profile_file=parent.relative_to(c.ROOT).as_posix(),
        child_profile=None,child_profile_file=None)
    if path.exists():
        result.update(child_profile=c.read(path),child_profile_file=path.relative_to(c.ROOT).as_posix())
    elif receipt['status']=='ok':
        raise ValueError('Completed profiled request missing bounded child sidecar')
    return result


def actual_entrypoint(child,envelope,label):
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=child.request(payload,envelope.timeout(300))
    c.save(c.OUT/'entrypoint_checks'/f'{label}.json',receipt)
    collect_profiles(child,receipt,envelope,startup=True)
    envelope.guard()
    c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed' and
        receipt['result']['challenger_quota']==child.challenger_quota and receipt['result']['counts']['complete_decisions']==52,
        'Actual corrected-entrypoint counterexamples failed; no retry')


def qualify(envelope,freeze):
    labels=[]
    for quota in (64,32):
        for mode in MODES:
            label=f'{quota}_{mode}'
            with make_worker(f'qualification_{label}',envelope,freeze,mode=mode,quota=quota) as child:
                actual_entrypoint(child,envelope,label)
            labels.append(label)
    result=dict(status='passed',workers=4,complete_decisions=208,labels=labels)
    c.save(c.OUT/'qualification_summary.json',result);return result


def decision(child,state,envelope):
    envelope.admit(); observer,before=rules.observation_payload(state),rules.serialize_snapshot(state)
    ledger.check_observer(before,observer); legal=rules.legal_actions(state)
    actions=[serialize_action(action) for action in legal]; request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',dict(player='search',observer=observer,before=before))
    def validate(result):
        c.require(type(result) is dict and set(result)==audit.RESULT_FIELDS and result.get('action') in actions,
                  'Exact unchanged scientific response required')
        return audit.validate_result(result,observer,'search',expected_quota=child.challenger_quota)
    c.require(envelope.remaining()>=1.,'Insufficient full move allowance')
    allowance=envelope.timeout(1.); c.require(allowance==1.,'No stage-clamped move dispatch')
    receipt=child.request({'player':'search','observer':observer},allowance,validate=validate)
    row=dict(arm='search',mode=child.runtime_mode,quota=child.challenger_quota,seat=state.current_player,
        before=before,observer=observer,after=None,action=None,trace=None,completed=receipt['status']=='ok',clock=receipt,
        parent_profile=None,parent_profile_file=None,child_profile=None,child_profile_file=None,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    try: row.update(collect_profiles(child,receipt,envelope))
    except BaseException as error: row['profile_capture_error']=repr(error)
    if row['completed']:
        result=receipt['result']
        c.require(result['observer_sha256']==c.digest(observer) and result['legal_actions_sha256']==legal_digest(legal) and
            result['legal_count']==len(legal),'Scientific observer/legal binding changed')
        row.update(action=result['action'],trace=result['diagnostics'])
        row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,deserialize_action(row['action'])))
    return row


def run_panel(envelope,freeze,plan,stage):
    schedule=plan[stage+'_schedule']; expected=48 if stage=='overhead' else 416
    c.require(len(schedule)==expected and [row['index'] for row in schedule]==list(range(expected)),
              'Fixed complete timing schedule required')
    rows=[]
    for quota in (64,32):
        with ExitStack() as stack:
            workers={mode:stack.enter_context(make_worker(f'{stage}_{quota}_{mode}',envelope,freeze,mode=mode,quota=quota)) for mode in MODES}
            for spec in (item for item in schedule if item['quota']==quota):
                row={**decision(workers[spec['mode']],decode_training_snapshot(plan_rules.snapshot_for_request(plan,spec)),envelope),**spec}
                c.save(c.OUT/stage/f"{spec['index']:04d}.json",row)
                envelope.guard()
                c.require(row['completed'],'Hard move failure preserved; no retry')
                c.require('profile_capture_error' not in row,'Timing evidence failure preserved; no retry')
                audit.check_clock(row['clock'])
                if len(rows)%24==0: print(c.canonical(dict(stage=stage,requests=len(rows)+1,total=expected)),flush=True)
                rows.append(row)
    summary=plan_rules.validate_panel(rows,plan,stage=stage,boundary=envelope.guard)
    c.save(c.OUT/f'{stage}_summary.json',summary); return summary


def verify_saved_row(row):
    path=c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()),'Unowned trusted receipt')
    c.require(c.read(path)==row['clock'] and path.name==row['clock']['request_id']+'.json','Raw trusted receipt changed')
    c.require(c.read(path.parent/'inputs'/path.name)==dict(player='search',observer=row['observer'],before=row['before']),
              'Saved input changed')
    if row['mode']=='profiled':
        parent=c.ROOT/row['parent_profile_file']; child=c.ROOT/row['child_profile_file']
        c.require(parent.resolve().is_relative_to((path.parent/'parent_profiles').resolve()) and
            parent.name==path.name and c.read(parent)==row['parent_profile'],'Parent timing sidecar changed')
        c.require(child.parent.resolve()==path.parent.resolve() and child.name==row['clock']['request_id']+'_child_profile.json' and
            c.read(child)==row['child_profile'],'Child timing sidecar changed')
    else:
        c.require(all(row.get(key) is None for key in ('parent_profile','parent_profile_file','child_profile','child_profile_file')),
                  'Baseline instrumentation unexpectedly present')


def verify_raw_transport(folder,freeze,envelope):
    """Bind actual closed response bytes and every startup/ordinary receipt.

    One extra bounded streaming decode preserves JSON byte order. Normalize
    only the terminal Windows CRLF expansion when comparing child pre-write
    UTF-8 evidence. No model calls and no expanded log is written.
    """
    folder=Path(folder);cleanup=c.read(folder/'cleanup.json')
    init=c.read(folder/'initialization.json');metadata=init['result']
    profiled=folder.name.endswith('_profiled');quota=metadata['search_simulations']
    expected={path.stem for path in folder.glob('request-*.json') if re.fullmatch(r'request-[0-9]{8}',path.stem)}
    seen=set();pending=bytearray();ready=False;count=0;startup_count=0
    def packet_line(raw):
        nonlocal ready,count,startup_count
        envelope.guard()
        c.require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Raw packet frame bound')
        packet=_decode(raw)
        if not ready:
            c.require(packet=={'type':'ready','metadata':metadata},'Exact raw READY metadata changed')
            ready=True;return
        c.require(set(packet)=={'type','request_id','result'} and packet['type']=='response',
                  'Completed diagnostic requires exact ordinary response packets')
        request_id=packet['request_id']
        c.require(request_id in expected and request_id not in seen,'Raw response inventory/request identity changed')
        receipt=c.read(folder/(request_id+'.json'))
        c.require(receipt['request_id']==request_id and receipt['status']=='ok' and receipt['result']==packet['result'],
                  'Raw response differs from original trusted receipt')
        startup=folder/'startup_inputs'/(request_id+'.json')
        ordinary=folder/'inputs'/(request_id+'.json')
        c.require(startup.is_file()!=ordinary.is_file(),'Exactly one saved payload type required')
        if startup.is_file():
            payload=c.read(startup);startup_count+=1
            if payload=={'operation':'entrypoint_checks'}:
                c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during timing')
            else:
                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
                c.require(quota==64 and payload=={'operation':'graph_parity','fixtures':plan['fixtures']},
                          'Exact original graph startup payload required')
        else:
            saved=c.read(ordinary);payload={'player':saved['player'],'observer':saved['observer']}
        if profiled:
            child=c.read(folder/(request_id+'_child_profile.json'))
            parent=c.read(folder/'parent_profiles'/(request_id+'.json'))
            normalized=raw[:-2]+b'\n' if raw.endswith(b'\r\n') else raw
            c.require(child['serialized_packet_utf8_sha256']==hashlib.sha256(normalized).hexdigest() and
                child['serialized_packet_utf8_bytes']==len(normalized),'Child serialized packet differs from exact raw stdout')
            c.require(child['request_id']==parent['request_id']==request_id and
                child['payload_sha256']==c.digest(payload) and child['result_sha256']==c.digest(packet['result']),
                'Child input/result sidecar binding changed')
            c.require(parent['request_raw_sha256']==child['request_raw_sha256'] and
                type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=c.MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'Actual parent written bytes differ from child raw request')
            if startup.is_file():
                analyze.validate_startup_profiles(parent,child,receipt,payload,quota,worker_pid=init['identity']['pid'])
        else:
            c.require(not (folder/(request_id+'_child_profile.json')).exists() and
                not (folder/'parent_profiles'/(request_id+'.json')).exists(),'Baseline contains timing sidecars')
        seen.add(request_id);count+=1
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1];packet_line(raw)
        c.require(len(pending)<=4*c.MiB,'Raw pending response frame bound')
        envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=16*c.MiB,max_compressed_bytes=16*c.MiB*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready and not pending and seen==expected,'Complete raw response/receipt inventory required')
    expected_startups=(1 if quota==64 else 0)+(1 if folder.name.startswith('qualification_') else 0)
    c.require(startup_count==expected_startups,'Startup request/sidecar inventory differs')
    return dict(status='passed',worker=folder.name,responses=count,startup_requests=startup_count,
        profiled=profiled,quota=quota,stdout_log=verified,exact_bytes_checked=True,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized')


def final_checks(envelope,freeze,plan,summaries):
    entries=[]
    for quota in (64,32):
        for mode in MODES:
            label=f'{quota}_{mode}'; receipt=c.read(c.OUT/'entrypoint_checks'/f'{label}.json')
            folder=c.OUT/'workers'/f'qualification_{label}'
            c.require(c.read(folder/f"{receipt['request_id']}.json")==receipt and receipt['status']=='ok' and
                receipt['result']['status']=='entrypoint_counterexamples_passed' and
                receipt['result']['challenger_quota']==quota and receipt['result']['counts']['complete_decisions']==52,
                'Actual entrypoint receipt changed')
            entries.append(label)
    c.save(c.OUT/'all_entrypoint_checks.json',dict(status='passed',workers=4,complete_decisions=208,labels=entries))
    collected=[]
    for stage,summary in summaries.items():
        paths=sorted((c.OUT/stage).glob('*.json')); expected=48 if stage=='overhead' else 416
        c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(expected)],'Timing inventory differs')
        rows=[c.read(path) for path in paths]
        for row in rows: envelope.guard();verify_saved_row(row)
        c.require(plan_rules.validate_panel(rows,plan,stage=stage,boundary=envelope.guard)==summary,'Independent panel reread differs')
        collected.extend(rows)
    logs=[]; graph=[];transport=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard();logs.append(verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline()))
        transport.append(verify_raw_transport(folder,freeze,envelope))
        metadata=c.read(folder/'initialization.json')['result']
        if folder.name.endswith('_profiled'):
            verify_instrumentation_identity(folder,freeze,metadata['search_simulations'],metadata)
        else:
            c.require(not (folder/'component_timing_identity.json').exists(),'Baseline instrumentation identity unexpectedly present')
        if metadata['search_simulations']==64: graph.append(verify_graph_evidence(folder,freeze,boundary=envelope.guard))
        else:
            c.require(metadata['search_simulations']==32 and 'inference_backend' not in metadata and
                not any(folder.glob('graph_*.json')),'Eager32 backend changed')
    expected_workers=4+4*len(summaries)
    c.require(len(logs)==expected_workers and len(graph)==expected_workers//2,'Fixed worker/graph inventory differs')
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(logs),receipts=logs))
    c.save(c.OUT/'all_graph_parity.json',dict(status='passed',workers=len(graph),receipts=graph))
    c.save(c.OUT/'all_raw_transport.json',dict(status='passed',workers=len(transport),receipts=transport))
    c.save(c.OUT/'component_analysis.json',analyze.summarize(collected,boundary=envelope.guard))
    seal=[]
    for directory in ('entrypoint_checks','overhead','diagnostic','workers'):
        for path in sorted((c.OUT/directory).rglob('*')):
            if path.is_file(): envelope.guard();seal.append(c.artifact(path))
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=seal))
    for item in seal: envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    c.require(predecessor_acceptance()==freeze['baseline_acceptance'],'Preserved earlier evidence changed')
    c.save(c.OUT/'verification_summary.json',dict(status='passed',requests=len(collected),workers=len(logs),graph_workers=len(graph),
        maximum_sampled_combined_worker_rss_bytes=envelope.peak_combined_rss,science_and_transitions_verified=True))


def run_stages(envelope,freeze,plan,*,qualification_started=False):
    if qualification_started: c.require(envelope.stage=='qualification','Startup qualification clock missing')
    else: envelope.begin('qualification')
    qualify(envelope,freeze); envelope.finish()
    envelope.begin('overhead');summaries={'overhead':run_panel(envelope,freeze,plan,'overhead')};envelope.finish()
    c.require(type(summaries['overhead']['qualified']) is bool,'Overhead admission must be Boolean')
    if summaries['overhead']['qualified']:
        envelope.begin('diagnostic');summaries['diagnostic']=run_panel(envelope,freeze,plan,'diagnostic');envelope.finish()
    envelope.begin('final');final_checks(envelope,freeze,plan,summaries);envelope.finish()
    c.save(c.OUT/'diagnostic_outcome.json',dict(status='completed_verified',instrumentation_qualified=summaries['overhead']['qualified'],
        diagnostic_completed='diagnostic' in summaries,summaries=summaries,fresh_games=0,strength_games=0,larger_data_deferred=True))
    return summaries


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Execution freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json');c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze changed')
    freeze,envelope=c.read(c.OUT/'freeze.json'),Envelope(claim);envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure='incomplete',None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope);envelope.admit()
        run_stages(envelope,freeze,c.read(c.OUT/'qualification_plan.json'),qualification_started=True)
        status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure,
            preserved_partial_evidence=True,partial_evidence_not_accepted=True))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,fresh_games=0,strength_games=0,larger_data_deferred=True,
        note='Acceptance additionally requires owning watchdog cleanup; diagnostic measurements never alter trusted clocks.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    return prepare() if args.operation=='prepare' else run(args.freeze_sha256)


if __name__=='__main__': raise SystemExit(main())



