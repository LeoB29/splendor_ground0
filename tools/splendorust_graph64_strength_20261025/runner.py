"""Fresh bounded qualified graph64 versus unchanged eager32 strength comparison.

Preparation and launch are gated on the separate optimization's fully verified
64-simulation acceptance and successful owning-watchdog cleanup.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
import hashlib
import struct
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
from . import common as c, audit, statistics, schedule as schedules
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit as scientific, qualification as original_qualification
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_timing_opt_20261024.sampler import snapshot_api_modules

BACKEND = 'torchscript_frozen_exact_forward_v1'


def require_player(mode,quota):
    c.require(type(quota) is int and ((mode=='baseline' and quota==32) or
        (mode=='candidate' and quota==64)), 'Only unchanged eager32 and qualified graph64 are authorized')

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

def unlaunched_preparation_exemption(path,schedule):
    """Only the explicitly preserved, never-frozen preparation may reuse seeds."""
    folder=c.ROOT/'outputs/splendorust_inference64_20261025_01'
    if Path(path).resolve()!=(folder/'operational_schedule.json').resolve():
        return None
    failure_path=folder/'preparation_failure.json'
    c.require(failure_path.is_file(),'Same-seed preparation lacks a preserved no-launch failure record')
    failure=c.read(failure_path)
    c.require(failure.get('status')=='preparation_failed_before_freeze' and
        type(failure.get('model_calls')) is int and failure['model_calls']==0 and
        type(failure.get('games')) is int and failure['games']==0,
        'Seed exemption requires a never-launched preparation with zero model calls/games')
    forbidden=('freeze.json','launch.json','watchdog.json','watchdog_claim.json','execution.json',
               'completion.json','workers','stages')
    c.require(not any((folder/name).exists() for name in forbidden),
              'Seed exemption forbidden after freeze, launch, execution or worker activity')
    c.require(c.read(path)==schedule,'Preserved preparation schedule differs from unchanged operational seeds')
    return dict(reason='Exact unchanged schedule saved by a preserved preparation that never froze or launched',
        schedule=c.artifact(path),failure=c.artifact(failure_path),
        required_absent=list(forbidden),model_calls=0,games=0)

def graph_parity(child,envelope,freeze):
    """Every graph instance proves exact outputs before any scientific decision."""
    item=freeze['graph_parity_plan']
    c.verify([item])
    plan=c.read(c.ROOT/item['path'])
    fixtures=plan['fixtures']
    c.require(len(fixtures)==11, 'Fixed graph parity inventory')
    receipt=child.request(dict(operation='graph_parity',fixtures=fixtures),envelope.timeout(60))
    path=child.output_dir/'graph_parity_clock.json'
    c.save(path,receipt)
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
    from importlib.metadata import version
    for receipt in freeze['exempt_preparation']:
        c.require(unlaunched_preparation_exemption(c.ROOT/receipt['schedule']['path'],
            c.read(c.PRIOR/'operational_schedule.json'))==receipt,'Preserved unlaunched preparation changed')
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
    c.require(statistics.runtime_contract() == freeze['statistics_runtime'], 'Statistics runtime changed')
    c.require(version('numpy')==freeze['numpy_version'] and version('torch')==freeze['torch_package_version'],
              'Pinned numerical package version changed')


def qualification_acceptance(*, verify_inputs=True):
    """Only the exact completed graph64/eager32 qualification admits collection."""
    names=('freeze.json','completion.json','execution.json','qualification_outcome.json',
        'qualification_summary.json','operational_summary.json','verification_summary.json',
        'all_entrypoint_checks.json','all_graph_parity.json','all_game_replay.json',
        'repeated_decision_summary.json','all_worker_log_integrity.json','graph_parity_plan.json',
        'operational_schedule.json','baseline_acceptance.json','independent_postrun_audit.json','completed_evidence_seal.json')
    saved={name:c.read(c.PRIOR/name) for name in names}
    complete,execution,outcome=(saved[name] for name in ('completion.json','execution.json','qualification_outcome.json'))
    sha=c.file_hash(c.PRIOR/'freeze.json')
    c.require(sha==c.PRIOR_FREEZE_SHA and complete['freeze_sha256']==execution['freeze_sha256']==sha,
              'Exact accepted qualification freeze identity changed')
    c.require(complete['status']=='completed_verified' and complete['failure'] is None and
        execution['status']=='completed' and execution['exit_code']==0 and not execution['cleanup_errors'] and
        execution['charged_job_seconds']<saved['freeze.json']['job_seconds'],
        'Qualification must finish verified with successful owning-watchdog cleanup')
    c.require(outcome['status']=='completed_verified' and outcome['inference64_qualified'] is True and
        outcome['saved_panel_qualified'] is True and outcome['strength_games_launched']==0 and
        saved['qualification_summary.json']['qualified'] is True and saved['operational_summary.json']['qualified'] is True,
        'Both saved64 timing panels and graph64/eager32 operational gates must pass')
    for name in ('verification_summary.json','all_entrypoint_checks.json','all_graph_parity.json',
                 'all_game_replay.json','repeated_decision_summary.json','all_worker_log_integrity.json'):
        c.require(saved[name]['status']=='passed','Incomplete accepted qualification evidence: '+name)
    c.require(saved['all_entrypoint_checks.json']['workers']==3 and saved['all_graph_parity.json']['workers']==14 and
        saved['all_game_replay.json']['games']==6 and saved['repeated_decision_summary.json']['requests']==12 and
        saved['all_worker_log_integrity.json']['workers']==29 and saved['operational_summary.json']['games']==6 and
        saved['verification_summary.json']['operational_games']==6 and
        saved['verification_summary.json']['operational_reproductions']==12,
        'Accepted qualification verification inventory changed')
    backend=saved['freeze.json']['unchanged_graph_backend_identity']
    c.require(len(saved['all_graph_parity.json']['receipts'])==14 and all(
        row['status']=='passed' and row['fixtures']==11 and row['graph_identity_sha256']==backend['graph_identity_sha256'] and
        row['weights_sha256']==backend['weights_sha256'] for row in saved['all_graph_parity.json']['receipts']),
        'Accepted graph/weight/parity instance inventory changed')
    independent=saved['independent_postrun_audit.json']
    c.require(independent['status']=='passed' and independent['freeze_sha256']==sha and
        type(independent['model_calls']) is int and independent['model_calls']==0 and independent['fresh_games']==0 and
        independent['evidence_modified'] is False and independent['exact_float_and_float32_bit_pairs']==154 and
        independent['closed_workers']==29 and independent['graph_workers']==14 and
        independent['frozen_source_artifacts']==len(saved['freeze.json']['inputs']) and
        independent['completed_seal_artifacts']==len(saved['completed_evidence_seal.json']['artifacts']),
        'Independent accepted qualification audit or evidence inventory changed')
    if verify_inputs:
        c.verify(saved['freeze.json']['inputs'])
        c.verify(saved['completed_evidence_seal.json']['artifacts'])
        c.verify(independent['audit_sources'])
    return dict(status='passed',qualification_freeze_sha256=sha,
        evidence=[c.artifact(c.PRIOR/name) for name in names],independent_audit_sources=independent['audit_sources'],
        unchanged_graph_backend_identity=backend,
        qualified_players={'challenger':{'mode':'candidate','quota':64},'opponent':{'mode':'baseline','quota':32}},
        saved_timing_rerun_required=False,actual_entrypoint_requalification_required=True)


def seed_inventory(schedule):
    targets = {int(row['seed_decimal']) for row in schedule}
    listed = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
        '-g', '*schedule*.json', '-g', '*seed_registry*.json'], cwd=c.ROOT,
        capture_output=True, text=True, check=True)
    checked, skipped, collisions = [], [], []
    def visit(value, path):
        if type(value) is int or isinstance(value, str) and value.isdecimal():
            if int(value) in targets: collisions.append(path)
        elif isinstance(value, list):
            for item in value: visit(item, path)
        elif isinstance(value, dict):
            for item in value.values(): visit(item, path)
    for name in sorted(listed.stdout.splitlines()):
        path = (c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()): continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(c.artifact(path)); continue
        visit(c.read(path), path.relative_to(c.ROOT).as_posix())
        checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh schedule overlaps registered prior evidence')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, collisions=collisions,
                scope='registered JSON names up to8MiB; not exhaustive logs')


def preparation_dependencies(prior,accepted,parity_plan,resource_estimate,paths=()):
    """Exact finite input binding shared by preparation and model-free tests."""
    inherited={item['path']:item for item in prior['inputs']}
    items={path:item for path,item in inherited.items() if not path.startswith('outputs/')}
    def bind(item):
        c.require(not Path(item['path']).is_absolute(),'Canonical repository-relative dependency required')
        if item['path'] in inherited:
            c.require(inherited[item['path']]==item,'Preserved accepted dependency changed')
        if item['path'] in items:
            c.require(items[item['path']]==item,'Conflicting frozen dependency')
        items[item['path']]=item
    for document in (accepted,parity_plan,resource_estimate,prior['exempt_preparation'],
        c.read(c.PRIOR/'baseline_acceptance.json'),original_qualification.build_plan(32),
        original_qualification.build_plan(64),c.read(original_qualification.PARITY_RECEIPT)):
        for item in nested_artifacts(document): bind(item)
    bind(c.artifact(original_qualification.PARITY_RECEIPT))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in prior['native_artifacts']: bind(item)
    for path in paths:
        if '__pycache__' not in path.parts:
            bind(c.artifact(path))
    return sorted(items.values(),key=lambda item:item['path'])


def prepare():
    # This is intentionally the first dependency gate: active/incomplete/failed
    # qualification cannot cause a comparison freeze or any model invocation.
    accepted = qualification_acceptance()
    c.require(not (c.OUT/'freeze.json').exists(), 'Already frozen; no replacement')
    c.OUT.mkdir(parents=True, exist_ok=True)
    sources = verify_runtime_sources()
    for name in ('prelaunch_tests.json', 'prelaunch_review.json'):
        receipt = c.read(c.OUT/name)
        c.require(receipt['status'] == 'passed', 'Prelaunch tests/review required')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, 'Launch requires7GiB free disk')
    prior = c.read(c.PRIOR/'freeze.json')
    resource_estimate=c.read(c.OUT/'preparation_resource_estimate.json')
    c.require(resource_estimate['status']=='passed' and resource_estimate['model_calls']==0 and
        resource_estimate['fresh_games']==0 and resource_estimate['outcomes_used_for_selection_or_resource_estimation'] is False and
        resource_estimate['source_freeze_sha256']==c.PRIOR_FREEZE_SHA, 'Bound outcome-blind preparation cost evidence required')
    schedule = schedules.make_schedule()
    c.save(c.OUT/'qualification_acceptance.json', accepted)
    c.save(c.OUT/'seed_inventory.json', seed_inventory(schedule))
    c.save(c.OUT/'schedule.json', schedule)
    parity_plan=c.read(c.PRIOR/'graph_parity_plan.json')
    c.save(c.OUT/'graph_parity_plan.json',parity_plan)
    c.save(c.OUT/'model_free_qualification.json', original_qualification.model_free_checks(64))
    paths = set((c.ROOT/'tools/splendorust_graph64_strength_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_graph64_strength*.py'))
    paths.add(c.PROTOCOL)
    paths.update(path for path in c.OUT.rglob('*') if path.is_file())
    inputs=preparation_dependencies(prior,accepted,parity_plan,resource_estimate,paths)
    freeze = dict(schema='splendorust_graph64_vs_eager32_strength_v1', time=c.now(),
        inputs=inputs,
        qualification_acceptance=accepted, python=prior['python'],
        python_executable_sha256=prior['python_executable_sha256'], torch_runtime=prior['torch_runtime'],
        numpy_version=prior['numpy_version'],torch_package_version=prior['torch_package_version'],
        torch_runtime_version=prior['torch_runtime_version'],exempt_preparation=prior['exempt_preparation'],
        graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        unchanged_graph_backend_identity=prior['unchanged_graph_backend_identity'],
        external_runtime=prior['external_runtime'], sampler_api_modules=prior['sampler_api_modules'],
        compression_runtime=prior['compression_runtime'], statistics_runtime=statistics.runtime_contract(),
        topology=prior['topology'], native_artifacts=prior['native_artifacts'], source_guard=sources,
        job_seconds=c.JOB_SECONDS, stage_seconds=c.STAGES, collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0, seconds_per_move=1., internal_seconds=.9, operational_margin_seconds=.7,
        challenger_quota=64, opponent_quota=32, operational_games=6, main_games=200, pairs=100,
        parent_class='tools.splendorust_timing_opt_20261024.common.CandidateStudyWorker',
        baseline_child_module='tools.splendorust_search_scale_20261024.worker',
        candidate_child_module='tools.splendorust_inference_opt_20261025.worker',
        player_modes={'challenger':'candidate','opponent':'baseline'},graph_parity_cases_per_candidate=11,
        profiles={role:c.artifact(c.profile_path(quota)) for role,quota in (('challenger',64),('opponent',32))},
        max_rss_bytes=2*c.GiB, max_evidence_bytes=c.MAX_EVIDENCE_BYTES, final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        final_reproductions_maximum=48,maximum_workers=462,maximum_graph_workers=231,resource_forecast_factor=1.35,
        larger_data_deferred=True, reference_promotion=False,
        failure_policy='unchanged established classifier; attributable player failures lose; infrastructure/scientific failures stop',
        authorization='Fresh paired200-game qualified graph64 versus unchanged eager32 strength comparison')
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, model_calls=0, inputs=len(inputs),
        freeze_sha256=c.file_hash(c.OUT/'freeze.json'))), flush=True)


def make_worker(name,envelope,freeze,*,mode,quota):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    require_player(mode,quota)
    c.require('torch' not in sys.modules,'Model-free supervisor required')
    envelope.admit()
    folder=c.OUT/'workers'/name
    envelope.register_worker(folder)
    cls=c.BaselineStudyWorker if mode=='baseline' else c.CandidateStudyWorker
    module='tools.splendorust_search_scale_20261024.worker' if mode=='baseline' else 'tools.splendorust_inference_opt_20261025.worker'
    child=cls(module,folder,
        native_artifacts=freeze['native_artifacts'],aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'CUDA_VISIBLE_DEVICES':'',
             'RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child)
        child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample()
        metadata=deepcopy(child.initialization['result'])
        if mode=='candidate':
            c.require(metadata.pop('inference_backend')==BACKEND,'Candidate inference backend changed')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,
            worker_pid=child.process.pid,process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota=mode,quota
        c.require(not child._failure,'Worker supervision failed at readiness')
        if mode=='candidate':
            graph_parity(child,envelope,freeze)
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise

def decision(child, state, envelope, role):
    c.require(role in ('challenger','opponent') and child.challenger_quota == (64 if role == 'challenger' else 32) and
              child.runtime_mode==('candidate' if role=='challenger' else 'baseline'),
              'Role/quota binding differs')
    require_player(child.runtime_mode,child.challenger_quota)
    envelope.admit()
    observer, before = rules.observation_payload(state), rules.serialize_snapshot(state)
    ledger.check_observer(before, observer)
    legal = rules.legal_actions(state)
    actions = [serialize_action(action) for action in legal]
    request_id = f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json', dict(player='search', observer=observer, before=before))
    def validate(result):
        if not isinstance(result,dict) or set(result) != scientific.RESULT_FIELDS or not isinstance(result['diagnostics'],dict):
            raise ValueError('PLAYER_MALFORMED_RESPONSE: response envelope fields')
        if result['action'] not in actions:
            raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return scientific.validate_result(result, observer, 'search', expected_quota=child.challenger_quota)
    c.require(envelope.remaining() >= 1., 'Insufficient stage time for a full one-second decision')
    allowance = envelope.timeout(1.)
    c.require(allowance == 1., 'Stage guard shortened the next move allowance; do not dispatch')
    receipt = child.request({'player':'search','observer':observer}, allowance, validate=validate)
    row = dict(arm='search',player_role=role,quota=child.challenger_quota,mode=child.runtime_mode,seat=state.current_player,
        observer=observer,before=before,after=None,action=None,trace=None,completed=receipt['status']=='ok',clock=receipt,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    if row['completed']:
        result = receipt['result']
        c.require(result['player']=='search' and result['observer_sha256']==c.digest(observer) and
            result['legal_actions_sha256']==legal_digest(legal) and result['legal_count']==len(legal),
            'Worker observation/legal-support binding differs')
        row.update(action=result['action'],trace=result['diagnostics'])
        row['after'] = rules.serialize_snapshot(rules.CorrectedEnv().step(state, deserialize_action(row['action'])))
    return row


def qualify(envelope,freeze):
    c.require(qualification_acceptance(verify_inputs=False)==freeze['qualification_acceptance'],
              'Qualification acceptance changed since comparison freeze')
    costs=[]
    for quota in (32,64):
        mode='candidate' if quota==64 else 'baseline'
        startup_begun=time.perf_counter()
        with make_worker(f'qualification_{quota}',envelope,freeze,mode=mode,quota=quota) as child:
            startup_seconds=time.perf_counter()-startup_begun
            begun=time.perf_counter()
            receipt=child.request({'operation':'entrypoint_checks'},envelope.timeout(300))
            c.save(c.OUT/f'entrypoint_checks_{quota}.json',receipt)
            envelope.guard()
            c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed',
                'Actual corrected-entrypoint counterexamples failed; no retry')
            c.require(receipt['result']['challenger_quota']==quota and
                      receipt['result']['counts']['complete_decisions']==52,'Actual entrypoint inventory differs')
            check_seconds=time.perf_counter()-begun
        begun=time.perf_counter()
        checked=verify_worker_logs(child.output_dir,boundary=envelope.guard,deadline=envelope.effective_deadline())
        log_seconds=time.perf_counter()-begun
        c.save(c.OUT/f'qualification_log_integrity_{quota}.json',checked)
        graph_seconds=0.
        if mode=='candidate':
            begun=time.perf_counter()
            graph_checked=verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
            graph_seconds=time.perf_counter()-begun
            c.save(c.OUT/'qualification_graph_integrity.json',graph_checked)
        hashing=measure_seal_cost([child.output_dir],envelope.guard)
        costs.append(dict(quota=quota,mode=mode,initialization_seconds=child.initialization['elapsed_seconds'],
                          complete_startup_and_parity_seconds=startup_seconds,
                          counterexample_seconds=check_seconds,log_seconds=log_seconds,
                          graph_proof_reread_seconds=graph_seconds,closed_worker_two_pass_seal_seconds=hashing['seconds']))
    begun=time.perf_counter()
    verify_runtime(freeze,envelope)
    source_seconds=time.perf_counter()-begun
    synthetic=[dict(**spec,winner=None,reason='natural',complete=True) for spec in schedules.stage_schedule('main')]
    begun=time.perf_counter()
    analysis=statistics.analyze_games(synthetic,evidence_gates_passed=True,boundary=envelope.guard)
    statistics.verify_analysis(synthetic,analysis,boundary=envelope.guard)
    statistics_seconds=time.perf_counter()-begun
    initialization=max(row['complete_startup_and_parity_seconds'] for row in costs)
    logs=max(row['log_seconds'] for row in costs)
    graph_reads=max(row['graph_proof_reread_seconds'] for row in costs)
    seal_reads=max(row['closed_worker_two_pass_seal_seconds'] for row in costs)
    # Whole entrypoint time/52 is a conservative observed per-decision check
    # allowance because it includes native parity and independent validation.
    validation=max(row['counterexample_seconds']/52 for row in costs)
    fixed=1.35*(source_seconds+statistics_seconds+2*logs+graph_reads+2*seal_reads+
        48*(initialization+1.+validation+logs+graph_reads+seal_reads))+30.
    result=dict(status='passed',saved_timing_rerun=False,actual_workers=2,costs=costs,
        source_seconds=source_seconds,statistics_seconds=statistics_seconds,
        initialization_seconds=initialization,validation_allowance_seconds=validation,
        final_graph_proof_rereads_charged=49,per_graph_proof_reread_seconds=graph_reads,
        final_worker_seal_reads_charged=50,per_worker_two_pass_seal_seconds=seal_reads,
        final_reproduction_maximum=48,forecast_seconds=fixed,
        synthetic_statistics_scope='200 synthetic draws used only to measure computation; discarded')
    c.save(c.OUT/'qualification_summary.json',result)
    return result


def game_folder(spec):
    return c.OUT/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def measure_seal_cost(folders, boundary):
    """Measure both real final-seal reads over closed operational evidence."""
    begun=time.perf_counter(); artifacts=[]
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                boundary(); artifacts.append(c.artifact(path))
    for item in artifacts:
        boundary(); c.verify([item])
    return dict(seconds=time.perf_counter()-begun,passes=2,files=len(artifacts),
                bytes=sum(item['bytes'] for item in artifacts),inventory_sha256=c.digest(artifacts))


def play_game(spec,envelope,freeze):
    schedules.validate_spec(spec)
    env=rules.CorrectedEnv(seed=int(spec['seed_decimal']))
    state=env.initial_state()
    initial=rules.serialize_snapshot(state)
    decode_training_snapshot(initial)
    folder=game_folder(spec)
    c.save(folder/'initial.json',dict(**spec,initial=initial))
    rows=[]; outcome=None; independent_seconds=0.; begun=time.perf_counter()
    with ExitStack() as stack:
        workers={role:stack.enter_context(make_worker(
            f"{spec['stage']}_g{spec['game_index']}_{role}",envelope,freeze,
            mode=spec[role+'_mode'],quota=spec[role+'_quota']))
            for role in ('challenger','opponent')}
        for ply in range(2000):
            if state.terminal: break
            seat=state.current_player
            role='challenger' if seat==spec['challenger_seat'] else 'opponent'
            row=decision(workers[role],state,envelope,role)
            row.update(ply=ply,**{key:spec[key] for key in ('stage','pair_index','game_index')})
            # Preserve the exact failure before budget/source/scientific checks.
            c.save(folder/'moves'/f'{ply:04d}.json',row)
            rows.append(row)
            envelope.guard()
            if row['completed']:
                t=time.perf_counter()
                scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',
                    before=row['before'],after=row['after'],expected_quota=row['quota'])
                independent_seconds+=time.perf_counter()-t
                state=decode_training_snapshot(row['after'])
            else:
                failure=audit.classify_failure(row['clock'])
                outcome=dict(reason='forfeit',winner=1-seat,loser=seat,failed_role=role,failed_seat=seat,failure=failure)
                break
            if not state.terminal and (row['after']==row['before'] or row['action']['action_type']=='PASS' and
                len(rows)>1 and rows[-2]['action'] and rows[-2]['action']['action_type']=='PASS'):
                raise ValueError('Unresolved actual-game deadlock; no official result')
        if outcome is None:
            c.require(state.terminal,'2,000-ply safety cap reached; incomplete')
            outcome=dict(reason='natural',winner=state.winner)
    # Both trees have confirmed cleanup before a game can be accepted.
    for role,child in workers.items():
        t=time.perf_counter()
        check=verify_worker_logs(child.output_dir,boundary=envelope.guard,deadline=envelope.effective_deadline())
        independent_seconds+=time.perf_counter()-t
        c.save(folder/f'log_integrity_{role}.json',check)
        if role=='challenger':
            t=time.perf_counter()
            graph_checked=verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
            independent_seconds+=time.perf_counter()-t
            c.save(folder/'graph_integrity_challenger.json',graph_checked)
    ending=dict(**spec,**outcome,final=rules.serialize_snapshot(state),complete=True,plies=len(rows),
                collection_seconds=time.perf_counter()-begun)
    t=time.perf_counter()
    checked=audit.audit_game(initial,rows,ending,spec,boundary=envelope.guard)
    independent_seconds+=time.perf_counter()-t
    c.save(folder/'result.json',ending)
    c.save(folder/'online_replay.json',checked)
    hashing=None
    if spec['stage']=='operational':
        hashing=measure_seal_cost([folder]+[child.output_dir for child in workers.values()],envelope.guard)
        independent_seconds+=hashing['seconds']
        c.save(folder/'hashing_measurement.json',hashing)
    c.save(folder/'cost.json',dict(independent_seconds=independent_seconds,
        collection_seconds=time.perf_counter()-begun,complete_decisions=sum(row['completed'] for row in rows),
        measured_two_pass_hash_seconds=None if hashing is None else hashing['seconds']))
    return ending,rows,independent_seconds


def final_evidence_forecast(schedule,boundary=lambda:None):
    """Conservative byte projection from closed files, without reading outcomes.

    A one-decision reproduction is charged an entire observed game's compressed
    worker logs, plus its largest request/input and an additional full move row.
    Final control files are projected separately; none consume an assumed-free
    portion of the fixed final evidence reserve.
    """
    ops=[spec for spec in schedule if spec['stage']=='operational']
    c.require(len(ops)==6,'Exactly six operational games required for byte forecast')
    game_paths=[]; worker_paths=[]; worker_bounds=[]; replay_bytes=0; log_receipt_bytes=[]; graph_receipt_bytes=[]
    largest_move=0
    for spec in ops:
        boundary(); folder=game_folder(spec)
        paths=[path for path in folder.rglob('*') if path.is_file()]
        game_paths.extend(paths)
        moves=list((folder/'moves').glob('*.json'))
        c.require(moves,'Operational move files absent from byte forecast')
        largest_move=max(largest_move,max(path.stat().st_size for path in moves))
        replay_bytes+=(folder/'online_replay.json').stat().st_size
        graph_receipt_bytes.append((folder/'graph_integrity_challenger.json').stat().st_size)
        for role in ('challenger','opponent'):
            log_receipt_bytes.append((folder/f'log_integrity_{role}.json').stat().st_size)
            worker=c.OUT/'workers'/f"{spec['stage']}_g{spec['game_index']}_{role}"
            files=[path for path in worker.rglob('*') if path.is_file()]
            inputs=list((worker/'inputs').glob('*.json'))
            c.require(inputs,'Operational worker inputs absent from byte forecast')
            receipts=[worker/path.name for path in inputs]
            c.require(all(path.is_file() for path in receipts),'Operational receipt inventory missing')
            logs=[path for path in files if path.suffix=='.gz']
            c.require(logs,'Compressed operational logs absent from byte forecast')
            variable=set(inputs+receipts+logs)
            static=[path for path in files if path not in variable]
            worker_paths.extend(files)
            worker_bounds.append(dict(
                static_bytes=sum(path.stat().st_size for path in static),
                entire_game_compressed_log_bytes=sum(path.stat().st_size for path in logs),
                largest_input_bytes=max(path.stat().st_size for path in inputs),
                largest_receipt_bytes=max(path.stat().st_size for path in receipts),
                reproduction_worker_files=len(static)+len(logs)+2))
    qualification_paths=[path for folder in (c.OUT/'workers').glob('qualification*')
                         for path in folder.rglob('*') if path.is_file()]
    observed_paths=game_paths+worker_paths+qualification_paths
    c.require(observed_paths,'No measured files for final seal forecast')
    # Reserve 128 extra path bytes for longer final audit/control names. JSON
    # bytes are measured using the actual canonical artifact representation.
    seal_item_bytes=max(len(c.canonical(dict(path=path.relative_to(c.ROOT).as_posix(),
        sha256='0'*64,bytes=2**63-1)).encode())+129 for path in observed_paths)
    single_worker=max(sum(item[key] for key in ('static_bytes','entire_game_compressed_log_bytes',
        'largest_input_bytes','largest_receipt_bytes')) for item in worker_bounds)
    reproduction_files=max(item['reproduction_worker_files'] for item in worker_bounds)
    reproduction_bytes=48*(single_worker+largest_move)
    projected_sealed_files=math.ceil((len(game_paths)+len(worker_paths))*206/6)+len(qualification_paths)+48*reproduction_files
    seal_bytes=projected_sealed_files*seal_item_bytes+1024
    replay_control_bytes=math.ceil(replay_bytes*206/6)
    log_control_bytes=(414+48)*max(log_receipt_bytes)
    graph_control_bytes=(207+48)*max(graph_receipt_bytes)
    fixed_control_bytes=8*c.MiB
    raw_final=reproduction_bytes+seal_bytes+replay_control_bytes+log_control_bytes+graph_control_bytes+fixed_control_bytes
    return dict(outcomes_used=False,conservative_factor=1.35,reproductions=48,
        observed_workers=len(worker_bounds),observed_game_files=len(game_paths),
        observed_worker_files=len(worker_paths),observed_worker_bounds=worker_bounds,
        largest_move_row_bytes=largest_move,single_reproduction_worker_bound_bytes=single_worker,
        projected_reproduction_bytes=reproduction_bytes,projected_sealed_files=projected_sealed_files,
        artifact_item_bound_bytes=seal_item_bytes,projected_seal_bytes=seal_bytes,
        projected_replay_control_bytes=replay_control_bytes,projected_log_control_bytes=log_control_bytes,
        projected_graph_control_bytes=graph_control_bytes,
        fixed_analysis_and_control_bytes=fixed_control_bytes,
        projected_final_evidence_bytes=math.ceil(1.35*raw_final),reserve_bytes=c.FINAL_RESERVE_BYTES)


def operational_gate(*,observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,
                     fixed_final_seconds,final_evidence_bytes,remaining_collection_seconds):
    values=(observed_seconds,observed_variable_bytes,current_bytes,independent_seconds,fixed_final_seconds,final_evidence_bytes)
    c.require(all(type(value) in (int,float) and math.isfinite(value) and value>=0 for value in values) and
              type(remaining_collection_seconds) in (int,float) and math.isfinite(remaining_collection_seconds),
              'Resource forecast requires finite measured quantities')
    factor=200/6*1.35
    final_factor=206/6*1.35
    collection=observed_seconds*factor
    evidence=current_bytes+observed_variable_bytes*factor
    final=independent_seconds*final_factor+fixed_final_seconds
    total_evidence=evidence+final_evidence_bytes
    passed=(collection<min(c.STAGES['main'],remaining_collection_seconds) and evidence<c.PREFINAL_EVIDENCE_BYTES and
            final<c.STAGES['final'] and final_evidence_bytes<c.FINAL_RESERVE_BYTES and total_evidence<c.MAX_EVIDENCE_BYTES)
    return dict(passed=passed,operational_games=6,observed_seconds=observed_seconds,
        observed_variable_bytes=observed_variable_bytes,projected_collection_seconds=collection,
        remaining_collection_seconds=min(c.STAGES['main'],remaining_collection_seconds),projected_prefinal_bytes=evidence,
        projected_final_seconds=final,arithmetic_factor=factor,final_arithmetic_factor=final_factor,
        projected_final_evidence_bytes=final_evidence_bytes,projected_total_evidence_bytes=total_evidence,
        final_evidence_reserve_bytes=c.FINAL_RESERVE_BYTES,total_evidence_cap_bytes=c.MAX_EVIDENCE_BYTES,outcomes_used=False)


def operational(envelope,freeze,schedule,qualification):
    start=time.perf_counter(); initial_bytes=bytes_used(); independent=0.; games=[]
    for spec in (row for row in schedule if row['stage']=='operational'):
        game,rows,cost=play_game(spec,envelope,freeze)
        games.append(game); independent+=cost
        c.require(game['reason']=='natural' and all(row['clock']['receipt_elapsed_seconds']<=.7 for row in rows),
                  'Operational natural completion or0.700-second margin failed')
        print(c.canonical(dict(stage='operational',completed_games=len(games),total=6)),flush=True)
    c.require(len(games)==6,'Operational schedule incomplete')
    final_bytes=final_evidence_forecast(schedule,envelope.guard)
    c.save(c.OUT/'final_evidence_forecast.json',final_bytes)
    used=bytes_used()
    gate=operational_gate(observed_seconds=time.perf_counter()-start,observed_variable_bytes=used-initial_bytes,
        current_bytes=used,independent_seconds=independent,fixed_final_seconds=qualification['forecast_seconds'],
        final_evidence_bytes=final_bytes['projected_final_evidence_bytes'],
        remaining_collection_seconds=min(envelope.start+c.COLLECTION_CUTOFF_SECONDS-time.monotonic(),
            envelope.wall_start+c.COLLECTION_CUTOFF_SECONDS-time.time()))
    c.save(c.OUT/'operational_gate.json',gate)
    c.require(gate['passed'],'Outcome-blind operational resource forecast failed; no expansion')
    return gate


def verify_saved_decision(row):
    require_player(row['mode'],row['quota'])
    c.require(row['mode']==('candidate' if row['player_role']=='challenger' else 'baseline'),
              'Saved scientific role/backend binding changed')
    path=c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()),'Clock outside owned evidence')
    c.require(c.read(path)==row['clock'] and path.name==row['clock']['request_id']+'.json','Trusted clock record changed')
    c.require(c.read(path.parent/'inputs'/path.name)==dict(player='search',observer=row['observer'],before=row['before']),
              'Saved player input changed')
    if row['completed']:
        scientific.audit_decision(row['observer'],row['action'],row['trace'],'search',
            before=row['before'],after=row['after'],expected_quota=row['quota'])
    else:
        audit.classify_failure(row['clock'])


def reproduce_decision(original,index,envelope,freeze):
    """The replay uses the original backend as well as its exact quota."""
    c.require(original['completed'],'Cannot replay an incomplete decision')
    require_player(original['mode'],original['quota'])
    with make_worker(f'audit_{index:02d}',envelope,freeze,mode=original['mode'],quota=original['quota']) as child:
        repeated=decision(child,decode_training_snapshot(original['before']),envelope,original['player_role'])
        c.save(c.OUT/'repeated_decisions'/f'{index:02d}.json',repeated)
        envelope.guard()
        c.require(repeated['completed'] and repeated['action']==original['action'] and
                  repeated['trace']==original['trace'],'Sampled complete decision reproduction differs')
        verify_saved_decision(repeated)
    return repeated


def final_checks(envelope,freeze,schedule):
    c.require(schedule==schedules.make_schedule(),'Frozen schedule derivation changed')
    for quota in (32,64):
        envelope.guard()
        receipt=c.read(c.OUT/f'entrypoint_checks_{quota}.json')
        folder=c.OUT/'workers'/f'qualification_{quota}'
        c.require(receipt['status']=='ok' and c.read(folder/f"{receipt['request_id']}.json")==receipt and
            receipt['result']['status']=='entrypoint_counterexamples_passed' and
            receipt['result']['challenger_quota']==quota and receipt['result']['counts']['complete_decisions']==52,
            'Preserved actual entrypoint receipt or inventory changed')
    records=[]; main_games=[]; replays=[]
    for spec in schedule:
        envelope.guard()
        folder=game_folder(spec)
        initial,game=c.read(folder/'initial.json'),c.read(folder/'result.json')
        expected=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(initial==dict(**spec,initial=expected),'Initial world/identity changed')
        paths=sorted((folder/'moves').glob('*.json'))
        c.require([path.name for path in paths]==[f'{i:04d}.json' for i in range(game['plies'])],
                  'Operational/main move inventory changed')
        rows=[c.read(path) for path in paths]
        for row in rows:
            envelope.guard(); verify_saved_decision(row)
        replays.append(audit.audit_game(initial['initial'],rows,game,spec,boundary=envelope.guard))
        if spec['stage']=='main':
            records.extend(rows); main_games.append(game)
    c.require(len(replays)==206 and len(main_games)==200,'Complete game inventory required')
    c.save(c.OUT/'all_game_replay.json',dict(status='passed',games=206,receipts=replays))
    c.save(c.OUT/'performance.json',audit.performance(records,main_games))
    sample=audit.select_replay_sample(records,maximum=48)
    c.save(c.OUT/'replay_sample_manifest.json',sample)
    selected=sample['decisions']; repetitions=[]
    c.require(len(selected)<=48 and all(sum(item['player_role']==role for item in selected)<=24
        for role in ('challenger','opponent')),'Complete-decision reproduction bound')
    for index,item in enumerate(selected):
        envelope.guard()
        original=next(row for row in records if all(row[key]==item[key]
            for key in ('pair_index','game_index','ply','player_role')))
        c.require(original['completed'],'Cannot reproduce an uncompleted timed-out decision')
        c.require(original['mode']==item['mode'] and original['quota']==item['quota'],
                  'Sampled reproduction backend/quota identity changed')
        reproduce_decision(original,index,envelope,freeze)
        repetitions.append(item)
    c.save(c.OUT/'repeated_decision_summary.json',dict(status='passed',requests=len(repetitions),identities=repetitions))
    logs=[]; graph_checks=[]
    for folder in sorted((c.OUT/'workers').iterdir()):
        envelope.guard()
        logs.append(verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline()))
        metadata=c.read(folder/'initialization.json')['result']
        if 'inference_backend' in metadata:
            graph_checks.append(verify_graph_evidence(folder,freeze,boundary=envelope.guard))
        else:
            c.require(metadata['search_simulations']==32 and not any(folder.glob('graph_*.json')),
                      'Unchanged eager32 worker evidence required')
    c.require(len(logs)==414+len(repetitions),'Two workers per game plus qualification/audit inventory required')
    c.save(c.OUT/'all_worker_log_integrity.json',dict(status='passed',workers=len(logs),receipts=logs))
    c.require(len(graph_checks)==207+sum(item['mode']=='candidate' for item in selected),
              'Graph instance inventory differs from qualification/game/sample roles')
    c.save(c.OUT/'all_graph_parity.json',dict(status='passed',workers=len(graph_checks),receipts=graph_checks,
        independent_full_outputs_and_float32_bits=True))
    seal=[]
    for subdir in ('games','workers','repeated_decisions'):
        for path in sorted((c.OUT/subdir).rglob('*')):
            if path.is_file():
                envelope.guard(); seal.append(c.artifact(path))
    c.save(c.OUT/'completed_game_seal.json',dict(artifacts=seal))
    analysis=statistics.analyze_games(main_games,evidence_gates_passed=True,boundary=envelope.guard)
    independent=statistics.verify_analysis(main_games,analysis,boundary=envelope.guard)
    c.save(c.OUT/'analysis.json',analysis)
    c.save(c.OUT/'independent_statistics.json',independent)
    for item in seal:
        envelope.guard();c.verify([item])
    verify_runtime(freeze,envelope)
    c.require(qualification_acceptance(verify_inputs=False)==freeze['qualification_acceptance'],
              'Preceding qualification acceptance changed')
    return analysis


def run_stages(envelope,freeze,schedule,*,qualification_started=False):
    if qualification_started:
        c.require(envelope.stage=='qualification','Startup qualification clock missing')
    else:
        envelope.begin('qualification')
    qualified=qualify(envelope,freeze)
    envelope.finish()
    envelope.begin('operational')
    operational(envelope,freeze,schedule,qualified)
    envelope.finish()
    envelope.begin('main')
    games=[spec for spec in schedule if spec['stage']=='main']
    c.require(len(games)==200,'Fixed200-game main schedule required')
    for index,spec in enumerate(games):
        play_game(spec,envelope,freeze)
        print(c.canonical(dict(stage='main',completed_games=index+1,total=200)),flush=True)
    envelope.finish()
    envelope.begin('final')
    result=final_checks(envelope,freeze,schedule)
    envelope.finish()
    return result


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Comparison freeze changed')
    claim=c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Watchdog freeze identity differs')
    envelope,freeze=Envelope(claim),c.read(c.OUT/'freeze.json')
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha))
    status,failure='incomplete',None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json',dict(sources=verify_runtime_sources()))
        verify_runtime(freeze,envelope)
        envelope.admit()
        schedule=c.read(c.OUT/'schedule.json')
        c.require(schedule==schedules.make_schedule(),'Frozen schedule changed')
        run_stages(envelope,freeze,schedule,qualification_started=True)
        status='completed_verified'
    except BaseException:
        failure=traceback.format_exc()
        c.save(c.OUT/'failure.json',dict(time=c.now(),stage=envelope.stage,error=failure))
    c.save(c.OUT/'completion.json',dict(status=status,time=c.now(),failure=failure,stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start,wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha,automatic_retries=0,larger_data_deferred=True,
        challenger_quota=64,opponent_quota=32,main_games_required=200,
        note='Acceptance also requires successful owning watchdog cleanup; no automatic player promotion.'))
    print(c.canonical(dict(status=status,stage=envelope.stage,failure=failure)),flush=True)
    return 0 if status=='completed_verified' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['prepare','run'])
    parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    return prepare() if args.operation=='prepare' else run(args.freeze_sha256)


if __name__=='__main__':
    raise SystemExit(main())

