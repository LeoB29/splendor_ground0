"""Independent saved scientific checks with root-job ownership and new encoding."""
from pathlib import Path
from types import SimpleNamespace
import time
import tempfile
import os
from . import common as c, private, collector, qualification, data, backend, storage
from .. import common as root
from tools.splendorust_gpu_probe_20261105.control import require_absent, native_fallback_messages

_legacy=private.load('audit')
_boundary=lambda:None


class JournalIndex(_legacy.JournalIndex):
    """Exact raw materialization, then the unchanged accepted streaming checker."""
    def __init__(self,path,route,scheduler,*,incomplete=False):
        self.stored_path=Path(path).resolve();self.materialized=None;self.seal=None
        with self.stored_path.open('rb') as stream:magic=stream.read(len(storage.JOURNAL_MAGIC))
        if magic==storage.JOURNAL_MAGIC:
            commit=c.read(self.stored_path.with_name(self.stored_path.name+'.seal_commit.json'))
            c.require(commit['status']=='committed' and commit['stored']==c.artifact(self.stored_path),'Completed authenticated journal seal')
            c.verify([commit['receipt']]);receipt=c.read(c.resolve(commit['receipt']))
            c.require(receipt['stored']==commit['stored'] and receipt['all_operations_closed'] is True,'Closed journal receipt binding')
            scratch=c.OUT/'audit_scratch';scratch.mkdir(parents=True,exist_ok=True)
            fd,name=tempfile.mkstemp(prefix='journal_',suffix='.jsonl',dir=scratch);self.materialized=Path(name)
            try:
                with os.fdopen(fd,'wb') as dest:
                    checked=storage.stream_journal(self.stored_path,dest);dest.flush();os.fsync(dest.fileno())
                c.require(checked==receipt['roundtrip'] and checked['raw_bytes']==receipt['raw']['bytes']
                    and checked['raw_sha256']==receipt['raw']['sha256'],'Raw byte identity reconstructed exactly')
                self.seal=receipt
                super().__init__(self.materialized,route,scheduler,incomplete=incomplete)
            except BaseException:
                # Preserve failed materialization for diagnosis, never erase failed evidence.
                raise
        else:
            c.require(incomplete,'Successful new runtime requires closed journal sealing')
            super().__init__(self.stored_path,route,scheduler,incomplete=incomplete)
        self.claim=c.read(c.CLAIM)

    def scan(self):
        super().scan()
        if self.seal is not None:
            c.require(self.journal_artifact['sha256']==self.seal['raw']['sha256'] and
                self.journal_artifact['bytes']==self.seal['raw']['bytes'],'Unchanged original scanner consumed every original byte')
            self.journal_artifact=c.artifact(self.stored_path)
        return self

    def complete_attribution(self):
        result=super().complete_attribution()
        if self.seal is not None:result['lossless_storage']=dict(raw=self.seal['raw'],stored=self.seal['stored'],exact_byte_roundtrip=True)
        return result

    def close(self,error=None):
        try:super().close(error)
        finally:
            if self.materialized is not None and error is None:
                c.require(self.materialized.resolve().parent==(c.OUT/'audit_scratch').resolve()
                    and self.materialized.name.startswith('journal_'),'Exact owned scratch path')
                self.materialized.unlink();self.materialized=None


def set_boundary(boundary):
    global _boundary
    _boundary=boundary


def guard_clock(claim=None):
    _boundary()
    claim=c.read(c.CLAIM) if claim is None else claim
    c.require(0 <= time.monotonic()-claim['monotonic_origin'] < c.TOTAL_SECONDS and
              0 <= time.time()-claim['unix_origin'] < c.TOTAL_SECONDS,'Original overall audit clock')


def stage_execution(plan,container,*,mode,cell=None):
    container=Path(container);owner=c.read(container/'owner.json');execution=c.read(container/'execution.json')
    frozen=c.artifact(root.PLAN);claim=c.read(c.CLAIM)
    c.require(claim['freeze']==frozen and root.read(root.PLAN)['collection_plan']==c.artifact(c.PLAN),'Root freezes exact collection plan')
    c.require(owner['identity']==execution['identity'] and owner['mode']==execution['mode']==mode and
        owner['plan_sha256']==execution['plan_sha256']==frozen['sha256'] and owner['windows_job_assigned'] is True,
        'Exact root-owned collection worker role and Windows job')
    c.require(owner['global_claim']==c.artifact(c.CLAIM),'Original global claim')
    c.verify([owner['phase_clock']]);phase=c.read(c.resolve(owner['phase_clock']))
    c.require(phase['claim']==owner['global_claim'],'Unextended original phase')
    for origin,end,deadline,phase_start,phase_end in (
            ('monotonic_origin','monotonic_end','monotonic_deadline','monotonic','monotonic_deadline'),
            ('unix_origin','unix_end','unix_deadline','unix_time','unix_deadline')):
        c.require(execution[origin]==owner[origin] and execution[deadline]==owner[deadline] and
            claim[origin] <= phase[phase_start] <= owner[origin] <= execution[end] < owner[deadline] <= phase[phase_end]
            and execution[end] < claim[origin]+c.MODEL_SECONDS,'Full original worker/phase/model dual-clock chain')
    c.require(execution['status']=='passed' and execution['worker_exit_code']==0 and execution['retries']==0
        and not execution['cleanup_errors'] and execution['error'] is None,'Actual successful exit and complete cleanup')
    c.require(execution['peak_rss_bytes']<=owner['maximum_rss_bytes']<=c.MAX_RSS,'Actual owned RSS bound')
    identities=execution['observed_owned_identities']+[execution['identity']]
    unique={(x['pid'],x['created']):x for x in identities}
    for identity in unique.values():require_absent(identity)
    c.require(not native_fallback_messages(container/'worker/stderr.log'),'No unsupported native fallback')
    startup=c.read(container/'worker/runtime_sources.json')
    c.require(startup['sources']==plan['source_guard'] and startup['identity']==execution['identity']
              and startup['original_claim']==owner['global_claim'] and startup['original_phase']==owner['phase_clock'],
              'Actual once-startup corrected guard bound to exact worker')
    result=c.read(container/'worker/result.json')
    c.require(result['status']=='passed' and result['mode']==mode and result['identity']==execution['identity'],
              'Actual worker return receipt')
    return dict(execution,status='completed'),list(unique.values())


def verify_gate_binding(plan,folder):
    gate=c.read(root.ORIGINAL_OUT/'collection/qualification_gate.json')
    c.require(c.artifact(c.PLAN)['sha256']==c.artifact(root.ORIGINAL_OUT/'collection/plan.json')['sha256'],
              'Byte-identical original scientific collection contract')
    c.require(gate['passed'] is True and gate['collection_plan']==c.artifact(root.ORIGINAL_OUT/'collection/plan.json'),
              'Exact original qualified production collector')
    c.verify([gate['audit']]);proof=c.read(c.resolve(gate['audit']))
    c.require(proof['status']=='passed' and proof['complete_decisions']==116 and proof['engineering_games']==2,
              'Actual fixture and complete compressed-game qualification')
    c.require(c.read(Path(folder)/'qualification_gate_binding.json')==gate,'Worker bound exact saved admission before models')
    return gate


_legacy.read=c.read
_legacy.JournalIndex=JournalIndex
_legacy.guard_audit_clock=guard_clock
_legacy.verify_gate_binding=verify_gate_binding
_legacy.private_checker=lambda name: SimpleNamespace(audit_stage_execution=stage_execution) if name=='batchfill' else (_ for _ in ()).throw(ValueError('CPU production has no GPU checker'))


def audit_cell(plan,spec,container):
    return _legacy.audit_cell(plan,spec,container)


def audit_qualification(plan,container):
    """Same 116 scientific fixture checks plus two genuine excluded trajectories."""
    guard_clock();container=Path(container);folder=container/'worker'
    execution,absent=stage_execution(plan,container,mode='qualification')
    c.require(plan['qualification']==qualification.build_plan() and plan['storage']==storage.contract(), 'Fixed qualifier and lossless representation')
    c.require(backend.source_equivalence()==plan['backend_source_equivalence'] and backend.capture_provenance()==plan['native_capture'], 'Exact accepted execution/capture sources')
    route='cpu';directory=folder/route;runtime=folder/'cpu_runtime'
    with _legacy.JournalIndex(runtime/'inference.jsonl',route,'original8') as journal:
        entrypoint=_legacy.audit_entrypoint(plan,directory/'entrypoint',journal);parity=[]
        for case in plan['cases']:
            spec=dict(game_id=case['case_id'],game_index=case['index'],purpose='engineering_qualification_only')
            row,_,_=_legacy.audit_decision(directory/'parity'/case['case_id'],spec,0,journal,before=case['snapshot'],
                owner='parity:'+case['case_id'],client='parity_'+str(case['slot']),slot=case['slot'])
            c.verify([case['expected_results'][route]]);source=c.read(c.resolve(case['expected_results'][route]))
            c.require(row['result']==source['result'] and row['after']==source['after'] and
                _legacy.bits(_legacy.prior.root_output(case,row['result']))==_legacy.bits(_legacy.prior.root_output(case,source['result'])), 'Entire accepted scientific parity')
            witness=dict(case_id=case['case_id'],actual=c.artifact(directory/'parity'/case['case_id']/'moves'/'000000.json'),
                expected=case['expected_results'][route],exact_full_result_and_after=True)
            c.require(c.read(directory/'parity'/case['case_id']/'comparison.json')==witness,'Exact parity witness');parity.append(witness)
        c.require({x.name for x in (directory/'parity').iterdir() if x.is_dir()}=={x['case_id'] for x in plan['cases']},'No extra or omitted parity cases')
        loader=plan['qualification']['loader'];cases=loader['cases']+[loader['private_variant']];rows=[]
        for index,case in enumerate(cases):
            spec=qualification.fixture_spec(case,index)
            row,_,_=_legacy.audit_decision(directory/'loader'/f'{index:02d}',spec,0,journal,before=case['snapshot'],
                owner=f'loader:{index}',client='loader',slot=0);rows.append(row)
        rowproof=data.qualify_captured_rows(loader,rows,boundary=_boundary)
        c.require(rowproof==c.read(directory/'loader_row_checks.json') and len(rows)==16,'Every actual loader counterexample')
        index=next(i for i,case in enumerate(cases) if case.get('id')=='noble_final_round');first=rows[index]
        last,_,_=_legacy.audit_decision(directory/'loader_terminal',first['spec'],1,journal,before=first['after'],owner='loader:terminal',client='loader',slot=0)
        document=c.read(directory/'fixture_target_game.json')
        c.require(document['initial']==first['before'] and document['final']==last['after'] and document['rows']==[
            c.artifact(directory/'loader'/f'{index:02d}'/'moves'/'000000.json'),c.artifact(directory/'loader_terminal'/'moves'/'000001.json')], 'Exact official fixture trajectory')
        full=_legacy.audit_full_loader_fixture(directory,document,first['spec'],[first,last])
        expected=dict(status='qualified',route=route,actual_entrypoint_decisions=67,accepted_saved_decisions=32,
            actual_loader_decisions=16,official_fixture_continuation_decisions=1,complete_decisions=116,
            entrypoint=c.artifact(directory/'entrypoint'/'qualification.json'),parity=parity,
            loader_rows=c.artifact(directory/'loader_row_checks.json'),full_loader=c.artifact(directory/'full_loader_checks.json'),
            actual_collector_and_loader_qualified=True,ordinary_games=0,training_updates=0,no_retry=True)
        c.require(c.read(directory/'qualification.json')==expected,'Exact complete CPU fixture accounting')
        engineering=folder/'engineering';specs=plan['engineering_games'];proofs=[]
        _legacy.audit_warmup(plan,engineering,journal)
        c.require({x.name for x in (engineering/'games').iterdir() if x.is_dir()}=={x['game_id'] for x in specs},'Only two excluded engineering games')
        for index,spec in enumerate(specs):
            assignment=c.read(engineering/'assignments'/f'{index:02d}.json')
            c.require(assignment['game_id']==spec['game_id'] and assignment['client_id']=='engineering_'+str(index) and assignment['slot']==index,'Fixed excluded engineering assignment')
            proofs.append(_legacy.audit_game(plan,dict(route='cpu'),spec,engineering/'games'/spec['game_id'],journal,assignment))
        manifest=data.build_manifest([p['document'] for p in proofs],specs,route='cpu',cell_id='engineering')
        c.require(c.read(engineering/'corpus_manifest.json')==manifest,'Exact engineering source corpus')
        loaded=data.load_corpus(manifest,expected_specs=specs,boundary=_boundary)
        c.require(loaded==c.read(engineering/'loader_result.json') and loaded['eligible_roots']>=32,'Same actual compressed complete-game loader')
        replays=_legacy.audit_replays(plan,dict(specifications=specs),engineering,journal,proofs,c.read(engineering/'cohort.json'))
        closure=_legacy.runtime_closure(plan,runtime,route,journal)
        initialization=_legacy.audit_initialization(container,runtime,route,journal.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0])
        accounting=journal.complete_attribution()
    c.require(c.read(folder/'cleanup.json')['success'] is True,'Qualification runtime cleanup')
    result=dict(status='passed',passed=True,complete_decisions=116,engineering_games=2,excluded_from_production=True,
        entrypoint=entrypoint,parity=parity,loader_row_checks=rowproof,full_loader_checks=full,
        engineering_targets=[p['target'] for p in proofs],engineering_game_proofs=proofs,engineering_replays=replays,
        runtime_closure=closure,initialization=initialization,journal=accounting,owned_processes_absent=absent,
        collection_plan=c.artifact(c.PLAN),qualification_summary=c.artifact(directory/'qualification.json'),
        engineering_result=c.artifact(engineering/'engineering_result.json'),storage=storage.contract(),model_calls=0)
    guard_clock();return result


def audit_recovery_qualification(plan,container):
    """Actual recovery entrypoint: all116 unchanged scientific counterexamples; no new game."""
    guard_clock();container=Path(container);folder=container/'worker'
    execution,absent=stage_execution(plan,container,mode='recovery_qualification')
    c.require(plan['qualification']==qualification.build_plan() and plan['storage']==storage.contract(), 'Fixed qualifier and lossless representation')
    c.require(backend.source_equivalence()==plan['backend_source_equivalence'] and backend.capture_provenance()==plan['native_capture'], 'Exact accepted execution/capture sources')
    route='cpu';directory=folder/route;runtime=folder/'cpu_runtime'
    with _legacy.JournalIndex(runtime/'inference.jsonl',route,'original8') as journal:
        entrypoint=_legacy.audit_entrypoint(plan,directory/'entrypoint',journal);parity=[]
        for case in plan['cases']:
            spec=dict(game_id=case['case_id'],game_index=case['index'],purpose='engineering_qualification_only')
            row,_,_=_legacy.audit_decision(directory/'parity'/case['case_id'],spec,0,journal,before=case['snapshot'],
                owner='parity:'+case['case_id'],client='parity_'+str(case['slot']),slot=case['slot'])
            c.verify([case['expected_results'][route]]);source=c.read(c.resolve(case['expected_results'][route]))
            c.require(row['result']==source['result'] and row['after']==source['after'] and
                _legacy.bits(_legacy.prior.root_output(case,row['result']))==_legacy.bits(_legacy.prior.root_output(case,source['result'])), 'Entire accepted scientific parity')
            witness=dict(case_id=case['case_id'],actual=c.artifact(directory/'parity'/case['case_id']/'moves'/'000000.json'),
                expected=case['expected_results'][route],exact_full_result_and_after=True)
            c.require(c.read(directory/'parity'/case['case_id']/'comparison.json')==witness,'Exact parity witness');parity.append(witness)
        c.require({x.name for x in (directory/'parity').iterdir() if x.is_dir()}=={x['case_id'] for x in plan['cases']},'No extra or omitted parity cases')
        loader=plan['qualification']['loader'];cases=loader['cases']+[loader['private_variant']];rows=[]
        for index,case in enumerate(cases):
            spec=qualification.fixture_spec(case,index)
            row,_,_=_legacy.audit_decision(directory/'loader'/f'{index:02d}',spec,0,journal,before=case['snapshot'],
                owner=f'loader:{index}',client='loader',slot=0);rows.append(row)
        rowproof=data.qualify_captured_rows(loader,rows,boundary=_boundary)
        c.require(rowproof==c.read(directory/'loader_row_checks.json') and len(rows)==16,'Every actual loader counterexample')
        index=next(i for i,case in enumerate(cases) if case.get('id')=='noble_final_round');first=rows[index]
        last,_,_=_legacy.audit_decision(directory/'loader_terminal',first['spec'],1,journal,before=first['after'],owner='loader:terminal',client='loader',slot=0)
        document=c.read(directory/'fixture_target_game.json')
        c.require(document['initial']==first['before'] and document['final']==last['after'] and document['rows']==[
            c.artifact(directory/'loader'/f'{index:02d}'/'moves'/'000000.json'),c.artifact(directory/'loader_terminal'/'moves'/'000001.json')], 'Exact official fixture trajectory')
        full=_legacy.audit_full_loader_fixture(directory,document,first['spec'],[first,last])
        expected=dict(status='qualified',route=route,actual_entrypoint_decisions=67,accepted_saved_decisions=32,
            actual_loader_decisions=16,official_fixture_continuation_decisions=1,complete_decisions=116,
            entrypoint=c.artifact(directory/'entrypoint'/'qualification.json'),parity=parity,
            loader_rows=c.artifact(directory/'loader_row_checks.json'),full_loader=c.artifact(directory/'full_loader_checks.json'),
            actual_collector_and_loader_qualified=True,ordinary_games=0,training_updates=0,no_retry=True)
        c.require(c.read(directory/'qualification.json')==expected,'Exact complete CPU fixture accounting')
        closure=_legacy.runtime_closure(plan,runtime,route,journal)
        initialization=_legacy.audit_initialization(container,runtime,route,journal.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0])
        accounting=journal.complete_attribution()
    c.require(c.read(folder/'cleanup.json')['success'] is True,'Actual qualifier runtime cleanup')
    result=dict(status='passed',passed=True,complete_decisions=116,new_games=0,training_updates=0,
        entrypoint=entrypoint,parity=parity,loader_row_checks=rowproof,full_loader_checks=full,
        runtime_closure=closure,initialization=initialization,journal=accounting,owned_processes_absent=absent,
        original_engineering_qualification=c.artifact(root.ORIGINAL_OUT/'collection/qualification_audit/worker/saved_audit.json'),
        collection_plan=c.artifact(c.PLAN),qualification_summary=c.artifact(directory/'qualification.json'),model_calls=0)
    guard_clock();return result
