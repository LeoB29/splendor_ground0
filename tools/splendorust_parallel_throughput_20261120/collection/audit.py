"""Independent original scientific checks over separate per-process journals."""
from contextlib import ExitStack
from pathlib import Path
import os
import tempfile
from . import common as c, private, data, storage

_legacy=private.load('audit')
_legacy.read=c.read
_boundary=lambda:None
_legacy.guard_audit_clock=lambda claim=None:_boundary()


class JournalIndex(_legacy.JournalIndex):
    def __init__(self,path,route='cpu',scheduler='original8'):
        self.stored_path=Path(path);self.materialized=None
        commit=c.read(self.stored_path.with_name(self.stored_path.name+'.seal_commit.json'))
        c.require(commit['status']=='committed' and commit['stored']==c.artifact(self.stored_path),'Closed original journal seal')
        c.verify([commit['receipt']]);self.seal=c.read(c.resolve(commit['receipt']))
        c.require(self.seal['stored']==commit['stored'] and self.seal['all_operations_closed'] is True,'Exact closed seal identity')
        scratch=c.OUT/'audit_scratch';scratch.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='journal_',suffix='.jsonl',dir=scratch);self.materialized=Path(name)
        with os.fdopen(fd,'wb') as stream:
            checked=storage.stream_journal(self.stored_path,stream);stream.flush();os.fsync(stream.fileno())
        c.require(checked==self.seal['roundtrip'],'Exact byte-for-byte journal reconstruction')
        super().__init__(self.materialized,route,scheduler)

    def scan(self):
        super().scan()
        c.require(self.journal_artifact['sha256']==self.seal['raw']['sha256'] and
            self.journal_artifact['bytes']==self.seal['raw']['bytes'],'Original scanner consumed every raw byte')
        self.journal_artifact=c.artifact(self.stored_path)
        return self

    def close(self,error=None):
        try:super().close(error)
        finally:
            if self.materialized is not None and error is None:
                c.require(self.materialized.resolve().parent==(c.OUT/'audit_scratch').resolve(),'Only owned derived scratch')
                self.materialized.unlink();self.materialized=None


def verify_execution(worker_folder,*,expected_mode):
    from .. import common as root
    from tools.splendorust_gpu_probe_20261105.control import require_absent, native_fallback_messages
    folder=Path(worker_folder);execution=root.read(folder.parent/'execution.json');owner=root.read(folder.parent/'owner.json')
    c.require(execution['status']=='passed' and execution['exit_code']==0 and not execution['cleanup_errors'] and
        execution['error'] is None and execution['identity']==owner['identity'] and
        execution['request']==owner['request'] and owner['request']['mode']==expected_mode,'Actual owned successful role/exit')
    c.require(owner['windows_job_assigned'] is True and owner['plan']==root.artifact(root.PLAN),'Exact frozen owner and Windows job')
    c.require(execution['monotonic_end']<owner['monotonic_deadline'] and execution['unix_end']<owner['unix_deadline'],'Original completed deadlines')
    for ident in execution['observed_owned_identities']+[execution['identity']]:require_absent(ident)
    c.require(not native_fallback_messages(folder/'stderr.log'),'No unsupported native fallback')
    result=root.read(folder/'result.json');startup=c.read(folder/'runtime_sources.json')
    c.require(result['status']=='passed' and result['identity']==owner['identity'] and
        startup['identity']==owner['identity'] and startup['sources']==root.read(root.PLAN)['collection']['source_guard'],
        'Actual result and once-startup corrected source guard')
    return dict(execution=root.artifact(folder.parent/'execution.json'),identity=owner['identity'],actual_exit=0,
        absent=execution['observed_owned_identities']+[execution['identity']])


def closure(plan,folder,journal):
    c.require(c.read(Path(folder)/'cleanup.json')['success'] is True,'Actual runtime cleanup')
    return _legacy.runtime_closure(plan,Path(folder)/'runtime','cpu',journal)


def initialization(folder,first_clock):
    folder=Path(folder);row=c.read(folder/'initialization.json');execution=c.read(folder.parent/'execution.json')
    c.require(row['status']=='passed' and row['no_retry'] is True and row['inside_measured_collection'] is False and
        row['started_perf_counter']<=row['ended_perf_counter']<=first_clock and
        execution['unix_start']<=row['started_unix']<=row['ended_unix']<=execution['unix_end'],
        'Original bounded actual runtime initialization before first operation')
    return c.artifact(folder/'initialization.json')


def check_cohort(spec,assigned,proofs,started,cohort,queue,committed,assembly_execution):
    """Pure saved checks used by independent audit and corruption fixtures."""
    c.require(len(assigned)==len(proofs)==len(committed)==16,'Every fixed queue entry and game')
    c.require(committed==[dict(position=i,spec=game,assignment=assigned[i]) for i,game in enumerate(spec['specifications'])],
        'Exact committed shared queue source and every claimed assignment')
    c.require(queue==dict(assignments=assigned,outcomes={p['game_id']:p['outcome'] for p in proofs},errors=[],never_started=[]),
        'Complete durable original queue outcomes')
    previous={}
    for index,(game,row,proof) in enumerate(zip(spec['specifications'],assigned,proofs)):
        key=(row['process_index'],row['slot'])
        c.require(row['queue_index']==index and row['game_id']==game['game_id']==proof['game_id'] and
            row['assigned_perf_counter']>=started['started_perf_counter'] and
            (index==0 or assigned[index-1]['assigned_perf_counter']<=row['assigned_perf_counter']) and
            (key not in previous or previous[key]<=row['assigned_perf_counter']),
            'Ascending fixed queue assigned only to idle persistent process/client')
        previous[key]=proof['finished']['ended_perf_counter']
    roots=sum(p['loaded']['eligible_roots'] for p in proofs);plies=sum(p['loaded']['plies'] for p in proofs)
    c.require(started['cell_id']==spec['cell_id'] and cohort['started_perf_counter']==started['started_perf_counter'] and
        cohort['started_unix']==started['started_unix']==min(x['assigned_unix'] for x in assigned) and
        started['started_perf_counter']==min(x['assigned_perf_counter'] for x in assigned),'Actual first dispatch clock')
    elapsed=cohort['ended_perf_counter']-started['started_perf_counter']
    c.require(cohort['status']=='completed' and cohort['cell_id']==spec['cell_id'] and cohort['route']=='cpu' and
        cohort['concurrency']==8 and cohort['games']==16 and cohort['plies']==plies and cohort['eligible_roots']==roots and
        cohort['full_cohort_completed'] is True and cohort['corpus_produced'] is False and
        cohort['engineering_targets_only'] is True and cohort['excluded_from_every_training_corpus'] is True and
        cohort['retries']==cohort['replacements']==cohort['training_updates']==0,'Entire independent cohort accounting')
    c.require(elapsed==cohort['wall_seconds']>0 and cohort['games_per_hour']==16*3600/elapsed and
        cohort['eligible_roots_per_hour']==roots*3600/elapsed and
        cohort['gameplay_end_perf_counter']==max(p['finished']['ended_perf_counter'] for p in proofs)<=cohort['ended_perf_counter'] and
        cohort['gameplay_end_unix']==max(p['finished']['ended_unix'] for p in proofs) and
        cohort['gameplay_seconds']==cohort['gameplay_end_perf_counter']-started['started_perf_counter'] and
        cohort['target_loader_seconds']==cohort['ended_perf_counter']-cohort['gameplay_end_perf_counter'] and
        started['started_unix']<=cohort['gameplay_end_unix']<=assembly_execution['unix_start']<=cohort['ended_unix']<=assembly_execution['unix_end'],
        'Primary clocks include natural tails, actual complete target loader and owned assembly')
    return dict(plies=plies,eligible_roots=roots,wall_seconds=elapsed)


def audit_cell(plan,spec,cell_folder,play_folders,replay_folder,*,boundary=lambda:None):
    global _boundary
    _boundary=boundary;folder=Path(cell_folder);replay_folder=Path(replay_folder)
    c.require(len(play_folders) in (1,4),'Actual one/four independent model service journals')
    executions=[verify_execution(p,expected_mode='play') for p in play_folders]
    executions.append(verify_execution(replay_folder,expected_mode='replays'))
    executions.append(verify_execution(folder/'assembly/worker',expected_mode='assemble'))
    release=c.read(folder/'release.json');c.verify(release['ready'])
    c.require(release['services']==len(play_folders) and release['clients']==8 and
        release['ready']==[c.artifact(Path(p)/'ready.json') for p in play_folders],'Every model service ready before single dispatch release')
    for index,p in enumerate(play_folders):
        ready=c.read(Path(p)/'ready.json')
        c.require(ready['process_index']==index and ready['clients']==(8 if len(play_folders)==1 else 2) and
            ready['warmup_complete'] is True and ready['ready_perf_counter']<=release['released_perf_counter'],
            'Exact actual model-service readiness')
    c.require([p.name for p in sorted((folder/'assignments').glob('*.json'))]==[f'{i:02d}.json' for i in range(16)],'All and only sixteen assignment files')
    c.require({p.name for p in (folder/'games').iterdir() if p.is_dir()}=={g['game_id'] for g in spec['specifications']},'No extra or replaced games')
    with ExitStack() as stack:
        journals=[stack.enter_context(JournalIndex(Path(p)/'runtime/inference.jsonl')) for p in play_folders]
        replay=stack.enter_context(JournalIndex(replay_folder/'runtime/inference.jsonl'))
        warmups=[]
        for p,journal in zip(play_folders,journals):warmups.append(_legacy.audit_warmup(plan,Path(p)/'warmup',journal))
        _legacy.audit_warmup(plan,replay_folder/'warmup',replay)
        game_proofs=[];assignments=[]
        for index,game in enumerate(spec['specifications']):
            boundary();assignment=c.read(folder/'assignments'/f'{index:02d}.json');assignments.append(assignment)
            c.require(assignment['queue_index']==index and assignment['game_id']==game['game_id'],'Fixed ascending shared queue')
            process=assignment['process_index'];c.require(process in range(len(journals)),'Exact source service index')
            c.require(release['released_perf_counter']<=assignment['assigned_perf_counter'],'No game starts before all services are ready')
            maximum=8 if len(journals)==1 else 2
            c.require(assignment['slot'] in range(maximum) and assignment['client_id']==f'worker_{assignment["slot"]}' and
                assignment['worker_index']==process*maximum+assignment['slot'],'Exact persistent two/eight-client routing')
            game_proofs.append(_legacy.audit_game(plan,spec,game,folder/'games'/game['game_id'],journals[process],assignment))
        for process,warmup in enumerate(warmups):
            actual=[a for a in assignments if a['process_index']==process]
            c.require(actual and warmup['calls'][-1]['clock']['ended_perf_counter']<min(a['assigned_perf_counter'] for a in actual),
                'Every admitted service warmed up before its first game dispatch')
            session=c.read(Path(play_folders[process])/'session_result.json')
            expected_outcomes={p['game_id']:p['outcome'] for p,a in zip(game_proofs,assignments) if a['process_index']==process}
            c.require(session['status']=='passed' and session['drained'] is True and session['errors']==[] and
                session['retries']==session['replacements']==0 and
                len(session['outcomes'])==len(expected_outcomes) and
                {o['game_id']:o for o in session['outcomes']}==expected_outcomes,'Every worker returned exactly its actual game outcomes')
            gate=c.read(c.OUT/'qualification_gate.json')
            c.require(c.read(Path(play_folders[process])/'qualification_gate_binding.json')==gate,'Exact original pre-model qualification gate')
        manifest=data.build_manifest([x['document'] for x in game_proofs],spec['specifications'],route='cpu',cell_id=spec['cell_id'])
        c.require(c.read(folder/'corpus_manifest.json')==manifest,'Complete actual corrected engineering manifest')
        loaded=data.load_corpus(manifest,expected_specs=spec['specifications'],boundary=boundary)
        c.require(loaded==c.read(folder/'loader_result.json'),'Every original loader check reproduced')
        expected_loaded=dict(status='actual_collector_loader_passed',manifest_sha256=c.digest(manifest),games=[x['loaded'] for x in game_proofs],
            game_count=16,eligible_roots=sum(x['loaded']['eligible_roots'] for x in game_proofs),training_updates=0,
            actual_training_loader_used=True,historical_training_loader_used=False,**data.IDENTITY)
        c.require(loaded==expected_loaded,'Independent game proofs reconstruct every loader result')
        target_manifest=c.read(folder/'target_manifest.json')
        c.require(target_manifest==dict(schema='gpu_collector_loaded_targets_v1',**data.IDENTITY,
            corpus=c.artifact(folder/'corpus_manifest.json'),loader=c.artifact(folder/'loader_result.json'),
            targets=[x['target'] for x in game_proofs],count=loaded['eligible_roots'],training_updates=0),'Every exact target artifact accounted')
        cohort=c.read(folder/'cohort.json')
        c.require(cohort['status']=='completed' and cohort['games']==16 and cohort['full_cohort_completed'] is True and
            cohort['excluded_from_every_training_corpus'] is True and cohort['corpus_produced'] is False,'Engineering cohort never released for training')
        from .queueing import inventory
        timing_proof=check_cohort(spec,assignments,game_proofs,c.read(folder/'started.json'),cohort,
            c.read(folder/'queue_outcomes.json'),inventory(folder/'queue.sqlite'),c.read(folder/'assembly/execution.json'))
        replay_proof=_legacy.audit_replays(plan,spec,folder,replay,game_proofs,cohort)
        closures=[closure(plan,p,j) for p,j in zip(play_folders,journals)]+[closure(plan,replay_folder,replay)]
        initializations=[initialization(p,j.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0]) for p,j in zip(play_folders,journals)]
        initializations.append(initialization(replay_folder,replay.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0]))
        accounting=[j.complete_attribution() for j in journals]+[replay.complete_attribution()]
    return dict(status='passed',games=16,game_proofs=game_proofs,model_replay=replay_proof,
        executions=executions,runtime_closures=closures,journals=accounting,initializations=initializations,timing=timing_proof,models=0,new_games=0)


def audit_qualification(plan,worker_folder,*,boundary=lambda:None):
    global _boundary
    _boundary=boundary;folder=Path(worker_folder);execution=verify_execution(folder,expected_mode='qualification')
    directory=folder/'cpu';parity=[]
    with JournalIndex(folder/'runtime/inference.jsonl') as journal:
        entrypoint=_legacy.audit_entrypoint(plan,directory/'entrypoint',journal)
        c.require({p.name for p in (directory/'parity').iterdir() if p.is_dir()}=={x['case_id'] for x in plan['cases']},'Only fixed32 qualification parity cases')
        for case in plan['cases']:
            boundary();spec=dict(game_id=case['case_id'],game_index=case['index'],purpose='engineering_qualification_only')
            row,_,_=_legacy.audit_decision(directory/'parity'/case['case_id'],spec,0,journal,before=case['snapshot'],
                owner='parity:'+case['case_id'],client='parity_'+str(case['slot']),slot=case['slot'])
            expected=case['expected_results']['cpu'];c.verify([expected]);prior=c.read(c.resolve(expected))
            from .identity import exact
            exact(row['result'],prior['result']);exact(row['after'],prior['after'])
            witness=dict(case_id=case['case_id'],actual=c.artifact(directory/'parity'/case['case_id']/'moves/000000.json'),expected=expected,exact_full_result_and_after=True)
            c.require(c.read(directory/'parity'/case['case_id']/'comparison.json')==witness,'Full exact saved parity witness');parity.append(witness)
        loader=plan['qualification']['loader'];cases=loader['cases']+[loader['private_variant']];rows=[]
        c.require({p.name for p in (directory/'loader').iterdir() if p.is_dir()}=={f'{i:02d}' for i in range(16)},'Only fixed16 actual loader decisions')
        from . import qualification
        for index,case in enumerate(cases):
            boundary();spec=qualification.fixture_spec(case,index)
            row,_,_=_legacy.audit_decision(directory/'loader'/f'{index:02d}',spec,0,journal,before=case['snapshot'],owner=f'loader:{index}',client='loader',slot=0)
            rows.append(row)
        rowproof=data.qualify_captured_rows(loader,rows,boundary=boundary)
        c.require(rowproof==c.read(directory/'loader_row_checks.json') and len(rows)==16,'Actual loader counterexamples')
        i=next(i for i,case in enumerate(cases) if case.get('id')=='noble_final_round');first=rows[i]
        last,_,_=_legacy.audit_decision(directory/'loader_terminal',first['spec'],1,journal,before=first['after'],owner='loader:terminal',client='loader',slot=0)
        document=c.read(directory/'fixture_target_game.json')
        c.require(document['initial']==first['before'] and document['final']==last['after'] and document['rows']==[
            c.artifact(directory/'loader'/f'{i:02d}'/'moves/000000.json'),c.artifact(directory/'loader_terminal/moves/000001.json')],
            'Actual full official fixture with exact two original moves')
        full=_legacy.audit_full_loader_fixture(directory,document,first['spec'],[first,last])
        expected=dict(status='qualified',route='cpu',actual_entrypoint_decisions=67,accepted_saved_decisions=32,
            actual_loader_decisions=16,official_fixture_continuation_decisions=1,complete_decisions=116,
            entrypoint=c.artifact(directory/'entrypoint/qualification.json'),parity=parity,
            loader_rows=c.artifact(directory/'loader_row_checks.json'),full_loader=c.artifact(directory/'full_loader_checks.json'),
            actual_collector_and_loader_qualified=True,ordinary_games=0,training_updates=0,no_retry=True)
        c.require(c.read(directory/'qualification.json')==expected,'Exact all116 actual qualification decisions')
        engineering=folder/'engineering';proofs=[]
        _legacy.audit_warmup(plan,engineering,journal)
        c.require({p.name for p in (engineering/'games').iterdir() if p.is_dir()}=={s['game_id'] for s in plan['engineering_games']},'Only two fixed excluded engineering games')
        c.require([p.name for p in sorted((engineering/'assignments').glob('*.json'))]==['00.json','01.json'],'Exactly two engineering assignments')
        for index,spec in enumerate(plan['engineering_games']):
            assignment=c.read(engineering/'assignments'/f'{index:02d}.json')
            c.require(assignment['queue_index']==assignment['worker_index']==assignment['slot']==index and
                assignment['client_id']=='engineering_'+str(index) and assignment['game_id']==spec['game_id'],'Fixed actual engineering assignment')
            proofs.append(_legacy.audit_game(plan,dict(route='cpu'),spec,engineering/'games'/spec['game_id'],journal,assignment))
        manifest=data.build_manifest([p['document'] for p in proofs],plan['engineering_games'],route='cpu',cell_id='engineering')
        c.require(c.read(engineering/'corpus_manifest.json')==manifest,'Actual engineering qualification manifest')
        loaded=data.load_corpus(manifest,expected_specs=plan['engineering_games'],boundary=boundary)
        c.require(loaded==c.read(engineering/'loader_result.json') and loaded['eligible_roots']>=32,'Engineering actual complete-game loader')
        replays=_legacy.audit_replays(plan,dict(specifications=plan['engineering_games']),engineering,journal,proofs,c.read(engineering/'cohort.json'))
        initialized=initialization(folder,journal.db.execute('SELECT MIN(submitted) FROM requests').fetchone()[0])
        closed=closure(plan,folder,journal);accounting=journal.complete_attribution()
    return dict(status='passed',passed=True,complete_decisions=116,engineering_games=2,entrypoint=entrypoint,parity=parity,
        loader_row_checks=rowproof,full_loader=full,runtime_closure=closed,journal=accounting,
        engineering_proofs=proofs,engineering_replays=replays,initialization=initialized,execution=execution,models=0,new_games=0)
