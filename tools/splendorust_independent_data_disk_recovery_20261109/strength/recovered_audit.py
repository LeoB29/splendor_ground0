"""Source-aware saved verification of the unchanged, interrupted 400-game study.

Every old row remains bound to its original freeze and owner. The one interrupted
game merges an immutable 26-ply prefix with a newly recorded suffix; no old row,
failure, process identity or actual stage exit is rewritten as a new success.
"""
from pathlib import Path
from . import common as c, lineage, design, analysis
from tools.splendorust_gpu_probe_20261105.control import require_absent


class MergedRows:
    def __init__(self,games,boundary=lambda:None):self.games,self.boundary=games,boundary
    def __iter__(self):
        for game in self.games:
            for ply in range(game['plies']):
                self.boundary();yield c.read(lineage.move_path(game,ply))


def expected_pair(spec, previous, prior_start):
    stage,index=spec['stage'],spec['game_index']
    start=0 if stage=='operational' else index if index%10==0 or previous is not None and previous['reason']=='forfeit' else prior_start
    c.require(type(start) is int and 0<=start<=index,'Prospective worker block start')
    prefix={route:f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS}
    if lineage.partial(spec):
        start=159
    names={route:f'{stage}_g{start:03d}_{route}' for route in c.VARIANTS}
    return start,names,prefix


def game_inventory(spec,ending):
    count=ending['plies'];folder=lineage.game_folder(spec)
    c.require(type(count) is int and 0<count<=c.MAX_PLIES,'Bounded exact game length')
    expected=[f'{n:06d}.json' for n in range(count)]
    if lineage.partial(spec):
        c.require(count>lineage.PREFIX_PLIES,'Interrupted nonterminal game must receive a suffix')
        old=lineage.old_game_folder(spec)
        c.require(not (old/'result.json').exists() and not (old/'online_replay.json').exists(),
                  'Original incomplete game is never relabelled completed')
        c.require([p.name for p in sorted((old/'moves').glob('*.json'))]==expected[:26] and
                  [p.name for p in sorted((folder/'moves').glob('*.json'))]==expected[26:],
                  'Exact 26 immutable old plies and contiguous new suffix; no copied or repeated move')
    else:
        c.require([p.name for p in sorted((folder/'moves').glob('*.json'))]==expected,'Exact contiguous source move inventory')


def read_games(schedule,envelope,freeze):
    from . import control
    games=[];counts={};previous={};starts={}
    for spec in schedule:
        envelope.guard();design.validate_spec(spec)
        folder=lineage.game_folder(spec);initial=design.initial_snapshot(spec)
        c.require(c.read(folder/'spec.json')==spec and c.read(folder/'initial.json')==dict(**spec,initial=initial),
                  'Original reserved setup retained in the correct evidence source')
        ending=c.read(folder/'result.json');game_inventory(spec,ending)
        start,names,prefix=expected_pair(spec,previous.get(spec['stage']),starts.get(spec['stage']))
        starts[spec['stage']]=start
        c.require(c.read(folder/'workers.json')==names,'Worker restart only at frozen block, guard loss or exact interrupted boundary')
        if lineage.partial(spec):
            c.require(c.read(lineage.old_game_folder(spec)/'workers.json')==prefix,'Original interrupted worker pair preserved')
        rows=[]
        for ply in range(ending['plies']):
            row=c.read(lineage.move_path(spec,ply))
            old_prefix=lineage.partial(spec) and ply<26
            source='old' if old_prefix else lineage.game_source(spec)
            name=(prefix if old_prefix else names)[row['variant']]
            c.require(type(row['ply']) is int and row['ply']==ply and lineage.source_of(row)==source,
                      'Exact source and global ply of every completed or failed decision')
            control.validate_row(row,spec,freeze,envelope,worker_name=name)
            key=(source,name);expected=counts.get(key,0)+3
            c.require(row['clock']['request_id']==f'request-{expected:08d}','Exact source-local persistent request sequence')
            counts[key]=expected-2;rows.append(row)
        proof=analysis.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
        c.require(proof==c.read(folder/'online_replay.json'),'Full independent corrected ledger replay of merged game')
        if spec['stage']=='operational':c.require(ending['reason']=='natural','Every original operational game remains natural')
        previous[spec['stage']]=ending;games.append(ending)
    c.require(len(games)==206 and sum(g['plies'] for g in games)<=c.MAX_LIVE_DECISIONS,'Complete fixed sample and original live-decision ceiling')
    return games,MergedRows(games,envelope.guard),counts


def expected_workers(games):
    pairs={};previous={};starts={}
    names={('old','qualification_'+r) for r in c.VARIANTS}|{('new','replay_'+r) for r in c.VARIANTS}
    for game in games:
        spec=design.validate_spec(game)
        start,pair,prefix=expected_pair(spec,previous.get(spec['stage']),starts.get(spec['stage']))
        starts[spec['stage']]=start;source=lineage.game_source(spec)
        pairs[(spec['stage'],spec['game_index'])]=(source,pair,prefix)
        names.update((source,n) for n in pair.values())
        if lineage.partial(spec):names.update(('old',n) for n in prefix.values())
        previous[spec['stage']]=game
    return names,pairs


def worker_segment(name,source):
    if source=='new' and name.startswith('main_g159_'):
        c.require(c.REPETITION==1,'Only repetition1 resumes a partial game')
        return 'resume_partial','resume_partial',None
    if name.startswith('qualification_'):return 'qualification','qualification',None
    if name.startswith('operational_'):return 'operational','operational',None
    if name.startswith('replay_'):return 'replay','replay',None
    c.require(name.startswith('main_g'),'Known live lifecycle')
    index=int(name.split('_')[1][1:]);block=index//10*10
    return f'main_{block:03d}','main_block',block


def segment_bounds(name,source,envelope):
    local,mode,block=worker_segment(name,source)
    base=c.LIVE_SOURCE_OUT if source=='old' else c.OUT
    study=c.parent.LIVE_SOURCE_OUT if source=='old' else c.STUDY_OUT
    start=c.read(base/'segments'/local/'start.json')
    root_name=f'{mode}_r{c.REPETITION}'+(f'_g{block:03d}' if block is not None else '')
    folder=study/'segments'/root_name
    execution=c.read(folder/'execution.json');owner=c.read(folder/'owner.json');claim=c.read(folder/'claim.json')
    failed=source=='old' and c.REPETITION==1 and local=='main_150'
    if failed:
        c.require(execution['status']=='failed' and execution['worker_exit_code']==1 and
                  not (base/'segments'/local/'complete.json').exists(), 'Original failed segment stays failed')
        result=c.read(folder/'result.json')
        c.require(result['status']=='failed' and 'Free disk reserve' in result['error'],'Exact original infrastructure stop retained')
        end=dict(monotonic=execution['ended_monotonic'],unix_time=execution['ended_unix'])
    else:
        c.require(execution['status']=='passed' and execution['worker_exit_code']==0 and execution['error'] is None,
                  'Each completed source segment actually exited successfully')
        end=c.read(base/'segments'/local/'complete.json')
        c.require(end['status']=='segment_complete' and end['start']==c.artifact(base/'segments'/local/'start.json') and
                  end['owner']==start['owner'] and end['resident_workers_after']==0,
                  'Source-local completion binds exact start and closed owner')
    c.require(execution['cleanup_errors']==[] and execution['identity']==owner['identity']==start['owner'],
              'Exact source-local actual owner and clean exit')
    for identity in execution['observed_owned_identities']+[execution['identity']]:require_absent(identity)
    c.verify([start['root_claim'],start['root_phase_clock'],start['segment_claim'],start['local_freeze']])
    c.require(start['root_claim']==c.artifact(study/'claim.json') and start['segment_claim']==c.artifact(folder/'claim.json') and
              start['local_freeze']==c.artifact(base/'freeze.json') and claim['claim']==start['root_claim'] and
              claim['phase_clock']==start['root_phase_clock'] and claim['repetition']==c.REPETITION and
              start['phase']==claim['mode']==mode and start['block_start']==claim['block_start']==block,
              'No substitution of source phase, freeze, repetition or owner')
    phase=c.read(c.ROOT/start['root_phase_clock']['path'])
    c.require(phase['claim']==start['root_claim'] and phase['stage']==('strength_replay' if mode=='replay' else 'strength'),
              'Exact shared original source phase')
    for key,origin in [('monotonic',envelope.start),('unix_time',envelope.wall_start)]:
        deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
        origin_key='original_monotonic_origin' if key=='monotonic' else 'original_unix_origin'
        c.require(start[origin_key]==origin and phase[key]<=start[key]<=end[key]<phase[deadline]==start[deadline] and
                  phase[deadline]<=origin+c.COLLECTION_CUTOFF_SECONDS,
                  'Original clocks cover both preserved and continued evidence')
    return start,end,failed


def verify_admissions(games,pairs,envelope):
    from . import control
    starts={'old':set(),'new':set()}
    for game in games:
        if game['stage']!='main':continue
        source,names,prefix=pairs[('main',game['game_index'])]
        starts[source].update(int(name.split('_')[1][1:]) for name in names.values())
        if lineage.partial(game):starts['old'].update(int(name.split('_')[1][1:]) for name in prefix.values())
    for source,indexes in starts.items():
        base=c.LIVE_SOURCE_OUT if source=='old' else c.OUT
        normal=indexes-({159} if source=='new' and c.REPETITION==1 else set())
        c.require({p.name for p in (base/'block_admissions').glob('*.json')}=={f'g{i:03d}.json' for i in normal},
                  'Exact fixed or classified-loss block admissions in each source')
        for index in normal:
            envelope.guard();proof=c.read(base/'block_admissions'/f'g{index:03d}.json')
            name=f'main_g{index:03d}_control';start,end,_=segment_bounds(name,source,envelope)
            previous=None if index==0 else c.artifact(lineage.game_folder(dict(stage='main',game_index=index-1))/'result.json')
            if source=='old' and index:
                previous=c.artifact(c.LIVE_SOURCE_OUT/f'games/main/g{index-1:03d}/result.json')
            c.require(proof['stage']=='main' and proof['repetition']==c.REPETITION and proof['start_game_index']==index and
                      proof['previous_game']==previous and proof['reason']==('fixed_block' if index%10==0 else 'next_unplayed_after_guard_loss') and
                      proof['no_retried_decision'] is True and proof['both_operational_complete']==control._both_operational_passed() and
                      proof['segment_start']==c.artifact(base/'segments'/start['segment']/'start.json'),
                      'Exact original admission semantics and operational gates')
            if index%10:c.require(c.read(c.ROOT/previous['path'])['reason']=='forfeit','Only a classified prior guard loss allows restart')
            prevkey=('operational',5) if index==0 else ('main',index-1)
            prevsource,prevnames,_=pairs[prevkey]
            prevbase=c.LIVE_SOURCE_OUT if prevsource=='old' else c.OUT
            wanted_closures=[c.artifact(prevbase/'workers'/prevnames[r]/'cleanup.json') for r in c.VARIANTS]
            c.require(proof['previous_closed_workers']==wanted_closures,'Exact previous source-local player pair was closed')
            control.verify_closed_before(proof['previous_closed_workers'],proof['unix_time'])
            for key in ('monotonic','unix_time'):
                deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
                origin='original_monotonic_origin' if key=='monotonic' else 'original_unix_origin'
                c.require(start[key]<=proof[key]<=end[key]<proof['effective_'+deadline]==start[deadline] and proof[origin]==start[origin],
                          'Block remains inside original clock')
            for route in c.VARIANTS:
                init=c.read(base/'workers'/f'main_g{index:03d}_{route}'/'initialization.json')
                c.require(init['identity']['created']>=proof['unix_time'],'Admission precedes worker creation')
    if c.REPETITION==1:
        spec=dict(stage='main',game_index=159);old=lineage.old_game_folder(spec);folder=lineage.game_folder(spec)
        proof=c.read(c.OUT/'resume_admission.json');continued=c.read(folder/'continuation.json')
        prefix=[c.artifact(old/'moves'/f'{i:06d}.json') for i in range(26)]
        old_names=c.read(old/'workers.json');closures=[c.artifact(c.LIVE_SOURCE_OUT/'workers'/old_names[r]/'cleanup.json') for r in c.VARIANTS]
        c.verify(prefix+closures)
        before=c.read(old/'moves/000025.json')['after']
        start,end,_=segment_bounds('main_g159_control','new',envelope)
        c.require(type(proof['repetition']) is int and proof['repetition']==1 and type(proof['game_index']) is int and proof['game_index']==159 and
                  type(proof['start_ply']) is int and proof['start_ply']==26 and proof['prefix']==prefix and proof['before_sha256']==c.digest(before) and
                  proof['source_initial']==c.artifact(old/'initial.json') and proof['source_spec']==c.artifact(old/'spec.json') and
                  proof['previous_closed_workers']==closures and proof['both_operational_complete']==control._both_operational_passed() and
                  proof['segment_start']==c.artifact(c.OUT/'segments/resume_partial/start.json') and
                  proof['active_decision'] is False and proof['failed_decision_retried'] is False and proof['game_replaced'] is False,
                  'Exact unplayed-ply admission, no reset or replacement')
        c.require(continued==dict(source_initial=c.artifact(old/'initial.json'),source_workers=c.artifact(old/'workers.json'),
                  source_freeze=c.artifact(c.LIVE_SOURCE_OUT/'freeze.json'),prefix=prefix,start_ply=26,
                  admission=c.artifact(c.OUT/'resume_admission.json'),original_rows_unchanged=True,original_decisions_retried=0,game_replacements=0),
                  'Full explicit immutable-prefix lineage')
        control.verify_closed_before(closures,proof['unix_time'])
        c.require(c.read(folder/'moves/000026.json')['before']==before,'First new move starts exactly after the last completed old move')
        for key in ('monotonic','unix_time'):
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            origin='original_monotonic_origin' if key=='monotonic' else 'original_unix_origin'
            c.require(start[key]<=proof[key]<=end[key]<proof['effective_'+deadline]==start[deadline] and proof[origin]==start[origin],
                      'Resume admission keeps original clocks')
        for route in c.VARIANTS:
            init=c.read(c.OUT/'workers'/f'main_g159_{route}'/'initialization.json')
            c.require(init['identity']['created']>=proof['unix_time'],'Resume admission precedes fresh qualified trees')
    else:
        c.require(not (c.OUT/'resume_admission.json').exists(),'No invented partial game in repetition0')


def verify_saved(root_envelope,rootfreeze,*,writing=False):
    from . import control,transport
    from tools.splendorust_independent_data_recovery_20261109.strength import control as oldcontrol,transport as oldtransport
    envelope=root_envelope if isinstance(root_envelope,control.AssessmentEnvelope) else control.AssessmentEnvelope(root_envelope)
    freeze=c.read(c.OUT/'freeze.json');oldfreeze=lineage.old_freeze();oldenv=lineage.old_boundary(envelope)
    c.require(rootfreeze==c.read(c.STUDY_OUT/'freeze.json'),'Exact separately frozen recovery')
    control.verify_runtime(freeze,envelope);oldcontrol.verify_runtime(oldfreeze,oldenv)
    for route in c.VARIANTS:
        c.require(freeze['checkpoint_bindings'][route]['weights_sha256']==oldfreeze['checkpoint_bindings'][route]['weights_sha256'],
                  'Same exact saved model weights across the interruption')
    specs,plan=control.load_inputs(freeze,envelope);oldspecs,oldplan=oldcontrol.load_inputs(oldfreeze,oldenv)
    c.require(specs==oldspecs and plan==oldplan,'Unchanged schedules and scientific qualification')
    old_segments={'qualification','operational'}|{f'main_{i:03d}' for i in range(0,160,10)}
    new_segments={'prepare','replay','saved'}|{f'main_{i:03d}' for i in range(160,200,10)}
    if c.REPETITION==1:new_segments.add('resume_partial')
    for base,wanted in ((c.LIVE_SOURCE_OUT,old_segments),(c.OUT,new_segments)):
        c.require({p.name for p in (base/'segments').iterdir() if p.is_dir()}==wanted,'Exact old and continued local segment inventories')
    for source,base in [('old',c.LIVE_SOURCE_OUT),('new',c.OUT)]:
        for stage in ('operational','main'):
            wanted=range(6) if source=='old' and stage=='operational' else range(160) if source=='old' and stage=='main' else range(159 if c.REPETITION==1 else 160,200) if source=='new' and stage=='main' else []
            c.require({p.name for p in (base/'games'/stage).glob('g*')}=={f'g{i:03d}' for i in wanted},'Exact old/new physical game inventories')
    games,rows,counts=read_games(specs,envelope,freeze);names,pairs=expected_workers(games)
    operational=[g for g in games if g['stage']=='operational'];main=[g for g in games if g['stage']=='main']
    ops=analysis.summarize_operational(operational,MergedRows(operational,envelope.guard),schedule=design.make_schedule('operational'),boundary=envelope.guard)
    c.require(ops==c.read(c.LIVE_SOURCE_OUT/'operational_analysis.json'),'Original operational proof independently unchanged')
    verify_admissions(games,pairs,envelope)
    selection=design.select_replay_sample(rows,maximum=96)
    c.require(selection==c.read(c.OUT/'replay_selection.json'),'Same outcome-blind full merged sample selection')
    replay_counts=control.verify_replays(selection,envelope,freeze,specs)
    replay_admit=c.read(c.OUT/'replay_admission.json')
    c.require(replay_admit==dict(replay_selection=c.artifact(c.OUT/'replay_selection.json'),previous_closed_workers=control.prior_pair_closures('replay',200),
              segment_start=c.artifact(c.OUT/'segments/replay/start.json'),required_seconds=control.replay_required_seconds(selection),forecast_policy='advisory_only',may_stop=False),
              'Original replay admission, blind selection and advisory estimate semantics')
    for name,count in replay_counts.items():counts['new',name]=count
    failures={}
    for game in main:
        if game['reason']=='forfeit':
            row=c.read(lineage.move_path(game,game['plies']-1));key=(lineage.source_of(row),Path(row['worker_folder']).name)
            c.require(key not in failures,'A failed worker is never reused');failures[key]=row
    qualifications={};worker_proofs=[]
    for source in ('old','new'):
        base=c.LIVE_SOURCE_OUT if source=='old' else c.OUT
        wanted={name for s,name in names if s==source}
        for directory in ('workers','qualification','closed_workers','closed_worker_costs','startup_costs'):
            observed={p.name if p.is_dir() else p.stem for p in (base/directory).iterdir()}
            c.require(observed==wanted,'Exact source-local used lifecycle inventory: '+source+'/'+directory)
    c.require(len(names)<=c.MAX_WORKER_LIFECYCLES,'Combined source-local lifecycle ceiling retained')
    for source,name in sorted(names):
        envelope.guard();base=c.LIVE_SOURCE_OUT if source=='old' else c.OUT
        ctl,trn,frz,env=(oldcontrol,oldtransport,oldfreeze,oldenv) if source=='old' else (control,transport,freeze,envelope)
        folder=base/'workers'/name;start,end,failedsegment=segment_bounds(name,source,envelope)
        init=c.read(folder/'initialization.json');place=c.read(folder/'placement_admission.json')['supervisor']
        c.require(frz['binding_created_unix']<=start['unix_time']<=init['identity']['created']<=end['unix_time'] and
                  (place['pid'],place['created'])==(start['owner']['pid'],start['owner']['created']), 'Exact worker belongs to its original segment and model binding')
        deadlines=c.read(folder/'initialization_deadlines.json')
        c.require(deadlines['original_monotonic_origin']==envelope.start and deadlines['original_unix_origin']==envelope.wall_start and
                  deadlines['stage_monotonic_deadline']==start['monotonic_deadline'] and deadlines['stage_unix_deadline']==start['unix_deadline'] and
                  deadlines['absolute_deadline']<=start['monotonic_deadline'],'Every worker retains original dual-clock bounds')
        qualified=ctl.qualification_evidence(name,env,frz,plan,writing=False)
        proof=trn.verify_closed_worker(folder,frz,env)
        c.require(proof==c.read(base/'closed_workers'/f'{name}.json') and proof['raw']['requests']==2+counts.get((source,name),0) and
                  proof['raw']['failed_requests']==int((source,name) in failures),'Exact full raw histories and closure, with no missing/unplayed request')
        if (source,name) in failures:
            row=failures[source,name]
            c.require(row['clock']['request_id']==f"request-{counts[source,name]+2:08d}",'Only last recorded request can lose')
        qualifications[source,name]=qualified
        worker_proofs.append(dict(source=source,worker=name,closed=c.artifact(base/'closed_workers'/f'{name}.json'),original_failed_segment_preserved=failedsegment))
    c.require(c.read(c.LIVE_SOURCE_OUT/'qualification_summary.json')==dict(accepted=True,workers={n:qualifications['old',n] for n in ['qualification_'+r for r in c.VARIANTS]}),
              'Original full qualification gate remains accepted')
    stats=analysis.analyze_main(main,True,boundary=envelope.guard)
    proof=analysis.verify_main(main,stats,boundary=envelope.guard)
    control.save_or_verify(c.OUT/'main_analysis.json',stats,writing);control.save_or_verify(c.OUT/'main_analysis_verification.json',proof,writing)
    timing=analysis.summarize_timing(rows,boundary=envelope.guard);control.save_or_verify(c.OUT/'timing_analysis.json',timing,writing)
    coverage=dict(status='verified_independent_data_vs_parent128_assessment',repetition=c.REPETITION,workers=len(names),graph_cases=11*len(names),
        scientific_qualification_decisions=67*len(names),completed_games=206,operational_games=6,main_games=200,main_pairs=100,
        live_decisions=sum(g['plies'] for g in games),exact_reproductions=len(selection['decisions']),classified_main_guard_losses=len(failures),
        original_failed_attempts_preserved=True,source_aware_worker_proofs=worker_proofs,partial_game_prefix_plies=26 if c.REPETITION==1 else 0,
        original_completed_main_games=159 if c.REPETITION==1 else 160,automatic_retries=0,game_replacements=0,strength_promotion=False,
        training=False,strength_and_timing_separate=True,all_raw_science_and_clocks_verified=True)
    control.save_or_verify(c.OUT/'verification_summary.json',coverage,writing)
    complete=c.read(c.OUT/'collection_completion.json')
    c.require(complete==dict(status='assessment_collected_and_reproduced',games=206,replays=len(selection['decisions']),repetition=c.REPETITION,
              freeze=c.artifact(c.OUT/'freeze.json'),replay_segment=c.artifact(c.OUT/'segments/replay/complete.json')),'Exact full merged sample and replay receipt')
    control.verify_runtime(freeze,envelope);oldcontrol.verify_runtime(oldfreeze,oldenv)
    return coverage
