"""Saved-only two-segment verification; interrupted evidence never becomes closed."""
from copy import deepcopy
from pathlib import Path
import psutil
from . import common as c,audit,schedule,analysis
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.splendorust_search256_strength_20261029_02 import runner as oldrunner,transport as oldtransport
from tools.splendorust_search256_recovery_20261030 import verification as previous,runner as previousrunner,transport as previoustransport

OLD_OUT=c.ROOT/'outputs/splendorust_search256_strength_20261029_02'
OLD_SHA='f85737b248e274e6be780461001b4294a1e937cccb70e78ac0e3afcd43878e53'
PREVIOUS_OUT=c.ROOT/"outputs/splendorust_search256_recovery_20261030_01"
PREVIOUS_SHA="af5174b5659914751776d636bf49cd13d7818cfaed88b4b161daa685da522a52"
CONTINUATION_GAME=294
PREFIX_PLIES=50
SEALED_DIRECTORIES=('workers','games','entrypoint_checks','closed_workers','closed_worker_costs',
    'phase_verification','phase_seals','repeated_decisions','replay_samples','placement_requests','placement_acks','memory_admissions',
    'recovery_forecasts')


class OldBoundary:
    def __init__(self,envelope):
        self.envelope=envelope;self.start=envelope.start;self.wall_start=envelope.wall_start
        self.end=envelope.end;self.placement_mode='isolated'
        self.placement_sequence=len(list((OLD_OUT/'placement_requests').glob('*.json')))
    def guard(self):return self.envelope.guard()
    def effective_deadline(self):return self.envelope.effective_deadline()
    def remaining(self):return self.envelope.remaining()
    def close_worker(self,folder,child=None):
        c.require(child is None and Path(folder).resolve().is_relative_to((OLD_OUT/'workers').resolve()),'Read-only old worker boundary')
        self.guard()


def save_or_verify(path,value,writing):
    if path.exists():c.require(c.read(path)==value,'Saved recovery verification differs: '+str(path))
    else:c.require(writing,'Missing saved verification');c.save(path,value)
    return value


class PreviousBoundary(OldBoundary):
    def __init__(self,envelope):
        super().__init__(envelope)
        self.placement_sequence=len(list((PREVIOUS_OUT/'placement_requests').glob('*.json')))
    def close_worker(self,folder,child=None):
        c.require(child is None and Path(folder).resolve().is_relative_to((PREVIOUS_OUT/'workers').resolve()),'Read-only previous recovery boundary')
        self.guard()


def load_context(freeze,envelope):
    envelope.guard();c.verify([freeze['recovery_manifest']]);manifest=c.read(c.ROOT/freeze['recovery_manifest']['path'])
    for key,value in dict(completed_main_games=294,operational_games=6,continuation_game_index=294,continuation_ply=50,
        original_attempt_status='incomplete',original_monotonic_origin=c.PRIOR_ORIGIN,original_unix_origin=c.PRIOR_WALL_START,
        prior_worker_trees=608,old_worker_trees=486,previous_worker_trees=120,new_worker_limit=260,maximum_combined_worker_trees=868,
        prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES,authorized_retries=0,
        historical_disclosed_retries=1).items():c.require(manifest[key]==value,'Exact clean continuation boundary '+key)
    c.require(manifest['main_monotonic_deadline']==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
        manifest['main_unix_deadline']==c.ORIGINAL_MAIN_UNIX_DEADLINE,'Original main deadline cannot restart')
    for key in ('old_freeze','original_claim','original_main_start','original_tree','original_main_admission','original_qualification_summary',
        'previous_tree','previous_freeze','previous_manifest','previous_failure','previous_completion','previous_execution',
        'previous_watchdog_execution','previous_retry','previous_retry_consumption'):c.verify([manifest[key]])
    c.require(manifest['old_freeze']==c.artifact(OLD_OUT/'freeze.json') and manifest['old_freeze']['sha256']==OLD_SHA,'Exact original02 freeze')
    c.require(manifest['previous_freeze']==c.artifact(PREVIOUS_OUT/'freeze.json') and
        manifest['previous_freeze']['sha256']==PREVIOUS_SHA,'Exact previous recovery freeze')
    priorfreeze=c.read(PREVIOUS_OUT/'freeze.json');priorboundary=PreviousBoundary(envelope)
    priorcontext=previous.load_context(priorfreeze,priorboundary)
    c.require(manifest['original_tree']==priorcontext['manifest']['original_tree'],'Original93k-file inventory retained unchanged')
    inventory=c.read(c.ROOT/manifest['previous_tree']['path']);pins={x['path']:x for x in inventory['files']}
    c.require(len(pins)==len(inventory['files'])==inventory['file_count']==23163 and
        sum(x['bytes'] for x in pins.values())==inventory['total_bytes']==1043690409,'Exact complete previous recovery inventory')
    c.require(all((c.ROOT/p).resolve().is_relative_to(PREVIOUS_OUT.resolve()) for p in pins),'Previous inventory is scoped to frozen recovery01')
    for identity in inventory['owner_identities_absent']:
        try:present=psutil.Process(identity['pid']).create_time()==identity['created']
        except psutil.NoSuchProcess:present=False
        c.require(not present,'An original or previous owning identity is still active')
    schedules=c.read(c.ROOT/freeze['schedule']['path'])
    c.require(schedules==priorcontext['schedules']==c.read(PREVIOUS_OUT/'schedule.json'),'Same complete406-game frozen schedule')
    context=dict(manifest=manifest,old_freeze=priorcontext['old_freeze'],previous_freeze=priorfreeze,
        previous_context=priorcontext,inventory=priorcontext['inventory'],inventory_map=priorcontext['inventory_map'],
        previous_inventory=inventory,previous_inventory_map=pins,schedules=schedules,
        schedule=schedules['operational']+schedules['main'],old_boundary=priorcontext['old_boundary'],previous_boundary=priorboundary)
    verify_previous_stop(context,envelope)
    return context


def source_game_folder(spec):
    root=OLD_OUT if spec['stage']=='operational' or spec['game_index']<235 else PREVIOUS_OUT if spec['game_index']<294 else c.OUT
    return root/'games'/spec['stage']/spec['candidate']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"


def source_move_path(spec,ply,context=None):
    game=spec['game_index']
    root=(OLD_OUT if spec['stage']=='operational' or game<235 or game==235 and ply<30 else
        PREVIOUS_OUT if game<294 or game==294 and ply<50 else c.OUT)
    return root/'games'/spec['stage']/spec['candidate']/f"p{spec['pair_index']:03d}_g{game}"/'moves'/f'{ply:04d}.json'


def verify_prior_admission(context,envelope):
    old=context['old_freeze'];proxy=context['old_boundary'];schedules=context['schedules']
    plan=c.read(c.ROOT/old['qualification_plan']['path']);qualified=oldrunner.verify_qualification(proxy,old,plan)
    ops,_,_=oldrunner.read_games(schedules['operational'],proxy,old)
    c.require(len(ops)==6 and all(g['reason']=='natural' for g in ops),'Retain six qualified operational games')
    gate=oldrunner.verify_resource_forecast(schedules['operational'],proxy)
    permit=c.read(OLD_OUT/'main_admission.json');c.require(permit['bindings']==oldrunner.main_admission_bindings() and permit['sample_size']==400 and permit['no_optional_stopping'],
        'Original exact fixed400 admission')
    oldrunner.validate_main_admission_time(permit,gate['inputs'],c.read(OLD_OUT/'stages/operational_end.json'),c.read(OLD_OUT/'stages/main_start.json'))
    for stage in ('qualification','operational'):
        start=c.read(OLD_OUT/'stages'/f'{stage}_start.json');end=c.read(OLD_OUT/'stages'/f'{stage}_end.json')
        c.require(start['monotonic']<=end['monotonic']<start['monotonic_deadline'] and start['unix_time']<=end['unix_time']<start['unix_deadline'],'Prior completed stage clocks')
    prior=context['previous_context'];previous.verify_prefix(prior,context['previous_boundary'])
    previous_qualification=previous.verify_recovery_qualification(prior,context['previous_boundary'],context['previous_freeze'])
    previous.read_games([schedules['main'][235]],context['previous_boundary'],context['previous_freeze'],prior)
    return dict(status='passed',qualification=qualified,previous_qualification=previous_qualification,operational_games=6,
        original_main_admission=context['manifest']['original_main_admission'],original_attempt_accepted=False,
        historical_retry_verified=True,new_retries=0,previous_attempt_accepted=False,
        source='unchanged original02 qualification and operational evidence plus previous qualified235 recovery seam')


def verify_prefix(context,envelope):
    spec=context['schedules']['main'][294];folder=PREVIOUS_OUT/'games/main/research/p147_g294'
    initial=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    c.require(c.read(folder/'initial.json')==dict(**spec,initial=initial) and not(folder/'result.json').exists(),'Exact stopped game294 initial state')
    paths=sorted((folder/'moves').glob('*.json'));c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(50)],'Exactly50 accepted prefix files')
    rows=[]
    for path in paths:
        envelope.guard();pin=context['previous_inventory_map'][path.relative_to(c.ROOT).as_posix()];c.verify([pin]);row=c.read(path)
        previousrunner.verify_saved_decision(row,context['previous_freeze']);rows.append(row)
    next_snapshot=audit.validate_prefix_chain(initial,rows,spec,boundary=envelope.guard)
    closures=[]
    for role in ('challenger','opponent'):
        worker=PREVIOUS_OUT/'workers'/f'main_research_g294_{role}'
        closure=previoustransport.verify_closed_worker(worker,context['previous_freeze'],context['previous_boundary'])
        c.require(closure==c.read(PREVIOUS_OUT/'closed_workers'/f'{worker.name}.json')==c.read(folder/f'closed_{role}.json'),
            'Current previous worker has authentic full clean closure')
        ids=[f'request-{i:08d}' for i in range(2,27)]
        c.require(sorted(p.stem for p in(worker/'inputs').glob('*.json'))==ids and
            [r['clock']['request_id'] for r in rows if r['player_role']==role]==ids and closure['raw']['requests']==26,
            'Exactly25 accepted ordinary requests per role and graph; no request27')
        c.require(not any((worker/p).exists() for p in ('request-00000027.json','inputs/request-00000027.json',
            'parent_profiles/request-00000027.json','request-00000027_child_profile.json','placement_requests/request-00000027.json')),
            'Next request27 never acquired input, clock, profile or dispatch receipt')
        closures.append(dict(role=role,proof=closure,receipt=c.artifact(PREVIOUS_OUT/'closed_workers'/f'{worker.name}.json')))
    result=dict(status='passed_clean_between_requests',initial=initial,rows=rows,next_snapshot=next_snapshot,spec=spec,
        move_artifacts=[c.artifact(p) for p in paths],original_initial=c.artifact(folder/'initial.json'),closed_workers=closures,
        next_request_never_started=True,new_retries=0)
    context['prefix']=result;return result


def expected_prefix_manifest(prefix,context):
    return dict(schema='search256_clean_saved_game_prefix_v1',previous_initial=prefix['original_initial'],previous_moves=prefix['move_artifacts'],
        previous_last_ply=49,continuation_first_ply=50,previous_next_request_id='request-00000027',previous_next_request_started=False,
        previous_closed_workers=[c.artifact(PREVIOUS_OUT/'closed_workers'/f'main_research_g294_{role}.json')
            for role in ('challenger','opponent')],new_decision_retries=0)


def audit_merged_game(initial,rows,ending,spec,context,freeze,boundary=lambda:None):
    if spec['game_index']!=294:return audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=boundary)
    prefix=context['prefix'];c.require(initial==prefix['initial'],'Exact previous game294 initial state')
    seam=audit.validate_segment_seam(rows,prefix['rows'],spec)
    proof=audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=boundary)
    return dict(proof,recovery_seam=seam,previous_attempt_accepted=False,clean_between_requests=True,new_move_retries=0)


def verify_live_row(row,context,freeze):
    from . import runner
    folder=(c.ROOT/row['worker_folder']).resolve()
    if folder.is_relative_to((OLD_OUT/'workers').resolve()):return oldrunner.verify_saved_decision(row,context['old_freeze'])
    if folder.is_relative_to((PREVIOUS_OUT/'workers').resolve()):return previousrunner.verify_saved_decision(row,context['previous_freeze'])
    c.require(folder.is_relative_to((c.OUT/'workers').resolve()),'Unknown execution-segment worker')
    return runner.verify_saved_decision(row,freeze)


def read_games(specs,envelope,freeze,context=None):
    from . import runner,transport
    context=load_context(freeze,envelope) if context is None else context
    games=[];paths=[];proofs=[]
    for spec in specs:
        envelope.guard()
        if spec['stage']=='operational' or spec['game_index']<294:
            oldgames,oldrows,oldproofs=previous.read_games([spec],context['previous_boundary'],context['previous_freeze'],context['previous_context'])
            games.extend(oldgames);paths.extend(oldrows.paths);proofs.extend(oldproofs);continue
        folder=source_game_folder(spec);ending=c.read(folder/'result.json')
        initial=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        c.require(c.read(folder/'initial.json')==dict(**spec,initial=initial),'Original prospective initial seed')
        start=50 if spec['game_index']==294 else 0
        suffix=sorted((folder/'moves').glob('*.json'))
        c.require([p.name for p in suffix]==[f'{i:04d}.json' for i in range(start,ending['plies'])],'Exact new segment move filenames')
        gamepaths=[source_move_path(spec,ply,context) for ply in range(ending['plies'])]
        rows=[c.read(p) for p in gamepaths]
        for row in rows:envelope.guard();verify_live_row(row,context,freeze)
        if start:
            if 'prefix' not in context:verify_prefix(context,envelope)
            c.require(c.read(folder/'prefix_manifest.json')==expected_prefix_manifest(context['prefix'],context),'Exact immutable old prefix references')
            proof=audit_merged_game(initial,rows,ending,spec,context,freeze,envelope.guard)
            runner.verify_clean_continuation(context,freeze,envelope,rows[50])
        else:proof=audit.audit_game(initial,rows,ending,spec,freeze=freeze,boundary=envelope.guard)
        c.require(proof==c.read(folder/'online_replay.json'),'Exact saved complete-game dictionary proof')
        if ending['reason']=='forfeit':
            saved=c.read(folder/'captured_fault.json');expected=runner.captured_fault(spec,rows[-1])
            c.require({k:v for k,v in saved.items() if k!='time'}=={k:v for k,v in expected.items() if k!='time'},'Research-guard attribution preserved')
        else:c.require(not(folder/'captured_fault.json').exists(),'Hidden administrative forfeit')
        for role in ('challenger','opponent'):
            worker=c.OUT/'workers'/f"main_research_g{spec['game_index']}_{role}"
            ids=[r['clock']['request_id'] for r in rows[start:] if r['player_role']==role]
            c.require(len(ids)==len(set(ids)) and set(ids)=={p.stem for p in(worker/'inputs').glob('*.json')},'Complete new ordinary-request inventory')
            closure=transport.verify_closed_worker(worker,freeze,envelope)
            c.require(closure==c.read(c.OUT/'closed_workers'/f'{worker.name}.json')==c.read(folder/f'closed_{role}.json'),'Exact new clean worker closure')
            proofs.append(closure)
        games.append(ending);paths.extend(gamepaths)
    return games,runner.SavedRows(paths,envelope.guard),proofs


def verify_previous_stop(context,envelope):
    """A clean resource stop is neither a player loss nor an ambiguous retry."""
    from .preparation_support import previous_exit_provenance,ram_wait_accounting
    provenance=previous_exit_provenance();ram=ram_wait_accounting()
    c.require(ram['charged_prior_ram_wait_seconds']==c.PRIOR_RAM_WAIT_SECONDS,'Conservative complete prior RAM-wait charge')
    failure=c.read(PREVIOUS_OUT/'failure.json');completion=c.read(PREVIOUS_OUT/'completion.json')
    c.require(failure['stage']=='main' and failure['error'].endswith('ValueError: Cumulative frozen RAM wait exceeded\n') and
        'envelope.admit()' in failure['error'],'Exact saved between-request RAM admission failure')
    c.require(completion['status']=='incomplete' and completion['failure']==failure['error'] and
        completion['newly_completed_games']==59 and completion['retained_completed_main_games']==235 and
        completion['freeze_sha256']==PREVIOUS_SHA and completion['automatic_retries']==0 and completion['disclosed_retry_recorded'],
        'Previous incomplete59-game segment and historical single retry preserved')
    for name in ('execution.json','watchdog_execution.json'):
        item=c.read(PREVIOUS_OUT/name)
        c.require(item['exit_code']==1 and item['freeze_sha256']==PREVIOUS_SHA and not item['cleanup_errors'] and
            item['monotonic_origin']==c.PRIOR_ORIGIN and item['unix_origin']==c.PRIOR_WALL_START and
            not item['cleanup_receipt']['unconfirmed'],'Previous actual failed owners and confirmed cleanup')
    c.require(not any((PREVIOUS_OUT/name).exists() for name in ('independent_review.json','stages/main_end.json')),
        'Prior incomplete main segment cannot become accepted')
    boundary=context['manifest']['boundary']
    c.require(boundary['game_index']==294 and boundary['ply']==50 and boundary['route']=='frozen' and
        boundary['new_request_id']=='request-00000003','Only exact never-started opponent move may continue')
    return dict(status='passed',resource_stop_preserved=True,previous_attempt_accepted=False,previous_actual_owner_exit_codes=[1,1,1],
        next_decision_never_started=True,historical_disclosed_retries=1,new_retries=0,exit_provenance=provenance,ram_wait_accounting=ram)


def verify_prior_preservation(context,envelope):
    """Every original file, including unclosed streams and unresolved input."""
    original=previous.verify_prior_preservation(context['previous_context'],context['previous_boundary'])
    pins=context['previous_inventory_map']
    actual=set()
    for path in PREVIOUS_OUT.rglob('*'):
        envelope.guard()
        if path.is_file():actual.add(path.relative_to(c.ROOT).as_posix())
    c.require(actual==set(pins),'No omitted, added, or discarded original evidence file')
    for item in pins.values():envelope.guard();c.verify([item])
    for item in context['manifest']['diagnosis_tree']:envelope.guard();c.verify([item])
    return dict(status='passed',original=original,previous_inventory=context['manifest']['previous_tree'],files=len(pins),
        bytes=context['previous_inventory']['total_bytes'],diagnosis_files=len(context['manifest']['diagnosis_tree']),
        original_attempt_status='incomplete',original_actual_exit_codes_unknown=True,
        unclosed_logs_preserved=True,unresolved_input_preserved=True,previous_attempt_status='incomplete',previous_ram_stop_preserved=True)


def verify_recovery_qualification(context,envelope,freeze):
    from . import qualification
    plan=c.read(c.ROOT/freeze['qualification_plan']['path']);receipts=[]
    c.require(plan==qualification.build_plan(),'Unchanged complete actual suite plan')
    for role,route in (('challenger','research'),('opponent','frozen')):
        folder=c.OUT/'workers'/f'main_research_g294_{role}'
        receipt=c.read(c.OUT/'entrypoint_checks'/f'{route}.json')
        c.require(receipt==c.read(folder/'entrypoint_clock.json') and receipt['request_id']=='request-00000002',
            'Retained replacement tree exact second startup request')
        proof=qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=envelope.guard)
        old=c.read(OLD_OUT/'entrypoint_checks'/f'{route}.json')
        c.require(old['status']=='ok' and c.digest(receipt['result'])==c.digest(old['result']),
            'Every unchanged full67 suite output equals its original qualified route')
        c.require({p.stem for p in(folder/'startup_inputs').glob('*.json')}=={'request-00000001','request-00000002'},
            'Only graph and full suite on retained tree')
        receipts.append(dict(route=route,clock=c.artifact(c.OUT/'entrypoint_checks'/f'{route}.json'),
            verification=proof,old_clock=c.artifact(OLD_OUT/'entrypoint_checks'/f'{route}.json')))
    result=dict(status='passed',scientific_decisions=134,parent_requests=4,graph_cases=22,
        worker_trees=2,retained_for_game294=True,original_qualification_reused=True,repeats_of_failed_decisions=0,receipts=receipts)
    c.require(c.read(c.OUT/'recovery_qualification_summary.json')==result,'Complete saved replacement qualification')
    c.require({p.name for p in(c.OUT/'entrypoint_checks').glob('*.json')}=={'frozen.json','research.json'},'Exact replacement suite inventory')
    return result


def verify_segment_admission(context,envelope,root,engine,first,last,prefix,prior_files):
    original=c.read(OLD_OUT/'assessment_gate.json')['inputs'];forecasts={}
    c.require({p.stem for p in(root/'recovery_forecasts').glob('*.json')}=={'preflight','qualified'},'Exactly two prospective outcome-blind resource gates')
    for name in ('preflight','qualified'):
        value=c.read(root/'recovery_forecasts'/f'{name}.json');inputs=value['inputs']
        c.require(value==dict(engine.recovery_projection(inputs),inputs=inputs) and value['passed'],'Independently recomputed frozen segment forecast')
        c.require(all(inputs[k]==original[k] for k in ('game_costs','worker_costs','hash_seconds')) and
            inputs['old_assessment_gate']==c.artifact(OLD_OUT/'assessment_gate.json') and inputs['prior_files']==prior_files and
            inputs['runtime_seconds']==c.read(root/'source_runtime_measurement.json')['seconds'],'Full original outcome-blind costs and prior inventory')
        forecasts[name]=inputs
    permit=c.read(root/'recovery_main_admission.json')
    expected=dict(freeze=c.artifact(root/'freeze.json'),qualification=c.artifact(root/'recovery_qualification_summary.json'),
        forecast=c.artifact(root/'recovery_forecasts/qualified.json'),original_main_admission=c.artifact(OLD_OUT/'main_admission.json'),
        retained_completed_games=first,retained_prefix_moves=prefix,remaining_games=400-first,
        maximum_disclosed_retries=1 if first==235 else 0,automatic_retries=0,no_optional_stopping=True)
    c.require(set(permit)==set(expected)|{'time','monotonic','unix_time','perf_counter_ns'} and
        all(permit[k]==v for k,v in expected.items()),'Exact frozen segment admission before game decisions')
    end=c.read(root/'stages/recovery_qualification_end.json');start=c.read(root/'stages/main_start.json')
    for key in ('monotonic','unix_time'):
        c.require(forecasts['preflight'][key]<=end[key]<=start[key]<=forecasts['qualified'][key]<=permit[key],
            'Preflight, qualification, original main stage and admission chronology')
    transition=c.read(root/'retained_worker_phase_transition.json')
    c.require(transition['stage']=='main' and transition['original_main_deadline']==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
        transition['original_main_unix_deadline']==c.ORIGINAL_MAIN_UNIX_DEADLINE and
        start['monotonic']<=transition['monotonic']<=permit['monotonic'] and start['unix_time']<=transition['unix_time']<=permit['unix_time'],
        'Retained idle trees transition within original stage clock')
    c.require(len(transition['workers'])==2 and {r['role'] for r in transition['workers']}=={'challenger','opponent'},'Exactly two retained qualified trees')
    for item in transition['workers']:
        folder=root/'workers'/f"main_research_g{first}_{item['role']}";init=c.read(folder/'initialization.json')
        c.require(item['worker']==folder.name and item['pid']==init['identity']['pid'] and item['request_counter']==2 and
            item['active_request'] is False and item['absolute_deadline']<=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
            init['identity']['created']>forecasts['preflight']['unix_time'],'No active clock reset; creation follows preflight')
    for spec in context['schedules']['main'][first:last+1]:
        envelope.guard();game=spec['game_index']
        if game>first:
            for role in ('challenger','opponent'):
                folder=root/'workers'/engine.game_worker_name(spec,role)
                init=c.read(folder/'initialization.json');admit=c.read(folder/'placement_admission.json')
                c.require(init['identity']['created']>permit['unix_time'] and
                    admit['supervisor']['monotonic_ns']>permit['perf_counter_ns'],'Worker created after qualified admission')
        path=root/'games/main/research'/f"p{spec['pair_index']:03d}_g{game}"/'moves'/f'{prefix if game==first else 0:04d}.json'
        row=c.read(path);c.require(row['parent_profile']['request_start_ns']>permit['perf_counter_ns'],'All segment decisions follow qualified admission')
    return dict(status='passed',admission=c.artifact(root/'recovery_main_admission.json'),original_main_deadline_not_reset=True,
        checked_game_indices=[first,last],outcome_blind_forecasts=[c.artifact(root/'recovery_forecasts'/f'{n}.json') for n in ('preflight','qualified')])


def verify_recovery_admission(context,envelope):
    from . import runner
    oldpermit=c.read(OLD_OUT/'main_admission.json')
    for spec in context['schedules']['main'][:236]:
        for role in ('challenger','opponent'):
            envelope.guard();folder=OLD_OUT/'workers'/runner.game_worker_name(spec,role)
            oldrunner.validate_main_startup(oldpermit,c.read(folder/'initialization.json'),c.read(folder/'placement_admission.json'))
    return dict(status='passed',previous=verify_segment_admission(context,envelope,PREVIOUS_OUT,previousrunner,235,294,30,context['inventory']['file_count']),
        current=verify_segment_admission(context,envelope,c.OUT,runner,294,399,50,
            context['inventory']['file_count']+context['previous_inventory']['file_count']))


def verify_stage_clocks(envelope):
    begin=c.read(PREVIOUS_OUT/'stages/recovery_qualification_start.json');end=c.read(PREVIOUS_OUT/'stages/recovery_qualification_end.json')
    owner=c.read(PREVIOUS_OUT/'guardian_claim.json');main=c.read(PREVIOUS_OUT/'stages/main_start.json')
    c.require(begin['monotonic']<=end['monotonic']<=main['monotonic'] and begin['unix_time']<=end['unix_time']<=main['unix_time'] and
        end['monotonic']<begin['monotonic_deadline']<=owner['owner_started_monotonic']+7200 and
        end['unix_time']<begin['unix_deadline']<=owner['owner_started_unix_time']+7200 and
        main['monotonic_deadline']==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and main['unix_deadline']==c.ORIGINAL_MAIN_UNIX_DEADLINE,
        'Previous accepted qualification and incomplete main retain original owning clocks')
    original=c.read(OLD_OUT/'stages/main_start.json');ending=c.read(c.OUT/'stages/main_end.json')
    c.require(original['monotonic_deadline']==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
        original['unix_deadline']==c.ORIGINAL_MAIN_UNIX_DEADLINE and
        original['monotonic']<=ending['monotonic']<original['monotonic_deadline'] and
        original['unix_time']<=ending['unix_time']<original['unix_deadline'],'Original main allowance includes the entire interruption')
    previous=None
    for stage in ('recovery_qualification','main','main_verification'):
        start=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        for key,origin in (('monotonic',envelope.start),('unix_time',envelope.wall_start)):
            deadline='monotonic_deadline' if key=='monotonic' else 'unix_deadline'
            c.require(start[key]<=end[key]<start[deadline]<=min(start[key]+c.STAGES[stage],origin+c.COLLECTION_CUTOFF_SECONDS),
                'Original nonresettable stage/model deadline '+stage)
            if previous is not None:c.require(previous[key]<=start[key],'Sequential stage chronology')
        if stage in ('recovery_qualification','main'):
            c.require(start['monotonic_deadline']<=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE and
                start['unix_deadline']<=c.ORIGINAL_MAIN_UNIX_DEADLINE,'Recovery cannot extend original main deadline')
        if stage=='recovery_qualification':
            claim=c.read(c.OUT/'guardian_claim.json')
            c.require(start['monotonic_deadline']<=claim['owner_started_monotonic']+7200 and
                start['unix_deadline']<=claim['owner_started_unix_time']+7200,'Replacement qualification includes all owning preparation time')
        previous=end
    final=c.read(c.OUT/'stages/final_start.json')
    c.require(previous['monotonic']<=final['monotonic'] and previous['unix_time']<=final['unix_time'] and
        final['monotonic_deadline']<=envelope.start+c.JOB_SECONDS and final['unix_deadline']<=envelope.wall_start+c.JOB_SECONDS,
        'Saved-only final begins after all model work under original global deadline')
    return dict(status='passed',original_main_deadline=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,
        original_model_deadline=envelope.start+c.COLLECTION_CUTOFF_SECONDS,original_global_deadline=envelope.start+c.JOB_SECONDS)


def reconcile_execution_counts(ordinary,parent_requests,replaycount,newworkers,closedworkers):
    c.require(all(type(x) is int and x>=0 for x in (ordinary,parent_requests,replaycount,newworkers,closedworkers)) and replaycount<=48,
        'Finite exact whole-sample execution counts')
    c.require(parent_requests==ordinary+872+2*replaycount,'Exact three-segment graph/suite/ordinary request reconciliation')
    c.require(newworkers==212+replaycount and closedworkers==816+replaycount and 608+newworkers<=868,
        'Complete closure count including preserved interrupted and replaced workers')
    return True


def verify_saved_attempt(envelope,freeze,*,writing,context=None):
    """Complete original, disclosed-retry and clean-continuation saved readback."""
    from . import runner,transport
    context=load_context(freeze,envelope) if context is None else context
    runner.verify_runtime(freeze,envelope)
    preserved=verify_prior_preservation(context,envelope);stopped=verify_previous_stop(context,envelope)
    prior=verify_prior_admission(context,envelope);prefix=verify_prefix(context,envelope)
    verify_recovery_qualification(context,envelope,freeze);admission=verify_recovery_admission(context,envelope)
    schedules=context['schedules'];schedule.validate_schedule(context['schedule'])
    ops,oprows,_=oldrunner.read_games(schedules['operational'],context['old_boundary'],context['old_freeze'])
    main,mainrows,phase=runner.verify_phase('main',schedules['main'],envelope,freeze,context,run_models=False)
    result=analysis.analyze_main(main,True,boundary=envelope.guard);checked=analysis.verify_main(main,result,boundary=envelope.guard)
    sensitivity=analysis.analyze_recovery_sensitivity(main,result,boundary=envelope.guard)
    sensitivity_checked=analysis.verify_recovery_sensitivity(main,result,sensitivity,boundary=envelope.guard)
    for name,value in (('main_analysis',result),('main_analysis_verification',checked),('recovery_sensitivity',sensitivity),
        ('recovery_sensitivity_verification',sensitivity_checked),('prior_preservation_verification',preserved),
        ('prior_admission_verification',prior),('previous_stop_verification',stopped),('recovery_admission_verification',admission)):
        save_or_verify(c.OUT/(name+'.json'),value,writing)
    def folders(root,specs):
        return {root/'games'/s['stage']/s['candidate']/f"p{s['pair_index']:03d}_g{s['game_index']}" for s in specs}
    for root,expected in ((OLD_OUT,folders(OLD_OUT,schedules['operational']+schedules['main'][:236])),
        (PREVIOUS_OUT,folders(PREVIOUS_OUT,schedules['main'][235:295])),(c.OUT,folders(c.OUT,schedules['main'][294:]))):
        c.require({p.parent for p in(root/'games').rglob('initial.json')}==expected,'No additional or discarded game in any execution segment')
    for directory in ('phase_verification','phase_seals'):
        c.require({p.stem for p in(c.OUT/directory).glob('*.json')}=={'main'},'Exact main verification inventory')
    oldworkers={'qualification_frozen','qualification_research'}|{runner.game_worker_name(s,r)
        for s in schedules['operational']+schedules['main'][:236] for r in ('challenger','opponent')}
    interrupted={'main_research_g235_challenger','main_research_g235_opponent'}
    prevworkers={runner.game_worker_name(s,r) for s in schedules['main'][235:295] for r in ('challenger','opponent')}
    newworkers={runner.game_worker_name(s,r) for s in schedules['main'][294:] for r in ('challenger','opponent')}
    for receipt in phase['reproductions']:newworkers.update(receipt['workers'])
    for root,expected in ((OLD_OUT,oldworkers),(PREVIOUS_OUT,prevworkers),(c.OUT,newworkers)):
        c.require({p.name for p in(root/'workers').iterdir() if p.is_dir()}==expected,'Every worker accounted for exactly in its own segment')
    c.require(len(oldworkers)==486 and len(prevworkers)==120 and len(newworkers)<=260,'Cumulative worker resource limits')
    rows=runner.SavedRows(oprows.paths+mainrows.paths,envelope.guard);games=ops+main
    save_or_verify(c.OUT/'performance.json',audit.performance(rows,games),writing)
    save_or_verify(c.OUT/'boundary_analysis.json',audit.summarize_boundary_rows(rows,boundary=envelope.guard),writing)
    faults=[c.read(source_game_folder(s)/'captured_fault.json') for s in context['schedule'] if(source_game_folder(s)/'captured_fault.json').exists()]
    save_or_verify(c.OUT/'all_player_faults.json',dict(count=len(faults),faults=faults,failed_move_retries=0,
        historical_unresolved_move_disclosed_retries=1,new_move_retries=0,resource_stop_as_player_loss=False,
        old_unresolved_request_classified_as_timeout=False),writing)
    proofs=[];parent_requests=32
    for root,names,engine,frozen,boundary in (
        (OLD_OUT,oldworkers-interrupted,oldtransport,context['old_freeze'],context['old_boundary']),
        (PREVIOUS_OUT,prevworkers,previoustransport,context['previous_freeze'],context['previous_boundary']),
        (c.OUT,newworkers,transport,freeze,envelope)):
        for name in sorted(names):
            envelope.guard();proof=engine.verify_closed_worker(root/'workers'/name,frozen,boundary)
            target=root/'closed_workers'/f'{name}.json';c.require(proof==c.read(target),'Every normal worker closure remains verified')
            parent_requests+=proof['raw']['requests'];proofs.append(c.artifact(target))
    save_or_verify(c.OUT/'all_closed_workers.json',dict(status='passed',normally_closed_workers=len(proofs),proofs=proofs,
        interrupted_workers_not_normalized=2,interrupted_prefix_proofs=context['previous_context']['prefix']['interrupted_workers']),writing)
    save_or_verify(c.OUT/'all_placement_integrity.json',dict(original=oldrunner.verify_exchanges(context['old_boundary'],context['old_freeze']),
        previous=previousrunner.verify_exchanges(context['previous_boundary'],context['previous_freeze']),
        recovery=runner.verify_exchanges(envelope,freeze)),writing)
    save_or_verify(c.OUT/'stage_clock_verification.json',verify_stage_clocks(envelope),writing)
    replaycount=sum(r['requests'] for r in phase['reproductions'])
    reconcile_execution_counts(len(rows),parent_requests,replaycount,len(newworkers),len(proofs))
    coverage=dict(complete=True,operational_games=6,main_games=400,actual_games=406,retained_completed_main_games=294,
        resumed_main_games=1,new_remaining_main_games=105,retained_prefix_moves=50,ordinary_game_decisions=len(rows),
        original_qualification_scientific_decisions=182,previous_qualification_scientific_decisions=134,recovery_qualification_scientific_decisions=134,
        total_verified_qualification_scientific_decisions=450,failed01_scientific_compute_separately_retained=91,
        original_qualification_parent_requests=52,previous_qualification_parent_requests=4,recovery_qualification_parent_requests=4,
        model_reproductions=replaycount,workers=len(newworkers),original02_workers=486,previous_recovery_workers=120,
        normally_closed_workers=len(proofs),preserved_interrupted_workers=2,graph_cases=11*(606+len(newworkers)),parent_requests=parent_requests,
        hard_guard_forfeits=len(faults),failed_decision_retries=0,historical_disclosed_unresolved_move_retries=1,new_move_retries=0,automatic_retries=0,
        training=False,original_attempt_status='incomplete',previous_attempt_status='incomplete',original_exit_codes_unknown=True,
        previous_actual_owner_exit_codes=[1,1,1],original_logs_rewritten=False,unresolved_original_request_invented_as_timeout=False,
        resource_stop_counted_as_player_loss=False,prior_worker_trees=608,combined_worker_trees=608+len(newworkers),maximum_combined_worker_trees=868,
        prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,maximum_cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS,prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES,
        failed01_parent_requests_separately_retained=26,failed01_passed_graph_cases_separately_retained=11,
        official_success_criteria_unchanged=True,historical_retry_sensitivity_reported=True)
    save_or_verify(c.OUT/'coverage.json',coverage,writing);runner.verify_runtime(freeze,envelope)
    return coverage


def final_checks(envelope,freeze,context=None):
    from . import runner
    coverage=verify_saved_attempt(envelope,freeze,writing=True,context=context)
    c.save(c.OUT/'verification_summary.json',dict(status='passed',coverage=coverage,source_inputs=len(freeze['inputs']),
        peak_combined_rss_bytes=envelope.peak_combined_rss,all_raw_placement_logs_graph_science_sources_verified=True,
        original_interrupted_closures_preserved_separately=True))
    names=('recovery_qualification_summary.json','source_runtime_measurement.json','recovery_main_admission.json',
        'retained_worker_phase_transition.json','clean_continuation_consumed.json','clean_continuation.json',
        'main_analysis.json','main_analysis_verification.json','recovery_sensitivity.json','recovery_sensitivity_verification.json',
        'prior_preservation_verification.json','prior_admission_verification.json','previous_stop_verification.json','recovery_admission_verification.json',
        'performance.json','boundary_analysis.json','all_player_faults.json','all_closed_workers.json',
        'all_placement_integrity.json','stage_clock_verification.json','coverage.json','verification_summary.json')
    paths=[c.OUT/name for name in SEALED_DIRECTORIES]+[c.OUT/name for name in names]
    c.save(c.OUT/'completed_evidence_seal.json',dict(artifacts=runner.evidence_seal(paths,envelope),
        immutable_prior_inventory=freeze['recovery_manifest']))
    return coverage
