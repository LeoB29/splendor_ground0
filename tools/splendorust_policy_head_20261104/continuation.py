"""Layered, saved-only admission of 48 complete source games.

The first 32-game raw audit is reused under exact preservation. Only the next
16 games, two complete workers and the rejected startup are newly audited.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import psutil
from . import common as c
from tools.splendorust_selfplay_pilot_resume_20261103 import continuation as base

OLD_OUT = base.OLD_OUT
OLD_LAUNCH = base.OLD_LAUNCH
PREVIOUS_OUT = c.ROOT/'outputs/splendorust_selfplay_pilot_resume_20261103_01'
PREVIOUS_OWNER = c.ROOT/'outputs/splendorust_selfplay_pilot_resume_owner_20261103_01'
NEW_OUT = c.ROOT/'outputs/splendorust_selfplay_pilot_resume2_20261103_01'
OLD_FREEZE_SHA = base.OLD_FREEZE_SHA
PREVIOUS_FREEZE_SHA = '859f95b2caaae0b3419c0f6caa0bcc33eef955e253124f068d5c4f47b8b2f87f'
QUALIFIED_FREEZE_SHA = base.QUALIFIED_FREEZE_SHA
ACCEPTANCE_SHA = base.ACCEPTANCE_SHA
COMPLETED_GAMES = 48
PREFIX_NAMES = base.PREFIX_NAMES + ('collect_004', 'collect_005')
FAILED_PREFIX_NAMES = ('collect_006',)
MANIFEST = NEW_OUT/'continuation_manifest.json'
PREFIX_REVIEW = NEW_OUT/'prelaunch/saved_prefix_review.json'


def old_freeze():
    return base.old_freeze()


def previous_freeze():
    c.require(c.file_hash(PREVIOUS_OUT/'freeze.json') == PREVIOUS_FREEZE_SHA,
              'Exact first-continuation freeze')
    value = c.read(PREVIOUS_OUT/'freeze.json')
    c.require(value['source_continuation'] == base.source_continuation() and
              value['job_seconds'] == 72*3600 and value['collection_cutoff_seconds'] == 60*3600,
              'First continuation authenticates the unchanged original32 prefix and old clocks')
    return value


def prefix_game_path(index):
    c.require(type(index) is int and 0 <= index < COMPLETED_GAMES, 'Only complete 48-game prefix')
    root = OLD_OUT if index < 32 else PREVIOUS_OUT
    return root/'games'/f'g{index:03d}'/'game.json'


def prefix_source_root(name):
    c.require(name in PREFIX_NAMES, 'Only seven successfully qualified source lifecycles')
    return OLD_OUT if name in base.PREFIX_NAMES else PREVIOUS_OUT


def source_continuation():
    original = old_freeze(); previous = previous_freeze()
    c.require(c.read(c.ROOT/original['game_schedule']['path']) ==
              c.read(c.ROOT/previous['game_schedule']['path']), 'Same complete fixed source schedule')
    return dict(schema='selfplay_saved_source_prefix_v2', completed_games=48,
        original_freeze=c.artifact(OLD_OUT/'freeze.json'), original_schedule=original['game_schedule'],
        previous_freeze=c.artifact(PREVIOUS_OUT/'freeze.json'), previous_schedule=previous['game_schedule'],
        games=[c.artifact(prefix_game_path(i)) for i in range(48)])


def require_absent(identities):
    return base.require_absent(identities)


def _identities():
    from tools.splendorust_selfplay_pilot_budget_20261102.workflow import identities
    docs = [c.read(PREVIOUS_OUT/n) for n in ('guardian_claim.json','watchdog_claim.json','launch.json',
             'guardian_owner.json','execution.json','watchdog_execution.json')]
    for path in sorted(PREVIOUS_OWNER.rglob('*.json')):
        docs.append(c.read(path))
    for name in ('collect_004','collect_005','collect_006'):
        docs += [c.read(PREVIOUS_OUT/'workers'/name/n) for n in ('initialization.json','cleanup.json')]
    combined = {(x['pid'],x['created']) for x in base._identities()} | identities(docs)
    return [dict(pid=pid,created=created) for pid,created in sorted(combined)]


def verify_stop():
    original = old_freeze(); previous = previous_freeze(); old_stop = base.verify_stop()
    completion=c.read(PREVIOUS_OUT/'completion.json');failure=c.read(PREVIOUS_OUT/'failure.json')
    c.require(completion['status']=='incomplete' and completion['completed_games']==48 and
        completion['preserved_source_games']==32 and completion['new_source_games']==16 and
        completion['freeze_sha256']==PREVIOUS_FREEZE_SHA and completion['automatic_retries']==0 and
        completion['coverage'] is None and failure['stage']=='collection' and
        'Entrypoint startup failure' in failure['error'], 'Exact incomplete startup-failure stop')
    c.require({p.name for p in (PREVIOUS_OUT/'games').iterdir()} == {f'g{i:03d}' for i in range(32,48)} and
        {p.name for p in (PREVIOUS_OUT/'workers').iterdir()} == {'collect_004','collect_005','collect_006'},
        'Exactly sixteen complete games and three actual previous workers')
    c.require(not any((PREVIOUS_OUT/n).exists() for n in
        ('fit','fit_execution.json','corpus_manifest.json','replays','independent_review.json')),
        'No downstream fit, replay or acceptance retained')
    for name in ('execution.json','watchdog_execution.json'):
        value=c.read(PREVIOUS_OUT/name)
        c.require(value['exit_code']==1 and not value['cleanup_errors'] and
            not value['cleanup_receipt']['unconfirmed'] and value['freeze_sha256']==PREVIOUS_FREEZE_SHA,
            'Previous actual failed owner exits and clean closure')
    owner_exit=c.read(PREVIOUS_OWNER/'pilot_guardian/exit.json')
    c.require(owner_exit['exit_code']==1 and not owner_exit['cleanup_errors'] and
              owner_exit['freeze_sha256']==PREVIOUS_FREEZE_SHA, 'Retained previous guardian exit')
    for name in ('collect_004','collect_005'):
        closure=c.read(PREVIOUS_OUT/'closed_workers'/f'{name}.json')
        c.require(closure['status']=='passed' and closure['worker']==name,'Two previously completed raw closures')
    for name in ('collect_004','collect_005','collect_006'):
        cleanup=c.read(PREVIOUS_OUT/'workers'/name/'cleanup.json')
        c.require(cleanup['status']=='passed' and not cleanup['alive'] and not cleanup['errors'],
                  'Previous worker owned-process cleanup')
    clock=c.read(PREVIOUS_OUT/'workers/collect_006/entrypoint_clock.json')
    c.require(clock['status']=='worker_error' and clock['accepted'] is False and clock['result'] is None and
        clock['request_id']=='request-00000002' and
        clock['detail']=="ValueError('STRENGTH_EXECUTION: Qualification available RAM below2GiB')",
        'Preserved rejected pre-suite RAM guard; not a failed game move')
    absent=_identities();require_absent(absent)
    claim=c.read(PREVIOUS_OUT/'guardian_claim.json')
    c.require(claim['monotonic_origin']==old_stop['original_monotonic_origin'] and
        claim['unix_origin']==old_stop['original_unix_origin'], 'Original elapsed origins never reset')
    collection=dict(old_stop['original_collection'])
    collection['cap_seconds']=72*3600
    collection['monotonic_deadline']=min(collection['monotonic']+72*3600,claim['monotonic_origin']+108*3600)
    collection['unix_deadline']=min(collection['unix_time']+72*3600,claim['unix_origin']+108*3600)
    forecast=c.read(PREVIOUS_OUT/'block_forecasts/005.json');snapshot=forecast['snapshot']
    waits=[snapshot['cumulative_ram_wait_seconds'],old_stop['prior_ram_wait_seconds']]
    for path in (PREVIOUS_OUT/'memory_admissions').glob('*.json'):
        value=c.read(path)
        for key in ('cumulative_ram_wait_seconds','cumulative_wait_seconds'):
            if key in value:waits.append(value[key])
    execution=c.read(PREVIOUS_OUT/'execution.json')
    tail=max(0.,execution['monotonic_end']-snapshot['monotonic_now'],execution['unix_end']-snapshot['unix_now'])
    return dict(exact_owner_identities_absent=absent,
        original_monotonic_origin=claim['monotonic_origin'], original_unix_origin=claim['unix_origin'],
        original_collection=old_stop['original_collection'], continuation_collection=collection,
        prior_ram_wait_seconds=max(waits)+tail,
        ram_wait_charge=dict(observed_cumulative_seconds=max(waits),conservative_unrecorded_tail_seconds=tail,
            policy='Charge the whole post-last-block interval to include any unrecorded final admission wait'),
        original_completed_games=32, previous_completed_games=16, completed_games=48,
        prior_source_lifecycles=7,preserved_failed_source_lifecycles=1,prior_actual_source_lifecycles=8,
        original_job_seconds=original['job_seconds'], original_model_seconds=original['collection_cutoff_seconds'],
        amended_job_seconds=120*3600, amended_model_seconds=108*3600, clock_windows_amended=True)


def _tree_inventory(root,boundary):
    return base._tree_inventory(root,boundary)


def prepare_continuation(*,boundary=lambda:None):
    c.require(c.SCOPE=='pilot' and c.OUT==NEW_OUT,'Exclusive second continuation output')
    if MANIFEST.exists():return verify_continuation(c.artifact(MANIFEST),boundary=boundary)
    previous=previous_freeze();stop=verify_stop()
    base.verify_prefix_review(previous['saved_prefix_review'],boundary=boundary)
    receipt=c.read(c.ROOT/previous['saved_prefix_review_execution']['path'])
    c.require(receipt['exit_code']==0 and receipt.get('error') is None and receipt['source_unchanged'] is True,
              'Actual successful first32 saved-prefix audit execution')
    trees=[];entries=list(previous['inputs'])
    for root in (OLD_OUT,OLD_LAUNCH,PREVIOUS_OUT,PREVIOUS_OWNER):
        tree,items=_tree_inventory(root,boundary);trees.append(tree);entries.extend(items)
    unique={}
    for item in entries:
        c.require(item['path'] not in unique or unique[item['path']]==item,'Preservation identity conflict')
        unique[item['path']]=item
    value=dict(schema='selfplay_pilot_continuation_manifest_v2',status='both_incomplete_attempts_preserved',
        original_freeze=c.artifact(OLD_OUT/'freeze.json'),previous_freeze=c.artifact(PREVIOUS_OUT/'freeze.json'),
        original_launch_plan=c.artifact(OLD_LAUNCH/'plan.json'),previous_owner_plan=c.artifact(PREVIOUS_OWNER/'plan.json'),
        reused_prefix_review=previous['saved_prefix_review'],reused_prefix_review_execution=previous['saved_prefix_review_execution'],
        source_continuation=source_continuation(),trees=trees,inputs=sorted(unique.values(),key=lambda x:x['path']),
        prior_evidence_bytes=sum(t['bytes'] for t in trees),**stop,next_game_index=48,next_block_index=6,
        failed_decision_retries=0,failed_qualification_repeated_in_new_tree=True,new_starting_checkpoint=False,
        original_status_remains_incomplete=True,previous_status_remains_incomplete=True,future_forecasts_advisory=True)
    c.save(MANIFEST,value)
    return verify_continuation(c.artifact(MANIFEST),boundary=boundary)


def verify_continuation(descriptor,*,boundary=lambda:None):
    c.require(descriptor==c.artifact(MANIFEST),'Exact second continuation manifest')
    value=c.read(MANIFEST);stop=verify_stop();previous=previous_freeze()
    c.require(value['schema']=='selfplay_pilot_continuation_manifest_v2' and
        value['original_freeze']==c.artifact(OLD_OUT/'freeze.json') and
        value['previous_freeze']==c.artifact(PREVIOUS_OUT/'freeze.json') and
        value['source_continuation']==source_continuation() and all(value[k]==v for k,v in stop.items()) and
        value['next_game_index']==48 and value['next_block_index']==6 and value['failed_decision_retries']==0 and
        value['original_status_remains_incomplete'] and value['previous_status_remains_incomplete'] and
        value['future_forecasts_advisory'] and value['new_starting_checkpoint'] is False,
        'Exact two-layer source continuation and explicit amended clocks')
    roots=(OLD_OUT,OLD_LAUNCH,PREVIOUS_OUT,PREVIOUS_OWNER)
    c.require([x['root'] for x in value['trees']]==[p.relative_to(c.ROOT).as_posix() for p in roots],
              'Exactly both original pilot and owner evidence trees')
    indexed={x['path']:x for x in value['inputs']}
    c.require(len(indexed)==len(value['inputs']),'Unique preserved inputs')
    for tree,root in zip(value['trees'],roots,strict=True):
        paths=sorted(p.relative_to(c.ROOT).as_posix() for p in root.rglob('*') if p.is_file())
        c.require(paths==tree['paths'] and len(paths)==tree['files'] and
            sum(indexed[x]['bytes'] for x in paths)==tree['bytes'],'Entire original evidence membership unchanged')
    c.require(value['prior_evidence_bytes']==sum(t['bytes'] for t in value['trees']) and
        all(indexed.get(x['path'])==x for x in previous['inputs']) and
        value['reused_prefix_review']==previous['saved_prefix_review'] and
        value['reused_prefix_review_execution']==previous['saved_prefix_review_execution'],
        'Full prior dependencies, proof reuse and cumulative byte accounting')
    for item in value['inputs']:boundary();c.verify([item])
    return value


class SavedGuard:
    def __init__(self,maximum_seconds=1200):
        claim=c.read(OLD_OUT/'guardian_claim.json')
        self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin']
        self.deadline=min(time.monotonic()+maximum_seconds,self.start+120*3600)
        self.wall_deadline=min(time.time()+maximum_seconds,self.wall_start+120*3600)
        self.placement_sequence=len(list((PREVIOUS_OUT/'placement_requests').glob('*.json')))
        self.last=-float('inf');self.peak_rss=0
    def remaining(self):return min(self.deadline-time.monotonic(),self.wall_deadline-time.time())
    def effective_deadline(self):return time.monotonic()+self.remaining()
    def guard(self):
        c.require(self.remaining()>0,'Bounded saved-only review and amended original-origin global window')
        now=time.monotonic()
        if now-self.last<1:return
        self.last=now;self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
        c.require(self.peak_rss<=2*c.GiB,'Saved-only reader2GiB memory cap')
    def close_worker(self,folder,child=None):
        c.require(child is None and folder.parent==PREVIOUS_OUT/'workers','Saved-only previous worker closure')
        self.guard()


def verify_failed_startup(freeze,guard):
    """Authenticate a rejected startup without converting it to qualification."""
    from tools.splendorust_selfplay_pilot_resume_20261103 import transport as t,adapters
    from tools.splendorust_selfplay_pilot_resume_20261103.worker import verify_execution_identity
    from tools.splendorust_search_scale_20261024 import audit as original_science
    folder=PREVIOUS_OUT/'workers/collect_006';route='candidate'
    logs=t.verify_worker_logs(folder,guard,route)
    init=c.read(folder/'initialization.json');metadata=init['result']
    identity=c.read(folder/'route_identity.json')
    c.require(identity==dict(route=route,module=t.OBSERVER_MODULE,
        native_artifacts=freeze['native_artifacts_by_route'][route],
        native_roles=freeze['native_roles_by_route'][route],metadata=metadata),
        'Rejected startup retains exact READY and native role identity')
    adapters.verify_saved_ready(folder,freeze,route=route)
    execution=verify_execution_identity(folder,freeze,metadata)
    native=adapters.verify_native_configuration_receipts(folder,route,boundary=guard.guard)
    graph=t.verify_graph_evidence(folder,freeze,boundary=guard.guard)
    cleanup=c.read(folder/'cleanup.json')
    c.require(not cleanup['alive'] and not cleanup['errors'] and cleanup['deadline_clamped_to_stage_and_global'],
              'Failed startup cleanup remains clean and bounded')
    require_absent(cleanup['confirmed'])
    ids=['request-00000001','request-00000002']
    c.require([p.stem for p in sorted(folder.glob('request-*.json')) if p.stem[8:].isdigit()]==ids and
        [p.stem for p in sorted((folder/'startup_inputs').glob('*.json'))]==ids and
        not list((folder/'inputs').glob('*.json')) and
        [p.stem for p in sorted((folder/'parent_profiles').glob('*.json'))]==ids and
        [p.stem for p in sorted((folder/'placement_requests').glob('*.json'))]==ids and
        {p.name for p in folder.glob('request-*_child_profile.json')}=={rid+'_child_profile.json' for rid in ids},
        'Exactly graph and failed suite requests, no ordinary decision')
    packets=[];pending=bytearray()
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1]
            packets.append(t._packet_evidence(raw))
            c.require(len(packets)<=3,'No extra raw failed-startup packet')
        c.require(len(pending)<=c.MAX_RESPONSE_BYTES,'Bounded failed startup packet');guard.guard()
    t.verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=c.MAX_LOG_BYTES,max_compressed_bytes=c.MAX_LOG_BYTES*16+128,
        deadline=guard.effective_deadline(),sink=sink)
    c.require(not pending and len(packets)==3 and packets[0]['packet']==dict(type='ready',metadata=metadata),
              'Exact complete READY, graph, error raw stream')
    for index,rid in enumerate(ids,1):
        receipt=c.read(folder/(rid+'.json'));payload=c.read(folder/'startup_inputs'/(rid+'.json'))
        parent=c.read(folder/'parent_profiles'/(rid+'.json'));child=c.read(folder/(rid+'_child_profile.json'))
        packet=packets[index]
        c.require(receipt['request_id']==rid and parent['request_id']==rid and child['request_id']==rid and
            child['payload_sha256']==c.digest(payload) and child['request_raw_sha256']==parent['request_raw_sha256'] and
            parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False and
            0<parent['request_raw_bytes']<=c.MiB, 'Failed-startup complete request identity and bytes')
        t._bind_parent_wire(parent,[packet],completed=receipt['status']=='ok')
        t._bind_child_wire(child,[packet],receipt)
        t.verify_placement_request(folder,rid,freeze,parent)
        if index==1:
            c.require(receipt['status']=='ok' and receipt['accepted'] is True and
                packet['packet']==dict(type='response',request_id=rid,result=receipt['result']) and
                payload==dict(operation='graph_parity',fixtures=c.read(c.ROOT/freeze['graph_parity_plan']['path'])['fixtures']),
                'Complete successful graph request remains accepted')
            t.validate_profiles(parent,child,receipt,payload,route,worker_pid=init['identity']['pid'])
        else:
            error="ValueError('STRENGTH_EXECUTION: Qualification available RAM below2GiB')"
            c.require(payload=={'operation':'entrypoint_checks'} and receipt['status']=='worker_error' and
                receipt['accepted'] is False and receipt['result'] is None and receipt['detail']==error and
                packet['packet'].get('type')=='error' and packet['packet'].get('request_id')==rid and
                packet['packet'].get('error')==error and c.read(folder/'entrypoint_clock.json')==receipt,
                'Exact RAM guard error remains rejected, without a scientific result')
            original_science.check_clock(receipt,limit=300.)
            start,arrival,_=t.boundaries._validate_parent(parent,receipt,completed=False)
            t.research_child_identity(child,parent,receipt,payload,512,route=route,completed=False,
                startup=True,worker_pid=init['identity']['pid'])
            c.require(child['profiles']==[] and child['result_sha256'] is None and
                child['outcome']=='failed_before_successful_emission','No scientific suite handle or result executed')
            t.research_boundaries(child,parent,start,arrival,startup=True)
            profiles=c.read(folder/'entrypoint_profiles.json')
            c.require(profiles['parent_profile']==parent and profiles['child_profile']==child and
                'profile_capture_error' not in profiles and 'placement_error' not in profiles,
                'Failed suite full profile evidence preserved')
    forensic_path=folder/'entrypoint_forensics.jsonl'
    events=[json.loads(line) for line in forensic_path.read_text(encoding='utf-8').splitlines()]
    c.require(len(events)==1 and events[0].get('sequence')==0 and events[0].get('previous_sha256') is None and
        events[0].get('event',{}).get('event')=='entrypoint_failed' and
        events[0]['event']['error']=="ValueError('STRENGTH_EXECUTION: Qualification available RAM below2GiB')" and
        'line 208, in handle' in events[0]['event']['traceback'] and
        'line 199, in boundary' in events[0]['event']['traceback'] and
        not (PREVIOUS_OUT/'qualification/collect_006.json').exists() and not (PREVIOUS_OUT/'games/g048').exists(),
        'Failure precedes all actual counterexample cases and the next source game')
    return dict(status='verified_preserved_rejected_startup',worker='collect_006',graph_cases=11,
        parent_requests=2,scientific_qualification_decisions=0,ordinary_decisions=0,accepted=False,
        logs=logs,graph=graph,native_configuration=native,execution_identity=execution,
        error_clock=c.artifact(folder/'entrypoint_clock.json'),forensics=c.artifact(forensic_path))


def saved_prefix_review(maximum_seconds=1200):
    c.require(c.SCOPE=='pilot' and c.OUT==NEW_OUT and not PREFIX_REVIEW.exists(),'New exclusive layered-prefix audit')
    from tools.splendorust_selfplay_pilot_resume_20261103 import qualification_control as old,transport,data
    guard=SavedGuard(maximum_seconds);guard.guard();manifest=prepare_continuation(boundary=guard.guard)
    freeze=previous_freeze();plan=c.read(c.ROOT/freeze['qualification_plan']['path'])
    specs=c.read(c.ROOT/freeze['game_schedule']['path'])
    prior=base.verify_prefix_review(freeze['saved_prefix_review'],boundary=guard.guard)
    counts=dict(prior['ordinary_request_counts']);counts.update(collect_004=0,collect_005=0)
    games=[];live=0;eligible=0
    for index in range(32,48):
        guard.guard();path=prefix_game_path(index);game=c.read(path);folder=path.parent;name=f'collect_{index//8:03d}'
        c.require([p.name for p in sorted((folder/'moves').glob('*.json'))]==
            [f'{i:06d}.json' for i in range(len(game['rows']))],'Complete contiguous preserved game row inventory')
        for row in game['rows']:
            guard.guard();c.require(c.read(folder/'moves'/f"{row['ply']:06d}.json")==row and
                Path(row['worker_folder'])==Path((PREVIOUS_OUT/'workers'/name).relative_to(c.ROOT)),
                'Exact previous source game, move and worker binding')
            old.validate_row(row,freeze,guard);live+=1;counts[name]+=1
        loaded=data.load_game(game,specs[index],boundary=guard.guard);eligible+=loaded['eligible_roots']
        games.append(dict(game_index=index,plies=loaded['plies'],eligible_roots=loaded['eligible_roots'],game=c.artifact(path)))
        del loaded,game
    for name in ('collect_004','collect_005'):
        old.qualification_evidence(name,guard,freeze,plan,writing=False)
        proof=transport.verify_closed_worker(PREVIOUS_OUT/'workers'/name,freeze,guard)
        c.require(proof==c.read(PREVIOUS_OUT/'closed_workers'/f'{name}.json'),'Previous full raw closure exactly reproduced')
        c.require(len(list((PREVIOUS_OUT/'workers'/name/'inputs').glob('*.json')))==counts[name],
                  'All successful previous ordinary requests belong to the16 complete games')
    failed=verify_failed_startup(freeze,guard)
    old.verify_exchanges(guard,freeze)
    verify_continuation(c.artifact(MANIFEST),boundary=guard.guard);guard.guard()
    c.require('torch' not in sys.modules,'Saved-only prefix audit imports no model runtime')
    return c.save(PREFIX_REVIEW,dict(schema='selfplay_saved_prefix_review_v2',approved=True,
        status='verified_preserved_source_prefix_only',manifest=c.artifact(MANIFEST),
        original_freeze=c.artifact(OLD_OUT/'freeze.json'),previous_freeze=c.artifact(PREVIOUS_OUT/'freeze.json'),
        reused_prefix_review=freeze['saved_prefix_review'],completed_games=48,newly_verified_games=16,
        live_decisions=prior['live_decisions']+live,eligible_roots=prior['eligible_roots']+eligible,
        newly_verified_live_decisions=live,newly_verified_eligible_roots=eligible,
        worker_lifecycles=7,actual_worker_lifecycles=8,preserved_failed_worker_lifecycles=1,
        graph_cases=77,scientific_qualification_decisions=469,capture_fixture_decisions=32,
        preserved_failed_graph_cases=11,preserved_failed_parent_requests=2,failed_startup=failed,
        ordinary_request_counts=counts,games=prior['games']+games,original_attempt_remains_incomplete=True,
        previous_attempt_remains_incomplete=True,models=0,new_games=0,
        source_inputs=[c.artifact(Path(__file__))],peak_rss_bytes=guard.peak_rss,maximum_seconds=maximum_seconds))


def verify_prefix_review(descriptor,*,boundary=lambda:None):
    c.require(descriptor==c.artifact(PREFIX_REVIEW),'Exact layered saved-prefix review')
    value=c.read(PREFIX_REVIEW)
    c.require(value['schema']=='selfplay_saved_prefix_review_v2' and value['approved'] is True and
        value['status']=='verified_preserved_source_prefix_only' and value['completed_games']==48 and
        value['newly_verified_games']==16 and value['worker_lifecycles']==7 and value['actual_worker_lifecycles']==8 and
        value['preserved_failed_worker_lifecycles']==1 and value['graph_cases']==77 and
        value['scientific_qualification_decisions']==469 and value['capture_fixture_decisions']==32 and
        value['preserved_failed_graph_cases']==11 and value['preserved_failed_parent_requests']==2 and
        value['failed_startup']['accepted'] is False and value['models']==value['new_games']==0 and
        value['original_attempt_remains_incomplete'] and value['previous_attempt_remains_incomplete'],
        'Complete layered verification without acceptance of the failed startup')
    c.verify(value['source_inputs']);verify_continuation(value['manifest'],boundary=boundary)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['preserve','review'])
    parser.add_argument('--maximum-seconds',type=int,default=1200);args=parser.parse_args()
    c.require(0<args.maximum_seconds<=1200,'Finite saved-only preparation cap')
    result=saved_prefix_review(args.maximum_seconds) if args.operation=='review' else prepare_continuation()
    print(c.canonical(result))
