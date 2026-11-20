"""Saved-only acceptance of the explicitly amended, segmented fixed sample.

Original failure/receipts are immutable. The one saved-response adoption is a
distinct event, never a fabricated successful clock or reproduction target.
All byte authentication uses the original bounded eight-thread implementation.
No model, search, environment transition, bootstrap or experimental test runs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import psutil

from tools.splendorust_search128_strength_20261028 import independent_review as old
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_search128_resume_20261029_01'
FIRST_RECOVERY=ROOT/'outputs/splendorust_search128_recovery_20261028_01'
FIRST_RECOVERY_SHA='524daba942dd6478ccbfc2a9293f391b1502ffca94163228aef8d1d3dba718f4'
CLOSURE=ROOT/'outputs/splendorust_search128_closure_check_20261029_01'
CLOSURE_SHA='647f06bdca298e4b1fcc4b4bdff205906c422707e4a1684f23e2fe409dc17dad'
ORIGINAL=ROOT/'outputs/splendorust_search128_strength_20261028_01'
ORIGINAL_SHA='9026f33a3a6dcc449bb28f025493825230f04d91298f21c2c5bb2382b69587c9'
ORIGIN_MONO,ORIGIN_UNIX=474768.187,1793218912.6661534
PRIOR_FINAL_SECONDS,PRIOR_RAM_WAIT_SECONDS=190.,200.
LIMIT=3600.
GiB=1024**3
ROLES=old.ROLES
POLICY=old.POLICY
require,canonical,digest=old.require,old.canonical,old.digest
expected_game,check_clock,selection=old.expected_game,old.check_clock,old.selection
distribution,check_statistics=old.distribution,old.check_statistics
save_new,identity_receipt,verify_identity_receipt=old.save_new,old.identity_receipt,old.verify_identity_receipt


class Review(ReviewRuntime):
    def __init__(self,seconds=LIMIT):super().__init__(ROOT,OUT,LIMIT,seconds=seconds)


class OriginalView:
    """Read old qualification with its unchanged verifier and explicit root."""
    def __init__(self,reader):self.reader=reader
    def read(self,name):return self.reader.read_path(ORIGINAL/name)
    def __getattr__(self,name):return getattr(self.reader,name)


def bound_read(r,item):
    r.verify(item)
    return r.read_path(r.physical(item['path']))


def expected_lifecycles():
    return {f'qualification_resume_{route}':dict(route=route,game_index=39,
        graph_request_id='request-00000001',entrypoint_request_id='request-00000002',
        ordinary_first_request_id='request-00000003') for route in ('research','frozen')}


def source_authentication(r,frozen,expected_sha):
    require(r.pin(OUT/'freeze.json')['sha256']==expected_sha,'Exact recovery freeze')
    pins={x['path']:x for x in frozen['inputs']}
    require(len(pins)==len(frozen['inputs']),'Unique full recovery source pins')
    for path in (Path(__file__),Path(old.__file__),Path(old.__file__).with_name('review_runtime.py')):
        key=path.relative_to(ROOT).as_posix()
        require(key in pins,'Frozen actual reader/helper implementation')
        r.verify(pins[key])
    require(sys.version==frozen['python'] and r.pin(Path(sys.executable))['sha256']==frozen['python_executable_sha256'],
            'Actual pinned reader runtime')


def bind_shared_clock(r,sha):
    claim=r.read('guardian_claim.json');final=r.read('stages/final_start.json')
    clock=r.read('postrun/review_owner_clock.json');completion=r.read('completion.json')
    require(claim['freeze_sha256']==clock['freeze_sha256']==sha and
            (claim['monotonic_origin'],claim['unix_origin'])==(ORIGIN_MONO,ORIGIN_UNIX),
            'Original dual guardian origins remain unchanged')
    require(final['cap_seconds']==14400-PRIOR_FINAL_SECONDS and completion['status']=='amended_recovered_verified',
            'Final allowance carries190 seconds already consumed')
    require(final['monotonic_deadline']==min(final['monotonic']+14400-PRIOR_FINAL_SECONDS,ORIGIN_MONO+86400) and
            final['unix_deadline']==min(final['unix_time']+14400-PRIOR_FINAL_SECONDS,ORIGIN_UNIX+86400),
            'No reset of original global cap or used finalization budget')
    mono=min(clock['review_monotonic_origin']+LIMIT,final['monotonic_deadline'],ORIGIN_MONO+86400)
    wall=min(clock['review_unix_origin']+LIMIT,final['unix_deadline'],ORIGIN_UNIX+86400)
    require(clock['deadline_monotonic']==mono and clock['deadline_unix']==wall and
            completion['independent_acceptance_deadline_monotonic']==final['monotonic_deadline'] and
            completion['independent_acceptance_deadline_unix']==final['unix_deadline'],'Shared review and acceptance deadline')
    require(clock['guardian_claim_sha256']==r.pin(OUT/'guardian_claim.json')['sha256'] and
            clock['completion_sha256']==r.pin(OUT/'completion.json')['sha256'],'Owner clock binds exact terminal evidence')
    require(clock['review_monotonic_origin']>=final['monotonic'] and clock['review_unix_origin']>=final['unix_time'],
            'Review starts after resumed final stage')
    r.shared_deadlines=(mono,wall);r.guard()
    return clock


def verify_bridge_bundle(r,manifest):
    bundle=bound_read(r,manifest['bridge']);bridge=bundle['bridge'];row=bundle['source_row']
    for item in bridge['consumed_inputs']:r.verify(item)
    actual=r.read_path(ORIGINAL/'games/main/p019_g38/moves/0039.json')
    require(row==actual and row['completed'] is False and row['clock']['accepted'] is False and
            row['clock']['status']=='resource_guard' and row['clock']['result'] is None and
            row['clock']['receipt_elapsed_seconds'] is None and row['after'] is None,
            'Original failed receipt is never promoted to successful')
    require(bridge['schema']=='splendorust_search128_saved_response_recovery_v1' and
            bridge['disposition']=='eligible_saved_response_adoption_under_explicit_amendment' and
            bridge['original_freeze_sha256']==ORIGINAL_SHA and bridge['original_attempt_status']=='incomplete' and
            bridge['original_clock_status']=='resource_guard' and bridge['original_clock_accepted'] is False and
            bridge['original_attempt_accepted'] is False and bridge['original_receipt_rewritten'] is False and
            bridge['original_row_rewritten'] is False and bridge['decision_clock_restarted'] is False and
            bridge['new_model_calls']==bridge['new_search_calls']==0 and bridge['recovered_transition_count']==1 and
            bridge['original_game_prefix_transitions']==39 and bridge['resume_ply']==40,
            'Explicit forensic amendment, not original acceptance or retry')
    require(bridge['result_sha256']==digest(bundle['result']) and bridge['action_sha256']==digest(bundle['result']['action']) and
            bridge['before_sha256']==digest(bundle['before']) and bridge['after_sha256']==digest(bundle['after']) and
            bridge['original_clock_sha256']==digest(row['clock']) and bundle['before']==row['before'],
            'Byte-bound saved science/transition bridge')
    timing=bridge['timing_evidence'];parent=row['parent_profile'];child=row['child_profile']
    require(parent['raw_arrivals_ns']==[timing['observed_reader_arrival_ns']] and parent['dropped_arrival_events']==0 and
            timing['original_start_ns']==parent['request_start_ns'] and
            timing['observed_complete_response_seconds']==(timing['observed_reader_arrival_ns']-timing['original_start_ns'])/1e9 and
            0<=timing['observed_complete_response_seconds']<=10 and
            timing['normalized_packet_sha256']==child['serialized_packet_utf8_sha256'] and
            timing['normalized_packet_bytes']==child['serialized_packet_utf8_bytes'] and
            child['result_sha256']==digest(bundle['result']) and child['outcome']=='response_flushed' and child['error'] is None,
            'Original observed packet timing and child byte binding retained')
    require(bridge['original_parent_response_hash_available'] is False and parent['response_raw_sha256'] is None,
            'No invented parent-consumed packet hash')
    return bundle


def validate_game_chain(spec,initial,ending,events,bundle):
    """Independently check the full saved chain without redoing transitions."""
    require(all(initial.get(k)==v and ending.get(k)==v for k,v in spec.items()) and ending['complete'] is True and
            type(ending['plies']) is int and 0<ending['plies']<=2000 and ending['plies']==len(events),
            'Complete fixed-schedule game')
    current=initial['initial'];ordinary=[];failed=None;bridges=0
    for ply,event in enumerate(events):
        require(failed is None and not current['state']['terminal'],'No event after an ending')
        if event['kind']=='recovery_bridge':
            require((spec['stage'],spec['game_index'],ply)==('main',38,39) and bridges==0 and
                    current==bundle['before'] and event['value']==bundle,
                    'One exact recovered ply39, never a second search/clock')
            current=bundle['after'];bridges+=1
            continue
        require(event['kind']=='ordinary','Known event kind')
        row=event['value'];role=row['player_role']
        require(row['before']==current and row['ply']==ply and all(row[k]==spec[k] for k in ('stage','pair_index','game_index')) and
                row['seat']==current['state']['current_player'] and role==('challenger' if row['seat']==spec['challenger_seat'] else 'opponent') and
                row['mode']==row['route']==ROLES[role] and row['quota']==(128 if role=='challenger' else 64) and row['arm']=='search' and
                row['placement_mode']=='isolated' and row['observation_mode']=='profiled' and
                not row.get('profile_capture_error') and not row.get('placement_error') and not row.get('transition_error'),
                'Exact ordinary identity/contiguous chain')
        fault=check_clock(row);ordinary.append(row)
        if fault:
            require(ply==len(events)-1,'Hard guard ends current game');failed=(row,fault)
        else:current=row['after']
    require(current==ending['final'],'Exact saved final state')
    if failed:
        row,kind=failed
        require(spec['stage']=='main' and ending['reason']=='forfeit' and ending['administrative_forfeit'] is True and
                ending['failure_policy']==POLICY and ending['failure']==kind and ending['loser']==ending['failed_seat']==row['seat'] and
                ending['winner']==1-row['seat'] and ending['failed_role']==row['player_role'],'Attributable9/10s guard loss only')
    else:
        require(ending['reason']=='natural' and current['state']['terminal'] and ending['winner']==current['state']['winner'] and
                ending.get('administrative_forfeit',False) is False and
                all(ending.get(k) is None for k in ('loser','failed_role','failed_seat','failure','failure_policy')),
                'Natural game without administrative attribution')
    require(bridges==int(spec['stage']=='main' and spec['game_index']==38),'Only scheduled game38 carries recovered event')
    return dict(ordinary=ordinary,failed=failed,bridges=bridges,final_sha256=digest(current),
                transitions=len(events)-int(failed is not None))


def owner_checks(r,frozen,sha,completion,identities):
    execution=r.read('execution.json');outer=r.read('watchdog_execution.json');os_exit=r.read('guardian_os_exit.json')
    guardian=r.read('guardian_claim.json');owner=r.read('guardian_owner.json');watchdog=r.read('watchdog.json')
    claim=r.read('watchdog_claim.json');launch=r.read('launch.json')
    require(all(x['freeze_sha256']==sha for x in (execution,outer,os_exit,guardian,watchdog,claim,launch)), 'Owning freeze chain')
    require(owner['guardian']==guardian and owner['watchdog']==dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']) and
            (launch['pid'],launch['created'])==(watchdog['runner_pid'],watchdog['runner_created']) and
            (claim['watchdog_pid'],claim['watchdog_created'])==(watchdog['watchdog_pid'],watchdog['watchdog_created']) and
            (claim['guardian_pid'],claim['guardian_created'])==(guardian['guardian_pid'],guardian['guardian_created']), 'Owning PID/create chain')
    require((guardian['monotonic_origin'],guardian['unix_origin'])==(ORIGIN_MONO,ORIGIN_UNIX), 'Original24h origins')
    require(os_exit['guardian_pid']==guardian['guardian_pid'] and abs(os_exit['guardian_created']-guardian['guardian_created'])<=.0001 and
            os_exit['exit_code']==0 and os_exit['observed_by']=='owned Process handle' and os_exit['cleanup_reason'] is None,'Retained guardian OS exit0')
    for end in (execution,outer):
        require(end['status']=='completed' and end['exit_code']==0 and end['cleanup_errors']==[] and
                end['cleanup_receipt']['errors']==end['cleanup_receipt']['unconfirmed']==[], 'Actual clean owner exit')
        charged=max(end['monotonic_end']-end['monotonic_origin'],end['unix_end']-end['unix_origin'])
        require(abs(charged-end['charged_job_seconds'])<1e-6 and charged<86400 and
                (end['monotonic_origin'],end['unix_origin'])==(ORIGIN_MONO,ORIGIN_UNIX),'Dual global original clock, including pause')
    require(outer['observed_process']=='owning_watchdog' and outer['exit_code_observed_by']=='guardian Popen handle' and outer['error'] is None,
            'Actual watchdog handle exit')
    for stage,cap in (('recovery_qualification',1800),('main',72000),('final',14400-PRIOR_FINAL_SECONDS)):
        start=r.read('stages/'+stage+'_start.json');end=r.read('stages/'+stage+'_end.json')
        cutoff=86400 if stage=='final' else 72000
        require(start['cap_seconds']==cap and start['monotonic_deadline']==min(start['monotonic']+cap,ORIGIN_MONO+cutoff) and
                start['unix_deadline']==min(start['unix_time']+cap,ORIGIN_UNIX+cutoff) and
                start['monotonic']<=end['monotonic']<start['monotonic_deadline'] and
                start['unix_time']<=end['unix_time']<start['unix_deadline'], 'Original shared stage deadlines')
    launcher=r.read('launcher_placement.json')
    require(launcher['freeze_sha256']==sha and launcher['logical_processors']==list(range(2,20)) and launcher['priority']=='Normal' and
            launcher['pid']==os_exit['owner_pid'],'Retained owner placement')
    identities.extend([dict(pid=guardian['guardian_pid'],created=guardian['guardian_created']),
        dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']),dict(pid=launch['pid'],created=launch['created'])])
    for root in (ORIGINAL,FIRST_RECOVERY):
        for name,key in (('owner_launch.json',None),('guardian_claim.json','guardian'),('watchdog_claim.json','watchdog'),('launch.json',None)):
            prior=r.read_path(root/name)
            identities.append(dict(pid=prior['pid'] if key is None else prior[key+'_pid'],
                                   created=prior['created'] if key is None else prior[key+'_created']))
    unique={(x['pid'],x['created']):x for x in identities}
    return dict(actual_experiment_os_exit_codes=dict(runner=0,watchdog=0,guardian=0),
        owned_experiment_identities_absent=[r.absent(x) for x in unique.values()],
        launch_owner_pid=launcher['pid'],launch_owner_actual_exit_required_after_acceptance=True,
        maximum_charged_seconds=max(execution['charged_job_seconds'],outer['charged_job_seconds']))


def authenticate_all(r,frozen):
    jobs=[(r.physical(x['path']),x) for x in frozen['inputs']]
    for name in ('completed_evidence_seal.json','analysis_seal.json'):
        jobs.extend((r.physical(x['path']),x) for x in r.read(name)['artifacts'])
    for key in ('torch_runtime','external_runtime'):
        jobs.extend((Path(x['absolute_path']),x) for x in frozen[key])
    jobs.extend((Path(x['path']),x) for x in frozen['compression_runtime']['files'])
    r.verify_many(jobs)


def review_all(r,expected_sha):
    frozen=r.read('freeze.json');source_authentication(r,frozen,expected_sha)
    for name in ('execution.json','watchdog_execution.json','guardian_os_exit.json'):
        terminal=r.read(name)
        require(terminal['exit_code']==0 and terminal['freeze_sha256']==expected_sha,'All actual experiment exits before review')
    completion=r.read('completion.json');verification=r.read('verification_summary.json');coverage=r.read('coverage.json')
    require(completion['status']=='amended_recovered_verified' and completion['failure'] is None and completion['freeze_sha256']==expected_sha and
            completion['automatic_retries']==0,'Verified amended collection/finalization prerequisite')
    require(frozen['schema']=='splendorust_search128_three_segment_resume_v1' and frozen['original_freeze']['sha256']==ORIGINAL_SHA and
            frozen['recovery1_freeze']['sha256']==FIRST_RECOVERY_SHA and frozen['closure_check']['sha256']==CLOSURE_SHA and
            frozen['main_games']==400 and frozen['operational_games']==6 and frozen['seconds_per_move']==10 and frozen['internal_seconds']==9 and
            frozen['job_seconds']==86400 and frozen['collection_cutoff_seconds']==72000 and frozen['max_workers']==864 and
            frozen['max_evidence_bytes']==16*GiB and frozen['final_reserve_bytes']==4*GiB and frozen['prefinal_evidence_bytes']==12*GiB and
            frozen['prior_final_seconds']==PRIOR_FINAL_SECONDS and frozen['prior_ram_wait_seconds']==PRIOR_RAM_WAIT_SECONDS and
            frozen['automatic_retries']==frozen['failed_decision_retries']==0 and frozen['source_attempts_remain_incomplete'] is True and
            frozen['original_failed_receipt_overwritten'] is False and frozen['training'] is False,'Frozen amended scope and original bounds')
    require(frozen['declared_worker_lifecycles']==expected_lifecycles() and frozen['continuation_game_index']==39 and
            frozen['completed_game_retries']==0 and frozen['qualified_prior_decisions']==322 and
            frozen['new_entrypoint_decisions']==119 and frozen['combined_qualification_decisions']==441,
            'Exact qualified first-game lifecycle; no completed-game replay')
    authenticate_all(r,frozen)
    pins={x['path']:x for x in frozen['inputs']}
    manifest=bound_read(r,frozen['resume_manifest']);old_freeze=bound_read(r,frozen['original_freeze'])
    bound_read(r,frozen['recovery1_freeze'])
    require(manifest['old_freeze']==frozen['original_freeze'] and manifest['original_attempt_status']=='incomplete' and
            manifest['recovery1_freeze']==frozen['recovery1_freeze'] and manifest['closure_check']==frozen['closure_check'] and
            manifest['first_recovery_status']=='incomplete' and
            manifest['original_failed_receipt_preserved'] is True and manifest['new_call_for_failed_move'] is False and
            manifest['move_clock_reset'] is False and manifest['completed_main_games']==39 and
            manifest['original_completed_main_games']==38 and manifest['continuation_game_index']==39,
            'Prospective preservation/adoption authorization')
    for root,key in ((ORIGINAL,'original_tree'),(FIRST_RECOVERY,'recovery1_tree'),(CLOSURE,'closure_tree')):
        require(all(pins.get(x['path'])==x for x in manifest[key]) and len(manifest[key])==manifest[key+'_files'] and
                sum(x['bytes'] for x in manifest[key])==manifest[key+'_bytes'],'Entire prior evidence byte inventory')
        r.exact_tree(root,[x['path'] for x in manifest[key]])
    for root in (ORIGINAL,FIRST_RECOVERY):
        old_completion=r.read_path(root/'completion.json')
        require(old_completion['status']=='incomplete' and not(root/'postrun/independent_completion_review.json').exists(),
                'Both historical incomplete attempts stay incomplete')
        for name in ('execution.json','watchdog_execution.json','guardian_os_exit.json'):
            require(r.read_path(root/name)['exit_code']==1,'Prior actual failure exits remain unchanged')
    closure=review_previous_closure(r,frozen,manifest)
    bundle=verify_bridge_bundle(r,manifest)
    seal=r.read('completed_evidence_seal.json')['artifacts'];sealmap={x['path']:x for x in seal}
    require(len(sealmap)==len(seal),'Unique final evidence seal')
    for name in ('workers','games','repeated_decisions','closed_workers','closed_worker_costs','placement_acks','placement_requests','recovery_entrypoint_checks','prior_saved_proofs'):
        prefix=(OUT/name).relative_to(ROOT).as_posix()+'/'
        r.exact_tree(OUT/name,[p for p in sealmap if p.startswith(prefix)])
    analysis_seal=r.read('analysis_seal.json')['artifacts']
    require({x['path'] for x in analysis_seal}=={(OUT/n).relative_to(ROOT).as_posix() for n in ('analysis.json','analysis_verification.json')},
            'Exact primary/independent statistics seal')
    require(verification['status']=='passed' and verification['all_raw_placement_logs_graph_science_sources_verified'] is True and
            verification['source_inputs']==len(pins) and verification['coverage']==completion['coverage']==coverage,
            'All saved scientific/transport/source verification')
    require(coverage['complete_planned_workload'] is True and coverage['planned_games']==coverage['verified_games']==406 and
            coverage['operational_games']==6 and coverage['main_games']==400 and coverage['unplayed_games']==0,
            'Complete fixed sample, no result-dependent replacement')
    schedule=[expected_game(i,stage) for stage,count in (('operational',6),('main',400)) for i in range(count)]
    require(r.read_path(ORIGINAL/'schedule.json')==schedule,'Unchanged independent fresh schedule')
    games_manifest=r.read('game_evidence_manifest.json')['games'];segments=r.read('segments_manifest.json')
    require(len(games_manifest)==406 and [x['spec'] for x in games_manifest]==schedule,'All exact406 game manifest entries')
    require(segments['original_freeze']==frozen['original_freeze'] and segments['recovery1_freeze']==frozen['recovery1_freeze'] and
            segments['recovery_freeze']==r.pin(OUT/'freeze.json') and segments['closure_check']==frozen['closure_check'] and
            segments['ordinary_timing_excludes_bridge'] is True,'Three explicit immutable execution segments')
    saved_bundle=bound_read(r,segments['bridge']);require(saved_bundle==bundle,'Exactly frozen saved response bridge')
    failed_original=bound_read(r,segments['original_failed_request'])
    require(failed_original==bundle['source_row'],'Original rejected row/clock separately represented')
    old_worker_ids={};qualification=old.qualification_review(OriginalView(r),old_freeze,old_worker_ids)
    byworker={(ORIGINAL/'workers'/name).relative_to(ROOT).as_posix():ids for name,ids in old_worker_ids.items()}
    worker_routes={path:path.rsplit('_',1)[-1] for path in byworker}
    games=[];ordinary_count=0;timing_acc=TimingAccumulator();administrative=[];selected={};selected_rows={};bridge_count=0;events_count=0
    replay=r.read('all_game_replay.json');faults=r.read('all_player_faults.json')
    require(replay['status']=='passed' and replay['games']==len(replay['receipts'])==406 and faults['failed_move_retries']==0,
            'Passed all406 independent ledger replay receipts')
    for index,entry in enumerate(games_manifest):
        r.guard();spec=entry['spec'];initial=bound_read(r,entry['initial']);ending=bound_read(r,entry['result'])
        events=[]
        for ply,event in enumerate(entry['events']):
            require(event['source_freeze_sha256'] in (ORIGINAL_SHA,FIRST_RECOVERY_SHA,expected_sha),'Explicit source freeze per event')
            if event['kind']=='recovery_bridge':
                require(event['source_freeze_sha256']==ORIGINAL_SHA and event['bridge']==segments['bridge'],'Single original bridge source')
                events.append(dict(kind='recovery_bridge',value=bound_read(r,event['bridge'])))
            else:
                require(event['kind']=='ordinary','Known game event')
                path=r.physical(event['row']['path']);expected_root={ORIGINAL_SHA:ORIGINAL,FIRST_RECOVERY_SHA:FIRST_RECOVERY,expected_sha:OUT}[event['source_freeze_sha256']]
                require(path.is_relative_to(expected_root/'games'),'Event artifact belongs to its declared execution segment')
                scheduled_root=(ORIGINAL if spec['stage']=='operational' or spec['game_index']<38 or spec['game_index']==38 and ply<39
                                else FIRST_RECOVERY if spec['game_index']==38 else OUT)
                require(expected_root==scheduled_root,'Exact prior38 games, saved game39 and unplayed361 continuation')
                row=bound_read(r,event['row']);events.append(dict(kind='ordinary',value=row))
                require(r.read_path(r.physical(row['worker_clock_file']))==row['clock'],'Actual saved original request receipt')
                folder=row['worker_folder'];require(r.physical(folder).is_relative_to(expected_root/'workers'),'Worker source segment')
                byworker.setdefault(folder,[]).append(row['clock']['request_id']);worker_routes[folder]=row['route']
        checked=validate_game_chain(spec,initial,ending,events,bundle)
        proof=bound_read(r,entry['online_replay'])
        if spec['stage']=='main' and spec['game_index']==38:
            require(entry['supplemental_proof_source']==frozen['closure_check'] and proof==closure['segmented_game']['proof'] and
                    entry['initial']==closure['segmented_game']['source_initial'] and entry['result']==closure['segmented_game']['source_result'],
                    'Game39 exact separate saved closure proof; neither failed attempt promoted')
        require(entry['online_replay']==replay['receipts'][index] and proof['status']=='passed' and proof['transitions']==checked['transitions'] and
                proof['failed_moves']==int(checked['failed'] is not None) and proof['final_sha256']==checked['final_sha256'],
                'Byte-bound full-game independent ledger proof')
        bridge_count+=checked['bridges'];events_count+=len(events);ordinary_count+=len(checked['ordinary']);games.append(ending)
        if checked['failed']:
            row,kind=checked['failed'];saved=r.read_path(r.physical(entry['result']['path']).parent/'captured_fault.json')
            require(saved['clock']==row['clock'] and saved['failure_classification']==kind and saved['failure_policy']==POLICY and
                    saved['no_retry'] is True and saved['collection_stopped'] is False and saved['hard_guard_seconds']==10 and
                    saved['internal_guard_seconds']==9,'Exact ordinary hard-guard administrative loss')
            administrative.append(saved)
        for row in checked['ordinary']:
            timing_acc.add(row)
            if row['completed'] and spec['stage']=='main':
                cell,item=selection(row)
                if cell not in selected or item['selection_sha256']<selected[cell]['selection_sha256']:
                    selected[cell]=item;selected_rows[cell]=row
    require(bridge_count==1 and events_count==ordinary_count+1 and
            coverage['live_decisions']==ordinary_count and coverage['live_game_events']==events_count and
            coverage['recovered_events']==1 and coverage['original_infrastructure_failures_preserved']==2 and
            coverage['qualification_ordinary_requests']==84 and coverage['qualification_scientific_decisions']==441 and
            coverage['recovery_actual_entrypoint_decisions']==238 and coverage['new_recovery_actual_entrypoint_decisions']==119 and
            coverage['failed_decision_retries']==0 and
            coverage['original_attempt_status']==coverage['first_recovery_status']=='incomplete' and
            len(administrative)==coverage['forfeits']==faults['count'] and faults['faults']==administrative,
            'One adoption, all actual ordinary events and fixed-sample faults represented')
    # The failed original request is an actual request but never a successful timing/reproduction row.
    failed_folder=bundle['source_row']['worker_folder'];byworker.setdefault(failed_folder,[]).append(bundle['source_row']['clock']['request_id'])
    worker_routes[failed_folder]='research'
    cells=list(product(ROLES,('0..19','20..39','40+'),(0,1),('neither','own','opponent','both')))
    chosen=[selected[x] for x in cells if x in selected]
    empty=[dict(player_role=x[0],phase=x[1],seat=x[2],private_category=x[3]) for x in cells if x not in selected]
    sample=r.read('replay_sample_manifest.json');repeat=r.read('repeated_decision_summary.json')
    require(sample['decisions']==chosen and sample['empty_cells']==empty and sample['maximum_decisions']==48 and
            sample['domain']==old.SAMPLE_DOMAIN and sample['scope']=='main_completed_cells' and
            sample['all_available_cells_selected'] is True and sample['failed_moves_excluded'] is True and
            sample['selection_uses_outcomes'] is False and sample['selection_uses_timing'] is False,
            'Exact original48-cell reproduction selector; recovered event excluded')
    require(repeat['status']=='passed' and repeat['requests']==len(chosen)==coverage['reproductions'] and repeat['identities']==chosen,
            'All predetermined completed-cell reproductions')
    for index,cell in enumerate(x for x in cells if x in selected):
        original=selected_rows[cell];row=r.read(f'repeated_decisions/{index:02d}.json')
        require(check_clock(row) is None and all(row[k]==original[k] for k in ('observer','before','action','trace','after')) and
                row['clock']['result']==original['clock']['result'] and row['player_role']==original['player_role'] and
                row['route']==original['route'],'Full saved reproduction equality without fresh inference')
        folder=(OUT/'workers'/f'audit_{index:02d}').relative_to(ROOT).as_posix()
        require(row['worker_folder']==folder,'Selected reproduction worker identity')
        byworker[folder]=[row['clock']['request_id']];worker_routes[folder]=row['route']
    worker_result=review_workers(r,frozen,expected_sha,segments,byworker,worker_routes,len(chosen),closure)
    require(worker_result['workers']==verification['workers'] and worker_result['parent_requests']==verification['parent_requests'] and
            worker_result['graph_cases']==verification['graph_cases'],'Complete combined worker/request/graph counts')
    final_contract=review_final_contract(r,bundle,ordinary_count,events_count,len(chosen))
    timing=review_timing(r,timing_acc,administrative)
    stats=check_statistics([g for g in games if g['stage']=='main'],r.read('analysis.json'),r.read('analysis_verification.json'))
    owning=owner_checks(r,frozen,expected_sha,completion,worker_result.pop('identities'))
    require(verification['peak_combined_rss_bytes']<=frozen['max_rss_bytes']==2*GiB,'Original aggregate worker RSS cap')
    waits=[r.read_path(p) for p in sorted((OUT/'memory_admissions').glob('*.json'))]
    require(all(0<=x['wait_seconds']<=x['cumulative_wait_seconds']<600 and x['cumulative_wait_seconds']>=PRIOR_RAM_WAIT_SECONDS and
                x['stable_seconds']==60 for x in waits),'Prior RAM wait carried, never reset')
    require(all(waits[i]['cumulative_wait_seconds']<waits[i+1]['cumulative_wait_seconds'] for i in range(len(waits)-1)), 'Cumulative RAM chronology')
    r.output_evidence_bytes=evidence_size(r)
    for name in ('failure.json','finalization_failure.json','watchdog_error.json','guardian_error.json','collection_stop.json','qualification_stop.json'):
        require(not(OUT/name).exists(),'New incomplete marker prevents acceptance: '+name)
    for name in ('runner_stderr.log','watchdog_stderr.log','guardian_stderr.log'):
        if (OUT/name).exists():require(r.pin(OUT/name)['bytes']==0,'Unexplained current owner stderr')
    r.counts.update(frozen_inputs=len(pins),sealed_artifacts=len(seal),sealed_bytes=sum(x['bytes'] for x in seal),
        original_preserved_files=len(manifest['original_tree']),first_recovery_preserved_files=len(manifest['recovery1_tree']),
        closure_checker_preserved_files=len(manifest['closure_tree']),games=406,operational_games=6,main_games=400,pairs=200,
        qualification_scientific_decisions=203,recovery_entrypoint_scientific_decisions=238,new_recovery_entrypoint_scientific_decisions=119,
        live_game_events=events_count,ordinary_timed_live_decisions=ordinary_count,recovered_events=1,
        reproductions=len(chosen),administrative_forfeits=len(administrative),output_bytes=r.output_evidence_bytes,**worker_result)
    return dict(counts=r.counts,statistics=stats,timing=timing,ownership=owning,qualification=qualification,final_contract=final_contract,
        recovered_response=bundle['bridge'],original_attempt_remains_incomplete=True,first_recovery_remains_incomplete=True,amended_recovered_benchmark=True,
        saved_scientific_raw_graph_placement_receipts_reused=True,scientific_replays_rerun=False,bootstrap_rerun=False,
        selected_reproduction_outputs_compared=True,
        limitations=['Both previous attempts and the original rejected request remain incomplete/unaccepted; only the separately resumed fixed sample may be accepted.',
            'One already emitted action is adopted from authenticated saved evidence, excluded from accepted-clock distributions and model reproductions.',
            'Scientific, raw, graph and placement validators are reused through authenticated passed receipts; no model or ledger replay is executed here.',
            'The original paired bootstrap is independently checked by a bound receipt, not rerun here.',
            'Acceptance remains conditional on actual reader, acceptance and retained owner exit0 under original shared deadlines.'])


class TimingAccumulator:
    """Only scalars/counters survive each game; never retain full callback rows."""
    def __init__(self):
        self.count=0
        self.times={role:[] for role in ROLES};self.statuses={role:Counter() for role in ROLES}
        self.grouped={};self.grouped_status={}

    def add(self,row):
        self.count+=1
        role=row['player_role'];self.statuses[role][row['clock']['status']]+=1
        if row['completed']:self.times[role].append(row['clock']['receipt_elapsed_seconds'])
        for key in ((row['stage'],role),(row['stage'],role,row['seat'])):
            self.grouped_status.setdefault(key,Counter())[row['clock']['status']]+=1
            if row['completed']:self.grouped.setdefault(key,[]).append(row['clock']['receipt_elapsed_seconds'])


def review_timing(r,acc,administrative):
    performance=r.read('performance.json');boundaries=r.read('boundary_analysis.json')
    require(performance['games']==406 and performance['natural_games']==406-len(administrative) and
            boundaries['rows']==acc.count and boundaries['observed_faults']==len(administrative), 'Ordinary timing denominators exclude one adoption')
    times,statuses=acc.times,acc.statuses
    timing={}
    for role in ROLES:
        values=times[role]
        expected=dict(completed_decisions=len(values),attempted_decisions=sum(statuses[role].values()),statuses=dict(statuses[role]),
            completed_arrival_seconds=distribution(values),completed_over_700ms=sum(x>.7 for x in values),
            completed_over_900ms=sum(x>.9 for x in values),completed_over_one_second=sum(x>1 for x in values),
            completed_at_or_above_one_second=sum(x>=1 for x in values))
        require(all(performance['players'][role][k]==v for k,v in expected.items()),'Independent accepted-clock timing arithmetic')
        timing[role]=expected
    old.verify_grouped_timing(performance,boundaries,acc.grouped,acc.grouped_status)
    require(performance['descriptive_thresholds_change_completion_or_stopping'] is False and
            boundaries['descriptive_thresholds_change_completion_or_stopping'] is False,'No retroactive deadline losses')
    return timing


def review_workers(r,frozen,sha,segments,byworker,routes,reproductions,closure):
    workers=segments['workers'];folders=[x['folder'] for x in workers]
    require(len(folders)==len(set(folders))==816+reproductions<=864,'Exact combined original/resumed/per-game/reproduction lifecycles')
    actual={(root/'workers'/p.name).relative_to(ROOT).as_posix() for root in (ORIGINAL,FIRST_RECOVERY,OUT) for p in (root/'workers').iterdir()}
    require(set(folders)==actual and set(byworker)<=set(folders),'No hidden or omitted worker tree')
    require(frozen['declared_worker_lifecycles']==expected_lifecycles(),'Exact frozen retained-worker lifecycles')
    expected_new=set(expected_lifecycles())|{f'main_g{i}_{role}' for i in range(40,400) for role in ROLES}|{f'audit_{i:02d}' for i in range(reproductions)}
    require({r.physical(p).name for p in folders if r.physical(p).is_relative_to(OUT/'workers')}==expected_new and
            {r.physical(p).name for p in folders if r.physical(p).is_relative_to(FIRST_RECOVERY/'workers')}==
            {'main_g38_challenger','main_g38_opponent'},'Exact old retained trees and new361 scheduled game lifecycles')
    closed=r.read('all_closed_workers.json')
    require(closed['status']=='passed' and closed['workers']==len(workers)==len(closed['proofs']),'All closed worker proof inventory')
    identities=[];parents=0;recovery_suites=0;first_suites=0;seen_proofs=[]
    for entry in workers:
        folder=entry['folder'];path=r.physical(folder);is_old=path.is_relative_to(ORIGINAL/'workers')
        is_first=path.is_relative_to(FIRST_RECOVERY/'workers')
        require(entry['source_freeze_sha256']==(ORIGINAL_SHA if is_old else FIRST_RECOVERY_SHA if is_first else sha),'Worker source freeze binding')
        proof=bound_read(r,entry['closed_proof']);seen_proofs.append(proof)
        if is_first:
            require(entry['supplemental_proof_source']==frozen['closure_check'] and
                    proof in closure['closed_workers'],'Exact separately verified first-recovery worker proof')
        expected=byworker.get(folder,[])
        require(entry['ordinary_request_ids']==expected,'Exact represented ordinary requests per lifecycle')
        require(proof['status']=='passed' and proof['raw']['status']==proof['logs']['status']==proof['graph']['status']=='passed' and
                proof['graph']['fixtures']==11 and proof['raw']['exact_bytes_checked'] is True and
                proof['raw']['original_trusted_receipts_unchanged'] is True and proof['placement_mode']=='isolated' and
                proof['placement_requests']==proof['raw']['requests'],'Passed immutable raw/log/graph/placement proof')
        if folder in routes:require(proof['route']==routes[folder],'Role quota route proof')
        checked=proof['raw']['requests_checked']
        require([x['request_id'] for x in checked if not x['startup']]==expected and
                len(checked)==proof['raw']['requests'] and proof['raw']['failed_requests']==sum(x['status']!='ok' for x in checked),
                'Every actual ordinary and failed original request represented')
        startups=[x for x in checked if x['startup']]
        expected_startups=2 if path.name.startswith('qualification_') or is_first and path.name.startswith('main_g38_') else 1
        require(len(startups)==proof['raw']['startup_requests']==expected_startups,'Exact graph/entrypoint administrative requests')
        if not is_old and expected_startups==2:
            receipt=r.read_path(path/(startups[1]['request_id']+'.json'));suite=receipt['result'];quota=128 if proof['route']=='research' else 64
            require(receipt['status']=='ok' and receipt['accepted'] is True and receipt['deadline_seconds']==300 and
                    receipt['received_before_deadline'] is True and 0<=receipt['receipt_elapsed_seconds']<=300 and
                    suite['status']=='entrypoint_counterexamples_passed' and suite['execution_route']==proof['route'] and
                    suite['challenger_quota']==quota and suite['world_pool']==3 and suite['original_counts']=={'complete_decisions':52} and
                    suite['counts']=={'complete_decisions':67 if quota==128 else 52} and len(suite['direct_same_quota'])==(15 if quota==128 else 0),
                    'Actual same-quota resumed-worker corrected entrypoint suite')
            if is_first:first_suites+=suite['counts']['complete_decisions']
            else:recovery_suites+=suite['counts']['complete_decisions']
        cleanup=r.read_path(path/'cleanup.json')
        require(cleanup['status']=='passed' and not cleanup['alive'] and not cleanup['errors'],'Every owned tree cleanly closed')
        identities.extend(dict(pid=x['pid'],created=x['created']) for x in cleanup['confirmed'])
        parents+=proof['raw']['requests']
    require(sorted(map(canonical,seen_proofs))==sorted(map(canonical,closed['proofs'])),'Exact all-closed receipt equality')
    require(recovery_suites==first_suites==119,'Both generations of actual startup suites')
    placement=r.read('all_placement_integrity.json')
    require(placement['status']=='passed' and all(placement[key]['status']=='passed' and
            placement[key]['owned_watchdog_and_runner'] is True and type(placement[key]['exchanges']) is int and
            placement[key]['exchanges']>0 for key in ('original','recovery1','recovery')),'All three saved placement/ownership segments')
    return dict(workers=len(workers),parent_requests=parents,graph_cases=11*len(workers),identities=identities)


def review_final_contract(r,bundle,ordinary_count,events_count,reproductions):
    """Reconcile sealed final summaries without repeating their model checks."""
    timing=r.read('recovery_timing.json')
    require(timing['ordinary_metrics_exclude_bridge'] is True and timing['source_receipt_not_rewritten'] is True and
            timing['excluded_original_clock']==bundle['source_row']['clock'] and
            timing['recovered_raw_arrival_evidence']==bundle['bridge']['timing_evidence'] and
            timing['ordinary_timed_live_requests']==ordinary_count and timing['logical_live_events']==events_count,
            'Separate original failed clock and forensic recovered arrival')
    prior=r.read('prior_qualification_verification.json')
    require(prior['status']=='passed' and prior['ordinary_requests']==84 and prior['scientific_decisions']==203 and
            prior['old_freeze_sha256']==ORIGINAL_SHA and len(prior['original_suites'])==2 and
            all(x['status']=='passed' for x in prior['original_suites']),'Original203 scientific checks retained')
    old_gate=bound_read(r,prior['prior_operational_gate'])
    require(old_gate['passed'] is True,'Original operational resource gate')
    summary=r.read('recovery_qualification_summary.json')
    require(summary['status']=='passed' and summary['scientific_decisions']==119 and summary['parent_requests']==2 and
            summary['original_scientific_decisions']==203 and summary['first_recovery_scientific_decisions']==119 and
            summary['prior_scientific_decisions']==322 and summary['combined_scientific_decisions']==441 and
            [x['route'] for x in summary['receipts']]==['research','frozen'] and
            summary['same_workers_retained_for_continuation'] is True and summary['failed_game_decision_retried'] is False,
            'Both actual resumed-worker suites')
    for entry in summary['receipts']:
        route=entry['route'];expected=52 if route=='frozen' else 67
        receipt=bound_read(r,entry['clock']);proof=entry['verification']
        require(receipt==r.read('recovery_entrypoint_checks/'+route+'.json') and receipt['status']=='ok' and
                receipt['accepted'] is True and receipt['result']['execution_route']==route and
                receipt['result']['counts']['complete_decisions']==expected and proof['status']=='passed' and
                proof['route']==route and proof['complete_decisions']==expected and
                proof['original_complete_decisions']==52 and proof['direct_same_quota']==expected-52 and
                proof['ordinary_handlers']==37 and proof['negative_handler_cases']==26 and
                proof['independent_saved_transitions']==37 and proof['model_calls']==0 and proof['qualification_rerun'] is False,
                'Bound saved actual-entrypoint verification and genuine quota count')
        r.verify(proof['forensic_artifact'])
    gate=r.read('recovery_resource_gate.json');final=r.read('reproduction_resource_gate.json')
    require(gate['passed'] is True and gate['remaining_scheduled_games']==361 and gate['outcomes_used'] is False and
            gate['move_timing_subtracted_seconds']==0 and gate['prior_final_seconds_charged']==PRIOR_FINAL_SECONDS and
            gate['prior_ram_wait_allocation_seconds']==PRIOR_RAM_WAIT_SECONDS and
            gate['original_monotonic_origin']==ORIGIN_MONO and gate['original_unix_origin']==ORIGIN_UNIX and
            gate['maximum_combined_workers']==864 and gate['qualification_extra_scientific_decisions']==119 and
            bound_read(r,gate['original_gate'])==old_gate and
            0<=gate['projected_collection_seconds']<gate['available_collection_seconds'] and
            gate['projected_collection_seconds']+gate['projected_final_seconds']<gate['available_global_seconds'] and
            3600<=gate['projected_final_seconds']<14400-PRIOR_FINAL_SECONDS and
            gate['aggregate_current_evidence_bytes']+gate['projected_additional_collection_bytes']<12*GiB and
            gate['projected_final_bytes']<4*GiB and
            gate['aggregate_current_evidence_bytes']+gate['projected_additional_collection_bytes']+gate['projected_final_bytes']<16*GiB,
            'Outcome-blind remaining-work bound under original clock')
    require(final['passed'] is True and final['selected_reproductions']==reproductions and
            final['prior_final_seconds_charged']==PRIOR_FINAL_SECONDS and final['independent_review_reserved_seconds']==3600 and
            3600<=final['projected_remaining_seconds']<final['available_seconds'] and final['projected_final_bytes']<4*GiB and
            final['current_evidence_bytes']+final['projected_final_bytes']<16*GiB,'Final reproduction and review resource reservation')
    return dict(original_scientific_decisions=203,first_recovery_scientific_decisions=119,new_recovery_scientific_decisions=119,combined_scientific_decisions=441,
                resource_receipts_bound=True,bridge_timing_separate=True)


def review_previous_closure(r,frozen,manifest):
    """Reuse the separately qualified proof, never accept either failed run."""
    require(frozen['closure_check']==manifest['closure_check'] and frozen['closure_check']['sha256']==CLOSURE_SHA and
            frozen['checker_os_exit']==manifest['checker_os_exit'],'Exact separate closure qualification')
    value=bound_read(r,frozen['closure_check']);terminal=bound_read(r,frozen['checker_os_exit'])
    require(value['status']=='qualified_saved_closure_checker_only' and value['freeze_sha256']==FIRST_RECOVERY_SHA and
            value['original_attempt_status']=='incomplete' and value['original_attempt_accepted'] is False and
            value['benchmark_accepted'] is False and len(value['original_failures_reproduced'])==2 and
            value['actual_entrypoint_decisions']==119 and value['graph_cases']==22 and value['ordinary_saved_moves']==14 and
            value['original_files_unchanged'] is True and value['new_model_calls']==value['new_games']==0 and
            value['original_clocks_rewritten'] is False,'Separate saved-only recovered proof scope')
    require(terminal['exit_code']==0 and terminal['observed_by']=='retained subprocess.Popen handle' and
            terminal['checker_pid']==value['reader']['pid'] and terminal['checker_created']==value['reader']['created'] and
            terminal['model_calls']==terminal['new_games']==0,'Actual closure-checker exit0')
    r.absent(value['reader'])
    for item in value['consumed_inputs']+value['imported_frozen_sources_and_runtime']+[value['checker_source']]:r.verify(item)
    game=value['segmented_game']
    require(game['status']=='passed_saved_game_only' and game['new_ordinary_moves']==14 and game['prior_accepted_moves']==39 and
            game['recovered_actions']==1 and game['model_calls']==0 and game['game_rerun'] is False and
            game['original_attempt_accepted'] is False and game['proof']['status']=='passed' and
            game['proof']['transitions']==54 and game['proof']['failed_moves']==0 and
            len(value['closed_workers'])==len(value['actual_entrypoint_verification'])==2 and
            all(x['status']=='passed' for x in value['closed_workers']+value['actual_entrypoint_verification']),
            'Exactly two retained worker proofs and one54-event saved game')
    return value


def guarded_publish(r,path,value):
    r.guard()
    if hasattr(r,'output_evidence_bytes'):
        size=len((canonical(value)+'\n').encode())
        require(r.output_evidence_bytes+size+1024**2<16*GiB,'Combined original/recovery publication evidence cap')
    save_new(path,value)
    if hasattr(r,'output_evidence_bytes'):r.output_evidence_bytes+=path.stat().st_size
    r.guard()


def failure_receipt(name,exc):
    path=OUT/'postrun'/name
    if not path.exists():save_new(path,dict(status='failed_no_acceptance',error=repr(exc),traceback=traceback.format_exc(),
        time=datetime.now(timezone.utc).isoformat(),models_run=False,games_run=False,original_attempt_unchanged=True))


def main_review(sha):
    r=Review()
    try:
        clock=bind_shared_clock(r,sha);frozen=r.read('freeze.json');source_authentication(r,frozen,sha)
        require(not(OUT/'postrun/independent_completion_review.json').exists() and not(OUT/'postrun/acceptance_invalidated.json').exists(),
                'Never overwrite an acceptance')
        result=review_all(r,sha)
        integrity=r.revalidate_identities();path=OUT/'postrun/independent_review_identities.json'
        guarded_publish(r,path,identity_receipt(r));identity_pin=r.pin(path)
        component=dict(status='passed_saved_evidence_component',accepted=False,freeze_sha256=sha,
            amended_recovered_benchmark=True,original_attempt_remains_incomplete=True,first_recovery_remains_incomplete=True,
            reviewer_pid=os.getpid(),reviewer_created=psutil.Process().create_time(),
            reviewer_source=r.pin(Path(__file__)),runtime_source=r.pin(Path(old.__file__).with_name('review_runtime.py')),
            shared_owner_clock=clock,maximum_review_seconds=LIMIT,identity_inventory=identity_pin,
            result=result,final_identity_validation=integrity,authentication_metrics=dict(r.metrics),
            peak_reader_rss_bytes=r._peak_rss,elapsed_seconds=time.monotonic()-r.start,
            consumed_inputs=sorted(r.consumed.values(),key=lambda x:x['path']),model_calls=0,search_calls=0,new_games=0,
            scientific_replays=0,bootstrap_executions=0,acceptance_requires_actual_reader_exit0=True)
        guarded_publish(r,OUT/'postrun/independent_review_component.json',component)
        print(canonical(dict(status=component['status'],counts=result['counts'],elapsed_seconds=time.monotonic()-r.start)))
        return 0
    except BaseException as exc:
        failure_receipt('independent_review_failure.json',exc)
        if (OUT/'postrun/independent_review_component.json').exists():failure_receipt('acceptance_invalidated.json',exc)
        print(canonical(dict(status='failed_no_acceptance',error=repr(exc))));return 1


def accept_exit(sha):
    r=Review()
    try:
        clock=bind_shared_clock(r,sha);frozen=r.read('freeze.json');source_authentication(r,frozen,sha)
        require(not(OUT/'postrun/independent_review_failure.json').exists() and not(OUT/'postrun/acceptance_invalidated.json').exists(),
                'Failed component cannot be accepted')
        component=r.read('postrun/independent_review_component.json');pin=r.pin(OUT/'postrun/independent_review_component.json')
        terminal=r.read('postrun/independent_review_os_exit.json')
        require(component['status']=='passed_saved_evidence_component' and component['accepted'] is False and
                component['amended_recovered_benchmark'] is True and component['original_attempt_remains_incomplete'] is True and
                component['first_recovery_remains_incomplete'] is True and
                component['freeze_sha256']==sha and component['shared_owner_clock']==clock and terminal['freeze_sha256']==sha and
                terminal['exit_code']==0 and terminal['observed_by']=='owned Process handle' and terminal['cleanup_reason'] is None and
                terminal['reader_pid']==component['reviewer_pid'] and abs(terminal['reader_created']-component['reviewer_created'])<=.0001 and
                terminal['script_sha256']==component['reviewer_source']['sha256']==r.pin(Path(__file__))['sha256'] and
                terminal['component_sha256']==pin['sha256'],'Actual exact reviewer handle exit0')
        absence=r.absent(dict(pid=component['reviewer_pid'],created=component['reviewer_created']))
        identities=bound_read(r,component['identity_inventory']);checked=verify_identity_receipt(r,identities)
        r.output_evidence_bytes=evidence_size(r)
        source_authentication(r,frozen,sha);r.revalidate_identities();r.guard()
        receipt=dict(status='accepted_amended_recovery_pending_final_owner_exit',accepted=True,freeze_sha256=sha,
            amended_recovered_benchmark=True,original_attempt_remains_incomplete=True,first_recovery_remains_incomplete=True,component=pin,
            reader_exit=r.pin(OUT/'postrun/independent_review_os_exit.json'),shared_owner_clock=clock,
            reader_absence=absence,final_identity_check=checked,result=component['result'],model_calls=0,new_games=0,search_calls=0,
            requires_actual_acceptance_exit0=True,requires_actual_owner_exit0=True,
            acceptance_invalid_if_marker_present='postrun/acceptance_invalidated.json',
            consumed_inputs=sorted(r.consumed.values(),key=lambda x:x['path']),time=datetime.now(timezone.utc).isoformat())
        guarded_publish(r,OUT/'postrun/independent_completion_review.json',receipt)
        print(canonical(dict(status=receipt['status'],accepted=True)));return 0
    except BaseException as exc:
        failure_receipt('acceptance_invalidated.json',exc)
        print(canonical(dict(status='failed_no_acceptance',error=repr(exc))));return 1


def evidence_size(r):
    total=0
    for root in (ORIGINAL,FIRST_RECOVERY,CLOSURE,OUT):
        for base,_,names in os.walk(root):
            r.guard();total+=sum((Path(base)/name).stat().st_size for name in names)
    require(total<16*GiB and shutil.disk_usage(ROOT).free>=5*GiB,'Combined evidence/disk bound')
    return total


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--accept-exit',action='store_true')
    args=parser.parse_args();raise SystemExit(accept_exit(args.freeze_sha256) if args.accept_exit else main_review(args.freeze_sha256))
