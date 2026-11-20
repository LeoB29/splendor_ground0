"""Bounded saved-only review for one fixed128-versus64 strength study.

First run --freeze-sha256 SHA under an external owner retaining the OS handle.
That owner must save postrun/independent_review_os_exit.json with reader_pid,
reader_created, exit_code, script_sha256, component_sha256, elapsed_seconds,
and observed_by='owned Process handle'. Only then run --accept-exit with the
same freeze SHA. The second mode reconciles receipts, not models or games.

Passed scientific/raw/graph/placement validators are reused and byte-bound;
this reader does not repeat learned decisions, ledger transitions or bootstrap.
It independently reconstructs schedule, scoring, sensitivity, live timing,
administrative fault identity and deterministic reproduction selection/equality.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import traceback
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_search128_strength_20261028_01'
LIMIT = 3600.
DOMAIN = 'splendorust_search128_strength_20261028_v1'
SAMPLE_DOMAIN = 'splendorust_search128_strength_20261028_audit_v1'
POLICY = 'research_hard_guard_9s_internal_10s_complete_response'
FAILURES = ('research_internal_hard_guard', 'research_outer_hard_guard')
ROLES = {'challenger': 'research', 'opponent': 'frozen'}
GiB = 1024**3


def require(value, message):
    if not value:
        raise ValueError('SEARCH128_REVIEW: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


from .review_runtime import ReviewRuntime

class Review(ReviewRuntime):
    def __init__(self, seconds=LIMIT):
        super().__init__(ROOT, OUT, LIMIT, seconds=seconds)


def expected_game(index, stage='main'):
    require(stage in ('operational','main') and type(index) is int and
            0 <= index < (6 if stage=='operational' else 400), 'Scheduled stage/local index')
    pair = index//2
    raw = json.dumps([DOMAIN, stage, pair], separators=(',', ':'), ensure_ascii=True).encode()
    return dict(stage=stage, pair_index=pair, game_index=index, challenger_seat=(pair+index%2)%2,
                seed_decimal=str(int.from_bytes(hashlib.sha256(raw).digest(), 'big')),
                challenger_quota=128, opponent_quota=64, challenger_mode='research', opponent_mode='frozen')


def distribution(values):
    ordered = sorted(values)
    def q(fraction):
        if not ordered: return None
        place = (len(ordered)-1)*fraction; lo, hi = math.floor(place), math.ceil(place)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(place-lo)
    return dict(count=len(values), mean=sum(values)/len(values) if values else None,
                min=min(values) if values else None, median=q(.5), p95=q(.95),
                max=max(values) if values else None)


def check_clock(row):
    clock = row['clock']; good = clock['status'] == 'ok'
    require(clock['clock'] == 'perf_counter_ns' and clock['deadline_seconds'] == 10., 'Actual ten-second raw clock')
    require(row['completed'] is good and clock['accepted'] is good, 'Completion/acceptance coupling')
    elapsed, received = clock['elapsed_seconds'], clock['receipt_elapsed_seconds']
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'Elapsed time')
    if received is not None:
        require(type(received) in (int, float) and math.isfinite(received) and 0 <= received <= elapsed+1e-6, 'Receipt timestamp')
    if good:
        require(received is not None and received <= 10 and clock['received_before_deadline'] is True and
                type(clock['result']) is dict and row['action'] == clock['result']['action'] and
                row['trace'] == clock['result']['diagnostics'] and type(row['after']) is dict, 'Completed scientific envelope')
        return None
    require(clock['result'] is None and row['action'] is None and row['trace'] is None and row['after'] is None,
            'Failed request cannot invent played science')
    if clock['status'] == 'timeout':
        require(elapsed >= 10 and clock['received_before_deadline'] is False and
                (received is None or received > 10), 'Actual complete-response hard guard')
        return 'research_outer_hard_guard'
    detail = str(clock.get('detail', ''))
    require(clock['status'] == 'worker_error' and 'STRENGTH_INTERNAL_DEADLINE:' in detail and
            not any(x in detail.lower() for x in ('source changed', 'ruleset', 'catalogue', 'resource', 'prerequisite', 'audit:')),
            'Only the explicit research hard guard may score an administrative loss')
    return 'research_internal_hard_guard'


def selection(row):
    seat, role, ply = row['seat'], row['player_role'], row['ply']
    view = row['observer']['observation']; require(view['current_player'] == seat, 'Observer acting seat')
    private = [any(r['source'] == 'deck' for r in view['players'][actor]['reserves']) for actor in (seat, 1-seat)]
    category = 'both' if all(private) else 'own' if private[0] else 'opponent' if private[1] else 'neither'
    phase = '0..19' if ply < 20 else '20..39' if ply < 40 else '40+'
    raw = json.dumps([SAMPLE_DOMAIN, 'main', row['pair_index'], row['game_index'], ply, role], separators=(',', ':'), ensure_ascii=True).encode()
    item = dict(stage='main', pair_index=row['pair_index'], game_index=row['game_index'], ply=ply,
                player_role=role, arm='search', quota=128 if role=='challenger' else 64, mode=ROLES[role], seat=seat,
                placement_mode='isolated', phase=phase, private_category=category,
                selection_sha256=hashlib.sha256(raw).hexdigest(), observer_sha256=digest(row['observer']))
    return (role, phase, seat, category), item


def check_statistics(games, result, verified):
    require(len(games)==400 and all(g['stage']=='main' for g in games), 'Primary statistics exclude all six operational games')
    require(result['challenger_simulations']==128 and result['reference_simulations']==64 and
            result['challenger_world_pool']==result['reference_world_pool']==3, 'Exact increased-compute question')
    points = [1 if g['winner'] is None else 2*int(g['winner'] == g['challenger_seat']) for g in games]
    pairs = [points[2*i]+points[2*i+1] for i in range(200)]
    mean = sum(points)/800
    seats = {str(s):sum(p for p,g in zip(points,games) if g['challenger_seat'] == s)/400 for s in (0,1)}
    admin = [g for g in games if g['reason'] == 'forfeit']; natural = sum(p for p,g in zip(points,games) if g['reason'] == 'natural')
    require(result['status'] == 'complete_verified_fixed_sample' and result['evidence_gates_passed'] is True and
            result['games'] == 400 and result['pairs'] == 200 and result['denominator'] == 800 and
            result['pair_numerators'] == pairs and result['numerator'] == sum(points) and result['search_mean_score'] == mean and
            result['by_search_seat'] == seats and result['search_wins'] == points.count(2) and
            result['draws'] == points.count(1) and result['search_losses'] == points.count(0), 'Independent fixed-sample score arithmetic')
    require(result['administrative_forfeits'] == len(admin) and result['natural_games'] == 400-len(admin) and
            result['administrative_failure_policy'] == POLICY and result['administrative_forfeits_by_player'] ==
            {role:sum(g['failed_role'] == role for g in admin) for role in ROLES}, 'Administrative counts')
    require(result['forfeits'] == [{k:g[k] for k in ('pair_index','game_index','loser','winner','failure','failure_policy','administrative_forfeit')}
                                 for g in admin], 'Analysis fault list')
    sensitivity = result['administrative_forfeit_sensitivity']
    require(sensitivity['administrative_games'] == len(admin) and sensitivity['complete_sample_games'] == 400 and
            sensitivity['denominator'] == 800 and sensitivity['lower_numerator'] == natural and
            sensitivity['upper_numerator'] == natural+2*len(admin) and
            sensitivity['all_administrative_candidate_losses'] == natural/800 and
            sensitivity['all_administrative_candidate_wins'] == (natural+2*len(admin))/800 and
            sensitivity['natural_only_strength_estimate'] is False, 'Independent full-sample sensitivity')
    boot = result['bootstrap']; ci = result['paired_ci95']
    require(boot['replicates'] == 50000 and boot['seed'] == 20261028128 and boot['generator'] == 'numpy.PCG64' and
            boot['resampling_unit'] == 'swapped_seat_pair' and boot['quantile_method'] == 'linear' and
            boot['draw_order'] == 'replicate_major_200_int64_indices', 'Fixed paired bootstrap specification')
    require(verified['status'] == 'passed' and verified['replicates'] == 50000 and
            verified['index_stream_sha256'] == boot['indices_sha256'] and
            verified['independent_python_integer_sums'] is True and verified['independent_linear_percentiles'] is True and
            verified['independent_full_sample_sensitivity'] is True, 'Saved independent statistics receipt')
    require(len(ci) == 2 and all(type(x) in (float,int) and math.isfinite(x) for x in ci) and 0 <= ci[0] <= ci[1] <= 1, 'Saved interval')
    radius = math.sqrt(math.log(40)/(2*200)); bound = [max(0.,mean-radius),min(1.,mean+radius)]
    require(result['bounded_pair_hoeffding_95'] == bound and result['bootstrap_degenerate'] is (ci[0] == ci[1]), 'Degenerate safeguard')
    candidate = mean >= .55 and ci[0] > .5 and min(seats.values()) >= .5
    control = mean <= .45 and ci[1] < .5 and max(seats.values()) <= .5
    if ci[0] == ci[1]: candidate = candidate and bound[0] > .5; control = control and bound[1] < .5
    classification = ('useful_evidence_for_128_under_research_guard' if candidate else
        'useful_evidence_for_64_under_research_guard' if control else
        'supported_128_advantage_not_meeting_useful_criteria' if ci[0]>.5 and (ci[0]!=ci[1] or bound[0]>.5) else
        'supported_64_advantage_not_meeting_useful_criteria' if ci[1]<.5 and (ci[0]!=ci[1] or bound[1]<.5) else 'inconclusive')
    require(result['classification'] == classification and result['automatic_promotion'] is False and
            result['one_second_deployability_established'] is False and
            result['administrative_forfeits_may_affect_strength_interpretation'] is bool(admin) and
            result['research_guard_seconds'] == {'internal':9.,'complete_response':10.}, 'Predeclared conclusion and limitations')
    return dict(score=mean, paired_ci95=ci, wins=points.count(2), draws=points.count(1), losses=points.count(0),
                administrative_forfeits=len(admin), classification=classification,
                sensitivity=sensitivity, bootstrap_reexecuted=False)


def review_all(r, expected_sha):
    frozen = r.read('freeze.json'); require(r.pin(OUT/'freeze.json')['sha256'] == expected_sha, 'Exact requested freeze')
    # Fail before a long read if the caller has not waited for all real owners.
    for name in ('execution.json','watchdog_execution.json','guardian_os_exit.json'):
        terminal=r.read(name)
        require(terminal['exit_code']==0 and terminal['freeze_sha256']==expected_sha,
                'All actual owning exits must precede this review')
    require(r.read('completion.json')['status']=='completed_verified','Collection/final verification must already be complete')
    require(frozen['phase'] == 'strength' and frozen['seconds_per_move'] == 10 and frozen['internal_seconds'] == 9 and
            frozen['main_games'] == 400 and frozen['operational_games']==6 and frozen['qualification_decisions']==203 and frozen['qualification_ordinary_requests']==84 and
            frozen['job_seconds'] == 86400 and frozen['collection_cutoff_seconds'] == 72000 and
            frozen['stage_seconds'] == {'qualification':3600,'operational':3600,'main':72000,'final':14400} and frozen['max_evidence_bytes'] == 16*GiB and
            frozen['final_reserve_bytes'] == 4*GiB and frozen['prefinal_evidence_bytes'] == 12*GiB and
            frozen['timing_thresholds_do_not_gate_admission'] is True and frozen['late_completed_moves_accepted'] is True and
            frozen['automatic_retries'] == frozen['failed_decision_retries'] == 0 and frozen['training'] is False, 'Frozen complete question/resources')
    pins = {x['path']:x for x in frozen['inputs']}; require(len(pins) == len(frozen['inputs']), 'Unique frozen inputs')
    authenticate_all(r,frozen)
    require(sys.version == frozen['python'] and r.pin(Path(sys.executable))['sha256'] == frozen['python_executable_sha256'], 'Reader runtime identity')
    for key in ('torch_runtime','external_runtime'):
        for item in frozen[key]:
            got = r.pin(Path(item['absolute_path']))
            require(got['sha256'] == item['sha256'] and got['bytes'] == item['bytes'], 'Pinned runtime bytes')
    completion = r.read('completion.json'); verification = r.read('verification_summary.json'); coverage = r.read('coverage.json')
    require(completion['status'] == 'completed_verified' and completion['failure'] is None and completion['phase'] == 'strength' and
            completion['freeze_sha256'] == expected_sha and completion['automatic_retries'] == 0 and
            completion['requires_all_three_clean_actual_os_exits'] is True, 'Complete fixed-sample result')
    require(verification['status'] == 'passed' and verification['all_raw_placement_logs_graph_science_sources_verified'] is True and
            verification['source_inputs'] == len(pins) and verification['training'] is False and
            verification['coverage'] == completion['coverage'] == coverage, 'Full verification receipt')
    require(coverage['status'] == 'full_workload_verified' and coverage['complete_planned_workload'] is True and
            coverage['phase']=='strength' and coverage['planned_games']==coverage['verified_games']==406 and
            coverage['unplayed_games']==0 and coverage['qualification_ordinary_requests']==84 and
            coverage['operational_games']==6 and coverage['main_games']==400 and
            coverage['admission_passed'] is True and coverage['continuation_after_forfeit']=='main_only', 'Fixed400 coverage')
    seal = r.read('completed_evidence_seal.json')['artifacts']; sealmap = {x['path']:x for x in seal}
    require(len(sealmap) == len(seal), 'Unique completion seal')
    for item in seal: r.verify(item)
    analysis_seal = r.read('analysis_seal.json')['artifacts']
    require({x['path'] for x in analysis_seal} == {(OUT/n).relative_to(ROOT).as_posix() for n in ('analysis.json','analysis_verification.json')}, 'Exact analysis seal')
    for item in analysis_seal: r.verify(item)
    for name in ('workers','games','qualification_rows','entrypoint_checks','repeated_decisions','closed_workers','closed_worker_costs','placement_requests','placement_acks'):
        prefix=(OUT/name).relative_to(ROOT).as_posix()+'/'
        r.exact_tree(OUT/name, [p for p in sealmap if p.startswith(prefix)])
    schedule = r.read('schedule.json'); require(schedule == [expected_game(i,stage) for stage,count in (('operational',6),('main',400)) for i in range(count)], 'Independent full fresh pair schedule')
    gate=r.read('assessment_gate.json')
    require(gate['passed'] is True and gate['resource_passed'] is True and gate['all_operational_games_natural'] is True and
            gate['observed_games']==6 and gate['projected_games']==400 and gate['conservative_factor']==1.5 and
            gate['independent_review_reserved_seconds']==3600 and gate['timing_used_for_admission'] is False and
            gate['move_timing_subtracted_seconds']==0 and gate['outcomes_used'] is False,
            'Outcome-blind measured resource gate; original clocks and independent reserve retained')
    replay = r.read('all_game_replay.json'); faults = r.read('all_player_faults.json')
    require(replay['status'] == 'passed' and replay['games'] == len(replay['receipts']) == 406 and
            faults['failed_move_retries'] == 0, 'All400 saved ledger replays')
    games=[]; selected={}; selected_rows={}; byworker={}; administrative=[]
    qualification_result=qualification_review(r,frozen,byworker)
    times={role:[] for role in ROLES}; statuses={role:Counter() for role in ROLES}; moves=0
    grouped_times={};grouped_statuses={}
    expected_folders={game_folder_name(spec) for spec in schedule}
    require({p.parent.relative_to(OUT).as_posix() for p in (OUT/'games').rglob('initial.json')}==expected_folders and
            {p.parent.relative_to(OUT).as_posix() for p in (OUT/'games').rglob('result.json')}==expected_folders,
            'No hidden extra or partial games')
    for game_number,spec in enumerate(schedule):
        r.guard();folder=game_folder_name(spec)
        # Both trees exist even when the first move forfeits before the other
        # player has an ordinary request. Its graph-only lifecycle still counts.
        for role in ROLES:byworker[f"{spec['stage']}_g{spec['game_index']}_{role}"]=[]
        game=r.read(folder+'/result.json'); initial=r.read(folder+'/initial.json')
        require(all(game.get(k)==v for k,v in spec.items()) and game['complete'] is True and
                type(game['plies']) is int and 0<game['plies']<=2000 and
                (game['winner'] is None or type(game['winner']) is int and game['winner'] in (0,1)) and
                all(initial.get(k)==v for k,v in spec.items()), 'Complete ending/initial schedule')
        paths=sorted((OUT/folder/'moves').glob('*.json'))
        require([p.name for p in paths]==[f'{i:04d}.json' for i in range(game['plies'])], 'Complete contiguous moves')
        previous=initial['initial']; failed=None
        for ply,path in enumerate(paths):
            row=r.read_path(path); moves+=1; role=row['player_role']; require(role in ROLES, 'Role')
            require(row['before']==previous and row['ply']==ply and all(row[k]==spec[k] for k in ('stage','pair_index','game_index')) and
                    row['seat']==previous['state']['current_player'] and role==('challenger' if row['seat']==spec['challenger_seat'] else 'opponent') and
                    row['mode']==row['route']==ROLES[role] and row['quota']==(128 if role=='challenger' else 64) and row['arm']=='search' and
                    row['placement_mode']=='isolated' and row['observation_mode']=='profiled' and
                    not row.get('profile_capture_error') and not row.get('placement_error') and not row.get('transition_error'), 'Saved live row identity and contiguous chain')
            fault=check_clock(row); statuses[role][row['clock']['status']]+=1
            for key in ((spec['stage'],role),(spec['stage'],role,row['seat'])):
                grouped_statuses.setdefault(key,Counter())[row['clock']['status']]+=1
                if fault is None:grouped_times.setdefault(key,[]).append(row['clock']['receipt_elapsed_seconds'])
            name=Path(row['worker_folder']).name; require(name==f"{spec['stage']}_g{spec['game_index']}_{role}", 'Per-game player lifecycle')
            byworker.setdefault(name,[]).append(row['clock']['request_id'])
            require(r.read_path(r.physical(row['worker_clock_file']))==row['clock'], 'Saved clock is original receipt')
            if fault:
                require(ply==len(paths)-1, 'No continuation after failed move within game');failed=(row,fault)
            else:
                previous=row['after']; times[role].append(row['clock']['receipt_elapsed_seconds'])
                if spec['stage']=='main':
                    cell,item=selection(row)
                    if cell not in selected or item['selection_sha256']<selected[cell]['selection_sha256']:
                        selected[cell]=item;selected_rows[cell]=row
        require(game['final']==previous, 'Saved final state chain')
        check=r.read(folder+'/online_replay.json'); require(check==replay['receipts'][game_number] and
            check['status']=='passed' and check['transitions']==len(paths)-int(failed is not None) and
            check['failed_moves']==int(failed is not None) and check['final_sha256']==digest(previous), 'Bound full-game ledger proof')
        if failed:
            row,kind=failed; saved=r.read(folder+'/captured_fault.json')
            require(spec['stage']=='main' and game['reason']=='forfeit' and game['administrative_forfeit'] is True and game['failure_policy']==POLICY and
                    game['failure']==kind and game['loser']==game['failed_seat']==row['seat'] and game['winner']==1-row['seat'] and
                    game['failed_role']==row['player_role'] and saved['clock']==row['clock'] and
                    saved['failure_classification']==kind and saved['failure_policy']==POLICY and saved['no_retry'] is True and
                    saved['collection_stopped'] is False and saved['hard_guard_seconds']==10 and saved['internal_guard_seconds']==9 and
                    all(saved[k]==row[k] for k in ('stage','pair_index','game_index','ply','player_role','seat')),
                    'Administrative loss exact original failed request, no retry')
            administrative.append(saved)
        else:
            require(game['reason']=='natural' and previous['state']['terminal'] and game['winner']==previous['state']['winner'] and
                    game.get('administrative_forfeit',False) is False and all(game.get(k) is None for k in ('loser','failed_role','failed_seat','failure','failure_policy')),
                    'Natural outcome carries no administrative loss')
            require(not(OUT/folder/'captured_fault.json').exists(), 'No hidden fault')
        games.append(game)
    require(moves==coverage['live_decisions'] and len(administrative)==coverage['forfeits']==faults['count'] and
            faults['faults']==administrative, 'Live and fault counts')
    sample=r.read('replay_sample_manifest.json'); repeat=r.read('repeated_decision_summary.json')
    cells=list(product(ROLES,('0..19','20..39','40+'),(0,1),('neither','own','opponent','both')))
    chosen=[selected[cell] for cell in cells if cell in selected]
    empty=[dict(player_role=x[0],phase=x[1],seat=x[2],private_category=x[3]) for x in cells if x not in selected]
    require(sample['decisions']==chosen and sample['empty_cells']==empty and sample['maximum_decisions']==48 and
            sample['domain']==SAMPLE_DOMAIN and sample['scope']=='main_completed_cells' and sample['all_available_cells_selected'] is True and
            sample['failed_moves_excluded'] is True and sample['selection_uses_outcomes'] is False and sample['selection_uses_timing'] is False,
            'Independently reconstructed deterministic completed-cell selection')
    require(repeat['status']=='passed' and repeat['requests']==len(chosen)==coverage['reproductions'] and repeat['identities']==chosen, 'Reproduction coverage')
    for index,cell in enumerate(cell for cell in cells if cell in selected):
        original=selected_rows[cell]; row=r.read(f'repeated_decisions/{index:02d}.json')
        require(check_clock(row) is None and all(row[k]==original[k] for k in ('observer','before','action','trace','after')) and
                row['clock']['result']==original['clock']['result'] and row['player_role']==original['player_role'] and
                row['route']==original['route'], 'Exact saved complete decision reproduction; no inference rerun')
        name=f'audit_{index:02d}';require(Path(row['worker_folder']).name==name,'Reproduction worker identity')
        byworker[name]=[row['clock']['request_id']]
    require(len(byworker)==814+len(chosen), 'Exact main/reproduction worker inventory')
    require({p.name for p in (OUT/'workers').iterdir()}==set(byworker), 'No additional or partial worker trees')
    closed=r.read('all_closed_workers.json'); require(closed['status']=='passed' and closed['workers']==len(closed['proofs'])==len(byworker)==verification['workers'],'Every worker closed')
    identities=[]; parents=0
    require({x['worker'] for x in closed['proofs']}==set(byworker), 'Exact closed worker names')
    for proof in closed['proofs']:
        name=proof['worker']; route=name.removeprefix('qualification_') if name.startswith('qualification_') else 'research' if name.endswith('_challenger') else 'frozen' if name.endswith('_opponent') else selected_rows[[x for x in cells if x in selected][int(name[6:])]]['route']
        require(proof==r.read('closed_workers/'+name+'.json') and proof['status']=='passed' and proof['route']==route and
                proof['raw']['status']==proof['logs']['status']==proof['graph']['status']=='passed' and proof['graph']['fixtures']==11 and
                proof['raw']['startup_requests']==(2 if name.startswith('qualification_') else 1) and
                proof['raw']['requests']==len(byworker[name])+(2 if name.startswith('qualification_') else 1) and
                proof['raw']['exact_bytes_checked'] is True and proof['raw']['original_trusted_receipts_unchanged'] is True and
                proof['placement_requests']==proof['raw']['requests'] and proof['placement_mode']=='isolated', 'Bound closed raw/log/graph/placement receipt')
        raw=proof['raw']['requests_checked'];require(len(raw)==proof['raw']['requests'] and
            [x['request_id'] for x in raw if not x['startup']]==byworker[name], 'All ordinary request identities represented')
        require(proof['raw']['failed_requests']==sum(x['status']!='ok' for x in raw), 'Raw failure count')
        cleanup=r.read('workers/'+name+'/cleanup.json');require(cleanup['status']=='passed' and not cleanup['alive'] and not cleanup['errors'],'Owned closure')
        for item in cleanup['confirmed']:identities.append(dict(pid=item['pid'],created=item['created']))
        parents+=proof['raw']['requests']
    require(parents==verification['parent_requests']==moves+900+2*len(chosen) and
            verification['graph_cases']==11*len(byworker) and len(byworker)<=862, 'Parent/graph/work bounds')
    placement=r.read('all_placement_integrity.json');require(placement['status']=='passed' and placement['owned_watchdog_and_runner'] is True,'Complete saved placement verification')
    performance=r.read('performance.json'); boundaries=r.read('boundary_analysis.json')
    require(performance['games']==406 and performance['natural_games']==406-len(administrative) and
            boundaries['rows']==moves and boundaries['observed_faults']==len(administrative), 'Timing summary denominators')
    timing={}
    for role in ROLES:
        expected=dict(completed_decisions=len(times[role]),attempted_decisions=sum(statuses[role].values()),statuses=dict(statuses[role]),
                      completed_arrival_seconds=distribution(times[role]),completed_over_700ms=sum(x>.7 for x in times[role]),
                      completed_over_900ms=sum(x>.9 for x in times[role]),completed_over_one_second=sum(x>1 for x in times[role]),
                      completed_at_or_above_one_second=sum(x>=1 for x in times[role]))
        require(all(performance['players'][role][k]==v for k,v in expected.items()), 'Independent per-role timing/count arithmetic')
        timing[role]=expected
    require(performance['descriptive_thresholds_change_completion_or_stopping'] is False and boundaries['descriptive_thresholds_change_completion_or_stopping'] is False,'Descriptive threshold scope')
    verify_grouped_timing(performance,boundaries,grouped_times,grouped_statuses)
    stats=check_statistics([g for g in games if g['stage']=='main'],r.read('analysis.json'),r.read('analysis_verification.json'))
    preservation=preservation_review(r,frozen,verification,pins)
    owning=owner_checks(r,frozen,expected_sha,completion,identities)
    require(verification['peak_combined_rss_bytes']<=frozen['max_rss_bytes']==2*GiB,'Sampled aggregate RSS cap')
    waits=[r.read_path(p) for p in sorted((OUT/'memory_admissions').glob('*.json'))]
    require(all(0<=x['wait_seconds']<=x['cumulative_wait_seconds']<600 and x['stable_seconds']==60 for x in waits),'Recorded bounded between-request waits')
    require(not waits or all(waits[i]['cumulative_wait_seconds']<waits[i+1]['cumulative_wait_seconds'] for i in range(len(waits)-1)),'Cumulative wait order')
    total=0
    for base,_,names in os.walk(OUT):
        r.guard();total+=sum((Path(base)/n).stat().st_size for n in names)
    require(total<16*GiB and shutil.disk_usage(ROOT).free>=5*GiB,'Completed evidence and maintained free-disk reserve')
    r.output_evidence_bytes=total
    for name in ('failure.json','finalization_failure.json','watchdog_error.json','guardian_error.json','collection_stop.json','qualification_stop.json','qualification_closure_failures.json'):
        require(not(OUT/name).exists(),'Incomplete/failure marker prevents whole-sample acceptance: '+name)
    for name in ('runner_stderr.log','watchdog_stderr.log','guardian_stderr.log'):
        if (OUT/name).exists():require(r.pin(OUT/name)['bytes']==0,'Unexplained owner stderr')
    r.counts.update(frozen_inputs=len(pins),sealed_artifacts=len(seal),sealed_bytes=sum(x['bytes'] for x in seal),
                    games=406,operational_games=6,main_games=400,pairs=200,qualification_scientific_decisions=203,qualification_ordinary_requests=84,live_decisions=moves,reproductions=len(chosen),workers=len(byworker),
                    graph_cases=11*len(byworker),parent_requests=parents,administrative_forfeits=len(administrative),output_bytes=total)
    return dict(statistics=stats,qualification=qualification_result,preservation=preservation,timing=timing,ownership=owning,counts=r.counts,
                saved_scientific_raw_graph_placement_receipts_reused=True,scientific_replays_rerun=False,
                bootstrap_rerun=False,selected_reproduction_outputs_compared=True,
                scope='Full frozen/sealed-byte readback and saved receipt reconciliation; independent fixed-sample score, fault, timing and reproduction selection/equality arithmetic. No model, search, ledger transition, bootstrap or test execution.',
                limitations=['The independent bootstrap receipt is authenticated and reconciled, not executed again.',
                             'Raw compression/scientific/graph/placement validity is established by the passed frozen validators and bound exact bytes, not reimplemented here.',
                             'The estimated strength is under the9s/10s research guard including administrative losses; no one-second deployment or unconstrained-strength claim.',
                             'Final acceptance requires an externally observed exit0 for this exact reviewer process and script.'])


def game_folder_name(spec):
    # Literal from the reviewed frozen runner.game_folder, checked at preparation.
    return f"games/{spec['stage']}/p{spec['pair_index']:03d}_g{spec['game_index']}"


def authenticate_all(r, frozen):
    """One bounded parallel pass, including every sealed byte; parsed JSON is reread."""
    jobs=[(r.physical(x['path']),x) for x in frozen['inputs']]
    for name in ('completed_evidence_seal.json','analysis_seal.json'):
        jobs.extend((r.physical(x['path']),x) for x in r.read(name)['artifacts'])
    for key in ('torch_runtime','external_runtime'):
        jobs.extend((Path(x['absolute_path']),x) for x in frozen[key])
    jobs.extend((Path(x['path']),x) for x in frozen['compression_runtime']['files'])
    r.verify_many(jobs)


def source_authentication(r, frozen, expected_sha):
    require(r.pin(OUT/'freeze.json')['sha256']==expected_sha,'Exact requested immutable freeze')
    pins={x['path']:x for x in frozen['inputs']}
    require(len(pins)==len(frozen['inputs']),'Unique source/dependency pins')
    for name in ('independent_review.py','review_runtime.py'):
        path=Path(__file__).parent/name; key=path.relative_to(ROOT).as_posix()
        require(key in pins,'Reviewer source must be explicitly frozen')
        r.verify(pins[key])
    require(sys.version==frozen['python'] and r.pin(Path(sys.executable))['sha256']==frozen['python_executable_sha256'],
            'Actual reader runtime')


def bind_shared_clock(r, sha):
    claim=r.read('guardian_claim.json'); final=r.read('stages/final_start.json')
    clock=r.read('postrun/review_owner_clock.json'); completion=r.read('completion.json')
    require(claim['freeze_sha256']==clock['freeze_sha256']==sha,'Shared owner clock freeze')
    require(final['cap_seconds']==14400 and completion['status']=='completed_verified', 'Original final-stage contract')
    mono=min(clock['review_monotonic_origin']+LIMIT,final['monotonic_deadline'],claim['monotonic_origin']+86400)
    wall=min(clock['review_unix_origin']+LIMIT,final['unix_deadline'],claim['unix_origin']+86400)
    require(clock['deadline_monotonic']==mono and clock['deadline_unix']==wall and
            completion['independent_acceptance_deadline_monotonic']==final['monotonic_deadline'] and
            completion['independent_acceptance_deadline_unix']==final['unix_deadline'], 'No fresh or extended acceptance budget')
    require(clock['guardian_claim_sha256']==r.pin(OUT/'guardian_claim.json')['sha256'] and
            clock['completion_sha256']==r.pin(OUT/'completion.json')['sha256'], 'Owner clock binds original claim/completion bytes')
    require(clock['review_monotonic_origin']>=final['monotonic'] and clock['review_unix_origin']>=final['unix_time'],
            'Review cannot precede the original final stage')
    r.shared_deadlines=(mono,wall);r.guard()
    return clock


def scientific_core(result, capture=True):
    # This is a literal identity/capture-only projection, never a quota/value edit.
    value=json.loads(canonical(result))
    for key in ('profile_id','profile_sha256','wrapper_id'):value.pop(key,None)
    value['diagnostics'].pop('profile_id',None)
    if not capture:
        for phase in value['diagnostics']['phases']:phase.pop('inference',None)
    return value


def qualification_review(r,frozen,byworker):
    r.verify(frozen['qualification_plan']); plan=r.read_path(r.physical(frozen['qualification_plan']['path']))
    require(plan['complete_decisions']==203 and plan['ordinary_requests']==84 and plan['total_parent_requests']==88 and
            plan['graph_cases']==22 and plan['world_pool']==3,'Exact actual128 qualification inventory')
    suites={}; counts={}
    for route,quota,extra in (('frozen',64,0),('research',128,15)):
        receipt=r.read('entrypoint_checks/'+route+'.json'); suite=receipt['result']; suites[route]=suite
        require(receipt['status']=='ok' and receipt['accepted'] is True and receipt['deadline_seconds']==300 and
                receipt['received_before_deadline'] is True and 0<=receipt['receipt_elapsed_seconds']<=300 and
                suite['status']=='entrypoint_counterexamples_passed' and suite['execution_route']==route and
                suite['challenger_quota']==quota and suite['reference_quota']==16 and suite['world_pool']==3 and
                suite['counts']=={'complete_decisions':52+extra} and suite['original_counts']=={'complete_decisions':52},
                'Actual startup suite is complete with correct unprojected quota')
        name='qualification_'+route;byworker[name]=[]
        require(r.read('workers/'+name+'/'+receipt['request_id']+'.json')==receipt,'Original actual-entrypoint receipt')
        path=OUT/'workers'/name/'entrypoint_forensics.jsonl'
        _,raw=r.pin(path,keep_bytes=True);events=[];previous=None
        require(len(raw)<=64*1024**2 and raw.endswith(b'\n'),'Bounded complete forensic chain')
        for index,line in enumerate(raw.splitlines()):
            r.guard();require(index<2048 and len(line)<=4*1024**2,'Forensic record bounds')
            item=json.loads(line)
            require(set(item)=={'sequence','previous_sha256','event'} and item['sequence']==index and item['previous_sha256']==previous,
                    'Exact startup forensic event chain')
            previous=digest(item);events.append(item['event'])
        require(not any(e.get('event') in ('failed','entrypoint_failed','worker_decision_failed') for e in events),'No failed startup accepted')
        ordinary=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='result']
        validated=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='validated']
        rejected=[e for e in events if e.get('operation')=='ordinary_worker_request' and e['event']=='rejected']
        direct=[e for e in events if e.get('operation')=='reference_choose' and e['event']=='result']
        same=[e for e in events if e.get('operation')=='direct_same_quota_choose' and e['event']=='result']
        require(len(ordinary)==len(validated)==37 and len(rejected)==26 and len(direct)==15 and len(same)==extra,
                '37 accepted handlers/26 intentional rejects/15 references plus genuine128 direct evidence')
        fixtures={x['id']:x for x in suite['fixtures']}; require(len(fixtures)==15,'All fifteen corrected fixtures')
        for case in fixtures:
            for arm in ('search','reference'):
                matching=[e for e in ordinary if e['case_id']==case and e['arm']==arm]
                require(len(matching)==1 and matching[0]['result']==fixtures[case]['results'][arm],'Startup summary binds actual raw handler output')
        same_summary={x['id']:x for x in suite['direct_same_quota']}
        require(len(same_summary)==extra and (not extra or set(same_summary)==set(fixtures)),'Genuine128 direct reproduction coverage')
        for event in same:
            expected=fixtures[event['case_id']]['results']['search'];item=same_summary[event['case_id']]
            require(event['request']['quota']==128 and event['result']==item['result']==dict(action=expected['action'],diagnostics=expected['diagnostics']) and
                    item['exact_callback_arrays_and_hashes'] is True,'Exact genuine128 direct complete arrays and action')
        require(len(suite['worker_rejections'])==26 and len(suite['actual_encoded_costs'])==30 and
                suite['native_contract']['status']=='passed' and suite['native_contract']['count']==28,
                'Original corrected-rule and native support coverage retained')
        counts[route]=dict(complete_decisions=52+extra,ordinary_handlers=37,intentional_rejects=26,direct_reference=15,direct_same_quota=extra)
    paths=sorted((OUT/'qualification_rows').glob('*.json'))
    require([p.name for p in paths]==[f'{i:04d}.json' for i in range(84)],'All84 exact ordinary qualification requests')
    results={};kinds=Counter()
    for spec,path in zip(plan['parity_requests'],paths,strict=True):
        row=r.read_path(path);route=spec['route'];receipt=row['clock'];result=receipt['result']
        require(row['spec']==spec and row['route']==route and row['completed'] is True and receipt['status']=='ok' and
                receipt['accepted'] is True and receipt['deadline_seconds']==10 and receipt['received_before_deadline'] is True and
                0<=receipt['receipt_elapsed_seconds']<=10 and result['action']==row['action'] and result['diagnostics']==row['trace'],
                'Exact successful ordinary qualification envelope')
        require(spec['quota']==(128 if route=='research' else 64) and spec['world_pool']==3 and
                result['simulations_quota']==spec['quota'] and
                result['observer_sha256']==digest(spec['observer'])==spec['observer_sha256'],'Qualification quota/observer binding')
        name='qualification_'+route;require(Path(row['worker_folder']).name==name,'Qualification worker route')
        byworker[name].append(receipt['request_id'])
        require(r.read_path(r.physical(row['worker_clock_file']))==receipt,'Original ordinary qualification receipt')
        kind=spec['kind'];kinds[kind]+=1
        if kind=='saved_reproduction':
            r.verify(spec['expected_record']);old=r.read_path(r.physical(spec['expected_record']['path']))
            require(old['completed'] and old['observer']==spec['observer'] and
                    scientific_core(result,False)==scientific_core(old['clock']['result'],False),'Exact historical64 science')
        elif kind=='live_edge':
            expected=next(x for x in suites[route]['fixtures'] if x['id']==spec['expected_case'])['results']['search']
            require(result==expected,'Full current actual edge reproduction')
        elif kind=='exact_repeat':require(result==results[spec['repeat_of']],'Full exact ordinary repeat')
        else:require(kind=='fresh_reserved','No unplanned qualification category')
        results[spec['index']]=result
    require(kinds==dict(saved_reproduction=24,fresh_reserved=36,live_edge=12,exact_repeat=12),'Fixed qualification categories')
    summary=r.read('qualification_summary.json')
    require(summary['qualified'] is True and summary['requests']==84 and summary['failure'] is None and
            summary['parity']['status']=='passed' and summary['parity']['counts']==dict(kinds) and summary['parity']['complete_decisions']==84 and
            summary['timing_gate']['passed'] is True and summary['timing_gate']['timing_used_for_admission'] is False,
            'Successful saved full qualification summary; margins descriptive only')
    return dict(scientific_decisions=203,ordinary_requests=84,administrative_requests=4,graph_cases=22,suites=counts,
                categories=dict(kinds),forensic_chains_reconstructed=True,scientific_transition_validators_reused=True)


def verify_grouped_timing(performance,boundaries,times,statuses):
    def expected(key):
        values=times.get(key,[]);state=statuses.get(key,Counter())
        return dict(completed_decisions=len(values),attempted_decisions=sum(state.values()),statuses=dict(state),
            completed_arrival_seconds=distribution(values),completed_over_700ms=sum(x>.7 for x in values),
            completed_over_900ms=sum(x>.9 for x in values),completed_over_one_second=sum(x>1 for x in values),
            completed_at_or_above_one_second=sum(x>=1 for x in values))
    for row in boundaries['groups']:
        key=(row['stage'],row['player_role']);require(all(row[k]==v for k,v in expected(key).items()),'Independent stage/role timing')
    for label,rows in (('performance',performance['by_stage_and_seat']),('boundary',boundaries['by_stage_and_seat'])):
        keys=[(x['stage'],x['player_role'],x['seat']) for x in rows]
        require(len(keys)==len(set(keys)) and set(keys)==set(product(('operational','main'),ROLES,(0,1))),label+' exact eight stage/role/seat cells')
        for row,key in zip(rows,keys):require(all(row[k]==v for k,v in expected(key).items()),label+' independent timing thresholds')


def preservation_review(r,frozen,verification,pins):
    r.verify(verification['prior_preservation']);proof=r.read_path(r.physical(verification['prior_preservation']['path']))
    require(proof['status']=='passed' and proof['scope']=='finite_referenced_dependencies' and
            proof['historical_statuses_changed'] is False and proof['mutation_of_previous_sources_or_evidence'] is False,
            'Finite historical preservation proof')
    require(proof['prior_manifest']==frozen['prior_preservation'],'Historical preservation exact frozen manifest')
    r.verify(proof['prior_manifest']);manifest=r.read_path(r.physical(proof['prior_manifest']['path']))
    require(manifest['scope']=='finite_referenced_dependencies' and manifest['earlier_failures_preserved'] is True and
            all(pins.get(x['path'])==x for x in manifest['artifacts']),'Every inherited source/evidence anchor authenticated')
    require(proof['prior_files']==len(manifest['artifacts']) and proof['prior_bytes']==sum(x['bytes'] for x in manifest['artifacts']),
            'Historical preservation counts')
    for row in manifest['historical_statuses']:
        require(r.read_path(r.physical(row['path']))[row['field']]==row['expected'],'Historical disposition unchanged')
    for name in manifest['required_absent']:require(not r.physical(name).exists(),'Historical failed review never relabelled')
    return dict(status='passed',files=proof['prior_files'],bytes=proof['prior_bytes'],scope=proof['scope'],
                unrelated_historical_games_reaudited=False)


def owner_checks(r,frozen,sha,completion,identities):
    execution=r.read('execution.json');outer=r.read('watchdog_execution.json');os_exit=r.read('guardian_os_exit.json')
    guardian=r.read('guardian_claim.json');owner=r.read('guardian_owner.json');watchdog=r.read('watchdog.json')
    claim=r.read('watchdog_claim.json');launch=r.read('launch.json')
    require(all(x['freeze_sha256']==sha for x in (execution,outer,os_exit,guardian,watchdog,claim,launch)), 'Actual owning freeze chain')
    require(owner['guardian']==guardian and owner['watchdog']==dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']) and
            (launch['pid'],launch['created'])==(watchdog['runner_pid'],watchdog['runner_created']) and
            (claim['watchdog_pid'],claim['watchdog_created'])==(watchdog['watchdog_pid'],watchdog['watchdog_created']) and
            (claim['guardian_pid'],claim['guardian_created'])==(guardian['guardian_pid'],guardian['guardian_created']),
            'Exact PID/creation ownership chain')
    require(os_exit['guardian_pid']==guardian['guardian_pid'] and abs(os_exit['guardian_created']-guardian['guardian_created'])<=.0001 and
            os_exit['exit_code']==0 and os_exit['observed_by']=='owned Process handle' and os_exit['cleanup_reason'] is None,
            'Actual retained guardian OS handle exit0')
    for end in (execution,outer):
        require(end['status']=='completed' and end['exit_code']==0 and end['cleanup_errors']==[] and
                end['cleanup_receipt']['errors']==end['cleanup_receipt']['unconfirmed']==[], 'Actual exit0 and clean owned closure')
        charged=max(end['monotonic_end']-end['monotonic_origin'],end['unix_end']-end['unix_origin'])
        require(abs(charged-end['charged_job_seconds'])<1e-6 and charged<86400 and
                end['monotonic_origin']==guardian['monotonic_origin'] and end['unix_origin']==guardian['unix_origin'],
                'Original dual nonreset clock')
    require(outer['observed_process']=='owning_watchdog' and outer['exit_code_observed_by']=='guardian Popen handle' and outer['error'] is None,
            'Watchdog actual owning handle')
    for key in ('elapsed_seconds','wall_elapsed_seconds'):require(0<=completion[key]<86400,'Original global bound')
    for stage,cap in frozen['stage_seconds'].items():
        start=r.read('stages/'+stage+'_start.json');end=r.read('stages/'+stage+'_end.json')
        require(start['cap_seconds']==cap and start['monotonic']<=end['monotonic']<start['monotonic_deadline'] and
                start['unix_time']<=end['unix_time']<start['unix_deadline'],'Bounded original stage clocks')
    launcher=r.read('launcher_placement.json')
    require(launcher['freeze_sha256']==sha and launcher['logical_processors']==list(range(2,20)) and launcher['priority']=='Normal' and
            launcher['pid']==os_exit['owner_pid'],'Actual retained launch-owner placement')
    identities.extend([dict(pid=guardian['guardian_pid'],created=guardian['guardian_created']),
        dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']),dict(pid=launch['pid'],created=launch['created'])])
    unique={(x['pid'],x['created']):x for x in identities}
    # The launch owner is deliberately alive until this reader and acceptance
    # terminate; its eventual OS exit0 is an external final condition.
    return dict(actual_experiment_os_exit_codes=dict(runner=0,watchdog=0,guardian=0),
        owned_experiment_identities_absent=[r.absent(x) for x in unique.values()],
        launch_owner_pid=launcher['pid'],launch_owner_actual_exit_required_after_acceptance=True,
        maximum_charged_seconds=max(execution['charged_job_seconds'],outer['charged_job_seconds']))


def identity_receipt(r):
    with r._lock:
        return dict(files=[dict(path=x['path'],identity=list(x['identity'])) for x in r._cache.values()],
                    directories=[dict(path=p,identity=list(marker)) for p,marker in r._directories.values()])


def verify_identity_receipt(r,value):
    from .review_runtime import identity
    require(type(value['files']) is list and type(value['directories']) is list,'Final authenticated identity inventory')
    seen=set()
    for item in value['files']:
        r.guard();key=r._key(item['path']);require(key not in seen,'Duplicate final file identity');seen.add(key)
        require(list(identity(r._stat(item['path'])))==item['identity'],'Authenticated bytes changed between component and acceptance: '+item['path'])
    seen.clear()
    for item in value['directories']:
        r.guard();key=r._key(item['path']);require(key not in seen,'Duplicate final directory identity');seen.add(key)
        current=r._stat(item['path'],directory=True)
        require([current.st_dev,current.st_ino,current.st_mode,getattr(current,'st_file_attributes',0)]==item['identity'],
                'Authenticated parent changed before acceptance')
    return dict(files=len(value['files']),directories=len(value['directories']),status='passed')


def guarded_publish(r,path,value):
    """An overrun while publishing never leaves an unqualified acceptance."""
    r.guard()
    if hasattr(r,'output_evidence_bytes'):
        # Prospective exact UTF-8 size, plus a small explicit allowance for the
        # retained owner's terminal receipts. No study file is deleted to fit.
        size=len((canonical(value)+'\n').encode('utf-8'))
        require(r.output_evidence_bytes+size+1024**2<16*GiB,'Independent publication would exceed original evidence cap')
    save_new(path,value)
    if hasattr(r,'output_evidence_bytes'):r.output_evidence_bytes+=path.stat().st_size
    r.guard()


def failure_receipt(name,exc):
    path=OUT/'postrun'/name
    if not path.exists():
        save_new(path,dict(status='failed_no_acceptance',error=repr(exc),traceback=traceback.format_exc(),
            time=datetime.now(timezone.utc).isoformat(),models_run=False,games_run=False))


def main_review(sha):
    r=Review()
    try:
        clock=bind_shared_clock(r,sha);frozen=r.read('freeze.json');source_authentication(r,frozen,sha)
        require(not(OUT/'postrun/independent_completion_review.json').exists() and
                not(OUT/'postrun/acceptance_invalidated.json').exists(),'No overwritten/recovered acceptance in this first review')
        result=review_all(r,sha)
        integrity=r.revalidate_identities()
        identities=OUT/'postrun/independent_review_identities.json'
        guarded_publish(r,identities,identity_receipt(r))
        identity_pin=r.pin(identities)
        component=dict(status='passed_saved_evidence_component',accepted=False,freeze_sha256=sha,
            reviewer_pid=os.getpid(),reviewer_created=psutil.Process().create_time(),
            reviewer_source=r.pin(Path(__file__)),runtime_source=r.pin(Path(__file__).with_name('review_runtime.py')),
            shared_owner_clock=clock,maximum_review_seconds=LIMIT,identity_inventory=identity_pin,
            result=result,final_identity_validation=integrity,authentication_metrics=dict(r.metrics),
            peak_reader_rss_bytes=r._peak_rss,elapsed_seconds=time.monotonic()-r.start,
            consumed_inputs=sorted(r.consumed.values(),key=lambda x:x['path']),
            model_calls=0,search_calls=0,new_games=0,scientific_replays=0,bootstrap_executions=0,
            acceptance_requires_actual_reader_exit0=True,acceptance_requires_shared_deadline=True)
        guarded_publish(r,OUT/'postrun/independent_review_component.json',component)
        print(canonical(dict(status=component['status'],counts=result['counts'],elapsed_seconds=time.monotonic()-r.start)))
        return 0
    except BaseException as exc:
        failure_receipt('independent_review_failure.json',exc)
        if (OUT/'postrun/independent_review_component.json').exists():failure_receipt('acceptance_invalidated.json',exc)
        print(canonical(dict(status='failed_no_acceptance',error=repr(exc))))
        return 1


def accept_exit(sha):
    r=Review()
    try:
        clock=bind_shared_clock(r,sha);frozen=r.read('freeze.json');source_authentication(r,frozen,sha)
        require(not(OUT/'postrun/independent_review_failure.json').exists() and
                not(OUT/'postrun/acceptance_invalidated.json').exists(),'No failed component may be accepted')
        component=r.read('postrun/independent_review_component.json');component_pin=r.pin(OUT/'postrun/independent_review_component.json')
        r.output_evidence_bytes=0
        for base,_,names in os.walk(OUT):
            r.guard();r.output_evidence_bytes+=sum((Path(base)/n).stat().st_size for n in names)
        terminal=r.read('postrun/independent_review_os_exit.json')
        require(component['status']=='passed_saved_evidence_component' and component['accepted'] is False and
                component['freeze_sha256']==sha and component['shared_owner_clock']==clock and
                terminal['freeze_sha256']==sha and terminal['exit_code']==0 and terminal['observed_by']=='owned Process handle' and
                terminal['cleanup_reason'] is None and terminal['reader_pid']==component['reviewer_pid'] and
                abs(terminal['reader_created']-component['reviewer_created'])<=.0001 and
                terminal['script_sha256']==component['reviewer_source']['sha256']==r.pin(Path(__file__))['sha256'] and
                terminal['component_sha256']==component_pin['sha256'], 'Actual exact reviewer handle exit0')
        absence=r.absent(dict(pid=component['reviewer_pid'],created=component['reviewer_created']))
        r.verify(component['identity_inventory']);identities=r.read_path(r.physical(component['identity_inventory']['path']))
        identity_check=verify_identity_receipt(r,identities)
        # Reauthenticate all reviewer/runtime source files after this final check.
        source_authentication(r,frozen,sha);r.revalidate_identities();r.guard()
        receipt=dict(status='accepted_saved_evidence_pending_final_owner_exit',accepted=True,freeze_sha256=sha,
            component=component_pin,reader_exit=r.pin(OUT/'postrun/independent_review_os_exit.json'),
            shared_owner_clock=clock,reader_absence=absence,final_identity_check=identity_check,
            result=component['result'],model_calls=0,new_games=0,search_calls=0,
            requires_actual_acceptance_exit0=True,requires_actual_owner_exit0=True,
            acceptance_invalid_if_marker_present='postrun/acceptance_invalidated.json',
            consumed_inputs=sorted(r.consumed.values(),key=lambda x:x['path']),
            time=datetime.now(timezone.utc).isoformat())
        guarded_publish(r,OUT/'postrun/independent_completion_review.json',receipt)
        print(canonical(dict(status=receipt['status'],accepted=True)))
        return 0
    except BaseException as exc:
        failure_receipt('acceptance_invalidated.json',exc)
        print(canonical(dict(status='failed_no_acceptance',error=repr(exc))))
        return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--accept-exit',action='store_true')
    args=parser.parse_args();raise SystemExit(accept_exit(args.freeze_sha256) if args.accept_exit else main_review(args.freeze_sha256))


