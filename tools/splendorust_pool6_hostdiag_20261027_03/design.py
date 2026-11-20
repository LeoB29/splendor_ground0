"""Prospective saved roots and request order; no models or fresh games."""
from copy import deepcopy
from . import common as c

PREVIOUS = c.ROOT / 'outputs/splendorust_pool6_admission_20261027_02'
PREVIOUS_FREEZE_SHA = '09551c1f9e8782e0d717576aa51f96bff47403637b56cb087fd843ff8c58c0e8'
PREVIOUS_REVIEW_SHA = 'e68f6dc7b34c49147a933303ebfc9bdcb97671951e15e143a71a0b4f990f615e'
PRESERVED_ATTEMPT = c.ROOT / 'outputs/splendorust_pool6_hostdiag_20261027_02'
PRESERVED_FREEZE_SHA = '49677386c480a0b5fdcd4d37b824065cf819f1559d7178471331a5ecca28cff1'
PRESERVED_REVIEW_SHA = 'c9f2d5d0a4709a41a1dca8435c17f70ae59f399f1c3d204ab00671d525073b8c'
PANEL = c.OUT / 'saved_panel.json'
MATRIX = c.OUT / 'request_matrix.json'
TARGETS = ((0,4),(0,6),(0,20),(1,41),(2,36),(2,39),(2,47),(3,26),(4,19))
CONTROL_PLIES = (2,0,18,39,34,37,45,24,17)
ROUNDS = 4


def fixed(path, value):
    if path.exists(): c.require(c.read(path) == value, 'Prepared artifact changed: '+str(path))
    else: c.save(path,value)
    return c.artifact(path)


def move_path(game,ply):
    return PREVIOUS / 'games/operational' / f'p{game//2:03d}_g{game}' / 'moves' / f'{ply:04d}.json'


def build_panel():
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import audit as ledger
    selected=[]; controls=[]; used=set(TARGETS)
    for pair,(game,ply) in enumerate(TARGETS):
        path=move_path(game,ply); target=c.read(path)
        c.require(target['completed'] and target['clock']['accepted'] and
            target['clock']['status']=='ok' and target['clock']['receipt_elapsed_seconds']>=.7,
            'Fixed target no longer a successful margin miss')
        c.require(target['trace']['inference_calls']==65,'Fixed target callback work')
        eligible=[]
        for p in sorted(path.parent.glob('*.json')):
            row=c.read(p)
            # Selection deliberately never examines control receipt latency.
            if (game,row['ply']) not in used and row['ply']<ply and row['route']==target['route'] and \
                    row['completed'] and row['clock']['status']=='ok' and row['trace'].get('inference_calls')==65:
                eligible.append((row['ply'],p,row))
        c.require(bool(eligible),'No unique earlier same-role structural control')
        control_ply,control_path,control=max(eligible,key=lambda x:x[0])
        c.require(control_ply==CONTROL_PLIES[pair],'Fixed deterministic control identity')
        used.add((game,control_ply))
        selected.append((pair,game,ply,path,target,'observed_slow_target'))
        controls.append((pair,game,control_ply,control_path,control,'fixed_structural_control'))
    positions=[]
    for index,(pair,game,ply,path,row,selection) in enumerate(selected+controls):
        before,observer=row['before'],row['observer']
        decode_training_snapshot(before);ledger.check_observer(before,observer)
        expectations={route:dict(expected_record=c.artifact(path)) if route==row['route'] else
            dict(expected_first_observation=True) for route in ('frozen','research')}
        positions.append(dict(position_index=index,case_id=f'g{game}_p{ply}',pair_index=pair,
            selection=selection,source_game=game,source_ply=ply,source_route=row['route'],
            source_record=c.artifact(path),observer=observer,observer_sha256=c.digest(observer),
            before_snapshot=before,before_sha256=c.digest(before),expectations=expectations,
            source_inference_calls=row['trace']['inference_calls'],
            source_phase_count=len(row['trace']['phases'])))
    c.require(len(positions)==18 and len({p['observer_sha256'] for p in positions})==18,
        'Eighteen unique unchanged saved roots')
    return dict(schema='pool6_hostdiag_saved_panel_v1',positions=positions,
        targets=[p['case_id'] for p in positions[:9]],controls=[p['case_id'] for p in positions[9:]],
        selection_rule='Fixed nine successful margin misses; for each in fixed order choose nearest earlier same-game/same-role completed65-callback root excluding every target and already-selected controls; no control timing filter',
        historical_parity='Exact original route only; other route validated scientifically and compared exactly across all repeats',
        new_games=0,new_trajectories=0)


def build_matrix(panel=None):
    panel=build_panel() if panel is None else panel
    requests=[]
    for stage,indices in (('overhead',list(range(9,18))),('diagnostic',list(range(18)))):
        for repeat in range(ROUNDS):
            order=indices[repeat:]+indices[:repeat]
            if repeat%2:order.reverse()
            for index in order:
                p=panel['positions'][index]
                routes=('frozen','research') if (repeat+index)%2==0 else ('research','frozen')
                for route in routes:
                    requests.append(dict(index=len(requests),sequence=len(requests),round=repeat,
                        position_index=index,stage=stage,observer_enabled=stage=='diagnostic' or repeat in (1,2),
                        route=route,kind='host_interference',case_id=p['case_id'],quota=64,
                        world_pool=3 if route=='frozen' else 6,seed_label='A',telemetry=False,
                        observer=deepcopy(p['observer']),observer_sha256=p['observer_sha256'],
                        before_snapshot=deepcopy(p['before_snapshot']),**p['expectations'][route]))
    c.require(len(requests)==216 and sum(x['stage']=='overhead' for x in requests)==72,
        'Fixed72 overhead and144 conditional diagnostic requests')
    return dict(schema='pool6_hostdiag_request_matrix_v1',requests=requests,rounds=ROUNDS,
        position_order='Cyclic shift by round; reverse odd rounds',
        arm_order='frozen first iff (round + position_index) is even',
        observer_order='Overhead OFF,ON,ON,OFF per cell; diagnostic always ON',
        no_warmup_dropping=True,no_retries=True,no_replacements=True)


def overhead_contract():
    return dict(requests=72,controls=9,players=2,rounds=4,per_player_mode_count=18,
        order=[False,True,True,False],maximum_mean_ratio=1.05,maximum_mean_added_seconds=.010,
        maximum_p95_added_seconds=.025,percentile='nearest_rank',
        all_correct_required=True,one_second_deadline_unchanged=True,
        per_player_all_conditions_required=True,trimmed_observations=0,
        failure_skips_full_diagnostic=True)


def host_observer_contract():
    return dict(interval_seconds=.1,maximum_buffer_bytes=c.HOST_OBSERVER_MAX_BYTES,
        external_to_player=True,active_decision_disk_writes=False,induced_load=False,
        measurements=['available physical memory','system commit headroom','process working set and private bytes',
            'process page-fault counts (soft and hard combined)','process CPU time; host-wide CPU utilization only when all observed intervals are consistent',
            'raw live-thread context-switch aggregates and signed net changes; true interval totals unavailable',
            'process I/O counters (not paging counters)'],
        host_cpu_policy=dict(required_for_measurement_acceptance=False,
            all_raw_samples_preserved=True,consistency_scope='every adjacent native sample and each derived span',
            invalid_window_scope='host-wide CPU unavailable for entire window and all subspans; no recovery by selecting endpoints',
            missing_value=None,offending_pairs_reported=True,clamp=False,discard_samples=False,reset=False,
            fatal_failures=['API failure','invalid counter type or shape','invalid timestamp','identity mismatch','invalid process counter']),
        limitations=['No direct runnable-queue delay measurement','Page-fault count does not identify hard faults',
            'Host-wide CPU readings may be inconsistent; report availability and denominators, never infer a reset cause',
            'Live-thread aggregates are non-monotonic and do not identify interval switch totals at any sign',
            'Observational timing association is not a causal load experiment'])


def build_plan():
    return dict(schema='pool6_hostdiag_bounded_plan_v3',phase=c.PHASE,
        purpose='Measure recurrence and host interference on saved roots after observer overhead qualification',
        positions=18,selected_slow_targets=9,controls=9,rounds=4,
        overhead_requests=72,conditional_diagnostic_requests=144,maximum_ordinary_requests=216,
        startup_complete_decisions=104,graph_cases=22,maximum_parent_requests=220,
        maximum_scientific_decisions=320,resident_workers=2,worker_replacements=0,
        candidate=dict(route='research',world_pool=6,simulations=64,seed_label='A'),
        control=dict(route='frozen',world_pool=3,simulations=64,seed_label='A'),
        seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_role='Descriptive recurrence only; both prior negative admissions remain negative',
        overhead_contract=overhead_contract(),host_observer_contract=host_observer_contract(),
        failure_policy='Stop at first timeout, mismatch, unexpected rejection or resource/integrity failure; preserve prefix. Overhead-gate failure skips full diagnostic and finalizes negative. No retries.',
        players_unchanged=True,payload_changes=False,existing_observer_unchanged=True,
        new_games=0,promotion=False,training=False,larger_data_deferred=True,external_testing='paused',
        resources=dict(job_seconds=c.JOB_SECONDS,collection_seconds=c.COLLECTION_CUTOFF_SECONDS,
            stage_seconds=c.STAGES,evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
            max_worker_rss=2*c.GiB,initial_disk=c.INITIAL_FREE_DISK_BYTES,maintained_disk=5*c.GiB,
            inherited_pre_request_ram_reserve=2*c.GiB,diagnostic_available_ram=c.DIAGNOSTIC_AVAILABLE_RAM_BYTES,
            available_ram_guard_scope='Before requests; no new active-request RAM stop or clock reset',
            stable_initial_and_recovery_seconds=60,cumulative_ram_wait_seconds=c.RAM_WAIT_SECONDS))


def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    fixed(c.OUT/'runtime_sources.json',verify_runtime_sources())
    panel=build_panel();matrix=build_matrix(panel)
    c.require(panel==c.read(PRESERVED_ATTEMPT/'saved_panel.json') and
        matrix==c.read(PRESERVED_ATTEMPT/'request_matrix.json'),'Same roots and exact request order as preserved attempt')
    fixed(PANEL,panel);fixed(MATRIX,matrix);fixed(c.OUT/'study_plan.json',build_plan())
    return dict(status='prepared_model_free',positions=18,overhead_requests=72,conditional_diagnostic_requests=144,new_games=0)


if __name__=='__main__':print(c.canonical(prepare()))
