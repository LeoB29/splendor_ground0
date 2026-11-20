"""Prospective four-root, four-fixed-worker core-placement diagnostic."""
from copy import deepcopy
from . import common as c
from tools.splendorust_pool6_hostdiag_20261027_03.design import host_observer_contract as inherited_observer_contract

PRESERVED_ATTEMPT = c.ROOT / 'outputs/splendorust_pool6_hostdiag_20261027_03'
PRESERVED_FREEZE_SHA = 'd3c4231c8676822dfff40a59d7f5abc8b9d0eef8b6089582cc8cc2fce7f794d1'
PRESERVED_REVIEW_SHA = 'a78c75e516bd1fe01148448befdfb6bb86f172de5d1b27bc3225f72210b65321'
PRESERVED_DIAGNOSIS_SHA = 'e48e4e7b578f354d7146de99f71c42911f39d65f9337e9099e75caffa3affd43'
PANEL = c.OUT / 'saved_panel.json'
MATRIX = c.OUT / 'request_matrix.json'
TARGETS = ('g2_p36', 'g2_p45')
CONTROLS = ('g2_p34', 'g2_p37')
CORES = (0, 2)
ROUTES = ('frozen', 'research')
ROUNDS = 12
ARMS = tuple(f'{route}_cpu{core}' for route in ROUTES for core in CORES)


def fixed(path, value):
    if path.exists(): c.require(c.read(path) == value, 'Prepared artifact changed: '+str(path))
    else: c.save(path,value)
    return c.artifact(path)


def build_panel():
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import audit as ledger
    c.require(c.file_hash(PRESERVED_ATTEMPT/'freeze.json')==PRESERVED_FREEZE_SHA,'Prior freeze changed')
    old=c.read(PRESERVED_ATTEMPT/'saved_panel.json')['positions'];by_id={p['case_id']:p for p in old}
    selected=[]
    for target_id in TARGETS:
        target=by_id[target_id]
        eligible=[p for p in old if p['selection']=='fixed_structural_control' and
            p['case_id'] not in TARGETS and p['case_id'] not in selected and
            p['source_game']==target['source_game'] and p['source_route']==target['source_route'] and
            p['source_ply']<target['source_ply'] and p['source_inference_calls']==65]
        c.require(bool(eligible),'No earlier eligible fixed control')
        selected.append(max(eligible,key=lambda p:p['source_ply'])['case_id'])
    c.require(tuple(selected)==CONTROLS,'Frozen structural control identities')
    references={}
    for path in sorted((PRESERVED_ATTEMPT/'records').glob('*.json')):
        row=c.read(path);key=(row['spec']['case_id'],row['route'])
        if key[0] in TARGETS+CONTROLS and key not in references:
            c.require(row['completed'] and row['clock']['accepted'] and row['clock']['status']=='ok',
                'Prior successful exact result required')
            references[key]=(path,row)
    positions=[]
    for index,case in enumerate(TARGETS+CONTROLS):
        p=deepcopy(by_id[case]);p['position_index']=index;p['pair_index']=index%2
        p['selection']='observed_slow_target' if index<2 else 'fixed_structural_control'
        decode_training_snapshot(p['before_snapshot']);ledger.check_observer(p['before_snapshot'],p['observer'])
        p['expectations']={}
        for route in ROUTES:
            path,row=references[case,route]
            c.require(row['observer']==p['observer'] and row['before']==p['before_snapshot'],'Exact saved root')
            p['expectations'][route]=dict(expected_record=c.artifact(path))
        positions.append(p)
    c.require(len({p['observer_sha256'] for p in positions})==4,'Four unique roots')
    slow_sources=[]
    for seq,case,route in ((80,TARGETS[0],'frozen'),(170,TARGETS[1],'research')):
        path=PRESERVED_ATTEMPT/'records'/f'{seq:04d}.json';row=c.read(path)
        c.require(row['spec']['case_id']==case and row['route']==route and
            row['completed'] and .7<=row['clock']['receipt_elapsed_seconds']<1.,'Saved slow source binding')
        slow_sources.append(c.artifact(path))
    return dict(schema='pool6_core_compare_saved_panel_v1',positions=positions,
        targets=list(TARGETS),controls=list(CONTROLS),slow_source_records=slow_sources,
        selection_rule='Two v3 margin crossings; nearest earlier same-game/same-source-role root among v3 fixed controls, excluding targets and already selected controls. No control-latency selection.',
        historical_parity='Exact accepted v3 full scientific result for both players on all four roots',
        new_games=0,new_trajectories=0)


def build_matrix(panel=None):
    panel=build_panel() if panel is None else panel
    requests=[]
    latin=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))
    for stage in ('overhead','diagnostic'):
        for repeat in range(ROUNDS):
            order=([2,3] if repeat%2==0 else [3,2]) if stage=='overhead' else latin[repeat%4]
            for index in order:
                p=panel['positions'][index]
                routes=ROUTES if (repeat//2+index)%2==0 else tuple(reversed(ROUTES))
                cores=CORES if repeat%2==0 else tuple(reversed(CORES))
                for route in routes:
                    pair_index=len(requests)//2
                    for core in cores:
                        requests.append(dict(index=len(requests),sequence=len(requests),round=repeat,
                            position_index=index,stage=stage,observer_enabled=stage=='diagnostic' or repeat%4 in (1,2),
                            route=route,logical_cpu=core,arm_id=f'{route}_cpu{core}',core_pair_index=pair_index,
                            kind='core_placement',case_id=p['case_id'],quota=64,
                            world_pool=3 if route=='frozen' else 6,seed_label='A',telemetry=False,
                            observer=deepcopy(p['observer']),observer_sha256=p['observer_sha256'],
                            before_snapshot=deepcopy(p['before_snapshot']),**p['expectations'][route]))
    c.require(len(requests)==288 and sum(x['stage']=='overhead' for x in requests)==96,
        'Fixed96 overhead and192 conditional diagnostic requests')
    return dict(schema='pool6_core_compare_request_matrix_v1',requests=requests,rounds=ROUNDS,
        position_order='Controls reverse each round; four-root Williams square repeated3 for full stage',
        arm_order='Adjacent CPU pair for each root/player; CPU0 leads even rounds, CPU2 odd; player lead alternates by floor(round/2)+position',
        observer_order='Overhead OFF,ON,ON,OFF repeated3 per cell; diagnostic always ON',
        no_warmup_dropping=True,no_retries=True,no_replacements=True)


def overhead_contract():
    return dict(requests=96,controls=2,players=2,cores=[0,2],rounds=12,per_arm_mode_count=12,
        order=[False,True,True,False]*3,maximum_mean_ratio=1.05,maximum_mean_added_seconds=.010,
        maximum_p95_added_seconds=.025,percentile='nearest_rank',
        all_correct_required=True,one_second_deadline_unchanged=True,
        per_arm_all_conditions_required=True,trimmed_observations=0,
        failure_skips_full_diagnostic=True)


def engineering_contract():
    return dict(completed_verified_full_stage_required=True,margin_seconds=.7,
        cpu2_all_full_responses_strictly_below_margin=True,
        maximum_per_player_mean_ratio=1.05,maximum_per_player_p95_added_seconds=.025,
        advantage_rule='After all eligibility conditions: CPU0 has a margin crossing absent on CPU2, or CPU2 p95 improves by at least25ms for either player',
        minimum_p95_gain_seconds=.025,statistical_significance_claim=False,
        eligibility_means='May recommend fresh admission checks only; no automatic player promotion or live games')


def host_observer_contract():
    value=deepcopy(inherited_observer_contract())
    value.update(maximum_compute_targets=12,supervisor_logical_processors=list(range(4,20)),
        worker_architecture='Four fixed resident trees; only one request active at a time; all idle trees monitored')
    value['limitations']+=['One persistent process instance per player/core; instance and core effects cannot be separated',
        'External applications may still use these cores; placement is not an exclusive reservation']
    return value


def build_plan():
    return dict(schema='pool6_core_compare_bounded_plan_v1',phase=c.PHASE,
        purpose='Test CPU placement sensitivity on two saved slow roots and fixed controls after per-arm observer overhead qualification',
        positions=4,selected_slow_targets=2,controls=2,rounds=12,logical_cpus=[0,2],arms=list(ARMS),
        overhead_requests=96,conditional_diagnostic_requests=192,maximum_ordinary_requests=288,
        startup_complete_decisions=208,graph_cases=44,maximum_parent_requests=296,
        maximum_scientific_decisions=496,resident_workers=4,worker_replacements=0,
        candidate=dict(route='research',world_pool=6,simulations=64,seed_label='A'),
        control=dict(route='frozen',world_pool=3,simulations=64,seed_label='A'),
        seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_role='Recorded engineering criterion, not stop at margin; previous negative admissions remain negative',
        overhead_contract=overhead_contract(),engineering_contract=engineering_contract(),host_observer_contract=host_observer_contract(),
        failure_policy='Stop at first timeout, mismatch, unexpected rejection or resource/integrity failure; preserve prefix. Any arm overhead-gate failure skips all full diagnostics and finalizes negative. No retries.',
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
    fixed(PANEL,panel);fixed(MATRIX,matrix);fixed(c.OUT/'study_plan.json',build_plan())
    return dict(status='prepared_model_free',positions=4,overhead_requests=96,conditional_diagnostic_requests=192,new_games=0)


if __name__=='__main__':print(c.canonical(prepare()))
