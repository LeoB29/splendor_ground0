"""Prospective eight-opening, independent-cell study; imports run no model."""
from copy import deepcopy
from . import common as c

PRIOR=c.ROOT/'outputs/splendorust_gpu_live_20261105_03'
PRIOR_REPORT=c.ROOT/'outputs/splendorust_gpu_live_report_20261105_03'
PRIOR_PLAN_SHA='33eb14d291ea7db50b94d458a7ab792c3c91d740eda8e941bfb1e84dcfa34878'
PRIOR_CLOSEOUT_SHA='8e5ccaa64537ca529d967a583c64381c2a05fe35eee84555073102d1dcacf7e1'
ORDER=(('cpu',1),('gpu',1),('gpu',2),('cpu',2),('cpu',4),('gpu',4),('gpu',8),('cpu',8))
RUNTIME_KEYS=('checkpoint','model_source','torch_version_expected','directml_version_expected',
 'device_name_contains','profiles','qualification','numerical','prior_panel','gpu_implementation',
 'gpu_physical_batch','client_slots','live_callback_batch','maximum_batch_wait_seconds','precision',
 'source_guard','ruleset_id','catalogue_sha256','turn_contract','internal_seconds','outer_seconds',
 'worker_affinity','supervisor_affinity','within_gpu_exact_replay_required','cross_backend_actions_descriptive')


def cells():
    return [dict(cell_id=f'{route}_c{count}',route=route,concurrency=count,
        groups=[list(range(start,start+count)) for start in range(0,8,count)]) for route,count in ORDER]


def build_cases(previous):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens,main_support
    from tools.splendorust_gpu_live_20261105_03.games import initial_snapshot
    result=[]
    for index,spec in enumerate(previous['games']['specifications']):
        path=PRIOR/'worker/games_cpu'/spec['game_id']/'initial.json'
        snapshot=c.read(path)
        c.require(c.read(path.parent/'spec.json')==spec,'Exact saved opening specification')
        c.require(index==spec['game_index'] and snapshot==previous['games']['initial_states'][index]['initial']
            ==initial_snapshot(spec),'Unchanged saved corrected initial state')
        state=guard.decode_training_snapshot(snapshot);observer=rules.observation_payload(state)
        c.require(not state.terminal and state.turn_index==0,'Opening requests only')
        _,_,mask=main_support(observer);tokens=public_root_tokens(observer)
        request=dict(player='search',observer=observer)
        result.append(dict(case_id=spec['game_id'],index=index,source_initial=c.artifact(path),
            source_spec=c.artifact(path.parent/'spec.json'),snapshot=snapshot,observer=observer,
            request=request,request_sha256=c.digest(request),expected_root_tokens=tokens,
            legal_policy_indices=[i for i,legal in enumerate(mask) if legal]))
    c.require(len(result)==8 and len({x['request_sha256'] for x in result})==8,'Eight distinct fixed requests')
    return result


def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    c.require(c.sha(PRIOR/'plan.json')==PRIOR_PLAN_SHA and c.sha(PRIOR_REPORT/'closeout.json')==PRIOR_CLOSEOUT_SHA,
              'Authenticated closed immediate-dispatch predecessor')
    previous=c.read(PRIOR/'plan.json');closed=c.read(PRIOR_REPORT/'closeout.json');c.verify(previous['inputs'])
    c.require(closed['outcome']=='gpu_and_cpu_qualification_passed_cpu_concurrent_game_wave_failed',
              'Prior qualification accepted, failed games preserved')
    c.verify([closed['independent_audit'],closed['independent_supplement']])
    execution=c.read(PRIOR/'execution.json');claim=c.read(PRIOR/'claim.json')
    for identity in execution['observed_owned_identities']:require_absent(identity)
    require_absent(dict(pid=claim['pid'],created=claim['created']))
    sources=verify_runtime_sources();c.require(sources==previous['source_guard'],'Same corrected source identities')
    runtime={key:deepcopy(previous[key]) for key in RUNTIME_KEYS}
    return dict(schema='splendorust_independent_concurrency_v1',time=c.now(),
        authorization='User authorized the proposed bounded concurrency diagnostic, preserving deadlines and checks',
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),closeout=c.artifact(PRIOR_REPORT/'closeout.json'),
            execution=c.artifact(PRIOR/'execution.json'),claim=c.artifact(PRIOR/'claim.json'),
            frozen_inputs=previous['inputs'],preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(PRIOR_REPORT.rglob('*')) if p.is_file()]),
        runtime=runtime,cases=build_cases(previous),cells=cells(),
        worker_affinity=runtime['worker_affinity'],supervisor_affinity=runtime['supervisor_affinity'],
        worker_limits=dict(qualification_seconds=1200,cell_seconds=300),total_seconds=c.TOTAL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,group_gap_seconds=0.,maximum_diagnostic_decisions=64,
        warmup=dict(calls_per_cell=5,case_index=0,client_id='warmup',slot=0,
            actual_service=True,inside_measured_panel=False,exact_within_route=True,
            internal_seconds=9.,outer_seconds=10.,failure_stops_study=True),
        qualification_decisions_per_route=102,qualification_not_counted_as_diagnostic_panel=True,
        internal_seconds=9.,outer_seconds=10.,deployment_target_seconds=1.,
        fixed_slot='case index0..7 for every concurrency; unused GPU rows remain zero',
        cell_failure_isolation='Only authenticated decision9/10s deadline failures permit later independent cells after exact cleanup',
        integrity_failure_policy='Stop study on source, status, startup, worker-cap, numerical, result, replay or cleanup failure',
        exact_same_route_baseline='Full scientific result and root bytes equal same-case C1; absent baseline prevents eligibility',
        cross_route_root_numeric_gate=True,cross_route_actions_descriptive=True,
        primary_timing='First group barrier timestamp through final group durable writes; includes between-group work; excludes model/native startup',
        eligibility='All8 complete and correct, same-route C1 exact, all numerical checks pass; partial cells have no completed-panel rate',
        selection='Highest completed8-request panel throughput per route among eligible cells; lower concurrency breaks exact ties',
        retries=0,new_games=0,training_updates=0,larger_data_deferred=True,automatic_followup_games=False,
        protocol=c.artifact(c.PROTOCOL),source_guard=sources)
