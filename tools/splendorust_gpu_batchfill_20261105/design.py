"""Predetermined saved states, two order-balanced blocks, no new games."""
from copy import deepcopy
from . import common as c

PRIOR=c.ROOT/'outputs/splendorust_gpu_throughput_20261105_01'
REPORT=c.ROOT/'outputs/splendorust_gpu_throughput_report_20261105_01'
PLAN_SHA='5c31bcbf65fc34766226b5f734336b3b0355a9a82ddb928dc0840ee3a60530d0'
AUDIT_SHA='e4b2d33972e5dfd48a86079be77c4e33bf220740345ee071d9c4c50d750c97ca'
CLOSEOUT_SHA='4dba90e81cca3e9ea8e642a939139f3fdf1adbaf704a17cb3469816ce7c73b8d'
PLIES=(0,12,24,40)
CONFIGS=(('cpu8','cpu',8,'original8'),('gpu8','gpu',8,'original8'),
         ('gpu16','gpu',16,'row_fifo32'),('gpu32','gpu',32,'row_fifo32'))

def cells():
    result=[]
    for block, order in enumerate((CONFIGS,tuple(reversed(CONFIGS)))):
        for name,route,count,scheduler in order:
            result.append(dict(cell_id=f'b{block+1}_{name}',configuration=name,block=block+1,
                route=route,concurrency=count,scheduler=scheduler,
                groups=[list(range(start,start+count)) for start in range(0,32,count)]))
    return result

def build_cases():
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens,main_support
    from tools.splendorust_gpu_throughput_20261105 import qualification as q, games
    cases=[]
    for ply in PLIES:
        for game in range(8):
            paths={route:PRIOR/'games'/route/'worker/games'/f'g{game:03d}'/'moves'/f'{ply:06d}.json' for route in ('cpu','gpu')}
            rows={route:c.read(path) for route,path in paths.items()}
            left,right=rows['cpu'],rows['gpu']
            c.require(left['before']==right['before'] and left['observer']==right['observer'], 'Matched saved state')
            snapshot=left['before']; state=guard.decode_training_snapshot(snapshot)
            observer=rules.observation_payload(state)
            c.require(not state.terminal and observer==left['observer'], 'Actual corrected nonterminal saved request')
            _,_,mask=main_support(observer); tokens=public_root_tokens(observer)
            request=dict(player='search',observer=observer);index=len(cases)
            case=dict(index=index,case_id=f'p{index:03d}',slot=game,source_game=f'g{game:03d}',source_ply=ply,
                snapshot=snapshot,observer=observer,observer_sha256=c.digest(observer),
                request=request,request_sha256=c.digest(request),expected_root_tokens=tokens,
                legal_policy_indices=[i for i,legal in enumerate(mask) if legal],
                source_before=c.artifact(paths['cpu']),expected_results={route:c.artifact(path) for route,path in paths.items()})
            for route,row in rows.items():
                c.require(row['ply']==ply and row['game_id']==case['source_game'] and row['result']['diagnostics']['inference_calls']>0,
                          'Predetermined neural decision exists on each route')
                q.validate_fixed_result(case,row['result'])
                c.require(games.actual_after(snapshot,row['result']['action'])==row['after'], 'Source transition independently reproduced')
            cases.append(case)
    c.require(len(cases)==32 and len({x['request_sha256'] for x in cases})==32, '32 distinct predetermined saved states')
    return cases

def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from . import backend
    for path,expected in ((PRIOR/'plan.json',PLAN_SHA),(REPORT/'audit_saved.json',AUDIT_SHA),(REPORT/'closeout.json',CLOSEOUT_SHA)):
        c.require(c.sha(path)==expected, 'Authenticated accepted predecessor')
    previous=c.read(PRIOR/'plan.json'); accepted=c.read(REPORT/'audit_saved.json')
    c.require(accepted['status']=='saved_throughput_study_verified', 'Accepted complete previous study')
    c.verify(previous['inputs'])
    for identity in accepted['exact_owned_processes_absent']: require_absent(identity)
    sources=verify_runtime_sources();c.require(sources==previous['source_guard'],'Unchanged corrected rule sources')
    runtime=deepcopy(previous['runtime'])
    runtime.pop('games',None);runtime.pop('gameplay_seconds_per_route',None)
    return dict(schema='splendorust_gpu_batchfill_v1',time=c.now(),
        authorization='User requested the focused scheduling and GPU batch-filling test before considering another software stack',
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),audit=c.artifact(REPORT/'audit_saved.json'),closeout=c.artifact(REPORT/'closeout.json'),
            frozen_inputs=previous['inputs'],
            preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(REPORT.rglob('*')) if p.is_file()]),
        runtime=runtime,cases=build_cases(),cells=cells(),source_guard=sources,
        scheduler_source_equivalence=backend.source_equivalence(),
        collision_qualification=dict(concurrencies=[16,32],callbacks_per_client=2,physical_batch=8,
            row_assignment='client_index modulo8',exact_against='accepted saved same-route root float32 outputs',
            qualification_only=True),
        worker_affinity=runtime['worker_affinity'],supervisor_affinity=runtime['supervisor_affinity'],
        worker_limits=dict(qualification_seconds=2700,cell_seconds=900),total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,group_gap_seconds=0.,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        warmup=deepcopy(previous['warmup']),maximum_diagnostic_decisions=256,qualification_decisions_per_route=67,
        primary_timing='First barrier release through final durable 32-case writes, including intergroup gaps; all concurrency adapters prepared before clock and reused across groups',
        case_mapping='case index i uses adapter i modulo concurrency; immutable row i modulo8',
        exact_reproduction='Every complete decision must equal accepted saved full same-route result, root bytes and independently reconstructed afterstate',
        comparison='Report two block rates separately and pooled total64/summedwall per configuration; descriptive within-block ratios; no statistical strength or sustained rate claim',
        useful_candidate='Both32-case blocks complete correctly; at least10percent greater throughput than fresh GPU8 in EACH block; CPU8 comparison reported separately',
        selection='No follow-up games or configuration selection; fixed four configurations each in both reversed blocks',
        failed_cells='No complete-workload rate; preserve failed prefixes and unstarted states. Only authenticated60/75 decision safeguards isolate a cell; technical failures stop study',
        late_completed_responses_accepted=True,descriptive_latency_threshold_seconds=[1.,9.,10.],
        retries=0,new_games=0,training_updates=0,larger_data_deferred=True,
        automatic_followup_games=False,automatic_followup_training=False,protocol=c.artifact(c.PROTOCOL))
