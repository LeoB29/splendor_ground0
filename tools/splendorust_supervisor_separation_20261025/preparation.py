"""Model-free admission and immutable freeze for the live reliability workload."""
import math
from pathlib import Path
import shutil
import subprocess

from . import common as c, placement, panel
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_boundary_timing_20261025.runner import nested_artifacts
from tools.splendorust_search_scale_20261024 import qualification as qualification


def acceptance(*, boundary=lambda: None):
    """Verify the finite qualified observer evidence, not the older full benchmark."""
    names = ('freeze.json', 'completion.json', 'execution.json', 'overhead_summary.json',
        'diagnostic_summary.json', 'diagnostic_outcome.json', 'verification_summary.json',
        'qualification_summary.json', 'all_entrypoint_checks.json', 'all_graph_parity.json',
        'all_raw_transport.json', 'all_worker_log_integrity.json', 'completed_evidence_seal.json',
        'independent_boundary_review.json')
    saved = {name: c.read(c.PRIOR / name) for name in names}
    frozen, done, outer = (saved[n] for n in ('freeze.json', 'completion.json', 'execution.json'))
    c.require(c.file_hash(c.PRIOR / 'freeze.json') == c.PRIOR_FREEZE_SHA and
        done['freeze_sha256'] == outer['freeze_sha256'] == c.PRIOR_FREEZE_SHA,
        'Exact qualified boundary observer freeze required')
    c.require(done['status'] == 'completed_verified' and done['failure'] is None and
        outer['status'] == 'completed' and outer['exit_code'] == 0 and not outer['cleanup_errors'] and
        outer['charged_job_seconds'] < frozen['job_seconds'], 'Observer completion/cleanup required')
    c.require(saved['overhead_summary.json']['qualified'] is True and
        saved['overhead_summary.json']['requests'] == 48 and
        all(saved['overhead_summary.json']['per_quota'][str(q)]['qualified'] is True for q in (32, 64)) and
        saved['diagnostic_summary.json']['qualified'] is True and
        saved['diagnostic_summary.json']['requests'] == 416 and
        saved['diagnostic_summary.json']['exact_scientific_groups'] == 104 and
        saved['diagnostic_outcome.json']['diagnostic_completed'] is True and
        saved['verification_summary.json']['status'] == 'passed' and
        saved['verification_summary.json']['requests'] == 464 and
        saved['verification_summary.json']['workers'] == 12 and
        saved['qualification_summary.json']['complete_decisions'] == 208 and
        saved['independent_boundary_review.json']['status'] == 'passed_scoped_review',
        'Complete positive observer qualification required')
    for name in ('all_entrypoint_checks.json', 'all_graph_parity.json', 'all_raw_transport.json',
                 'all_worker_log_integrity.json'):
        c.require(saved[name]['status'] == 'passed', 'Qualified observer verification changed')
    seal = saved['completed_evidence_seal.json']['artifacts']
    c.require(len(seal) == len({item['path'] for item in seal}) == 2014, 'Observer seal inventory')
    for item in seal:
        boundary()
        c.require((c.ROOT / item['path']).resolve().is_relative_to(c.PRIOR.resolve()), 'Observer seal scope')
        c.verify([item])
    for item in frozen['inputs']:
        boundary(); c.verify([item])
    return dict(status='passed', observer_freeze_sha256=c.PRIOR_FREEZE_SHA,
        evidence=[c.artifact(c.PRIOR / n) for n in names], frozen_inputs=frozen['inputs'],
        sealed_evidence=seal, qualified_players={'challenger': {'mode': 'candidate', 'quota': 64},
        'opponent': {'mode': 'baseline', 'quota': 32}},
        observer_module='tools.splendorust_boundary_timing_20261025.worker',
        full_old_strength_seal_reaudit=False, prior_failures_reclassified=False)


def seed_inventory(rows):
    targets = {int(row['seed_decimal']) for row in rows}
    result = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
        '-g', '*schedule*.json', '-g', '*seed_registry*.json'], cwd=c.ROOT,
        capture_output=True, text=True, check=True)
    checked, skipped, collisions = [], [], []
    def visit(value, path):
        if type(value) is int or isinstance(value, str) and value.isdecimal():
            if int(value) in targets: collisions.append(path)
        elif isinstance(value, list):
            for child in value: visit(child, path)
        elif isinstance(value, dict):
            for child in value.values(): visit(child, path)
    for name in sorted(result.stdout.splitlines()):
        path = (c.ROOT / name).resolve()
        if path.is_relative_to(c.OUT.resolve()): continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(c.artifact(path)); continue
        visit(c.read(path), path.relative_to(c.ROOT).as_posix())
        checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh live workload seed collision')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, collisions=collisions,
        scope='Registered manifest/schedule/seed JSON up to8MiB; no exhaustive log claim')



DOMAIN = 'supervisor_separation20261025v1'
LIVE = c.ROOT/'outputs/splendorust_live_boundary_20261025_01'
LIVE_SHA = '50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b'


def operational_schedule():
    import hashlib
    import json
    rows=[]
    for pair in range(3):
        encoded=json.dumps([DOMAIN,'operational',pair],separators=(',',':'),ensure_ascii=True)
        seed=str(int.from_bytes(hashlib.sha256(encoded.encode()).digest(),'big'))
        for game in (0,1):
            rows.append(dict(stage='operational',pair_index=pair,game_index=2*pair+game,
                challenger_seat=(pair+game)%2,seed_decimal=seed,challenger_quota=64,opponent_quota=32,
                challenger_mode='candidate',opponent_mode='baseline'))
    return rows


def validate_game_spec(value):
    c.require(type(value) is dict and type(value.get('game_index')) is int and 0<=value['game_index']<6,
        'Six-game operational schedule identity')
    expected=operational_schedule()[value['game_index']]
    c.require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in expected.items()),'Frozen operational specification differs')
    return expected


validate_spec = validate_game_spec
make_schedule = operational_schedule


def validate_schedule(rows):
    c.require(rows==operational_schedule() and all(validate_game_spec(r)==r for r in rows),'Exact six-game operational schedule')
    return rows


def history_acceptance():
    names=('freeze.json','completion.json','execution.json','coverage.json','verification_summary.json','collection_stop.json')
    docs={n:c.read(LIVE/n) for n in names}
    c.require(c.file_hash(LIVE/'freeze.json')==LIVE_SHA and docs['completion.json']['freeze_sha256']==LIVE_SHA and
        docs['execution.json']['freeze_sha256']==LIVE_SHA and docs['completion.json']['status']=='captured_fault_verified' and
        docs['execution.json']['status']=='completed' and docs['execution.json']['exit_code']==0 and
        not docs['execution.json']['cleanup_errors'] and docs['verification_summary.json']['status']=='passed',
        'Exact preserved live captured-fault attempt required')
    inputs=[x for x in docs['freeze.json']['inputs'] if not x['path'].startswith('outputs/')]
    c.verify(inputs)
    return dict(status='passed',freeze_sha256=LIVE_SHA,evidence=[c.artifact(LIVE/n) for n in names],
        frozen_nonoutput_inputs=inputs,full_live_seal_reaudit=False,primary_panel_records_verified_separately=True)


def resource_forecast():
    """Model-free projection including every startup, graph proof and reproduction."""
    costs_path=LIVE/'qualification_summary.json'; costs=c.read(costs_path)
    start=max(r['complete_startup_and_parity_seconds'] for r in costs['costs'])
    suite=max(r['counterexample_seconds'] for r in costs['costs'])
    validation=suite/52
    cleanup=max(r['log_and_raw_transport_seconds']+r['graph_proof_reread_seconds']+
        r['closed_worker_two_pass_seal_seconds'] for r in costs['costs'])
    paths=[LIVE/'stages'/f'operational_{edge}.json' for edge in ('start','end')]
    a,b=map(c.read,paths); ops=max(b['monotonic']-a['monotonic'],b['unix_time']-a['unix_time'])
    # Per-request placement inventory outside clocks plus 5s per mode handshake
    # is charged explicitly. Original request work remains bounded by1s.
    stages=dict(qualification=1.5*4*(start+suite+cleanup)+60+30+4*5,
        timing=1.5*(192*(1+validation+.015)+8*(start+cleanup))+8*5+20,
        operational=ops*1.5+30,
        final=1.5*(48*(start+1+validation+cleanup+.015)+
            costs['source_seconds']+costs['acceptance_seconds'])+50)
    bytepath=LIVE/'final_evidence_forecast.json'; old=c.read(bytepath)
    # Entire previous final projection already charged206-game control files;
    # retain that conservative bound, then add placement receipts and controls.
    final=math.ceil(old['projected_final_evidence_bytes']*1.35+16*c.MiB)
    pre=160*c.MiB
    value=dict(status='passed',model_calls=0,fresh_games=0,outcomes_used_for_selection_or_resource_estimation=False,
        sources=[c.artifact(p) for p in [costs_path,*paths,bytepath]],
        measured_startup_graph_seconds=start,measured_52_suite_seconds=suite,
        measured_closed_worker_verification_seconds=cleanup,measured_six_operational_seconds=ops,
        projected_stage_seconds=stages,projected_total_seconds=sum(stages.values()),
        projected_prefinal_bytes=pre,projected_final_bytes=final,
        inventories=dict(actual_entrypoint_workers=4,actual_entrypoint_decisions=208,timing_requests=192,
            timing_workers=8,games_maximum=6,game_workers_maximum=12,reproduction_requests_maximum=48,
            workers_maximum=72,graph_workers_maximum=60),
        methods=['Worst measured startup including graph proofs and52-suite;1.5time margin.',
            'Every timing request charged1s plus measured validation and15ms placement evidence.',
            'All48 reproductions charged separate startup/graph/decision/cleanup.',
            'Previous206-game final byte forecast times1.35 plus16MiB placement/control reserve.',
            'Initial60s stable RAM and explicit bounded mode handshakes included.'],
        limitations=['Hard caps override forecasts; optional600s RAM wait is charged, not guaranteed to fit.',
            'Previous timings do not establish placement causality or guarantee current host performance.'])
    return validate_resource_forecast(value)


def validate_resource_forecast(value):
    c.require(value['status']=='passed' and value['model_calls']==value['fresh_games']==0 and
        value['outcomes_used_for_selection_or_resource_estimation'] is False,'Model-free resource forecast required')
    finite=lambda x:type(x) in (int,float) and math.isfinite(x) and x>=0
    stages=value['projected_stage_seconds']
    c.require(set(stages)==set(c.STAGES) and all(finite(stages[k]) and stages[k]<v for k,v in c.STAGES.items()),'Projected stage budget exceeded')
    total=value['projected_total_seconds']
    c.require(finite(total) and sum(stages.values())<=total<c.JOB_SECONDS and
        sum(v for k,v in stages.items() if k!='final')<c.COLLECTION_CUTOFF_SECONDS,'Projected job budget exceeded')
    pre,final=value['projected_prefinal_bytes'],value['projected_final_bytes']
    c.require(finite(pre) and finite(final) and pre<c.PREFINAL_EVIDENCE_BYTES and final<c.FINAL_RESERVE_BYTES and
        pre+final<c.MAX_EVIDENCE_BYTES,'Projected evidence budget exceeded')
    c.require(value['inventories']['workers_maximum']==72 and value['inventories']['graph_workers_maximum']==60,
        'Complete startup and reproduction worker inventory required')
    return value


def dependencies(prior,documents,paths):
    inherited={x['path']:x for x in prior['inputs']}
    items={p:x for p,x in inherited.items() if not p.startswith('outputs/')}
    def bind(item):
        c.require(not Path(item['path']).is_absolute(),'Repository-relative frozen input')
        c.require(item['path'] not in items or items[item['path']]==item,'Conflicting dependency')
        c.require(item['path'] not in inherited or inherited[item['path']]==item,'Frozen input changed')
        items[item['path']]=item
    for doc in documents:
        for item in nested_artifacts(doc):bind(item)
    for path in paths:bind(c.artifact(path))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']:bind(item)
    for item in prior['native_artifacts']:bind(item)
    return sorted(items.values(),key=lambda x:x['path'])


def prepare():
    from .runner import verify_runtime
    c.require(not (c.OUT/'freeze.json').exists(),'No replacement freeze')
    accepted=acceptance();history=history_acceptance()
    for name in ('prelaunch_tests_final.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name)
        c.require(receipt['status']=='passed','Prelaunch tests and independent review required');c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free>=7*c.GiB,'7GiB free disk required initially')
    source_guard=verify_runtime_sources();prior=c.read(c.PRIOR/'freeze.json')
    plan=panel.build_plan();panel.validate_plan(plan)
    resource=resource_forecast();rows=validate_schedule(operational_schedule())
    c.save(c.OUT/'preparation_resource_estimate.json',resource)
    c.save(c.OUT/'qualification_acceptance.json',accepted);c.save(c.OUT/'history_acceptance.json',history)
    c.save(c.OUT/'panel.json',plan);c.save(c.OUT/'schedule.json',rows)
    c.save(c.OUT/'seed_inventory.json',seed_inventory(rows))
    parity=c.read(c.PRIOR/'graph_parity_plan.json');c.save(c.OUT/'graph_parity_plan.json',parity)
    c.save(c.OUT/'model_free_qualification.json',qualification.model_free_checks(64))
    paths=set((c.ROOT/'tools/splendorust_supervisor_separation_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_supervisor_separation*.py'))
    paths.update(p for p in c.OUT.rglob('*') if p.is_file());paths.add(c.PROTOCOL)
    # Explicitly preserve both offline-reader attempts, including unsealed derived receipts.
    for version in ('','_v2'):
        paths.add(c.ROOT/f'tools/splendorust_saved_failure_diagnosis_20261025{version}.py')
    for attempt in ('01','02'):
        folder=c.ROOT/f'outputs/splendorust_saved_failure_diagnosis_20261025_{attempt}'
        paths.update(p for p in folder.glob('*.json') if p.is_file())
    copied=('python','python_executable_sha256','torch_runtime','external_runtime','sampler_api_modules',
        'compression_runtime','topology','native_artifacts','numpy_version','torch_package_version',
        'torch_runtime_version','unchanged_graph_backend_identity')
    freeze=dict(schema='splendorust_supervisor_separation_v1',time=c.now(),**{k:prior[k] for k in copied},
        inputs=dependencies(prior,[accepted,history,plan,parity,resource,qualification.build_plan(32),
            qualification.build_plan(64),c.read(qualification.PARITY_RECEIPT)],paths),
        qualification_acceptance=accepted,history_acceptance=history,source_guard=source_guard,
        placement=placement.plan(prior['topology']),graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0,seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        margin_is_forfeit=False,operational_games=6,main_games=0,pairs=3,
        challenger_quota=64,opponent_quota=32,player_modes={'challenger':'candidate','opponent':'baseline'},
        actual_entrypoint_workers=4,actual_entrypoint_decisions=208,timing_requests=192,timing_workers=8,
        final_reproductions_maximum=48,maximum_workers=72,maximum_graph_workers=60,
        max_rss_bytes=2*c.GiB,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        profile_child_module='tools.splendorust_boundary_timing_20261025.worker',
        scientific_backends={'64':'tools.splendorust_inference_opt_20261025.worker','32':'tools.splendorust_search_scale_20261024.worker'},
        observer_scope='unchanged qualified boundary observer plus symmetric placement verification',
        sidecar_wait_maximum_seconds=1.,instrumentation_time_subtracted=False,per_inference_instrumentation=False,
        failure_policy='First attributable timing failure stops without games; first live failure loses game and stops. Verify collected prefix; integrity failures remain incomplete.',
        failed_decision_retries=0,strength_inference=False,reference_promotion=False,larger_data_deferred=True)
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical(dict(prepared=True,model_calls=0,inputs=len(freeze['inputs']),freeze_sha256=c.file_hash(c.OUT/'freeze.json'))),flush=True)
