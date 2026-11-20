"""Model-free admission and immutable freeze for the live reliability workload."""
import math
from pathlib import Path
import shutil
import subprocess

from . import common as c, schedule
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_boundary_timing_20261025.runner import nested_artifacts, verify_runtime
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


def resource_forecast():
    """Charge the complete prior live workload and additional observation evidence."""
    old = c.ROOT / 'outputs/splendorust_graph64_strength_20261025_01'
    sources, elapsed = [], {}
    for stage in c.STAGES:
        paths = [old/'stages'/f'{stage}_{edge}.json' for edge in ('start', 'end')]
        a, b = map(c.read, paths)
        elapsed[stage] = max(b['monotonic']-a['monotonic'], b['unix_time']-a['unix_time'])
        sources.extend(c.artifact(p) for p in paths)
    old_seal = c.read(old/'completed_game_seal.json')
    old_bytes = sum(item['bytes'] for item in old_seal['artifacts'])
    sources.append(c.artifact(old/'completed_game_seal.json'))
    final_bytes = c.read(old/'final_evidence_forecast.json')['projected_final_evidence_bytes']
    sources.append(c.artifact(old/'final_evidence_forecast.json'))
    projected = {k: elapsed[k]*1.5 for k in c.STAGES}
    projected['qualification'] += 60
    projected['final'] += 120
    # Extra raw transport, timestamp sidecars and row/analysis copies are charged
    # separately from the inherited complete game/worker evidence, with margin.
    added_boundary = 384*c.MiB
    pre = math.ceil(old_bytes*1.35 + added_boundary + 32*c.MiB)
    final = math.ceil(final_bytes*1.5 + 64*c.MiB)
    value = dict(status='passed', model_calls=0, fresh_games=0,
        outcomes_used_for_selection_or_resource_estimation=False,
        source_freeze_sha256='556480a2501f4fb4cf829fe518aa4ba1ccbf10f12ee088fccf627408dfd69560',
        sources=sources, measured_stage_seconds=elapsed, measured_sealed_bytes=old_bytes,
        projected_stage_seconds=projected, projected_total_seconds=sum(projected.values()),
        projected_prefinal_bytes=pre, projected_final_bytes=final,
        inventories=dict(games_maximum=206, actual_entrypoint_workers=2, actual_entrypoint_decisions=104,
            reproduction_requests_maximum=48, workers_maximum=462, graph_workers_maximum=255),
        methods=['Prior complete206-game stage times times1.5, plus60squalification and120sfinal allowance.',
            'Prior sealed bytes times1.35, plus384MiBboundary evidence and32MiBpreparation allowance.',
            'Prior final-byte forecast times1.5 plus64MiB for extra seal/boundary analysis.',
            'Fresh six-game outcome-blind time/evidence forecast must pass before200-game expansion.'],
        limitations=['Forecast is not a guarantee; hard stage/global/evidence guards remain binding.',
            'No full prior benchmark seal re-audit: only its frozen inventory metadata informs size.',
            'First attributable player failure ends collection and requires verified prefix finalization.'])
    return validate_resource_forecast(value)


def validate_resource_forecast(value):
    c.require(value['status']=='passed' and value['model_calls']==value['fresh_games']==0 and
        value['outcomes_used_for_selection_or_resource_estimation'] is False, 'Model-free resource forecast required')
    stages = value['projected_stage_seconds']
    finite = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    c.require(set(stages)==set(c.STAGES) and all(finite(stages[k]) and stages[k]<v for k,v in c.STAGES.items()),
        'Projected stage budget exceeded')
    total = value['projected_total_seconds']
    c.require(finite(total) and sum(stages.values()) <= total < c.JOB_SECONDS and
        sum(v for k,v in stages.items() if k!='final') < c.COLLECTION_CUTOFF_SECONDS,
        'Projected job/collection budget exceeded')
    pre, final = value['projected_prefinal_bytes'], value['projected_final_bytes']
    c.require(finite(pre) and finite(final) and pre<c.PREFINAL_EVIDENCE_BYTES and
        final<c.FINAL_RESERVE_BYTES and pre+final<c.MAX_EVIDENCE_BYTES, 'Projected evidence budget exceeded')
    return value


def dependencies(prior, accepted, parity, resource, paths):
    inherited = {item['path']: item for item in prior['inputs']}
    items = {path:item for path,item in inherited.items() if not path.startswith('outputs/')}
    def bind(item):
        c.require(not Path(item['path']).is_absolute(), 'Repository-relative input required')
        c.require(item['path'] not in inherited or inherited[item['path']]==item, 'Frozen dependency changed')
        c.require(item['path'] not in items or items[item['path']]==item, 'Conflicting dependency')
        items[item['path']] = item
    for doc in (accepted, parity, resource, qualification.build_plan(32), qualification.build_plan(64),
                c.read(qualification.PARITY_RECEIPT)):
        for item in nested_artifacts(doc): bind(item)
    bind(c.artifact(qualification.PARITY_RECEIPT))
    for quota in (16,32,64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in prior['native_artifacts']: bind(item)
    for path in paths: bind(c.artifact(path))
    return sorted(items.values(),key=lambda item:item['path'])


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'No replacement freeze')
    accepted = acceptance()
    for name in ('prelaunch_tests_final.json', 'prelaunch_review.json'):
        receipt = c.read(c.OUT/name)
        c.require(receipt['status']=='passed', 'Prelaunch tests and independent review required')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, '7GiB free disk required initially')
    source_guard = verify_runtime_sources()
    prior = c.read(c.PRIOR/'freeze.json')
    resource = resource_forecast()
    c.save(c.OUT/'preparation_resource_estimate.json', resource)
    rows = schedule.validate_schedule(schedule.make_schedule())
    c.save(c.OUT/'qualification_acceptance.json', accepted)
    c.save(c.OUT/'seed_inventory.json', seed_inventory(rows))
    c.save(c.OUT/'schedule.json', rows)
    parity = c.read(c.PRIOR/'graph_parity_plan.json')
    c.save(c.OUT/'graph_parity_plan.json', parity)
    c.save(c.OUT/'model_free_qualification.json', qualification.model_free_checks(64))
    paths = set((c.ROOT/'tools/splendorust_live_boundary_20261025').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_live_boundary*.py'))
    paths.update(p for p in c.OUT.rglob('*') if p.is_file())
    paths.add(c.PROTOCOL)
    copied = ('python','python_executable_sha256','torch_runtime','external_runtime','sampler_api_modules',
        'compression_runtime','topology','native_artifacts','numpy_version','torch_package_version',
        'torch_runtime_version','unchanged_graph_backend_identity')
    freeze = dict(schema='splendorust_live_boundary_reliability_v1', time=c.now(),
        inputs=dependencies(prior,accepted,parity,resource,paths), **{k:prior[k] for k in copied},
        qualification_acceptance=accepted, source_guard=source_guard,
        graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        job_seconds=c.JOB_SECONDS, stage_seconds=c.STAGES, collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0, seconds_per_move=1., internal_seconds=.9, margin_seconds=.7,
        margin_is_diagnostic_only=True, operational_games=6, main_games=200, pairs=100,
        challenger_quota=64, opponent_quota=32, player_modes={'challenger':'candidate','opponent':'baseline'},
        actual_entrypoint_workers=2, actual_entrypoint_decisions=104,
        final_reproductions_maximum=48, maximum_workers=462, maximum_graph_workers=255,
        max_rss_bytes=2*c.GiB, max_evidence_bytes=c.MAX_EVIDENCE_BYTES, final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        resource_forecast_factor=1.35, profile_child_module='tools.splendorust_boundary_timing_20261025.worker',
        scientific_backends={'64':'tools.splendorust_inference_opt_20261025.worker',
            '32':'tools.splendorust_search_scale_20261024.worker'},
        observer_scope='unchanged qualified minimal parent/child boundaries', sidecar_wait_maximum_seconds=1.,
        instrumentation_time_subtracted=False, per_inference_instrumentation=False,
        failure_policy='First established attributable player failure loses game and ends collection; verify played prefix. Infrastructure/scientific failures remain incomplete.',
        failed_decision_retries=0, strength_inference=False, reference_promotion=False, larger_data_deferred=True)
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, model_calls=0, inputs=len(freeze['inputs']),
        freeze_sha256=c.file_hash(c.OUT/'freeze.json'))), flush=True)
