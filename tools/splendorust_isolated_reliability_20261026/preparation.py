"""Finite prior acceptance, empirical resources and a fresh immutable freeze."""
import math
from pathlib import Path
import shutil
import subprocess

from . import common as c, placement, schedule
from tools import future_training_rules_20261021 as corrected
from tools.splendorust_boundary_timing_20261025.runner import nested_artifacts
from tools.splendorust_search_scale_20261024 import qualification


QUALIFIED = c.ROOT / 'outputs/splendorust_supervisor_qualification_20261026_01'
QUALIFIED_SHA = 'cc5f19859865e41844689deee48cf1e089db59fdbb6b46af14da3711ac34d2e4'
QUALIFIED_REVIEW_SHA = 'df09400ed6a5b3313c951344bb6abe6bb783f14ef89af3494ea637009b572d82'
QUALIFIED_REVIEW_EXECUTION_SHA = '793ee91c1ad7adf26eb5620b9ce74d05d154f612b1dfaba6281821365ef4ad4e'
LIVE = c.ROOT / 'outputs/splendorust_live_boundary_20261025_01'
LIVE_SHA = '50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b'
STOPPED_SHA = 'b6e9073a83d2bf00e4d1ac16d249e9bc75fc0219730f7d4781fb264d80fbf855'
ACCEPTANCE_NAMES = (
    'freeze.json', 'completion.json', 'execution.json', 'qualification_summary.json',
    'timing_validation.json', 'timing_gate.json', 'timing_lifecycle_verification.json',
    'control_timeout_manifest.json', 'coverage.json', 'verification_summary.json',
    'all_worker_integrity.json', 'all_placement_integrity.json', 'all_game_replay.json',
    'performance.json', 'repeated_decision_summary.json', 'replay_sample_manifest.json',
    'completed_evidence_seal.json', 'independent_results_review.json',
    'independent_review_execution.json',
)
REVIEW_COVERED_NAMES = tuple(name for name in ACCEPTANCE_NAMES if name not in (
    'all_placement_integrity.json', 'independent_results_review.json', 'independent_review_execution.json'))


def validate_accepted_documents(docs):
    """Exact positive isolated qualification, with original failures preserved."""
    frozen, done, outer = (docs[n] for n in ('freeze.json', 'completion.json', 'execution.json'))
    review = docs['independent_results_review.json']
    c.require(done['freeze_sha256'] == outer['freeze_sha256'] == review['freeze_sha256'] == QUALIFIED_SHA,
              'Exact isolated qualification identity required')
    c.require(done['status'] == 'isolated_qualified_verified' and done['failure'] is None and
              outer['status'] == 'completed' and outer['exit_code'] == 0 and not outer['cleanup_errors'] and
              outer['charged_job_seconds'] < frozen['job_seconds'], 'Clean qualified owning exit required')
    c.require(review['status'] == 'passed_scoped_review' and review['consumed_inputs_readback'] == 'passed' and
              review['isolated_qualified'] is True and
              docs['independent_review_execution.json']['exit_code'] == 0 and
              docs['independent_review_execution.json']['error'] is None and
              docs['independent_review_execution.json']['stdout_sha256'] == QUALIFIED_REVIEW_SHA,
              'Successful bound independent qualification review required')
    checks = docs['qualification_summary.json']
    c.require(checks['status'] == 'passed' and checks['workers'] == 4 and checks['actual_decisions'] == 208 and
              {(x['placement_mode'], x['quota'], x['complete_decisions']) for x in checks['suites']} ==
              {(mode, quota, 52) for mode in ('original', 'isolated') for quota in (32, 64)},
              'Both players under both placements must have actual entrypoint qualification')
    timing = docs['timing_validation.json']; gate = docs['timing_gate.json']
    c.require(timing['status'] == 'passed' and timing['complete'] is True and
              timing['requests'] == timing['completed'] == 192 and
              timing['isolated_completed'] == timing['control_completed'] == 96 and
              timing['fully_replicated_isolated_groups'] == timing['fully_replicated_groups'] == 48 and
              timing['isolated_historical_comparisons'] == timing['control_historical_comparisons'] == 46 and
              timing['historical_comparisons'] == 92 and timing['control_timeout_count'] == 0,
              'Complete exact isolated/control panel acceptance required')
    c.require(gate['qualified'] is True and gate['complete_inventory'] is True and
              gate['isolated_completed'] == 96 and gate['retained_control_timeouts'] == 0 and
              all(gate['per_quota'][str(q)]['qualified'] is True and
                  gate['per_quota'][str(q)]['distributions']['isolated']['count'] == 48 and
                  gate['per_quota'][str(q)]['distributions']['isolated']['max'] < .7 for q in (32, 64)),
              'Both isolated players must have passed their frozen absolute margin')
    lifecycle = docs['timing_lifecycle_verification.json']
    c.require(lifecycle['status'] == 'passed' and lifecycle['workers'] == 8 and
              lifecycle['control_timeouts'] == lifecycle['continuation_workers'] == 0 and
              docs['control_timeout_manifest.json']['count'] == 0,
              'Accepted qualification lifecycle differs')
    coverage = docs['coverage.json']; verified = docs['verification_summary.json']
    c.require(done['coverage'] == coverage and coverage['isolated_qualified'] is True and
              coverage['operational_margin_passed'] is True and coverage['verified_games'] == 6 and
              coverage['unplayed_games'] == 0 and coverage['complete_planned_workload'] is True and
              verified['status'] == 'passed' and verified['isolated_qualified'] is True and
              verified['games'] == 6 and verified['live_requests'] == 338 and
              verified['workers'] == 62 and verified['graph_workers'] == 31 and verified['reproductions'] == 38,
              'Complete verified six-game isolated qualification required')
    performance = docs['performance.json']; repeated = docs['repeated_decision_summary.json']
    c.require(performance['games'] == performance['natural_games'] == 6 and not performance['forfeits'] and
              all(performance['players'][r]['completed_decisions'] == 169 and
                  performance['players'][r]['attempted_decisions'] == 169 and
                  performance['players'][r]['completed_arrival_seconds']['max'] < .7
                  for r in ('challenger', 'opponent')) and
              repeated['status'] == 'passed' and repeated['requests'] == 38 and
              repeated['no_failed_decision_retries'] is True and
              len(docs['replay_sample_manifest.json']['decisions']) == 38,
              'Accepted live decisions and actual38selected reproductions required')
    for name in ('all_worker_integrity.json', 'all_placement_integrity.json', 'all_game_replay.json'):
        c.require(docs[name]['status'] == 'passed', 'Accepted integrity receipt changed')
    c.require(frozen['history_acceptance']['freeze_sha256'] == LIVE_SHA and
              frozen['previous_attempt_acceptance']['freeze_sha256'] == STOPPED_SHA and
              frozen['previous_attempt_acceptance']['preserved_outcome'] == 'timing_failure_verified' and
              frozen['previous_attempt_acceptance']['old_qualification_reclassified'] is False and
              frozen['previous_attempt_acceptance']['old_full_paired_comparison_passed'] is False,
              'Earlier live failure and stopped qualification must remain preserved')
    c.require(len(frozen['inputs']) == 4885 and
              frozen['placement'] == placement.plan(frozen['topology']) and
              frozen['seconds_per_move'] == 1. and frozen['internal_seconds'] == .9,
              'Accepted runtime/placement/clock identity differs')
    return True


def validate_review_bindings(docs, artifacts, review_source):
    """Bind reviewed raw summary bytes; do not overstate the review's scope."""
    review = docs['independent_results_review.json']
    consumed = review['consumed_inputs']
    mapping = {item['path']: item for item in consumed}
    c.require(len(mapping) == len(consumed), 'Duplicate reviewed input identity')
    covered = []
    for name in ACCEPTANCE_NAMES:
        item = artifacts[name]
        if name in REVIEW_COVERED_NAMES:
            c.require(item['path'] in mapping, 'Required summary absent from independent review: '+name)
        if item['path'] in mapping:
            c.require(mapping[item['path']] == item, 'Independently reviewed summary bytes changed: '+name)
            covered.append(name)
    execution = docs['independent_review_execution.json']
    c.require(artifacts['independent_results_review.json']['sha256'] == QUALIFIED_REVIEW_SHA and
        artifacts['independent_review_execution.json']['sha256'] == QUALIFIED_REVIEW_EXECUTION_SHA,
        'Pinned independent review or execution receipt changed')
    c.require(review_source['path'] == execution['script'] and
        review_source['sha256'] == execution['script_sha256'] and
        execution['stdout_file'] == artifacts['independent_results_review.json']['path'] and
        execution['stdout_sha256'] == QUALIFIED_REVIEW_SHA,
        'Independent review source or stdout binding changed')
    return dict(status='passed', independently_reviewed_summary_names=covered,
        preserved_additional_summary_names=[name for name in ACCEPTANCE_NAMES if name not in covered],
        additional_summaries_not_claimed_independently_read=True,
        review_execution_sha256=QUALIFIED_REVIEW_EXECUTION_SHA,
        review_source_sha256=review_source['sha256'])


def acceptance(*, boundary=lambda: None):
    """Verify one finite accepted qualification and its already frozen ancestry."""
    docs = {name: c.read(QUALIFIED / name) for name in ACCEPTANCE_NAMES}
    c.require(c.file_hash(QUALIFIED / 'freeze.json') == QUALIFIED_SHA and
              c.file_hash(QUALIFIED / 'independent_results_review.json') == QUALIFIED_REVIEW_SHA,
              'Pinned isolated qualification evidence changed')
    artifacts = {name: c.artifact(QUALIFIED/name) for name in ACCEPTANCE_NAMES}
    review_source = c.artifact(c.ROOT / 'tools/splendorust_supervisor_qualification_review_20261026.py')
    review_binding = validate_review_bindings(docs, artifacts, review_source)
    validate_accepted_documents(docs)
    seal = docs['completed_evidence_seal.json']['artifacts']
    c.require(len(seal) == len({item['path'] for item in seal}) == 4392, 'Accepted4392artifact seal required')
    for item in seal:
        boundary()
        c.require((c.ROOT / item['path']).resolve().is_relative_to(QUALIFIED.resolve()), 'Accepted seal scope')
        c.verify([item])
    frozen = docs['freeze.json']
    for item in frozen['inputs']:
        boundary(); c.verify([item])
    return dict(status='passed', freeze_sha256=QUALIFIED_SHA,
        evidence=[artifacts[name] for name in ACCEPTANCE_NAMES],
        frozen_inputs=frozen['inputs'], sealed_evidence=seal,
        report=c.artifact(c.ROOT / 'docs/engineering/splendorust_supervisor_qualification_20261026_results.md'),
        review_source=review_source, independent_review_binding=review_binding,
        accepted_placement=frozen['placement'], accepted_topology=frozen['topology'],
        qualified_players={'challenger': {'mode': 'candidate', 'quota': 64},
                           'opponent': {'mode': 'baseline', 'quota': 32}},
        actual_correctness_decisions=208, saved_timing_attempts=192, fresh_qualification_games=6,
        accepted_reproductions=38, observer_module='tools.splendorust_boundary_timing_20261025.worker',
        original_controls_and_timing_panel_repeated=False, earlier_failures_reclassified=False,
        full_old_strength_seal_reaudit=False)


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
        visit(c.read(path), path.relative_to(c.ROOT).as_posix()); checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh isolated reliability seed collision')
    return dict(status='passed', checked=checked, skipped_oversized=skipped, collisions=collisions,
        scope='Registered manifest/schedule/seed JSON up to8MiB; no exhaustive raw-log claim')


def _duration(start, end):
    value = max(end['monotonic']-start['monotonic'], end['unix_time']-start['unix_time'])
    c.require(math.isfinite(value) and value > 0, 'Positive recorded stage duration required')
    return value


def project_resources(measured):
    """Outcome-blind empirical projection; actual hard caps remain authoritative."""
    game = measured['maximum_complete_game_seconds']
    independent = measured['maximum_game_independent_seconds']
    startup = measured['maximum_startup_and_parity_seconds']
    check = measured['per_decision_validation_seconds']
    closed = measured['maximum_closed_worker_check_seconds']
    # Complete per-game measurement includes both worker starts, graph proof,
    # online ledger/raw/log checks and two-pass hashing. Do not add starts twice.
    stages = dict(qualification=1.5*measured['previous_qualification_stage_seconds']+60.,
        operational=1.5*6*game+15., main=1.5*194*game+30.,
        final=max(1.5*(200*independent+48*(startup+1.+check+closed+.015)+2*closed+120.),
                  1.5*measured['previous_final_stage_seconds']*48/38))
    pre = math.ceil(1.35*(200*measured['maximum_game_evidence_bytes']+
                         measured['isolated_qualification_evidence_bytes'])+64*c.MiB)
    files = math.ceil(1.35*(200*measured['maximum_game_evidence_files']+
                          48*measured['maximum_reproduction_files']+2000))
    current_final = (48*measured['maximum_reproduction_bytes']+
        files*measured['maximum_seal_item_bytes']+
        measured['previous_final_control_bytes']*200/6)
    final_bytes = math.ceil(max(1.35*current_final,
                              1.35*measured['older_206_game_final_bytes'])+32*c.MiB)
    return dict(status='passed', model_calls=0, fresh_games=0,
        outcomes_used_for_selection_or_resource_estimation=False, measured=measured,
        projected_stage_seconds=stages, projected_total_seconds=sum(stages.values()),
        projected_prefinal_bytes=pre, projected_final_bytes=final_bytes,
        projected_sealed_files=files,
        inventories=dict(total_games=200, operational_games=6, main_games=194,
            actual_entrypoint_workers=2, actual_entrypoint_decisions=104, timing_requests=0,
            game_workers_maximum=400, reproduction_requests_maximum=48,
            workers_maximum=450, graph_workers_maximum=249),
        methods=['Maximum of six current complete game costs, times1.5, for every one of200games.',
            'Complete game cost includes two startups, graph proof, online validation and two-pass hashing.',
            'Final charges independent verification for all200games,48 worst-startup reproductions and120s source/ancestry allowance.',
            'Final also has a measured previous-final-stage floor scaled38to48reproductions.',
            'Prefinal uses maximum current game-plus-two-workers evidence times200 with1.35margin and64MiBfixedreserve.',
            'Final bytes take the larger current metadata projection or conservative earlier206gameforecast, then add32MiB.',
            'All450workers and249possible graphworkers are included; no timing-panel/control rerun.'],
        limitations=['Forecasts are empirical, not guarantees for rare stalls, long games or slow startup.',
            'The first-six resource gate must recompute remaining194game feasibility before main collection.',
            'RAM waits consume fixed clocks; hard stage/global/disk/evidence limits always override projection.'])


def resource_forecast():
    sources = []
    def read(path):
        sources.append(c.artifact(path)); return c.read(path)
    seal = read(QUALIFIED / 'completed_evidence_seal.json')['artifacts']
    prefix = QUALIFIED.relative_to(c.ROOT).as_posix()+'/'
    costs = [read(p) for p in sorted((QUALIFIED / 'games/operational').glob('*/cost.json'))]
    c.require(len(costs) == 6, 'Exactly six accepted operational cost measurements')
    old = read(LIVE / 'qualification_summary.json')
    old_bytes = read(LIVE / 'final_evidence_forecast.json')
    def duration(stage):
        return _duration(read(QUALIFIED / 'stages' / (stage+'_start.json')),
                         read(QUALIFIED / 'stages' / (stage+'_end.json')))
    game_groups = []
    for index in range(6):
        stems = (prefix+f'games/operational/p{index//2:03d}_g{index}/',
                 prefix+f'workers/operational_g{index}_challenger/',
                 prefix+f'workers/operational_g{index}_opponent/')
        game_groups.append([item for item in seal if item['path'].startswith(stems)])
    repro_names = {item['path'].split('/workers/')[1].split('/')[0] for item in seal
                   if item['path'].startswith(prefix+'workers/audit_')}
    repro_groups = [[item for item in seal if item['path'].startswith(prefix+'workers/'+name+'/')]
                    for name in sorted(repro_names)]
    c.require(len(repro_groups) == 38 and all(game_groups), 'Measured complete game/reproduction inventory')
    final_names = ('all_worker_integrity.json', 'all_placement_integrity.json', 'all_game_replay.json',
        'performance.json', 'boundary_analysis.json', 'verification_summary.json',
        'repeated_decision_summary.json', 'replay_sample_manifest.json')
    final_artifacts = [c.artifact(QUALIFIED / name) for name in final_names]; sources.extend(final_artifacts)
    measured = dict(maximum_complete_game_seconds=max(x['collection_seconds'] for x in costs),
        maximum_game_independent_seconds=max(x['independent_seconds'] for x in costs),
        maximum_startup_and_parity_seconds=max(x['complete_startup_and_parity_seconds'] for x in old['costs']),
        per_decision_validation_seconds=max(x['counterexample_seconds'] for x in old['costs'])/52,
        maximum_closed_worker_check_seconds=max(x['log_and_raw_transport_seconds']+
            x['graph_proof_reread_seconds']+x['closed_worker_two_pass_seal_seconds'] for x in old['costs']),
        previous_qualification_stage_seconds=duration('qualification'),
        previous_final_stage_seconds=duration('final'),
        maximum_game_evidence_bytes=max(sum(x['bytes'] for x in group) for group in game_groups),
        maximum_game_evidence_files=max(map(len, game_groups)),
        isolated_qualification_evidence_bytes=sum(x['bytes'] for x in seal
            if x['path'].startswith(prefix+'workers/qualification_isolated_')),
        maximum_reproduction_bytes=max(sum(x['bytes'] for x in group) for group in repro_groups)+
            max(x['bytes'] for x in seal if x['path'].startswith(prefix+'repeated_decisions/')),
        maximum_reproduction_files=max(map(len, repro_groups))+1,
        maximum_seal_item_bytes=max(len(c.canonical(x).encode())+2 for x in seal)+96,
        previous_final_control_bytes=sum(x['bytes'] for x in final_artifacts),
        older_206_game_final_bytes=old_bytes['projected_final_evidence_bytes'])
    result = project_resources(measured)
    result['sources'] = sorted({x['path']: x for x in sources}.values(), key=lambda x: x['path'])
    return validate_resource_forecast(result)


def validate_resource_forecast(value):
    c.require(value['status'] == 'passed' and value['model_calls'] == value['fresh_games'] == 0 and
              value['outcomes_used_for_selection_or_resource_estimation'] is False, 'Model-free resource forecast required')
    finite = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    c.require(all(finite(x) for x in value['measured'].values()), 'Finite nonnegative measurements required')
    stages = value['projected_stage_seconds']
    c.require(set(stages) == set(c.STAGES) and
              all(finite(stages[k]) and stages[k] < cap for k, cap in c.STAGES.items()), 'Projected stage budget exceeded')
    total = value['projected_total_seconds']
    c.require(finite(total) and sum(stages.values()) <= total < c.JOB_SECONDS and
              sum(v for k, v in stages.items() if k != 'final') < c.COLLECTION_CUTOFF_SECONDS,
              'Projected total or collection window exceeded')
    pre, final = value['projected_prefinal_bytes'], value['projected_final_bytes']
    c.require(finite(pre) and finite(final) and pre < c.PREFINAL_EVIDENCE_BYTES and
              final < c.FINAL_RESERVE_BYTES and pre+final < c.MAX_EVIDENCE_BYTES, 'Projected evidence budget exceeded')
    expected = dict(total_games=200, operational_games=6, main_games=194,
        actual_entrypoint_workers=2, actual_entrypoint_decisions=104, timing_requests=0,
        game_workers_maximum=400, reproduction_requests_maximum=48, workers_maximum=450, graph_workers_maximum=249)
    c.require(value['inventories'] == expected, 'Exact200game/450worker inventory required')
    return value


def dependencies(prior, documents, paths):
    inherited = {item['path']: item for item in prior['inputs']}
    items = dict(inherited)
    def bind(item):
        path = Path(item['path'])
        c.require(not path.is_absolute() and (c.ROOT/path).resolve().is_relative_to(c.ROOT.resolve()),
                  'Repository-relative frozen input required')
        c.require(item['path'] not in items or items[item['path']] == item, 'Frozen input changed or conflicting dependency')
        items[item['path']] = item
    for document in documents:
        for item in nested_artifacts(document): bind(item)
    for path in paths: bind(c.artifact(path))
    for quota in (16, 32, 64):
        bind(c.artifact(c.profile_path(quota)))
        for item in c.profile_for(quota)['artifacts']: bind(item)
    for item in prior['native_artifacts']: bind(item)
    return sorted(items.values(), key=lambda item: item['path'])


def prepare():
    from .runner import verify_runtime
    c.require(not (c.OUT/'freeze.json').exists(), 'No replacement reliability freeze')
    for name in ('prelaunch_tests_verified.json', 'prelaunch_review.json'):
        receipt = c.read(c.OUT/name)
        c.require(receipt['status'] == 'passed', 'Passed tests and independent review required')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, '7GiB free disk required initially')
    source_guard = corrected.verify_runtime_sources()
    accepted = acceptance(); prior = c.read(QUALIFIED/'freeze.json')
    resource = resource_forecast(); rows = schedule.validate_schedule(schedule.make_schedule())
    c.save(c.OUT/'preparation_resource_estimate.json', resource)
    c.save(c.OUT/'qualification_acceptance.json', accepted)
    c.save(c.OUT/'schedule.json', rows)
    c.save(c.OUT/'seed_inventory.json', seed_inventory(rows))
    parity = c.read(QUALIFIED/'graph_parity_plan.json'); c.save(c.OUT/'graph_parity_plan.json', parity)
    c.save(c.OUT/'model_free_qualification.json', qualification.model_free_checks(64))
    paths = set((c.ROOT/'tools/splendorust_isolated_reliability_20261026').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_isolated_reliability*.py'))
    paths.update(p for p in c.OUT.rglob('*') if p.is_file()); paths.add(c.PROTOCOL)
    copied = ('python', 'python_executable_sha256', 'torch_runtime', 'external_runtime', 'sampler_api_modules',
        'compression_runtime', 'topology', 'native_artifacts', 'numpy_version', 'torch_package_version',
        'torch_runtime_version', 'unchanged_graph_backend_identity')
    freeze = dict(schema='splendorust_isolated_reliability_20261026_v1', time=c.now(),
        **{key: prior[key] for key in copied},
        inputs=dependencies(prior, [accepted, resource, parity, qualification.build_plan(32),
            qualification.build_plan(64), c.read(qualification.PARITY_RECEIPT)], paths),
        qualification_acceptance=accepted, source_guard=source_guard,
        ruleset_id=corrected.RULESET_ID, catalogue_sha256=corrected.CATALOGUE_SHA256,
        turn_contract=corrected.TURN_CONTRACT, checkpoint_use='explicit_historical_pretrained_warm_start_unchanged',
        placement=prior['placement'], placement_mode='isolated',
        graph_parity_plan=c.artifact(c.OUT/'graph_parity_plan.json'),
        job_seconds=c.JOB_SECONDS, stage_seconds=c.STAGES, collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        automatic_retries=0, seconds_per_move=1., internal_seconds=.9, margin_seconds=.7,
        margin_is_forfeit=False, margin_is_admission_gate=False, total_games=200,
        operational_games=6, main_games=194, pairs=100, operational_pairs=3, main_pairs=97,
        challenger_quota=64, opponent_quota=32, player_modes={'challenger': 'candidate', 'opponent': 'baseline'},
        actual_entrypoint_workers=2, actual_entrypoint_decisions=104, timing_requests=0,
        final_reproductions_maximum=48, maximum_workers=450, maximum_graph_workers=249,
        max_rss_bytes=2*c.GiB, max_evidence_bytes=c.MAX_EVIDENCE_BYTES, final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        profile_child_module='tools.splendorust_boundary_timing_20261025.worker',
        scientific_backends={'64': 'tools.splendorust_inference_opt_20261025.worker',
                             '32': 'tools.splendorust_search_scale_20261024.worker'},
        observer_scope='unchanged qualified boundary observer and symmetric saved placement verification',
        sidecar_wait_maximum_seconds=1., instrumentation_time_subtracted=False, per_inference_instrumentation=False,
        resource_gate='After first6fresh games, recompute remaining194game resource cost without outcomes.',
        failure_policy='First attributable player fault loses that game and stops collection; verify played prefix. Infrastructure/evidence faults remain incomplete.',
        slower_valid_moves='700ms is descriptive only; retain every valid move without changing the deadline or stopping rule.',
        failed_decision_retries=0, strength_inference=False, reference_promotion=False, larger_data_deferred=True)
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, model_calls=0, inputs=len(freeze['inputs']),
        freeze_sha256=c.file_hash(c.OUT/'freeze.json'))), flush=True)
