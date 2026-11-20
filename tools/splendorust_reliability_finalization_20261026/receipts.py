"""Reconcile pinned successful receipts; never rerun science or runtime checks.

Only JSON receipts/manifests are read and authenticated here. Whole-tree and
source byte preservation, new owning cleanup and final acceptance belong to the
outer finalization pipeline. No old reader/runtime module is imported.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from . import common as c

ORIGINAL = 'outputs/splendorust_isolated_reliability_20261026_01'
V1 = 'outputs/splendorust_reliability_recovery_20261026_01'
V2 = 'outputs/splendorust_reliability_recovery_v2_20261026_01'
FREEZE_SHAS = {
    ORIGINAL: 'a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497',
    V1: '6be928519a8b59b27ebbb385c4a18f1f07701037817595fce96389d16952fcbe',
    V2: '6877bd2ca7fe912ca0d7402e7eedbd5f8117bc0cb26650e5752f40b2e760dbd2',
}
REQUIRED_RECEIPTS = tuple(
    f'{root}/{name}' for root, names in (
        (ORIGINAL, ('freeze.json','completion.json','execution.json','failure.json','watchdog.json',
            'performance.json','repeated_decision_summary.json','replay_sample_manifest.json','all_game_replay.json',
            'all_worker_log_integrity.json','all_raw_transport.json','all_graph_parity.json',
            'all_placement_integrity.json','completed_evidence_seal.json','qualification_summary.json')),
        (V1, ('freeze.json','execution.json','watchdog_error.json','watchdog_claim.json','watchdog.json')),
        (V2, ('freeze.json','completion.json','execution.json','failure.json','watchdog_claim.json','watchdog.json',
            'runtime_verification.json','saved_review.json','consumed_inputs.json','independent_receipt_review.json',
            'partial_recovery_report.json','corrected_runtime_sources.json')),
    ) for name in names)
KNOWN_SHA = {f'{root}/freeze.json': sha for root, sha in FREEZE_SHAS.items()}
KNOWN_SHA.update({
    f'{ORIGINAL}/completion.json':'256c47139e0b5e7123edcc829252591847d3026c8bde6b5993a7f157567f9724',
    f'{ORIGINAL}/execution.json':'634fa38887af747f666993b2d4f867b61495246aaa8cf36b7224ce0899dd8cc5',
    f'{ORIGINAL}/failure.json':'08c0b04cbe13d5beafbeac80fdf550b29bdd9461422fc3c5544e830485affd2e',
    f'{V2}/runtime_verification.json':'4f93af1c13c50a29b4ab172a2997217580f8ca7e48e9b24520e6cacdf1fcc13b',
    f'{V2}/saved_review.json':'23279ca303d08e5ebe7e274749cdb170e9091367e18d822dbf35b5200446171c',
    f'{V2}/consumed_inputs.json':'d46966c82cd78b0ad7b7d4ab57794e62a87a37d43b9887dc5d4149698d895179',
    f'{V2}/independent_receipt_review.json':'d91d426076a9d0031c14ffc4732b589411636708213282762808d5f02ec761ad',
    f'{V2}/completion.json':'20156ca68618df666eb969f12d33a08ecaaf3471149bf0afdae40be84bab4a21',
    f'{V2}/execution.json':'84dc6d2a11de7efaae6ce3ee73b820f2b2558037ef5beace39b34f33ab11c164',
    f'{V2}/failure.json':'9e65a3667d2383af26b831c6e47c0920d1e1c570f49abb7b204a50522b906053',
})
SCIENCE_COUNTS = dict(games=200, live_decisions=11020, actual_entrypoint_decisions=104,
    reproduced_completed_decisions=48, worker_count=450, graph_workers=225, worker_requests=11295,
    original_snapshot_artifacts=74579, sealed_artifacts=74528, sealed_bytes=700262089, preserved_player_faults=0)
ABSENT_PATHS = tuple(f'{root}/{name}' for root, names in (
    (ORIGINAL, ('coverage.json','verification_summary.json','stages/final_end.json','collection_stop.json')),
    (V1, ('completion.json','auditor_placement.json','runtime_verification.json','saved_review.json')),
    (V2, ('original_preservation.json','recovered_coverage.json','verification_summary.json')),
) for name in names)


def require(value, message):
    if not value:
        raise ValueError('FINALIZATION_RECEIPTS: ' + message)


def index(items, boundary=lambda: None):
    """Validate manifest metadata, without reading the listed artifacts."""
    require(type(items) is list, 'Artifact manifest must be a list')
    result, folded = {}, set()
    for item in items:
        boundary()
        require(type(item) is dict and set(item) == {'path','bytes','sha256'}, 'Artifact schema differs')
        path, size, sha = item['path'], item['bytes'], item['sha256']
        require(type(path) is str and '\\' not in path and ':' not in path and path and
            bool(PurePosixPath(path).parts) and PurePosixPath(path).as_posix() == path and not PurePosixPath(path).is_absolute() and
            all(part not in ('.','..','') for part in PurePosixPath(path).parts), 'Noncanonical artifact path')
        require(path not in result and path.casefold() not in folded, 'Duplicate or case-aliased artifact')
        require(type(size) is int and size >= 0 and type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha), 'Artifact size/hash differs')
        result[path] = item
        folded.add(path.casefold())
    return result


def same_subset(subset, whole, label):
    require(all(whole.get(path) == item for path, item in subset.items()), label + ' artifact bindings differ')


def load_pinned(path, pins, envelope):
    envelope.guard()
    require(path in pins, 'Required receipt was not frozen: ' + path)
    item = pins[path]
    require(item['bytes'] <= 32*1024**2, 'Receipt JSON exceeds bounded metadata size')
    require(path not in KNOWN_SHA or item['sha256'] == KNOWN_SHA[path], 'Known successful/failure receipt pin differs')
    data = bytearray()
    with (c.ROOT/path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            envelope.guard()
            data.extend(chunk)
            require(len(data) <= item['bytes'], 'Receipt grew beyond frozen bytes')
    require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'Pinned receipt changed: ' + path)
    envelope.guard()
    return json.loads(data)


def check_cleanup(execution):
    require(execution['cleanup_errors'] == [] and execution['cleanup_receipt']['errors'] == [] and
            execution['cleanup_receipt']['unconfirmed'] == [], 'Recorded owner cleanup was incomplete')


def validate_failures(documents):
    """Only the three disclosed interruptions qualify; never relabel success."""
    read = lambda root, name: documents[f'{root}/{name}']
    old, old_execution, old_failure = (read(ORIGINAL, name) for name in ('completion.json','execution.json','failure.json'))
    require(old['status'] == 'incomplete' and old['stage'] == 'final' and old['coverage'] is None and
        old_execution['status'] == 'runner_failed' and old_execution['exit_code'] == 1 and old['automatic_retries'] == 0 and
        old['freeze_sha256'] == FREEZE_SHAS[ORIGINAL] and old['failure'] == old_failure['error'] and
        old_failure['stage'] == 'final' and old_failure['partial_evidence_not_accepted'] is True and
        old_failure['error'].endswith('ValueError: Stage/global real-clock deadline\n') and
        'line 199, in verify_runtime' in old_failure['error'], 'Original source-guard interruption differs')
    first, first_error = read(V1,'execution.json'), read(V1,'watchdog_error.json')
    require(first['status'] == 'failed' and first['exit_code'] == 0 and
        first_error['error'].endswith('ValueError: RECOVERY: Auditor launched an unexpected child\n') and
        'in resource_guard' in first_error['error'], 'First failed bootstrap differs')
    complete, execution, failure = (read(V2, name) for name in ('completion.json','execution.json','failure.json'))
    require(complete['status'] == 'incomplete' and complete['original_status'] == 'incomplete' and
        execution['status'] == 'watchdog_deadline' and execution['exit_code'] == 0 and
        execution['bootstrap_relaunches'] == 1 and execution['model_or_decision_retries'] == 0 and
        1800 <= execution['charged_job_seconds'] < 1801, 'Second recovery status/deadline differs')
    require(failure['phase'] == 'final_original_snapshot_readback_begin' and failure['original_attempt_preserved'] is True and
        failure['error'].endswith('ValueError: RECOVERY: Recovery UTC/monotonic deadline\n'), 'Only final preservation remained')
    frames = re.findall(r'File "([^\"]+)", line (\d+), in (\w+)', failure['error'])
    expected = [('runner.py','85','main'),('runner.py','49','execute'),('runner.py','17','finish_snapshot'),
                ('preparation.py','15','inventory'),('common.py','75','guard'),('common.py','28','require')]
    require([(Path(path).name, line, function) for path,line,function in frames] == expected and
        all(Path(path).resolve().parent == c.ROOT/'tools/splendorust_reliability_recovery_v2_20261026' for path,_,_ in frames),
        'Recovery interruption was not the first final inventory')
    for root, record in ((ORIGINAL,old_execution),(V1,first),(V2,execution)):
        check_cleanup(record)
        require(record['freeze_sha256'] == FREEZE_SHAS[root], 'Failure freeze identity differs')
        if root != ORIGINAL:
            require(record['automatic_retries'] == 0, 'Recovery retry differs')
    for root in (V1,V2):
        claim = read(root,'watchdog_claim.json')
        record = first if root == V1 else execution
        require(claim['freeze_sha256'] == FREEZE_SHAS[root] and
            claim['monotonic_origin'] == record['monotonic_origin'] == 297762.765 and
            claim['unix_origin'] == record['unix_origin'] == 1793041905.7336407, 'Recovery deadline was reset')
    return dict(original_status='incomplete',original_exit_code=1,first_recovery_status='failed',
                second_recovery_status='incomplete',second_recovery_execution='watchdog_deadline',
                exit_zero_after_job_cleanup_is_not_acceptance=True,all_recorded_failures_preserved=True)


def validate_science(saved, original_performance, repeated, sample, independent):
    require(saved['status'] == 'passed_saved_evidence_review' and saved['freeze_sha256'] == FREEZE_SHAS[ORIGINAL] and
        all(type(saved[key]) is int and saved[key] == value for key,value in SCIENCE_COUNTS.items()), 'Successful saved science counts differ')
    require(saved['original_completion_status'] == 'incomplete' and saved['original_exit_code'] == 1 and
        saved['original_snapshot_byte_authentication'] == 'passed' and saved['boundary_failures'] == [] and
        saved['no_overall_recovered_acceptance_claim'] is True and saved['no_causal_or_strength_claim'] is True and
        saved['final_snapshot_readback'] == 'required_from_owning_recovery_pipeline_before_acceptance', 'Saved component scope differs')
    coverage = saved['coverage']
    require(coverage['status'] == 'full_workload_verified' and coverage['planned_games'] == coverage['verified_games'] == 200 and
        coverage['unplayed_games'] == 0 and coverage['verified_games_by_stage'] == {'main':194,'operational':6} and
        coverage['complete_planned_workload'] is True and coverage['failure'] is None and
        coverage['stopped_on_first_attributable_failure'] is False and coverage['placement_mode'] == 'isolated' and
        coverage['engineering_margin_is_descriptive_only'] is True and coverage['no_strength_claim'] is True,
        'Saved coverage is not the full isolated workload')
    performance = saved['performance']
    require(performance == original_performance and performance['natural_games'] == performance['games'] == 200 and
        performance['forfeits'] == [] and performance['stage_counts'] == {'main':194,'operational':6} and
        performance['engineering_margin_is_descriptive_only'] is True and performance['engineering_margin_changes_completion_or_stopping'] is False and
        performance['no_strength_claim'] is True, 'Preserved performance scope differs')
    for role,quota,slow,maximum in (('challenger',64,1,.8542834),('opponent',32,0,.5134891)):
        player=performance['players'][role]
        require(player['quota'] == quota and player['attempted_decisions'] == player['completed_decisions'] == 5510 and
            player['statuses'] == {'ok':5510} and player['completed_arrival_seconds']['count'] == 5510 and
            player['completed_arrival_seconds']['max'] == maximum and player['completed_at_or_above_engineering_margin'] == slow,
            'Saved per-player timing/counts differ')
    require(repeated['status'] == 'passed' and repeated['requests'] == 48 and repeated['prefix_only'] is False and
        repeated['identities'] == sample['decisions'] and len(sample['decisions']) == 48 and sample['empty_cells'] == [] and
        sample['selection_uses_outcomes'] is False and sample['selection_uses_timing'] is False and
        saved['replay_cap'] == saved['replay_populated_cells'] == 48 and saved['replay_empty_cells'] == 0,
        'Existing48 completed reproductions or frozen selection differ')
    require(independent['passed_components']['saved_scientific_evidence'] == SCIENCE_COUNTS, 'Independent scientific receipt disagrees')
    return coverage, performance


def reconcile_manifests(freeze, original, first, second, consumed, saved, runtime, corrected, envelope):
    original_inputs = index(original['inputs'],envelope.guard)
    index(first['inputs'],envelope.guard)
    second_inputs = index(second['inputs'],envelope.guard)
    snapshot = index(second['original_snapshot'],envelope.guard)
    require(len(snapshot) == 74579 and index(freeze['original_snapshot'],envelope.guard) == snapshot and
        index(first['original_snapshot'],envelope.guard) == snapshot, 'Three original snapshot manifests disagree')
    require(index(freeze['preservation_trees'][ORIGINAL],envelope.guard) == snapshot, 'Final preservation omits original snapshot entries')
    artifacts = index(consumed['artifacts'],envelope.guard)
    require(consumed['count'] == len(artifacts) == 74635, 'Successful consumed manifest count differs')
    consumed_original = {path:item for path,item in artifacts.items() if path.startswith(ORIGINAL+'/')}
    require(consumed_original == snapshot, 'Consumed manifest omits/changes original evidence or metadata')
    validators = {path:item for path,item in artifacts.items() if not path.startswith(ORIGINAL+'/')}
    require(len(validators) == saved['imported_validator_sources'] == 56 and all(path.endswith('.py') for path in validators),
        'Scientific imported-source inventory differs')
    frozen_validators = {**original_inputs, **second_inputs}
    same_subset(validators, frozen_validators, 'Consumed validator source')
    source_preservation = index(freeze['source_preservation'],envelope.guard)
    same_subset(validators, source_preservation, 'Final validator source preservation')
    bindings = runtime['imported_validator_bindings']
    bound = index(bindings['sources'],envelope.guard)
    require(bindings['status'] == 'passed' and len(bound) == 50 and len(bindings['functions']) == 19,
        'Successful runtime validator inventory differs')
    same_subset(bound, original_inputs, 'Runtime frozen sources')
    same_subset(bound, validators, 'Runtime/science source agreement')
    functions=set()
    for entry in bindings['functions']:
        envelope.guard()
        identity=(entry['module'],entry['function'])
        require(identity not in functions and bound.get(entry['source']['path']) == entry['source'] and
            entry['source']['path'] == entry['module'].replace('.','/')+'.py', 'Runtime callable ownership receipt differs')
        functions.add(identity)
    require(runtime['preimport_frozen_python_sources'] == sum(path.endswith('.py') and not path.startswith('outputs/') for path in original_inputs),
        'Preimport runtime source count differs')
    require(len(corrected) == 8 and all(original_inputs.get(path,{}).get('sha256') == sha for path,sha in corrected.items()),
        'Corrected startup source receipt differs')
    return dict(original_files=len(snapshot),consumed_entries=len(artifacts),scientific_validator_sources=len(validators),
                runtime_validator_sources=len(bound),runtime_callable_bindings=len(functions),metadata_agreement='passed')


def validate_receipts(freeze,envelope):
    """Return reused component evidence, conditional on final byte preservation."""
    envelope.guard()
    pins = index(freeze['receipt_pins'],envelope.guard)
    require(set(pins) == set(REQUIRED_RECEIPTS), 'Required receipt pin inventory differs')
    documents = {path:load_pinned(path,pins,envelope) for path in REQUIRED_RECEIPTS}
    read=lambda root,name:documents[f'{root}/{name}']
    protected={}
    for root,items in freeze['preservation_trees'].items():
        envelope.guard()
        current=index(items,envelope.guard)
        require(all(path.startswith(root+'/') for path in current), 'Protected tree contains another tree')
        require(not set(protected)&set(current), 'Overlapping protected tree manifests')
        protected.update(current)
    require(set(freeze['preservation_trees']) == {ORIGINAL,V1,V2}, 'All three prior attempts must be preserved')
    same_subset(pins,protected,'Receipt/whole-tree preservation')
    require(all(path not in protected and not (c.ROOT/path).exists() for path in ABSENT_PATHS), 'Original missing acceptance marker was replaced')
    failures=validate_failures(documents)
    original,first,second=(read(root,'freeze.json') for root in (ORIGINAL,V1,V2))
    require(second['original_freeze_sha256'] == first['original_freeze_sha256'] == FREEZE_SHAS[ORIGINAL] and
        second['bootstrap_relaunches'] == 1 and second['model_or_decision_retries'] == 0 and
        second['shared_monotonic_origin'] == 297762.765 and second['shared_unix_origin'] == 1793041905.7336407,
        'Frozen recovery ancestry/budget differs')
    for name in ('completion.json','execution.json','failure.json'):
        require(first['original_pins'][name] == second['original_pins'][name] == pins[f'{ORIGINAL}/{name}']['sha256'],
                'Historical original failure pins differ')
    independent=read(V2,'independent_receipt_review.json')
    require(independent['status'] == 'partial_recovery_only' and independent['accepted'] is False and
        independent['no_recovered_acceptance'] is True and independent['original_status'] == 'incomplete' and
        independent['v2_status'] == 'incomplete' and independent['v2_execution_status'] == 'watchdog_deadline' and
        independent['cleanup_errors'] == [] and independent['exit_code_zero_after_owning_cleanup_is_not_acceptance'] is True,
        'Independent partial review was mislabelled as acceptance')
    require(independent['incomplete_work']['failure_phase'] == 'final_original_snapshot_readback_begin' and
        independent['incomplete_work']['final_inventory_completed'] is False and
        independent['incomplete_work']['final_original_byte_readback_started'] is False,
        'Independent failure scope differs')
    independent_inputs=index(independent['consumed_receipts_and_source_artifacts'],envelope.guard)
    all_preserved={**protected,**index(freeze['source_preservation'],envelope.guard)}
    same_subset(independent_inputs,all_preserved,'Independent partial review input')
    for path in (f'{V2}/saved_review.json',f'{V2}/runtime_verification.json',f'{V2}/consumed_inputs.json',
                 f'{V2}/freeze.json',f'{ORIGINAL}/freeze.json',f'{ORIGINAL}/completion.json',f'{V2}/failure.json'):
        require(independent_inputs.get(path) == pins[path], 'Independent partial review did not bind a reused receipt')
    runtime,saved=read(V2,'runtime_verification.json'),read(V2,'saved_review.json')
    require(runtime['status'] == 'passed_runtime_and_prior_recovery' and runtime['original_freeze'] == pins[f'{ORIGINAL}/freeze.json'] and
        runtime['original_inputs_rechecked_from_start'] == len(original['inputs']) == 9330 and
        runtime['preceding_frozen_inputs'] == 4885 and runtime['preceding_sealed_artifacts'] == 4392 and
        runtime['preceding_qualification_exact_acceptance'] == 'passed' and runtime['previous_partial_loop_reused'] is False and
        runtime['original_incomplete_status_unchanged'] is True and runtime['original_outputs_written'] is False,
        'Successful complete runtime/prior receipt differs')
    require(all(runtime[key] == 0 for key in ('model_calls','model_imports','fresh_decisions','fresh_games','workers_launched')) and
        all(runtime[key] == 'passed' for key in ('compression_runtime','physical_host_and_isolated_supervisor','python_and_numerical_packages','sampler_api_owners')) and
        runtime['native_artifacts'] == len(original['native_artifacts']) and runtime['torch_runtime_files'] == len(original['torch_runtime']) and
        runtime['external_runtime_files'] == len(original['external_runtime']), 'Runtime receipt scope/counts differ')
    coverage,performance=validate_science(saved,read(ORIGINAL,'performance.json'),read(ORIGINAL,'repeated_decision_summary.json'),
        read(ORIGINAL,'replay_sample_manifest.json'),independent)
    manifest=reconcile_manifests(freeze,original,first,second,read(V2,'consumed_inputs.json'),saved,runtime,
                                read(V2,'corrected_runtime_sources.json'),envelope)
    require(independent['passed_components']['consumed_manifest_entries'] == manifest['consumed_entries'] and
        independent['passed_components']['original_runtime_inputs_from_index_zero'] == 9330, 'Independent source/manifest counts differ')
    seal=index(read(ORIGINAL,'completed_evidence_seal.json')['artifacts'],envelope.guard)
    require(len(seal) == 74528 and sum(item['bytes'] for item in seal.values()) == 700262089, 'Original completed seal metadata differs')
    same_subset(seal,protected,'Original completed seal')
    games=read(ORIGINAL,'all_game_replay.json')
    require(games['status'] == 'passed' and games['games'] == len(games['receipts']) == 200 and
        sum(row['transitions'] for row in games['receipts']) == 11020 and all(row['status'] == 'passed' and row['failed_moves'] == 0 for row in games['receipts']),
        'Passed original game-science receipt inventory differs')
    for name,count in (('all_worker_log_integrity.json',450),('all_raw_transport.json',450),('all_graph_parity.json',225),('all_placement_integrity.json',450)):
        value=read(ORIGINAL,name)
        require(value['status'] == 'passed' and value['workers'] == len(value['receipts']) == count and
            len({row['worker'] for row in value['receipts']}) == count, 'Original worker proof inventory differs')
    raw=read(ORIGINAL,'all_raw_transport.json')
    require(sum(row['requests'] for row in raw['receipts']) == 11295 and all(row['status'] == 'passed' and row['failed_requests'] == 0 for row in raw['receipts']),
        'Original raw request/failure inventory differs')
    require(all(row['status'] == 'passed' and row['fixtures'] == 11 for row in read(ORIGINAL,'all_graph_parity.json')['receipts']),
        'Original graph receipt inventory differs')
    require(read(ORIGINAL,'all_placement_integrity.json')['all_workers_isolated'] is True and
        all(row['placement_mode'] == 'isolated' for row in read(ORIGINAL,'all_placement_integrity.json')['receipts']), 'Placement proof scope differs')
    qualified=read(ORIGINAL,'qualification_summary.json')
    require(qualified['status'] == 'passed' and qualified['actual_workers'] == 2 and qualified['complete_decisions'] == 104,
        'Original startup receipt differs')
    partial=read(V2,'partial_recovery_report.json')
    require(partial['status'] == 'partial_recovery_only_overall_incomplete' and partial['observed_player_faults'] == 0 and
        all(partial[key] == 0 for key in ('new_games','new_models','new_decision_reproductions')), 'Partial report scope differs')
    envelope.guard()
    return dict(status='passed_reused_receipt_reconciliation',schema='splendorust_finalization_receipts_v1',
        **SCIENCE_COUNTS,coverage=coverage,performance=performance,failures=failures,manifest_reconciliation=manifest,
        reused_runtime_inputs=9330,reused_prior_qualification_inputs=4885,reused_prior_sealed_artifacts=4392,
        receipt_artifacts=list(pins.values()),new_games=0,new_models=0,new_reproductions=0,
        full_science_or_runtime_rerun=False,full_original_tree_byte_pass=False,
        no_overall_acceptance_claim=True,requires_final_byte_preservation_and_clean_owning_exit=True,
        no_causal_or_strength_claim=True,owned_process_identities=independent['owned_process_identities'])
