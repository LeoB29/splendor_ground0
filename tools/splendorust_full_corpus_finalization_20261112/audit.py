"""Final saved-only acceptance across two preserved, incomplete owner attempts.

Only the two unfinished saved-reader roles obtain new execution receipts. An
absent predecessor is never assigned an invented exit code or clean completion.
"""
import os
from pathlib import Path
import sys

from . import common as c


REPLACEMENTS = ('independent_mixed_terminal_r1', 'reader_mixed_terminal_r0')
JOBS = REPLACEMENTS + ('final_acceptance',)


def source_execution_folders():
    from tools.splendorust_full_corpus_recovery_20261111.audit import expected_executions
    return expected_executions()


def execution_routes(expected=None, replacements=None):
    """Exact original role allocation, with only two missing saved roles routed anew."""
    expected = source_execution_folders() if expected is None else list(expected)
    wanted = {name: c.OUT/'jobs'/name/'execution.json' for name in REPLACEMENTS}
    replacements = wanted if replacements is None else replacements
    c.require(replacements == wanted, 'Only the two named unfinished saved readers may be replaced')
    c.require(len(expected) == len(set(expected)), 'Unique expected source execution roles')
    missing = {c.SOURCE/'segments'/name for name in REPLACEMENTS}
    c.require(missing <= set(expected), 'Both unfinished roles exist in the original fixed plan')
    rows = []
    for folder in expected:
        folder = Path(folder)
        c.require(folder.is_relative_to(c.SOURCE), 'Every accepted role belongs to preserved source')
        replaced = folder in missing
        rows.append(dict(slot=folder.relative_to(c.SOURCE).as_posix(),
            source='finalization' if replaced else 'preserved_recovery',
            path=replacements[folder.name] if replaced else folder/'execution.json'))
    return rows


def passed_execution(value):
    c.require(value['status'] == 'passed' and type(value['worker_exit_code']) is int and
              value['worker_exit_code'] == 0 and value['cleanup_errors'] == [] and
              value['error'] is None, 'Every accepted execution has an observed successful exit and clean cleanup')
    c.require(type(value['observed_owned_identities']) is list and type(value['identity']) is dict,
              'Every accepted execution retains exact observed ownership')
    return value


def source_interruption(require_absent=None):
    if require_absent is None:
        from tools.splendorust_gpu_probe_20261105.control import require_absent
    c.require(not any((c.SOURCE/name).exists() for name in ('execution.json', 'completion.json', 'failure.json')),
              'Unknown recovery-owner exit is preserved without an invented terminal receipt')
    unfinished = c.SOURCE/'segments'/REPLACEMENTS[0]
    c.require(not (unfinished/'execution.json').exists() and not (unfinished/'result.json').exists() and
              not (c.SOURCE/'segments'/REPLACEMENTS[1]).exists() and not (c.SOURCE/'final_audit').exists(),
              'Only the recorded unfinished sixth independent reader and unstarted final readers remain')
    claim = c.read(c.SOURCE/'claim.json')
    owner = c.read(unfinished/'owner.json')
    root = dict(pid=claim['pid'], created=claim['created'])
    c.require((root, owner['identity']) == c.SOURCE_IDENTITIES,
              'Exactly the recorded interrupted root and last controller identities')
    for identity in (root, owner['identity']):
        require_absent(identity)
    return dict(status='incomplete_preserved', actual_root_exit_code=None,
        actual_root_exit_observed=False, clean_root_exit_claimed=False,
        root_identity=root, interrupted_controller_identity=owner['identity'],
        exact_identities_absent=True, source_claim=c.artifact(c.SOURCE/'claim.json'),
        unfinished_owner=c.artifact(unfinished/'owner.json'),
        explanation='Both recorded identities are absent; disappearance does not establish an exit code or clean owner completion.')


def verify_execution_inventory(routes, boundary=lambda: None):
    expected_source = {row['path'] for row in routes if row['source'] == 'preserved_recovery'}
    actual_source = set()
    for path in c.SOURCE.rglob('execution.json'):
        boundary(); value = c.read(path)
        if ('worker_exit_code' in value and 'observed_owned_identities' in value and 'plan_sha256' in value) or path.parent.parent == c.SOURCE/'segments':
            actual_source.add(path)
    c.require(actual_source == expected_source, 'All and only originally completed source execution roles remain')
    expected_new = {c.OUT/'jobs'/name/'execution.json' for name in REPLACEMENTS}
    actual_new = {path for path in c.OUT.rglob('execution.json')
                  if not path.is_relative_to(c.OUT/'jobs/final_acceptance')}
    c.require(actual_new == expected_new, 'Exactly two new completed saved-reader execution receipts')
    c.require({p.name for p in (c.OUT/'jobs').iterdir() if p.is_dir()} == set(JOBS),
              'Exactly three planned saved-only finalization jobs')
    c.require(not list(c.SOURCE.rglob('failure.json')) and not list(c.OUT.rglob('failure.json')),
              'No concealed recovery or finalization failure')


def verify_execution(path, boundary=lambda: None):
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    path = Path(path); boundary()
    value = passed_execution(c.read(path)); owner = c.read(path.parent/'owner.json')
    c.require(value['identity'] == owner['identity'] and owner['windows_job_assigned'] is True,
              'Actual execution binds its exact owned job')
    if path.is_relative_to(c.OUT):
        from .saved import validate_passed_execution
        c.require(path.parent.parent == c.OUT/'jobs' and path.parent.name in REPLACEMENTS,
                  'Only completed new saved-reader jobs enter this final audit')
        result = c.read(path.parent/'worker/result.json')
        validate_passed_execution(value, owner, result, mode=path.parent.name, new=True)
        c.require(value['plan_sha256'] == owner['plan_sha256'] == c.file_hash(c.OUT/'freeze.json'),
                  'New actual execution and owner bind the exact finalization freeze')
        c.verify([result['runtime_sources']], boundary=boundary)
        limits = c.original_limits()
        for deadline in ('monotonic_deadline', 'unix_deadline'):
            c.require(value[deadline] == owner[deadline] <= limits[deadline],
                      'New saved execution remains clipped to original reader/final/global deadlines')
    if 'segment_claim' in value:
        item = value['segment_claim']; c.verify([item], boundary=boundary)
        c.require(item == c.artifact(path.parent/'claim.json'), 'Exact source segment claim')
        claim = c.read(c.ROOT/item['path']); phase_item = claim['phase_clock']
        c.verify([phase_item], boundary=boundary)
        phase = c.read(c.ROOT/phase_item['path'])
        start_mono, start_wall = owner['created_monotonic'], owner['created_unix']
        end_mono, end_wall = value['ended_monotonic'], value['ended_unix']
        c.require(phase['claim'] == claim['claim'] and phase['monotonic'] <= start_mono and
                  phase['unix_time'] <= start_wall, 'Source execution starts inside its original phase')
        if claim.get('reader_clock') is not None:
            c.verify([claim['reader_clock']], boundary=boundary)
            reader = c.read(c.ROOT/claim['reader_clock']['path'])
            c.require(reader['claim'] == claim['claim'] and reader['original_phase'] == phase_item and
                      reader['monotonic'] <= start_mono <= end_mono < reader['monotonic_deadline'] and
                      reader['unix_time'] <= start_wall <= end_wall < reader['unix_deadline'],
                      'Every accepted source independent reader kept its original shorter reader clock')
    else:
        start_mono, start_wall = value['monotonic_origin'], value['unix_origin']
        end_mono, end_wall = value['monotonic_end'], value['unix_end']
        phase = value
        c.require(value.get('retries') == 0, 'No saved-job retry hidden in a completed receipt')
    c.require(start_mono <= end_mono < phase['monotonic_deadline'] and
              start_wall <= end_wall < phase['unix_deadline'], 'Actual successful exit lies inside its original clocks')
    identities = value['observed_owned_identities'] + [value['identity']]
    for identity in identities:
        boundary(); require_absent(identity)
    return c.artifact(path), identities


def acceptance_path(matchup):
    base = c.OUT if matchup == 'mixed_terminal' else c.SOURCE
    return base/'strength'/matchup/'report/independent_acceptance.json'


def verify_cohort_coverage(boundary=lambda: None):
    from tools.splendorust_full_corpus_recovery_20261111 import common as source
    total_replays = total_games = total_ops = retry_total = 0
    proofs = []
    for matchup in source.MATCHUPS:
        for rep in (0, 1):
            boundary(); name = f'independent_{matchup}_r{rep}'
            result_path = (c.OUT/'jobs'/name/'worker/result.json' if name in REPLACEMENTS else
                           c.SOURCE/'segments'/name/'result.json')
            result = c.read(result_path)
            c.require(result['status'] == 'passed' and result['error'] is None, 'Every one of six independent cohort readers passed')
            coverage = result['result']
            saved_path = c.SOURCE/'strength'/matchup/f'r{rep}'/'verification_summary.json'
            c.require(coverage == c.read(saved_path), 'Independent reader exactly reproduces original saved-cohort coverage')
            c.require(coverage['status'] == 'verified_amended_full_corpus_assessment' and
                      coverage['matchup'] == matchup and coverage['repetition'] == rep and
                      coverage['main_games'] == 200 and coverage['operational_games'] == 6 and
                      coverage['all_completed_raw_science_and_clocks_verified'] is True and
                      coverage['all_new_workers_closed'] is True and coverage['game_replacements'] == 0 and
                      coverage['automatic_retries'] == 0 and coverage['training'] is False,
                      'Complete unchanged scientific cohort acceptance')
            selection = c.read(c.SOURCE/'strength'/matchup/f'r{rep}'/'replay_selection.json')
            c.require(coverage['exact_reproductions'] == len(selection['decisions']) <= 96,
                      'All selected source reproductions retained')
            total_replays += coverage['exact_reproductions']; total_games += coverage['main_games']
            total_ops += coverage['operational_games']; retry_total += coverage['authorized_potentially_interrupted_retry_count']
            proofs.append(c.artifact(result_path))
    c.require((total_games, total_ops, total_replays, retry_total) == (1200, 36, 463, 1),
              'Exact completed source sample, all463 replays, and original single retry')
    return dict(main_games=total_games, operational_games=total_ops, exact_reproductions=total_replays,
                authorized_potentially_interrupted_retry_count=retry_total, independent_cohort_results=proofs)


def finalize_family(folder, boundary=lambda: None):
    from tools.splendorust_full_corpus_recovery_20261111 import family
    folder = Path(folder).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Final family artifacts stay outside preserved source')
    folder.mkdir(parents=True, exist_ok=True)
    records = {}; artifacts = {}
    for matchup in family.GROUPS:
        boundary(); analysis_path = c.SOURCE/'strength'/matchup/'report/analysis.json'
        accept_path = acceptance_path(matchup)
        items = dict(analysis=c.artifact(analysis_path), independent_acceptance=c.artifact(accept_path))
        c.verify(list(items.values()), boundary=boundary)
        acceptance = c.read(accept_path)
        c.require(acceptance['original_failed_attempt_preserved'] is True,
                  'Original failed full-corpus attempt stays preserved in every matchup')
        records[matchup] = dict(analysis=c.read(analysis_path), acceptance=acceptance,
                               analysis_descriptor=items['analysis'])
        artifacts[matchup] = items
    summary = family.synthesize(records)
    summary.update(accepted_group_artifacts=artifacts, saved_only_finalization=True,
        recovery_attempt_status='incomplete', recovery_owner_actual_exit_code=None,
        recovery_owner_exit_observed=False, new_models=0, new_games=0)
    item = c.save(folder/'family_summary.json', summary)
    figure = folder/'family_strength.png'
    c.require(not figure.exists(), 'No replacement finalization figure')
    family.chart(summary, figure)
    document = folder/'results.md'
    with document.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(family.markdown(summary))
        stream.write('\nThe recovery owner stopped before final acceptance, with its actual exit code unknown. '
                     'This separate saved-only finalization retains that incomplete attempt and completes only the missing verification. '
                     'The new finalization owner still requires an observed successful exit and cleanup.\n')
        stream.flush(); os.fsync(stream.fileno())
    boundary()
    return dict(status='family_summary_complete', summary=item, figure=c.artifact(figure),
        report=c.artifact(document), comparisons=3, main_games=1200, operational_games=36,
        model_calls=0, new_games=0, automatic_promotion=False, charts_require_visual_inspection=True)


def new_inventory(folder, boundary=lambda: None):
    """Inventory stable finalization evidence; the active final job is separately sealed on exit."""
    path = folder/'finalization_evidence.jsonl'; count = total = 0
    excluded = {c.OUT/'stage.json', c.OUT/'stage.tmp', c.OUT/'stage.pending',
                c.OUT/'runner_stdout.log', c.OUT/'runner_stderr.log'}
    with path.open('xb') as stream:
        for item_path in sorted(c.OUT.rglob('*')):
            if not item_path.is_file() or item_path.is_relative_to(c.OUT/'jobs/final_acceptance') or item_path in excluded:
                continue
            boundary(); before = item_path.stat(); item = c.artifact(item_path); after = item_path.stat()
            c.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Stable finalization evidence')
            stream.write((c.canonical(item)+'\n').encode()); count += 1; total += item['bytes']
        stream.flush(); os.fsync(stream.fileno())
    return dict(inventory=c.artifact(path), files=count, bytes=total,
        exclusions=['active final_acceptance job subtree', 'mutable root stage and tmp', 'retained owner stdout/stderr'])


def final(plan, folder, boundary=lambda: None):
    """Run all final acceptance obligations without any model or gameplay route."""
    from tools.splendorust_full_corpus_recovery_20261111 import common as source, recovery
    from tools.splendorust_full_corpus_recovery_20261111.boundary import check_source_boundary, verify_retry_consumption
    folder = Path(folder).resolve()
    c.require(folder == c.OUT/'jobs/final_acceptance/worker', 'Exact final saved-only worker output')
    c.require('torch' not in sys.modules, 'Final acceptance starts without model runtime')
    c.verify(plan['inputs'], boundary=boundary)
    interruption = source_interruption()
    reuse = recovery.validate_reuse(c.read(c.SOURCE/'reuse.json'), boundary=boundary)
    recovery.validate_recovery_claim(c.read(c.SOURCE/'claim.json'))
    source_boundary = check_source_boundary(boundary=boundary, deep=True)
    retry = verify_retry_consumption(boundary=boundary)
    routes = execution_routes(); verify_execution_inventory(routes, boundary)
    original = c.read(c.ORIGINAL/'execution.json')
    c.require(original['status'] == 'failed' and original['error'] is not None, 'Original full-corpus failure remains incomplete')
    receipts = []; identities = {}
    for path in [c.ROOT/item['path'] for item in original['segments']] + [row['path'] for row in routes]:
        item, observed = verify_execution(path, boundary); receipts.append(item)
        for identity in observed: identities[identity['pid'], identity['created']] = identity
    corpus = c.read(source.OLD_OUT/'collection/corpus_acceptance.json')
    c.require(corpus.get('status') in ('passed', 'accepted', 'verified') and
              c.read(source.SOURCE_ACCEPTANCE)['approved'] is True and
              c.read(source.SOURCE_CLOSEOUT)['status'] == 'completed_verified', 'Corrected source corpus acceptance retained')
    from tools.splendorust_full_corpus_20261110 import fit
    fit.validate_commitment(c.read(source.SELECTION), c.artifact(c.ORIGINAL/'all_fits_committed.json'), boundary=boundary)
    training = c.read(c.SOURCE/'training_audit/worker/training_audit.json')
    c.require(training['status'] == 'all_training_saved_verified' and
              training['new_optimizer_calls'] == training['recovery_model_calls'] == 0 and
              training['scientific_optimizer_calls'] == 7128 and training['optimizer_calls'] == 7134,
              'Accepted saved fitting audit preserves all original optimization counts')
    offline = c.read(c.SOURCE/'offline_analysis/worker/arithmetic_verification.json')
    c.require(offline['status'] == 'independent_saved_arithmetic_verified' and
              offline['model_calls'] == offline['optimizer_calls'] == 0, 'Accepted saved offline arithmetic retains no model calls')
    coverage = verify_cohort_coverage(boundary)
    family_result = finalize_family(folder/'family', boundary)
    for base, names in ((c.ORIGINAL, ('qualification', 'corpus', 'fit', 'offline')),
                        (c.SOURCE, ('strength', 'strength_replay'))):
        for name in names:
            ended = c.read(base/'stages'/f'{name}_end.json'); started = c.read(c.ROOT/ended['start']['path'])
            c.require(started['monotonic'] <= ended['monotonic'] < started['monotonic_deadline'] and
                      started['unix_time'] <= ended['unix_time'] < started['unix_deadline'], 'Completed phases kept original dual clocks')
    phase = c.read(c.SOURCE/'stages/strength_start.json'); old = c.read(c.ORIGINAL/'stages/strength_start.json')
    c.require({key: phase[key] for key in old if key != 'claim'} == {key: old[key] for key in old if key != 'claim'},
              'Original strength phase never restarted')
    preserved_original = recovery.original_inventory_verify(boundary=boundary)
    preserved_source = c.verify_source_inventory(plan, boundary=boundary)
    physical = new_inventory(folder, boundary)
    total_new = sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())
    c.require(total_new + preserved_source['bytes'] + reuse['preserved_bytes'] < source.MAX_EVIDENCE_BYTES,
              'All three evidence trees obey unchanged original logical evidence cap')
    boundary(); c.require('torch' not in sys.modules, 'Final saved acceptance never loaded model runtime')
    result = dict(status='passed', approved=True, blocking_findings=[], scope='saved_only_finalization_of_amended_recovered_full_corpus_study',
        saved_only_finalization=True, amended_recovered_study=True, original_attempt_status='incomplete',
        original_failed_attempt_preserved=True, recovery_attempt_status='incomplete', source_interruption=interruption,
        source_owner_actual_exit_code=None, source_owner_exit_observed=False, source_clean_exit_claimed=False,
        finalization_freeze=c.artifact(c.OUT/'freeze.json'), recovery_freeze=c.artifact(c.SOURCE/'freeze.json'),
        original_freeze=c.artifact(c.ORIGINAL/'freeze.json'), preserved_original=preserved_original,
        preserved_recovery=preserved_source, physical_inventory=physical, successful_actual_exits=receipts,
        source_execution_routes=[dict(row, path=c.artifact(row['path'])) for row in routes],
        exact_owned_identities_absent=list(identities.values()), replaced_saved_roles=list(REPLACEMENTS),
        coverage=coverage, family_result=family_result,
        strength_acceptances=[c.artifact(acceptance_path(group)) for group in source.MATCHUPS],
        corpus_acceptance=c.artifact(source.OLD_OUT/'collection/corpus_acceptance.json'),
        selection=c.artifact(source.SELECTION), all_fits_committed=c.artifact(c.ORIGINAL/'all_fits_committed.json'),
        cache_acceptance=c.artifact(source.CACHE_ACCEPTANCE), source_boundary=source_boundary, retry_consumption=retry,
        authorized_potentially_interrupted_retry_count=1, completed_decisions_retried=0, replacement_games=0,
        source_games=1152, main_strength_games=1200, operational_strength_games=36, exact_reproductions=463,
        new_fits=0, new_predictions=0, new_models=0, new_games=0, new_optimizer_calls=0,
        no_automatic_promotion=True, reference='original_ef8a', owner_exit_still_requires_observation=True,
        finalization_owner_exit_status='pending_actual_observation', charts_require_visual_inspection=True)
    c.save(folder/'independent_acceptance.json', result)
    return result
