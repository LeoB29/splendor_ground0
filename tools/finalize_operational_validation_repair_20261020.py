"""Exclusive, outcome-closed acceptance of the authorized final-validation repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/live_game_operational_recovery_20261020_02_validation_02'
SOURCE = ROOT / 'outputs/live_game_operational_recovery_20261020_02'
REPORT = ROOT / 'docs/engineering/live_game_operational_final_validation_repair_20261020_result.md'
CAP, CPU = 43200, 2


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def artifact(path):
    path = path.resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def closed(identity):
    require(type(identity.get('pid')) is int and identity['pid'] > 0 and
            type(identity.get('created')) in (int, float), 'Process identity schema')
    try:
        return psutil.Process(identity['pid']).create_time() != identity['created']
    except psutil.NoSuchProcess:
        return True


def boundary():
    launch = read(SOURCE / 'launch.json')
    require(launch['maximum_seconds'] == CAP, 'Original cap unchanged')
    remaining = min(launch['epoch_seconds'] + CAP - time.time(),
                    launch['monotonic_start'] + CAP - time.monotonic())
    size = sum(p.stat().st_size for base in (SOURCE, BASE) for p in base.rglob('*') if p.is_file())
    free = shutil.disk_usage(BASE).free
    require(remaining > 0 and size < 4 * 1024**3 - 32 * 1024**2 and free >= 5 * 1024**3,
            'Original finalization resource limits')
    return {'remaining_seconds': remaining, 'combined_evidence_bytes': size,
            'disk_free_bytes': free, 'original_maximum_seconds': CAP}


def check_artifacts(rows, check_clock=None):
    require(len({row['path'] for row in rows}) == len(rows), 'Unique artifact paths')
    for row in rows:
        if check_clock is not None:
            check_clock()
        path = ROOT / row['path']
        require(path.resolve().is_relative_to(ROOT) and artifact(path) == row, 'Bound artifact changed')


def source_preservation(freeze, check_clock=None):
    check_artifacts(freeze['inputs'], check_clock)
    snapshot = read(ROOT / freeze['source_inventory']['path'])
    require(snapshot['source_base'] == SOURCE.relative_to(ROOT).as_posix() and snapshot['outcomes_closed'] is True,
            'Source inventory scope')
    actual = {p.relative_to(ROOT).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    require(actual == {row['path'] for row in snapshot['files']}, 'Exact preserved source inventory')
    check_artifacts(snapshot['files'], check_clock)
    original = read(SOURCE / 'freeze.json')
    check_artifacts(original['inputs'], check_clock)
    require(read(SOURCE / 'coordinator_hold.json')['status'] == 'preserved_hold', 'Historical hold retained')
    return {'source_files': len(snapshot['files']), 'original_frozen_inputs': len(original['inputs']),
            'all_hashes_and_sizes_match': True, 'historical_hold_retained': True}


def operational_report(analysis, execution, preservation):
    require(analysis['status'] == 'passed' and analysis['outcomes_closed'] is True and
            analysis['strength_analysis_games'] == 0 and analysis['larger_data_deferred'] is True and
            analysis['operational_schedule']['games'] == 18 and analysis['operational_schedule']['pairs'] == 9,
            'Accepted operational analysis scope')
    local = analysis['per_stage']['games:local']
    opponent = analysis['per_stage']['games:opponent']
    require(local['statuses'].get('forfeit_timeout') == 4 and
            analysis['scope']['original_timeout_forfeits_preserved'] == 4,
            'All four local timeouts retained')
    coverage = analysis['actual_continuation_verification']
    require(coverage['roots'] == 361 and coverage['full_stress_roots'] == 45 and
            coverage['sample_only_roots'] == 316 and
            analysis['source_preservation']['all_hashes_and_sizes_match'] is True,
            'Complete prescribed replay and source checks')
    rows = []
    for label, data in [('Local four-candidate planner', local), ('SplendoRust', opponent)]:
        timing = data['completed_receipt_seconds']
        failed = sum(n for status, n in data['statuses'].items() if status != 'completed')
        fmt = lambda v: 'n/a' if v is None else f'{v:.3f}'
        rows.append(f"| {label} | {data['statuses'].get('completed', 0)} / {data['attempted']} | "
                    f"{fmt(timing['mean'])} | {fmt(timing['p95'])} | {fmt(timing['max'])} | {failed} |")
    cost = analysis['resources']
    replay = coverage['replayed_union']
    combined = analysis['combined_memory_waiting']
    return '\n'.join([
        '# Operational validation completed with four retained timeouts', '',
        'The eighteen-game operational stage and its prescribed correctness checks passed independent validation. '
        'All four earlier local planner timeouts remain recorded. This validates the saved evidence; it does not '
        'establish failure-free operation at 120 seconds or playing strength. Matchup outcomes remain closed.', '',
        'Acceptance additionally requires the repair directory\'s `finalization_validation.json` to say `passed` '
        'and no `hold.json` in that directory. The original coordinator hold remains preserved historical evidence.', '',
        '| Player | Completed / attempted moves | Mean seconds | 95th percentile | Maximum completed seconds | Failures |',
        '|---|---:|---:|---:|---:|---:|', *rows, '',
        f"The original twelve validation decisions and six exact hidden-state repeat pairs were authenticated and reused. "
        f"The game audit covered {coverage['roots']} roots: {coverage['full_stress_roots']} full and "
        f"{coverage['sample_only_roots']} sampled. Its unique replay union contains {replay['worlds']} worlds, "
        f"{replay['continuations']} continuations and {replay['transitions']} transitions. "
        f"Stress coverage was {coverage['covered_cells']} of {coverage['cells']} prescribed cells; absent cells remain untested.", '',
        f"All {preservation['source_files']} completed-source files and {preservation['original_frozen_inputs']} original "
        'frozen inputs were unchanged. The new entry wrapper called the frozen numerical analyzer unchanged. '
        'It separately verified the repair provenance and fresh memory admission. The launcher authenticated its '
        'observed console helper and checked owned-process closure. The failed first launch and its original memory '
        'admission, empty log and diagnostics were retained.', '',
        f"The original production segment took {cost['original_attempt_elapsed_seconds']/3600:.3f} hours, "
        f"the first recovery {cost['first_recovery_elapsed_seconds']/3600:.3f} hours, and the second recovery's "
        f"production {cost['production_elapsed_seconds']/3600:.3f} hours. Elapsed time from the second recovery "
        f"launch through the independent check was {analysis['final_validation_repair']['elapsed_from_original_launch_seconds']/3600:.3f} hours, "
        'including the stopped-validation interval and repair preparation. That elapsed interval is not all active computation. '
        'No clock or deadline was reset.', '',
        f"Combined memory admission used {combined['admissions']} gates and charged {combined['charged_seconds']/60:.2f} minutes, "
        'including the supplemental stable window. All admission and validation time remained within the original '
        'twelve-hour recovery cap. Player clocks remained at 120 seconds; no move or forfeit was replayed as a new decision.', '',
        'Unsampled continuations received structural and arithmetic checks, not exhaustive replay. Local replay shares '
        'frozen feature and support kernels. The external bot\'s native search edges and neural outputs retain the '
        'qualified trust boundary. Resource observations are sampled. These operational games and the separate six-game '
        'pilot remain excluded from any future fresh strength sample.', '',
        'Next: examine the four recorded local timeouts and determine the appropriate operational policy before committing '
        'to a fresh matched strength benchmark. The larger-data study remains deferred. No further games, player changes, '
        'training or outcome opening are authorized by this report.', '',
        'Evidence: `outputs/live_game_operational_recovery_20261020_02_validation_02/`, including the repair freeze, '
        'launcher plan and execution, independent analysis, supplemental admission and finalization receipts.', ''])


def finalize(expected_freeze, expected_plan, expected_execution):
    psutil.Process().cpu_affinity([CPU])
    require(not (BASE / 'hold.json').exists() and not REPORT.exists() and
            not (BASE / 'finalization.json').exists() and not (BASE / 'finalization_validation.json').exists(),
            'Exclusive new acceptance with no repair hold')
    require(artifact(BASE / 'repair_freeze.json')['sha256'] == expected_freeze and
            artifact(BASE / 'launcher_plan.json')['sha256'] == expected_plan and
            artifact(BASE / 'execution.json')['sha256'] == expected_execution, 'Expected acceptance inputs')
    freeze = read(BASE / 'repair_freeze.json')
    require(artifact(Path(__file__)) in freeze['inputs'], 'Frozen finalizer identity')
    launch = read(SOURCE / 'launch.json')
    def check_clock():
        require(0 <= time.time() - launch['epoch_seconds'] < CAP and
                0 <= time.monotonic() - launch['monotonic_start'] < CAP, 'Original finalization clocks')
    boundary()
    preservation = source_preservation(freeze, check_clock)
    # The execution schema is deliberately checked by the launcher-specific validator below.
    from tools.operational_final_validation_repair_20261020 import validate_execution_for_finalization
    execution = validate_execution_for_finalization(expected_freeze, expected_plan, expected_execution)
    analysis = read(BASE / 'independent_analysis.json')
    report = operational_report(analysis, execution, preservation)
    boundary()
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(report)
    paths = [BASE / name for name in ('repair_freeze.json', 'launcher_plan.json', 'launch.json', 'execution.json',
              'analyzer.log', 'independent_analysis.json', 'memory_admission.json', 'memory_admission.jsonl')]
    paths += [REPORT, Path(__file__), ROOT / freeze['source_inventory']['path']]
    artifacts = [artifact(path) for path in paths]
    final = {'schema': 'operational_validation_repaired_finalization_v1',
        'status': 'operational_checks_passed_with_preserved_timeouts', 'time': now(),
        'games': 18, 'pairs': 9, 'preserved_local_timeout_forfeits': 4,
        'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True,
        'source_preservation': preservation, 'original_coordinator_hold_retained': True,
        'combined_memory_waiting': analysis['combined_memory_waiting'],
        'artifacts': artifacts, 'resources': boundary(),
        'acceptance_requires': 'This directory finalization_validation.json status passed and hold.json absent',
        'additional_games_or_training_authorized': False}
    save(BASE / 'finalization.json', final)
    require(read(BASE / 'finalization.json') == final, 'Finalization delivery')
    check_artifacts(artifacts, check_clock)
    require(not (BASE / 'hold.json').exists(), 'No repair hold')
    validation = {'schema': 'operational_validation_repaired_acceptance_v1', 'status': 'passed', 'time': now(),
        'finalization': artifact(BASE / 'finalization.json'), 'bound_artifacts_rechecked': len(artifacts),
        'resources': boundary(), 'original_coordinator_hold_is_preserved_historical_evidence': True,
        'acceptance_invalid_if_repair_hold_exists': True, 'outcomes_closed': True}
    save(BASE / 'finalization_validation.json', validation)
    require(read(BASE / 'finalization_validation.json') == validation, 'Acceptance delivery')
    boundary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repair-freeze-sha256', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    try:
        finalize(args.repair_freeze_sha256, args.plan_sha256, args.execution_sha256)
    except BaseException as error:
        failure = BASE / 'sealed/finalization_failure_01.json'
        if not failure.exists():
            save(failure, {'time': now(), 'traceback': traceback.format_exc(), 'outcomes_closed': True})
        if not (BASE / 'hold.json').exists():
            save(BASE / 'hold.json', {'status': 'preserved_hold', 'time': now(),
                 'diagnostic': artifact(failure), 'outcomes_closed': True})
        print(json.dumps({'status': 'preserved_hold', 'category': type(error).__name__, 'outcomes_closed': True}))
        return 1
    print(json.dumps({'status': 'passed', 'games': 18, 'pairs': 9, 'preserved_timeouts': 4, 'outcomes_closed': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
