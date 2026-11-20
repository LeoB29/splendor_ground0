"""Read-only operational timing summary; no model work or matchup outcomes."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
ACCEPTED = ROOT/'outputs/live_game_operational_recovery_20261020_02_validation_02'
OUTPUT = ROOT/'outputs/local_timeout_policy_review_20261020_01/aggregate_completed.json'
EXPECTED_TIMEOUTS = {'games/g006/t0000', 'games/g007/t0007', 'games/g017/t0006', 'games/g019/t0007'}


def artifact(path):
    path = path.resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summary(rows):
    completed = [row for row in rows if row['status'] == 'completed']
    def descriptive(values):
        return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
                'min': min(values), 'max': max(values)} if values else {'count': 0}
    return {'attempted': len(rows), 'statuses': dict(Counter(row['status'] for row in rows)),
        'completed_receipt_seconds': descriptive([row['receipt_seconds'] for row in completed]),
        'completed_request_cpu_seconds': descriptive([row['request_cpu_seconds'] for row in completed]),
        'completed_cpu_to_supervisor_wall_ratio': descriptive([
            row['request_cpu_seconds']/row['supervisor_seconds'] for row in completed]),
        'completed_over_100_seconds': sum(row['receipt_seconds'] > 100 for row in completed),
        'completed_over_110_seconds': sum(row['receipt_seconds'] > 110 for row in completed),
        'maximum_sampled_tree_rss_mib': max(row['peak_sampled_tree_rss_mib'] for row in rows)}


def main():
    psutil.Process().cpu_affinity([2])
    assert not OUTPUT.exists()
    acceptance = read(ACCEPTED/'finalization_validation.json')
    final = read(ACCEPTED/'finalization.json')
    assert acceptance['status'] == 'passed' and not (ACCEPTED/'hold.json').exists()
    assert acceptance['finalization'] == artifact(ACCEPTED/'finalization.json')
    assert artifact(ACCEPTED/'independent_analysis.json') in final['artifacts']
    assert artifact(ACCEPTED/'repair_freeze.json') in final['artifacts']
    repair = read(ACCEPTED/'repair_freeze.json')
    assert repair['original_freeze'] == artifact(SOURCE/'freeze.json')
    assert repair['source_inventory'] == artifact(ROOT/repair['source_inventory']['path'])
    original = read(SOURCE/'freeze.json')
    snapshot = read(ROOT/repair['source_inventory']['path'])
    index = {}
    for item in original['inputs']+snapshot['files']:
        assert item['path'] not in index or index[item['path']] == item
        index[item['path']] = item
    used = {}
    def checked(path):
        value = artifact(path)
        assert index.get(value['path']) == value, 'Changed or unbound source input'
        used[value['path']] = value
        return read(path)
    schedule_source = ROOT/'tools/analyze_live_game_operational_20261019.py'
    assert index[schedule_source.relative_to(ROOT).as_posix()] == artifact(schedule_source)
    used[schedule_source.relative_to(ROOT).as_posix()] = artifact(schedule_source)
    from tools import analyze_live_game_operational_20261019 as core
    specs = {spec['game']: spec for spec in core.schedule()}
    rows = []
    paths = sorted((SOURCE/'decision_attempts/games').glob('g*/t*.json'))
    for path in paths:
        record = checked(path)
        if record['player'] != 'local':
            continue
        clock, init = record['clock'], record['initialization']
        game, ply = int(path.parent.name[1:]), int(path.stem[1:])
        assert clock['deadline_seconds'] == 120 and init['deadline_seconds'] == 60 and init['accepted'] is True
        receipt_path = ROOT/record['receipt']['path']
        assert artifact(receipt_path) == record['receipt']
        receipt = checked(receipt_path)
        assert receipt['status'] == clock['status'] and receipt['accepted'] == clock['accepted']
        relative = record['receipt']['path']
        segment = ('recovery02' if 'recovery_20261020_02/' in relative else
                   'recovery01' if 'recovery_20261020_01/' in relative else 'original')
        row = {'id': record['id'], 'game': game, 'ply': ply, 'fit': specs[game]['fit'],
               'segment': segment, 'status': record['status'], 'supervisor_seconds': clock['elapsed_seconds'],
               'receipt_seconds': clock['receipt_elapsed_seconds'],
               'request_cpu_seconds': clock['telemetry']['request_cpu_seconds'],
               'peak_sampled_tree_rss_mib': clock['telemetry']['peak_sampled_rss_bytes']/2**20,
               'initialization_seconds': init['elapsed_seconds'], 'receipt': record['receipt']}
        if record['status'] == 'forfeit_timeout':
            assert clock['status'] == 'timeout' and clock['accepted'] is False
            assert clock['receipt_elapsed_seconds'] is None and clock['received_before_deadline'] is False
            assert receipt['result'] is None
        else:
            assert record['status'] == 'completed' and clock['status'] == 'ok' and clock['accepted'] is True
            assert clock['received_before_deadline'] is True and clock['receipt_elapsed_seconds'] <= 120
        rows.append(row)
    failures = [row for row in rows if row['status'] != 'completed']
    assert {row['id'] for row in failures} == EXPECTED_TIMEOUTS
    assert len(rows) == 365 and len(failures) == 4 and len({row['game'] for row in failures}) == 4
    aggregate = summary(rows)
    accepted = read(ACCEPTED/'independent_analysis.json')['per_stage']['games:local']
    assert aggregate['statuses'] == accepted['statuses'] and aggregate['attempted'] == accepted['attempted']
    assert abs(aggregate['completed_receipt_seconds']['mean']-accepted['completed_receipt_seconds']['mean']) < 1e-10
    assert aggregate['completed_receipt_seconds']['max'] == accepted['completed_receipt_seconds']['max']
    assert 'torch' not in sys.modules
    # Reauthenticate every inspected source after summarizing; all outputs are elsewhere.
    assert all(artifact(ROOT/item['path']) == item for item in used.values())
    result = {'schema': 'operational_deadline_record_review_v1', 'time': datetime.now(timezone.utc).isoformat(),
        'accepted_validation': artifact(ACCEPTED/'finalization_validation.json'),
        'accepted_analysis': artifact(ACCEPTED/'independent_analysis.json'), 'inspector': artifact(Path(__file__)),
        'all': aggregate,
        'by_fit': {str(fit): summary([row for row in rows if row['fit'] == fit]) for fit in sorted({row['fit'] for row in rows})},
        'by_segment': {segment: summary([row for row in rows if row['segment'] == segment])
                       for segment in ('original', 'recovery01', 'recovery02')},
        'early_plies_0_through_7': summary([row for row in rows if row['ply'] <= 7]),
        'timeouts': failures, 'local_operational_rows': rows,
        'timeout_ended_games': 4, 'operational_games': 18,
        'inspected_source_artifacts': sorted(used.values(), key=lambda row: row['path']),
        'inspected_source_hashes_rechecked': True, 'outcomes_closed': True,
        'strength_outcomes_inspected': False, 'new_games': 0, 'model_inference_executed': False,
        'interpretation': ['Timing means exclude censored timeout completion times.',
            'CPU measurements are sampled process-tree deltas, not exact productive-compute time.',
            'Observed timeout frequencies are descriptive correlated operational counts, not future probabilities.',
            'By-fit, stage and early-move groups are post hoc descriptions, not controlled causal comparisons.']}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'local_attempts': len(rows), 'timeouts': len(failures),
                      'source_artifacts_rechecked': len(used), 'outcomes_closed': True}))


if __name__ == '__main__':
    main()
