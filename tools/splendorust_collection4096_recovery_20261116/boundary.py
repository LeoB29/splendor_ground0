"""Immutable shutdown boundary; never rewrites or upgrades the original attempt."""
from pathlib import Path
from . import common as c
from .collection import common as cc

ORIGINAL = c.ROOT / 'outputs/splendorust_collection4096_20261115_01'
CONTAINER = ORIGINAL / 'collection/cells/c041'
WORKER = CONTAINER / 'worker'
CELL = WORKER / 'cell'
ORIGINAL_PLAN = ORIGINAL / 'collection/plan.json'
INTERRUPTED = ('g0666', 40)
EXPECTED_OPERATION = 935


def selected_replays(plan):
    spec = next(x for x in plan['cells'] if x['cell_id'] == 'c041')
    rows = []
    for game in spec['specifications']:
        ending = cc.read(CELL/'games'/game['game_id']/'result.json')
        for chosen in ending['model_replay_selection']:
            rows.append(dict(game_id=game['game_id'], ply=chosen['ply'],
                original_slot=ending['slot'], replay_slot=(ending['slot']+1)%8,
                source=c.artifact(CELL/'games'/game['game_id']/'moves'/f'{chosen["ply"]:06d}.json')))
    c.require(len(rows) == 48 and len({(x['game_id'],x['ply']) for x in rows}) == 48,
              'Exactly the original 48 distinct selected replays')
    return rows


def validate_boundary(report):
    c.require(report['schema'] == 'collection4096_shutdown_boundary_v1', 'Explicit shutdown boundary schema')
    c.require(report['original_attempt_incomplete'] is True and report['completed_games'] == 672 and
        report['fully_verified_games'] == 656 and report['completed_replays'] == 32 and
        report['interrupted_verification_decisions'] == 1 and report['unstarted_replays'] == 15,
        'Exact fixed failure inventory')
    c.verify(report['artifacts'])
    expected = sorted(p.relative_to(c.ROOT).as_posix() for p in CONTAINER.rglob('*') if p.is_file())
    recorded = sorted(x['path'] for x in report['artifacts'] if c.resolve(x).is_relative_to(CONTAINER))
    c.require(expected == recorded, 'Preserved original c041 file inventory unchanged')
    c.require(report['selected_replays'] == selected_replays(c.read(ORIGINAL_PLAN)), 'Original complete replay selection unchanged')
    c.require((report['selected_replays'][32]['game_id'],report['selected_replays'][32]['ply']) == INTERRUPTED,
              'Only the disclosed interrupted verification decision')
    c.require(report['active_operation']['operation_id'] == EXPECTED_OPERATION and
        report['active_operation']['client_id'] == 'replay_g0666', 'Exact original active operation')
    c.require(c.read(WORKER/'runtime/active_decisions.json')['active'] == [report['active_operation']] and
        report['original_cell_deadline_unix'] == c.read(CONTAINER/'owner.json')['unix_deadline'] and
        report['original_execution'] == c.artifact(CONTAINER/'execution.json') and
        report['original_replay_attempt'] == c.artifact(CELL/'model_replays/g0666/000040/attempts/000040.json'),
        'Boundary metadata reconstructed from original failure receipts')
    paths = {c.resolve(x) for x in report['artifacts']}
    c.require({ORIGINAL/'freeze.json', ORIGINAL/'claim.json', ORIGINAL_PLAN,
        ORIGINAL/'stage.json', ORIGINAL/'execution.json', ORIGINAL/'failure.json',
        ORIGINAL/'qualification_gate.json', ORIGINAL/'collection/qualification_gate.json'} <= paths,
        'Required original source and qualification receipts remain preserved')
    return True


def inspect_boundary():
    plan = c.read(ORIGINAL_PLAN)
    execution = c.read(CONTAINER/'execution.json'); owner = c.read(CONTAINER/'owner.json')
    c.require(c.read(ORIGINAL/'stage.json')['status'] == 'incomplete' and
        c.read(ORIGINAL/'execution.json')['status'] == 'incomplete', 'Original attempt remains incomplete')
    c.require(execution['status'] == 'failed' and execution['worker_exit_code'] == 1073807364 and
        execution['retries'] == 0 and not execution['cleanup_errors'] and execution['identity'] == owner['identity'],
        'Exact interrupted original worker, never a successful execution')
    c.require(not (ORIGINAL/'collection/cells/c042').exists(), 'No later original cell started')
    c.require({p.name for p in (ORIGINAL/'collection/cells').iterdir() if p.is_dir()} ==
              {f'c{i:03d}' for i in range(42)}, 'Exact original completed-plus-interrupted prefix')
    cohort = cc.read(CELL/'cohort.json'); queue = cc.read(CELL/'queue_outcomes.json')
    c.require(cohort['status'] == 'completed' and cohort['games'] == 16 and cohort['plies'] == 878 and
        cohort['eligible_roots'] == 878 and cohort['retries'] == cohort['replacements'] == 0 and
        len(queue['outcomes']) == 16 and not queue['errors'] and not queue['never_started'],
        'All 16 original games naturally completed before interruption')
    c.require(cc.read(CELL/'loader_result.json')['status'] == 'actual_collector_loader_passed', 'Original loader completed')
    selection = selected_replays(plan)
    for index, item in enumerate(selection):
        dest = CELL/'model_replays'/item['game_id']/f'{item["ply"]:06d}'
        if index < 32:
            c.require(cc.read(dest/'comparison.json')['exact_full_result_capture_and_after'] is True,
                      'Exact completed replay prefix')
        elif index == 32:
            c.require((item['game_id'],item['ply']) == INTERRUPTED, 'Exact interrupted replay boundary')
            c.require(sorted(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()) ==
                ['attempts/000040.json','requests/000040.json'], 'Interrupted verification has no completed response or clock')
        else:
            c.require(not dest.exists(), 'Remaining 15 replay decisions were never started')
    for name in ('worker_result.json','result.json','cleanup.json'):
        c.require(not (WORKER/name).exists(), 'Never fabricate missing original worker closure '+name)
    c.require(not (CELL/'model_replay_summary.json').exists(), 'Original replay summary absent')
    active = c.read(WORKER/'runtime/active_decisions.json')
    c.require(len(active['active']) == 1 and active['identity'] == owner['identity'], 'One exact interrupted owned operation')
    operation = active['active'][0]
    c.require(operation['operation_id'] == EXPECTED_OPERATION and operation['client_id'] == 'replay_g0666' and
        operation['kind'] == 'ordinary' and operation['deadline_unix'] == operation['started_unix']+75. and
        operation['started_unix'] < execution['unix_end'] < operation['deadline_unix'],
        'Shutdown interrupted verification before its deadline')
    attempt = cc.read(CELL/'model_replays/g0666/000040/attempts/000040.json')
    c.require(attempt['no_retry'] is True and attempt['slot'] == selection[32]['replay_slot'] and
        attempt['started_perf_counter'] == operation['started_perf'], 'Interrupted request retains original no-retry origin')
    paths = [ORIGINAL/'freeze.json',ORIGINAL/'claim.json',ORIGINAL/'stage.json',ORIGINAL/'execution.json',
        ORIGINAL/'failure.json',ORIGINAL_PLAN,ORIGINAL/'qualification_gate.json',
        ORIGINAL/'collection/qualification_gate.json']
    paths += sorted(p for p in CONTAINER.rglob('*') if p.is_file())
    paths += [c.resolve(owner['phase_clock'])]
    report = dict(schema='collection4096_shutdown_boundary_v1', original_attempt_incomplete=True,
        completed_games=672, fully_verified_games=656, completed_replays=32,
        interrupted_verification_decisions=1, unstarted_replays=15, selected_replays=selection,
        active_operation=operation, original_execution=c.artifact(CONTAINER/'execution.json'),
        original_cell_deadline_unix=owner['unix_deadline'], original_replay_attempt=c.artifact(
            CELL/'model_replays/g0666/000040/attempts/000040.json'),
        artifacts=[c.artifact(p) for p in sorted(set(paths))],
        original_games_retried=0, original_clocks_reset=False, models_launched=0)
    validate_boundary(report)
    return report
