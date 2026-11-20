"""Read-only identity and lifecycle checks for the interrupted search256 attempt.

This module deliberately imports no player, evaluator or inherited study runtime.
Its observations establish absence of recorded identities, never an exit code or
the cause of the interruption. Frozen receipt hashes belong to the outer manifest.
"""
import json
import math
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
OLD_OUT = ROOT / 'outputs/splendorust_search256_strength_20261029_02'
FIRST_OUT = ROOT / 'outputs/splendorust_search256_strength_20261029_01'
OLD_FREEZE_SHA = 'f85737b248e274e6be780461001b4294a1e937cccb70e78ac0e3afcd43878e53'
INTERRUPTED_WORKERS = ('main_research_g235_challenger', 'main_research_g235_opponent')
TREE_ROLES = {'python', 'reference', 'search', 'console:python',
              'console:reference', 'console:search'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Regular retained identity receipt')
    _require(path.stat().st_size <= 4 * 1024**2, 'Bounded retained identity receipt')
    return json.loads(path.read_text(encoding='utf-8'))


def _identity(pid, created):
    _require(type(pid) is int and pid > 0 and type(created) in (int, float)
             and math.isfinite(created) and created > 0, 'Exact PID and creation identity')
    return {'pid': pid, 'created': created}


def _process_tree(receipt, *, startup=False):
    tree = receipt['telemetry']['process_tree']
    _require(type(tree) is list and len(tree) == 6 and
             {row['role'] for row in tree} == TREE_ROLES, 'Exact six-member retained worker tree')
    result = {row['role']: dict(_identity(row['pid'], row['created']),
                               parent_pid=row['parent_pid']) for row in tree}
    _require(len({row['pid'] for row in result.values()}) == 6, 'Unique worker-tree PIDs')
    player = result['python']
    if startup:
        _require(receipt.get('identity') == _identity(player['pid'], player['created']),
                 'Startup receipt belongs to recorded worker identity')
    for role in TREE_ROLES - {'python'}:
        parent = result[role.split(':', 1)[1]] if role.startswith('console:') else player
        _require(result[role]['parent_pid'] == parent['pid'], 'Retained helper ancestry')
    return result


def recorded_old_identities():
    """Read and crosscheck the three owners and twelve retained worker identities."""
    guardian = _read(OLD_OUT / 'guardian_claim.json')
    watchdog = _read(OLD_OUT / 'watchdog_claim.json')
    runner = _read(OLD_OUT / 'launch.json')
    owner = _read(OLD_OUT / 'guardian_owner.json')
    for value in (guardian, watchdog, runner):
        _require(value['freeze_sha256'] == OLD_FREEZE_SHA, 'Exact interrupted freeze identity')
    guardian_id = _identity(guardian['guardian_pid'], guardian['guardian_created'])
    watchdog_id = _identity(watchdog['watchdog_pid'], watchdog['watchdog_created'])
    runner_id = _identity(runner['pid'], runner['created'])
    _require(owner['guardian'] == guardian and owner['watchdog'] == watchdog_id,
             'Guardian ownership receipts agree')
    _require(_identity(watchdog['guardian_pid'], watchdog['guardian_created']) == guardian_id,
             'Watchdog guarded exact retained owner')
    result = [guardian_id, watchdog_id, runner_id]
    for worker_name, route, quota in ((INTERRUPTED_WORKERS[0], 'research', 256),
                                      (INTERRUPTED_WORKERS[1], 'frozen', 128)):
        folder = OLD_OUT / 'workers' / worker_name
        initial = _read(folder / 'initialization.json')
        latest = _read(folder / 'request-00000016.json')
        placement = _read(folder / 'placement_admission.json')
        _require(initial['accepted'] is True and initial['status'] == 'ok' and
                 latest['accepted'] is True and latest['status'] == 'ok' and
                 latest['request_id'] == 'request-00000016', 'Accepted retained identity observations')
        init_tree, tree = _process_tree(initial, startup=True), _process_tree(latest)
        _require(init_tree == tree, 'Startup and latest worker/helper identities agree')
        startup = initial['result']
        _require(startup['freeze_sha256'] == OLD_FREEZE_SHA and startup['route'] == route and
                 startup['request_quotas'] == [quota] and startup['pid'] == tree['python']['pid'],
                 'Exact interrupted player startup identity')
        _require(startup['native_pids'] == {name: tree[name]['pid'] for name in ('reference', 'search')},
                 'Native helpers belong to recorded worker')
        _require(tree['python']['parent_pid'] == runner_id['pid'] and
                 _identity(placement['player_pid'], placement['player_created']) == initial['identity'] and
                 _identity(placement['supervisor']['pid'], placement['supervisor']['created']) == runner_id,
                 'Interrupted workers belong to retained runner')
        result.extend(_identity(tree[role]['pid'], tree[role]['created']) for role in sorted(tree))
    _require(len(result) == 15 and len({row['pid'] for row in result}) == 15,
             'Exactly fifteen distinct retained owner/worker/helper identities')
    return result


def _absent(identity):
    """PID reuse establishes old-identity absence; inaccessible processes do not."""
    try:
        current = psutil.Process(identity['pid'])
        created = current.create_time()
    except psutil.NoSuchProcess:
        return dict(identity, alive=False)
    _require(type(created) in (int, float) and math.isfinite(created),
             'Readable current process creation identity')
    _require(abs(created - identity['created']) > .0001, 'Recorded old process is still alive')
    return dict(identity, alive=False)


def old_owner_identities():
    """Return exact retained identities only after checking all are now absent."""
    return [_absent(identity) for identity in recorded_old_identities()]


def verify_old_absence():
    """Re-read the retained evidence and repeat absence checks immediately before launch."""
    return old_owner_identities()


def _tree_count(folder):
    workers = folder / 'workers'
    entries = sorted(workers.iterdir())
    _require(all(p.is_dir() and not p.is_symlink() for p in entries),
             'Only regular worker lifecycle directories')
    names = [p.name for p in entries]
    counts = {stage: sum(name.startswith(stage + '_') for name in names)
              for stage in ('qualification', 'operational', 'main')}
    _require(sum(counts.values()) == len(entries), 'No unclassified prior worker lifecycle')
    counts.update(worker_trees=len(entries),
        graph_builds=sum((p / 'graph_identity.json').is_file() for p in entries),
        graph_parities=sum((p / 'graph_parity.json').is_file() for p in entries),
        cleanup_receipts=sum((p / 'cleanup.json').is_file() for p in entries),
        unclosed_workers=[p.name for p in entries if not (p / 'cleanup.json').is_file()])
    return counts


def old_tree_counts():
    """Count prior lifecycles without opening games or interpreting any outcome."""
    first, second = _tree_count(FIRST_OUT), _tree_count(OLD_OUT)
    _require(first == dict(qualification=2, operational=0, main=0, worker_trees=2,
        graph_builds=2, graph_parities=1, cleanup_receipts=2, unclosed_workers=[]),
        'Exact first failed-attempt lifecycle counts')
    _require(second == dict(qualification=2, operational=12, main=472, worker_trees=486,
        graph_builds=486, graph_parities=486, cleanup_receipts=484,
        unclosed_workers=list(INTERRUPTED_WORKERS)), 'Exact interrupted-attempt lifecycle counts')
    return dict(attempt01=first, attempt02=second, used_worker_trees=488,
        used_graph_builds=488, completed_graph_parities=487,
        recovery_replacement_trees=2, remaining_complete_game_trees=328,
        reserved_replay_trees=48, maximum_new_worker_trees=378,
        maximum_combined_worker_trees=866)
