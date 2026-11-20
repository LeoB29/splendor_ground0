"""Recovery resources; original scientific helpers retain their immutable globals."""
from pathlib import Path
import shutil
import time

from tools.matched_strength_20261020.common import *
from tools.matched_strength_20261020 import common as original

SOURCE = original.OUTPUT
SOURCE_FREEZE_SHA256 = '41244628326e23ea6a8685752f367bd4ca5e13809cfb47d983d0cd6f139b248e'
SOURCE_REVIEW = ROOT/'outputs/matched_strength_interruption_review_20261022_01'
OUTPUT = ROOT/'outputs/matched_strength_recovery_20261022_01'
PREPARATION = ROOT/'outputs/matched_strength_recovery_20261022_preparation'
MAX_SECONDS = 12*3600
POST_GAME_RESERVE = 6*3600
ORIGINAL_POST_GAME_RESERVE = 18*3600
PARTIAL_GAME, NEXT_PLY = 59, 43
COMPLETED_GAMES = original.game_order(30)[:-1]
RETRY_IDENTITY = 'games/g059/t0043'
AMENDMENT = {'retry_identity': RETRY_IDENTITY, 'new_allowance_seconds': 120,
             'maximum_retries': 1, 'partial_search_reused': False,
             'original_frozen_status': 'incomplete'}


@lru_cache(maxsize=1)
def preserved_source_bytes():
    """Immutable source size, authenticated before workers and rechecked at finish.

    Source files are never writable recovery outputs. External mutations cause
    the full source inventory/hash verification to fail, never acceptance.
    """
    inventory = read(SOURCE_REVIEW/'source_inventory.json')
    require(inventory['schema'] == 'preserved_interrupted_benchmark_inventory_v1' and
            inventory['source'] == SOURCE.relative_to(ROOT).as_posix() and
            type(inventory['total_bytes']) is int and inventory['total_bytes'] >= 0 and
            inventory['total_bytes'] == sum(x['bytes'] for x in inventory['files']), 'Immutable source size receipt')
    return inventory['total_bytes']


def boundary(base, deadline, *, memory_headroom=True):
    launch = read(Path(base)/'launch.json'); prior = read(SOURCE/'launch.json')
    require(launch['maximum_seconds'] == MAX_SECONDS and prior['maximum_seconds'] == 72*3600,
            'Original/recovery execution caps changed')
    mono, epoch = time.monotonic(), time.time()
    require(mono >= prior['monotonic_start'] and mono >= launch['monotonic_start'],
            'Monotonic continuity lost; no clock substitution')
    recovery_remaining = min(deadline-mono, launch['monotonic_start']+MAX_SECONDS-mono,
                             launch['epoch_seconds']+MAX_SECONDS-epoch)
    original_remaining = min(prior['monotonic_start']+prior['maximum_seconds']-mono,
                             prior['epoch_seconds']+prior['maximum_seconds']-epoch)
    result = {'time': now(), 'remaining_seconds': min(recovery_remaining, original_remaining),
              'recovery_remaining_seconds': recovery_remaining, 'original_remaining_seconds': original_remaining,
              'free_disk_bytes': shutil.disk_usage(ROOT).free,
              'artifact_bytes': preserved_source_bytes()+directory_bytes(base), 'memory': memory()}
    if result['remaining_seconds'] <= 0: raise ResourceHold('Original/recovery dual-clock ceiling reached')
    if result['free_disk_bytes'] < MIN_FREE: raise ResourceHold('Five-GiB disk reserve reached')
    if result['artifact_bytes'] >= config()['evidence_cap_bytes']-EVIDENCE_RESERVE:
        raise ResourceHold('Combined original/recovery evidence ceiling reached')
    if memory_headroom and (result['memory']['available_physical'] < 1024**3 or
                           result['memory']['available_page'] < 3*1024**3):
        raise ResourceHold('Insufficient live memory headroom')
    return result


def work_budget(receipt, reserve):
    """Protect eighteen original hours and six recovery hours before game moves."""
    original_reserve = ORIGINAL_POST_GAME_RESERVE if reserve == POST_GAME_RESERVE else reserve
    return min(receipt['recovery_remaining_seconds']-reserve,
               receipt['original_remaining_seconds']-original_reserve)


def verify_source_inventory(inventory, *, boundary=lambda: None):
    require(inventory['schema'] == 'preserved_interrupted_benchmark_inventory_v1' and
            inventory['source'] == SOURCE.relative_to(ROOT).as_posix(), 'Interrupted source inventory identity')
    actual = {p.relative_to(ROOT).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    require(actual == {x['path'] for x in inventory['files']}, 'Original evidence inventory changed')
    for item in inventory['files']:
        boundary(); require(artifact(ROOT/item['path']) == item, 'Original evidence changed: '+item['path'])
