"""Read R2 evidence under its original identity; use a separate audit abort flag."""
from pathlib import Path
import shutil
import time

from tools.equal_budget_breadth_arena_128_recovery_02 import common as predecessor

ROOT = predecessor.ROOT
DATA = predecessor.OUTPUT
AUDIT_OUTPUT = predecessor.SOURCE / 'recovery_03'
DATA_BINDING = '1aaf9b20ac2fcb3392d5b05b1cbf3be8bc0f89cd89c1a306e35dcb43f880d435'


def __getattr__(name):
    # Provenance, sample resolution, and source bindings are unchanged R2 logic.
    return getattr(predecessor, name)


def budget_boundary(base=DATA):
    base = Path(base).resolve()
    if base == DATA.resolve():
        control = AUDIT_OUTPUT
    else:
        predecessor.require(not base.is_relative_to(predecessor.SOURCE.resolve()),
                            'Unexpected production verification base')
        control = base
    if (control / 'abort.json').exists():
        raise predecessor.RecoveryAborted('Audit recovery abort requested; preserve evidence')
    if base == DATA.resolve() and time.time() > predecessor.DEADLINE_EPOCH:
        raise predecessor.BudgetExceeded('Original five-day deadline reached, including downtime')
    if shutil.disk_usage(ROOT).free < predecessor.CONFIG['disk_reserve_bytes']:
        raise predecessor.BudgetExceeded('Five-GiB disk reserve reached')
