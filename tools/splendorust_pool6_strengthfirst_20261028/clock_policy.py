"""Declared generous hard guards; no additional search or deployment admission."""
import time
from tools.splendorust_sampling_stability_20261027.audit import check_clock as original_check_clock
SECONDS_PER_MOVE=10.
INTERNAL_SECONDS=9.
def require(value,message):
    if not value:raise ValueError('STRENGTH_EXECUTION: '+message)
def boundary(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: 9s internal hard guard exceeded')
def check_clock(receipt,*,limit=SECONDS_PER_MOVE):return original_check_clock(receipt,limit=limit)
