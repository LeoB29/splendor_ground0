"""Pilot I/O with read-only scientific constants from the qualified live route."""
from .. import common as parent
from tools.splendorust_sixworld512_20261114_v2 import common as inherited

PACKAGE='tools.splendorust_parallel_throughput_20261115.live'
MiB=1024**2
GiB=1024**3
MODEL_RSS=2*GiB
MAX_PLIES=2000
GAMES_PER_BLOCK=16
IDENTITY=inherited.IDENTITY
PRIOR_OUT=inherited.PRIOR_OUT
PRIOR_FREEZE=inherited.PRIOR_FREEZE
PRIOR_FREEZE_SHA=inherited.PRIOR_FREEZE_SHA
PRIOR_CLOSEOUT=inherited.PRIOR_CLOSEOUT
PRIOR_CLOSEOUT_SHA=inherited.PRIOR_CLOSEOUT_SHA

def __getattr__(name):
    return getattr(parent,name)
