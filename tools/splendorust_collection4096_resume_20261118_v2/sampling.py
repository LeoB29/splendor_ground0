"""Qualified collator for discarded qualification; no scientific fit schedule."""
from copy import deepcopy
from tools.splendorust_policy_head_20261104 import data as accepted
from tools.splendorust_policy_head_20261104 import training as accepted_training
from . import common as c,schedule
DOMAIN=STUDY='splendorust_collection4096_20261115_v1'
IDENTITY=deepcopy(accepted.IDENTITY)
SEEDS=tuple(schedule.FIT_SEEDS[f'r{r}'] for r in (0,1))

def fits():return []
def fit_identity(*args,**kwargs):raise ValueError('Scientific fitting is outside this collection-only freeze')
def _sha(value):
    c.require(type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef'),'Complete hash')
    return value
def collate_inputs(examples,weights):return accepted_training.collate_inputs(examples,weights)
def collate(examples,weights):return accepted_training.collate(examples,weights)