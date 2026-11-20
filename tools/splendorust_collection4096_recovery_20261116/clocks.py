"""Reboot-safe watchdog clocks; original absolute study bounds never restart.

The virtual monotonic origin keeps unchanged elapsed-budget arithmetic valid
after a reboot. No scientific decision clock is resumed or rebased here.
The boot-time estimate is diagnostic: wall-clock corrections may change it.
UTC and monotonic deadlines are fixed at launch and are never recalculated.
"""
import math
import time
import psutil
from . import common as c

# Exact original receipt, kept explicit rather than trusting a supplied plan.
ORIGINAL_FREEZE_SHA = 'ac15c01e9e4e2f36b3de6f525bf6a268c271535496df0e39c553945882a5eb88'
SUPPLEMENTAL_SECONDS = 3600


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def original_deadlines():
    original = c.ORIGINAL_OUT
    freeze = c.artifact(original/'freeze.json')
    c.require(freeze['sha256'] == ORIGINAL_FREEZE_SHA, 'Exact original collection freeze')
    plan = c.read(original/'freeze.json')
    root = c.read(original/'claim.json')
    launcher = c.read(original/'launcher_claim.json')
    phase = c.read(original/'stages/collection_start.json')
    c.require(root['freeze'] == launcher['freeze'] == freeze and
              root['monotonic_origin'] == launcher['monotonic_origin'] and
              root['unix_origin'] == launcher['unix_origin'], 'Original launcher/root clock agreement')
    c.require(plan['total_seconds'] == root['total_seconds'] == 168*3600 and
              plan['model_seconds'] == root['model_seconds'] == 144*3600 and
              plan['stage_seconds']['collection'] == phase['cap_seconds'] == 120*3600,
              'Unchanged original collection allowances')
    origin = root['unix_origin']
    c.require(_finite(origin) and phase['stage'] == 'collection' and
              phase['claim'] == c.artifact(original/'claim.json') and
              phase['unix_deadline'] == min(phase['unix_time']+120*3600, origin+144*3600) and
              phase['monotonic_deadline'] == min(phase['monotonic']+120*3600, root['monotonic_origin']+144*3600),
              'Original collection stage clipped to original model cutoff')
    return dict(schema='collection4096_original_deadlines_v1', original_unix_origin=origin,
        global_unix_deadline=origin+168*3600, model_unix_deadline=origin+144*3600,
        collection_unix_origin=phase['unix_time'], collection_unix_deadline=phase['unix_deadline'],
        original_bindings=[freeze,c.artifact(original/'claim.json'),
                           c.artifact(original/'launcher_claim.json'),c.artifact(original/'stages/collection_start.json')])


def _now(mono=None, wall=None, boot=None):
    c.require((mono is None) == (wall is None), 'Both sampled clocks supplied together')
    if mono is None: mono,wall = time.monotonic(),time.time()
    if boot is None: boot=psutil.boot_time()
    c.require(all(_finite(x) for x in (mono,wall,boot)) and mono >= 0 and boot > 0,
              'Finite current-boot clock samples')
    return float(mono),float(wall),float(boot)


def create_limits(*, mono=None, wall=None, boot=None, original=None):
    bound=original_deadlines()
    c.require(original is None or original == bound, 'Exact bound original deadlines')
    original=bound
    mono,wall,boot=_now(mono,wall,boot)
    c.require(original['original_unix_origin'] <= wall < original['global_unix_deadline'],
              'Recovery inside the unexpired original overall clock')
    anchor=dict(monotonic=mono,unix=wall,boot_unix=boot,
                boot_unix_diagnostic_only=True,deadline_reanchoring_allowed=False)
    virtual_origin=mono-(wall-original['original_unix_origin'])
    value=dict(monotonic_origin=virtual_origin,unix_origin=original['original_unix_origin'],
        monotonic_deadline=virtual_origin+c.TOTAL_SECONDS,unix_deadline=original['global_unix_deadline'],
        recovery_anchor=anchor,recovery_anchor_sha256=c.digest(anchor),
        original_deadlines=original,original_deadlines_sha256=c.digest(original),
        downtime_charged=True,per_decision_clocks_rebased=False)
    check(value,mono=mono,wall=wall,boot=boot)
    return value


def validate(claim):
    anchor=claim['recovery_anchor'];old=claim['original_deadlines']
    c.require(claim['recovery_anchor_sha256']==c.digest(anchor) and
              claim['original_deadlines_sha256']==c.digest(old), 'Immutable recovery clock bindings')
    c.require(old['schema']=='collection4096_original_deadlines_v1' and
              all(_finite(old[k]) for k in ('original_unix_origin','global_unix_deadline',
                  'model_unix_deadline','collection_unix_origin','collection_unix_deadline')),
              'Finite original absolute bounds')
    start=old['original_unix_origin']
    c.require(old['global_unix_deadline']==start+c.TOTAL_SECONDS and
              old['model_unix_deadline']==start+c.MODEL_SECONDS and
              start <= old['collection_unix_origin'] < old['collection_unix_deadline'] and
              old['collection_unix_deadline']==min(old['collection_unix_origin']+120*3600,old['model_unix_deadline']),
              'No original global/model/collection extension')
    c.require(all(_finite(anchor[k]) for k in ('monotonic','unix','boot_unix')) and
              anchor['monotonic']>=0 and anchor['boot_unix']>0 and start<=anchor['unix'] and
              anchor['boot_unix_diagnostic_only'] is True and anchor['deadline_reanchoring_allowed'] is False,
              'Immutable recovery anchor; boot estimate is diagnostic only')
    c.require(claim['unix_origin']==start and
              abs(claim['monotonic_origin']-(anchor['monotonic']-(anchor['unix']-start)))<=1e-6 and
              claim['unix_deadline']==old['global_unix_deadline'] and
              abs(claim['monotonic_deadline']-claim['monotonic_origin']-c.TOTAL_SECONDS)<=1e-6 and
              claim['downtime_charged'] is True and claim['per_decision_clocks_rebased'] is False,
              'Virtual origin charges every elapsed second; no decision rebase')


def check(claim, *, mono=None, wall=None, boot=None, model=False):
    validate(claim);mono,wall,boot=_now(mono,wall,boot);anchor=claim['recovery_anchor']
    c.require(mono>=anchor['monotonic'] and wall>=claim['unix_origin'] and
              mono<claim['monotonic_deadline'] and wall<claim['unix_deadline'],
              'Original global deadline or recovery origin violated')
    if model:
        c.require(mono<claim['monotonic_origin']+c.MODEL_SECONDS and
                  wall<claim['original_deadlines']['model_unix_deadline'], 'Original model cutoff expired')
    return True


def mapped_mono(claim, wall):
    return claim['monotonic_origin']+(wall-claim['unix_origin'])


def phase_limits(claim,name,*,mono=None,wall=None,boot=None):
    mono,wall,boot=_now(mono,wall,boot);check(claim,mono=mono,wall=wall,boot=boot,model=name!='final')
    c.require(name in ('supplemental','collection','corpus','final'),'Recovery phase name')
    cap=SUPPLEMENTAL_SECONDS if name=='supplemental' else c.STAGES[name]
    old=claim['original_deadlines'];start=wall;startmono=mono
    cutoff=old['global_unix_deadline'] if name=='final' else old['model_unix_deadline']
    if name in ('collection','supplemental'):cutoff=min(cutoff,old['collection_unix_deadline'])
    if name=='collection':
        start=old['collection_unix_origin'];startmono=mapped_mono(claim,start)
    deadline=min(start+cap,cutoff)
    value=dict(stage=name,monotonic=startmono,unix_time=start,cap_seconds=cap,
        monotonic_deadline=min(startmono+cap,mapped_mono(claim,deadline)),unix_deadline=deadline,
        original_collection_clock=c.artifact(c.ORIGINAL_OUT/'stages/collection_start.json'))
    validate_phase(claim,value)
    c.require(mono<value['monotonic_deadline'] and wall<value['unix_deadline'],'Original phase already expired')
    return value


def validate_phase(claim,phase):
    validate(claim);name=phase['stage'];old=claim['original_deadlines']
    c.require(name in ('supplemental','collection','corpus','final'),'Exact recovery phase')
    cap=SUPPLEMENTAL_SECONDS if name=='supplemental' else c.STAGES[name]
    cutoff=old['global_unix_deadline'] if name=='final' else old['model_unix_deadline']
    if name in ('collection','supplemental'):cutoff=min(cutoff,old['collection_unix_deadline'])
    c.require(type(phase['cap_seconds']) is int and phase['cap_seconds']==cap,
              'Unchanged phase cap except explicit one-hour verification supplement')
    if name=='collection':
        c.require(phase['unix_time']==old['collection_unix_origin'] and
                  abs(phase['monotonic']-mapped_mono(claim,phase['unix_time']))<=1e-6,
                  'Collection phase starts at original saved origin, never at recovery')
    else:
        c.require(phase['unix_time']>=claim['unix_origin'] and
                  phase['monotonic']>=claim['recovery_anchor']['monotonic'], 'Fresh bounded recovery phase')
    deadline=min(phase['unix_time']+cap,cutoff)
    c.require(phase['unix_deadline']==deadline and
              abs(phase['monotonic_deadline']-min(phase['monotonic']+cap,mapped_mono(claim,deadline)))<=1e-6,
              'Phase clocks clipped without UTC or monotonic extension')


validate_claim = validate
