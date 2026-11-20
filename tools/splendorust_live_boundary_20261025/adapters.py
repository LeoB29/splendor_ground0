"""Live lifecycle using the immutable, qualified minimal boundary observer.

Player mode and observation mode are separate. Every worker uses the qualified
observer, while the original scientific role remains graph64 or eager32.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
import time

from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from tools.splendorust_boundary_timing_20261025 import analyze as observer_audit
from tools.splendorust_boundary_timing_20261025.runner import (
    verify_instrumentation_identity, validate_graph_parity, verify_graph_evidence, BACKEND)

OBSERVER_MODULE='tools.splendorust_boundary_timing_20261025.worker'
SIDECAR_WAIT_SECONDS=1.


def require_player(mode,quota):
    c.require(type(quota) is int and ((mode=='baseline' and quota==32) or
              (mode=='candidate' and quota==64)), 'Only unchanged eager32 and qualified graph64 are authorized')


def collect_profiles(child,receipt,envelope):
    """Preserve available evidence without extending or resetting move clocks.

    A caught failure or killed worker may lack a child sidecar. That disclosed
    absence is not a second failure. Any malformed or unbound evidence is an
    integrity error for the caller to preserve before stopping the study.
    """
    result=dict(parent_profile=None,parent_profile_file=None,child_profile=None,
                child_profile_file=None,profile_capture_status='pending')
    try:
        profile=deepcopy(child.last_parent_profile)
        c.require(type(profile) is dict and profile['request_id']==receipt['request_id'],
                  'Parent observer request identity')
        parent=child.output_dir/'parent_profiles'/f"{receipt['request_id']}.json"
        result.update(parent_profile=profile,parent_profile_file=parent.relative_to(c.ROOT).as_posix())
        c.save(parent,profile)
        path=child.output_dir/f"{receipt['request_id']}_child_profile.json"
        begun=time.monotonic();wait_end=begun+min(SIDECAR_WAIT_SECONDS,max(0.,envelope.remaining()))
        while not path.exists() and not getattr(child,'_closed',False) and time.monotonic()<wait_end:
            envelope.guard();time.sleep(min(.005,max(0.,wait_end-time.monotonic())))
        if path.exists():
            c.require(path.stat().st_size<=512*1024+1,'Bounded child observer sidecar')
            result.update(child_profile=c.read(path),child_profile_file=path.relative_to(c.ROOT).as_posix(),
                          profile_capture_status='complete')
        elif receipt['status']=='ok':
            raise ValueError('Completed request missing required child observer sidecar')
        else:
            result['profile_capture_status']='missing_child_after_failed_request'
        return result
    except BaseException as error:
        result.update(profile_capture_status='evidence_error',profile_capture_error=repr(error))
        return result


def _require_profiles(profiles):
    c.require('profile_capture_error' not in profiles,'Observer evidence integrity error: '+str(profiles.get('profile_capture_error')))


def graph_parity(child,envelope,freeze):
    item=freeze['graph_parity_plan'];c.verify([item]);plan=c.read(c.ROOT/item['path']);fixtures=plan['fixtures']
    c.require(len(fixtures)==11,'Fixed graph parity inventory')
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=child.request(payload,envelope.timeout(60))
    c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_profiles(child,receipt,envelope)
    _require_profiles(profiles);envelope.guard()
    c.require(receipt['status']=='ok','Graph output parity failed before decisions; no fallback/retry')
    validate_graph_parity(receipt['result'],fixtures)
    observer_audit.validate_startup_profiles(profiles['parent_profile'],profiles['child_profile'],
        receipt,payload,child.challenger_quota,worker_pid=child.process.pid)
    c.require(c.read(child.output_dir/f"{receipt['request_id']}.json")==receipt,'Graph parity raw receipt changed')
    verify_graph_evidence(child.output_dir,freeze,boundary=envelope.guard)
    return receipt


def make_worker(name,envelope,freeze,*,mode,quota):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    require_player(mode,quota)
    c.require('torch' not in sys.modules,'Model-free supervisor required')
    envelope.admit();folder=c.OUT/'workers'/name;envelope.register_worker(folder)
    child=ProfiledStudyWorker(OBSERVER_MODULE,folder,
        native_artifacts=freeze['native_artifacts'],aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']],initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(),max_message_bytes=4*c.MiB,max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB,max_rss_bytes=2*c.GiB,resource_guard=envelope.guard,
        env={'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT),'STUDY_WORKER_DIR':str(folder),
             'STUDY_SEARCH_SIMULATIONS':str(quota),'STUDY_BOUNDARY_PROFILE':'1','CUDA_VISIBLE_DEVICES':'',
             'RAYON_NUM_THREADS':'1','TOKIO_WORKER_THREADS':'1'})
    try:
        envelope.own_worker(child);child.study_after_close=lambda:envelope.close_worker(folder,child)
        telemetry=child._sample();metadata=deepcopy(child.initialization['result'])
        if mode=='candidate':c.require(metadata.pop('inference_backend')==BACKEND,'Candidate inference backend changed')
        validate_ready_metadata(metadata,expected_sources=SOURCE_HASHES,worker_pid=child.process.pid,
            process_tree=telemetry['process_tree'],expected_quota=quota)
        child.runtime_mode,child.challenger_quota,child.observation_mode=mode,quota,'profiled'
        c.require(not child._failure,'Worker supervision failed at readiness')
        verify_instrumentation_identity(folder,freeze,quota,child.initialization['result'])
        if mode=='candidate':graph_parity(child,envelope,freeze)
        envelope.guard();return child
    except BaseException:
        child.close();raise


def actual_entrypoint(child,envelope,label=None):
    """Return the original52-decision receipt; caller saves its summary path."""
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=child.request(payload,envelope.timeout(300))
    profiles=collect_profiles(child,receipt,envelope)
    _require_profiles(profiles);envelope.guard()
    c.require(receipt['status']=='ok' and receipt['result']['status']=='entrypoint_counterexamples_passed' and
        receipt['result']['challenger_quota']==child.challenger_quota and
        receipt['result']['counts']['complete_decisions']==52,'Actual corrected-entrypoint counterexamples failed; no retry')
    observer_audit.validate_startup_profiles(profiles['parent_profile'],profiles['child_profile'],
        receipt,payload,child.challenger_quota,worker_pid=child.process.pid)
    return receipt


def decision(child,state,envelope,role):
    c.require(role in ('challenger','opponent') and child.challenger_quota==(64 if role=='challenger' else 32) and
        child.runtime_mode==('candidate' if role=='challenger' else 'baseline') and child.observation_mode=='profiled',
        'Role/quota/observation binding differs')
    require_player(child.runtime_mode,child.challenger_quota);envelope.admit()
    observer,before=rules.observation_payload(state),rules.serialize_snapshot(state)
    ledger.check_observer(before,observer);legal=rules.legal_actions(state)
    actions=[serialize_action(action) for action in legal];request_id=f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json',dict(player='search',observer=observer,before=before))
    def validate(result):
        if not isinstance(result,dict) or set(result)!=scientific.RESULT_FIELDS or not isinstance(result['diagnostics'],dict):
            raise ValueError('PLAYER_MALFORMED_RESPONSE: response envelope fields')
        if result['action'] not in actions:
            raise ValueError('PLAYER_ILLEGAL_ACTION: response action')
        return scientific.validate_result(result,observer,'search',expected_quota=child.challenger_quota)
    c.require(envelope.remaining()>=1.,'Insufficient stage time for a full one-second decision')
    allowance=envelope.timeout(1.);c.require(allowance==1.,'Stage guard shortened the next move allowance; do not dispatch')
    receipt=child.request({'player':'search','observer':observer},allowance,validate=validate)
    row=dict(arm='search',player_role=role,quota=child.challenger_quota,mode=child.runtime_mode,
        observation_mode='profiled',seat=state.current_player,observer=observer,before=before,after=None,
        action=None,trace=None,completed=receipt['status']=='ok',clock=receipt,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    row.update(collect_profiles(child,receipt,envelope))
    if row['completed']:
        result=receipt['result']
        c.require(result['player']=='search' and result['observer_sha256']==c.digest(observer) and
            result['legal_actions_sha256']==legal_digest(legal) and result['legal_count']==len(legal),
            'Worker observation/legal-support binding differs')
        row.update(action=result['action'],trace=result['diagnostics'])
        row['after']=rules.serialize_snapshot(rules.CorrectedEnv().step(state,deserialize_action(row['action'])))
    # The caller must save this row before evaluating any observer-integrity
    # error or classifying the original failure. Never revise the clock/result.
    return row


def verify_raw_transport(folder,freeze,envelope):
    from .raw_transport import verify_raw_transport as verify
    return verify(folder,freeze,envelope)
