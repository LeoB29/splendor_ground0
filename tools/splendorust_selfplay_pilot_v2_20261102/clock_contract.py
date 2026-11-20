"""Explicit request QPC, inherited owner, and wall-clock domains.

This module has no model imports, clock conversion, slack, or waits. A parent
observation is sampled after reading the complete response, never inferred from
a child's timestamp. Owner deadlines are compared only with time.monotonic.
"""
from __future__ import annotations
import math
import time

INTERNAL_SECONDS = 9.0
EXTERNAL_SECONDS = 10.0


class ClockContractError(ValueError):
    def __init__(self, branch):
        self.branch = branch
        super().__init__('SELFPLAY_CLOCK_V2: ' + branch)


def require(value, branch):
    if not value:
        raise ClockContractError(branch)


def number(value, branch):
    require(type(value) in (int, float) and math.isfinite(value), branch)
    return value


def runtime_contract():
    def info(name):
        value = time.get_clock_info(name)
        return dict(api='time.' + name, implementation=value.implementation,
                    monotonic=value.monotonic, adjustable=value.adjustable,
                    resolution_seconds=value.resolution)
    return dict(schema='selfplay_explicit_clock_contract_v2',
                request_clock=info('perf_counter'), owner_clock=info('monotonic'),
                wall_clock=info('time'), internal_seconds=INTERNAL_SECONDS,
                complete_response_seconds=EXTERNAL_SECONDS,
                parent_observation_after_complete_read=True,
                no_clock_domain_conversion=True, added_tolerance_seconds=0.0)


def sample():
    # Each named domain remains independent even when this Python version happens
    # to implement two of them with the same operating-system clock.
    return dict(request_qpc=time.perf_counter(), owner_monotonic=time.monotonic(), unix=time.time())


def validate_sample(value):
    require(type(value) is dict and set(value) == {'request_qpc', 'owner_monotonic', 'unix'}, 'clock_sample_fields')
    for key in value:
        number(value[key], 'finite_' + key)
    return value


def start_fields(reading=None):
    value = validate_sample(sample() if reading is None else reading)
    return dict(clock_contract=runtime_contract(), request_qpc_start=value['request_qpc'],
                request_qpc_deadline=value['request_qpc'] + EXTERNAL_SECONDS,
                internal_qpc_deadline=value['request_qpc'] + INTERNAL_SECONDS,
                owner_monotonic_start=value['owner_monotonic'], unix_start=value['unix'],
                unix_deadline=value['unix'] + EXTERNAL_SECONDS)


def validate_start(start, expected_contract=None):
    require(start.get('schema') == 'selfplay_learned_request_start_v2', 'request_start_schema')
    require(start.get('clock_contract') == (runtime_contract() if expected_contract is None else expected_contract),
            'request_clock_contract')
    for key in ('request_qpc_start', 'request_qpc_deadline', 'internal_qpc_deadline',
                'owner_monotonic_start', 'unix_start', 'unix_deadline'):
        number(start.get(key), 'finite_' + key)
    require(start['request_qpc_deadline'] == start['request_qpc_start'] + EXTERNAL_SECONDS and
            start['internal_qpc_deadline'] == start['request_qpc_start'] + INTERNAL_SECONDS and
            start['unix_deadline'] == start['unix_start'] + EXTERNAL_SECONDS,
            'unchanged_request_9_10_seconds')
    return start


def validate_admission(start, *, owner_start, owner_unix_start, owner_deadline, unix_deadline, expected_contract=None):
    validate_start(start, expected_contract)
    for key, value in [('owner_start', owner_start), ('owner_unix_start', owner_unix_start),
                       ('owner_deadline', owner_deadline), ('owner_unix_deadline', unix_deadline)]:
        number(value, 'finite_' + key)
    require(owner_start <= start['owner_monotonic_start'] and
            start['owner_monotonic_start'] + EXTERNAL_SECONDS <= owner_deadline,
            'full_request_fits_original_owner_clock')
    require(owner_unix_start <= start['unix_start'] and start['unix_deadline'] <= unix_deadline,
            'full_request_fits_original_wall_clock')
    return True


def completion_fields(start, reading=None):
    validate_start(start)
    value = validate_sample(sample() if reading is None else reading)
    require(start['request_qpc_start'] <= value['request_qpc'] < start['request_qpc_deadline'], 'child_request_qpc_deadline')
    require(start['owner_monotonic_start'] <= value['owner_monotonic'], 'child_owner_clock_order')
    require(start['unix_start'] <= value['unix'] < start['unix_deadline'], 'child_wall_deadline')
    return dict(clock_contract=start['clock_contract'], finished_qpc=value['request_qpc'],
                finished_owner_monotonic=value['owner_monotonic'], finished_unix=value['unix'],
                elapsed_seconds=value['request_qpc'] - start['request_qpc_start'])


def validate_pending(start, observed, expected_contract=None):
    validate_start(start, expected_contract); validate_sample(observed)
    require(start['request_qpc_start'] <= observed['request_qpc'] < start['request_qpc_deadline'], 'pending_request_qpc_window')
    require(start['owner_monotonic_start'] <= observed['owner_monotonic'], 'pending_owner_clock_order')
    require(start['unix_start'] <= observed['unix'] < start['unix_deadline'], 'pending_wall_window')
    return True


def validate_received(start, response, *, observed, start_artifact, request_index,
                      case_id, mode, candidate_manifest, expected_contract=None):
    validate_start(start, expected_contract)
    validate_sample(observed)
    require(type(request_index) is int and 0 <= request_index < 6 and
            type(start.get('request_index')) is int and start['request_index'] == request_index and
            start.get('case_id') == case_id and start.get('mode') == mode and
            start.get('candidate_manifest') == candidate_manifest, 'start_request_identity')
    require(response.get('schema') == 'selfplay_learned_response_v2' and
            type(response.get('request_index')) is int and response['request_index'] == request_index and
            response.get('start') == start_artifact and response.get('case_id') == case_id and
            response.get('mode') == mode and response.get('candidate_manifest') == candidate_manifest,
            'complete_response_identity')
    require(response.get('clock_contract') == start['clock_contract'], 'complete_response_clock_contract')
    for key in ('finished_qpc', 'finished_owner_monotonic', 'finished_unix', 'elapsed_seconds'):
        number(response.get(key), 'finite_' + key)
    require(observed['request_qpc'] < start['request_qpc_deadline'], 'parent_request_qpc_deadline')
    require(observed['unix'] < start['unix_deadline'], 'parent_wall_deadline')
    require(start['request_qpc_start'] <= response['finished_qpc'] <= observed['request_qpc'], 'request_qpc_order')
    require(start['owner_monotonic_start'] <= response['finished_owner_monotonic'] <= observed['owner_monotonic'],
            'owner_clock_order')
    require(start['unix_start'] <= response['finished_unix'] <= observed['unix'], 'wall_clock_order')
    require(response['elapsed_seconds'] == response['finished_qpc'] - start['request_qpc_start'], 'child_elapsed_qpc')
    return dict(status='passed', receipt_elapsed_seconds=observed['request_qpc']-start['request_qpc_start'],
                request_qpc_and_owner_monotonic_compared_separately=True)


def validate_ack(start, response, ack, expected_contract=None):
    require(ack.get('schema') == 'selfplay_learned_parent_receipt_v2' and ack.get('accepted') is True and
            ack.get('clock_contract') == start.get('clock_contract') and ack.get('deadline_seconds') == EXTERNAL_SECONDS and
            ack.get('response_complete_before_deadline') is True and ack.get('original_request_clock_preserved') is True,
            'parent_receipt_contract')
    observed = dict(request_qpc=ack.get('parent_observed_qpc'),
                    owner_monotonic=ack.get('parent_observed_owner_monotonic'), unix=ack.get('parent_observed_unix'))
    proof = validate_received(start, response, observed=observed, start_artifact=ack.get('start'),
        request_index=ack.get('request_index'), case_id=start.get('case_id'), mode=start.get('mode'),
        candidate_manifest=start.get('candidate_manifest'), expected_contract=expected_contract)
    require(ack.get('receipt_elapsed_seconds') == proof['receipt_elapsed_seconds'], 'parent_elapsed_qpc')
    return proof


def validate_sequence(previous_ack, next_start):
    validate_start(next_start)
    for start_key, ack_key in [('request_qpc_start', 'parent_observed_qpc'),
                               ('owner_monotonic_start', 'parent_observed_owner_monotonic'),
                               ('unix_start', 'parent_observed_unix')]:
        number(previous_ack.get(ack_key), 'finite_previous_' + ack_key)
        require(next_start[start_key] >= previous_ack[ack_key], 'sequential_' + start_key)
    return True


def require_inherited_deadlines(owner_deadline, unix_deadline, reading=None):
    number(owner_deadline, 'finite_inherited_owner_deadline'); number(unix_deadline, 'finite_inherited_wall_deadline')
    value = validate_sample(sample() if reading is None else reading)
    require(value['owner_monotonic'] < owner_deadline, 'original_owner_deadline')
    require(value['unix'] < unix_deadline, 'original_wall_deadline')
    return True
