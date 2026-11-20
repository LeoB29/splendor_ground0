"""New bounded qualification; historical tools and evidence remain immutable."""
from pathlib import Path

from tools.local_player_qualification_128.common import (
    ROOT, MAX_SECONDS, MAX_BYTES, MIN_FREE, MAX_WORKER_RSS, ResourceHold,
    now, read, canonical, digest, file_hash, artifact, save_new, verify, memory,
    directory_bytes, boundary, execution_lease,
)

OUTPUT = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
PRIOR_LOCAL = ROOT / 'outputs/local_player_qualification_20261018_recovery_01'
PRIOR_EXTERNAL = ROOT / 'outputs/external_strength_screen_20261018_02'
PROFILE_ID = 'four_candidate128_corrected_rules_20261018_v1'
RULESET_ID = 'splendor_base_qualified_rules_20261018_v1'
CATALOGUE_SHA256 = 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
AUDIT_SECONDS = 600
INITIALIZATION_SECONDS = 60
FINAL_RESERVE_SECONDS = 600
FITS = (161001, 161002, 161003)


def build_cases(entries):
    if len(entries) != 23 or {e['fit'] for e in entries[:18]} != set(FITS):
        raise ValueError('Require the unchanged 18 core and five edge positions')
    cases = []
    for allowance in (1, 120):
        for index, entry in enumerate(entries):
            cases.append({'index': len(cases), 'root_index': index, 'fit': entry['fit'],
                          'breadth': False, 'allowance_seconds': allowance,
                          'kind': 'core' if index < 18 else 'edge', 'hidden_variant': False})
    seen = set()
    for index, entry in enumerate(entries[:18]):
        if entry['fit'] in seen:
            continue
        seen.add(entry['fit'])
        cases.append({'index': len(cases), 'root_index': index, 'fit': entry['fit'],
                      'breadth': False, 'allowance_seconds': 120,
                      'kind': 'repeat', 'hidden_variant': True})
    if len(cases) != 49:
        raise ValueError('Incorrect fixed schedule')
    return cases


def verify_artifacts(items):
    for item in items:
        p = ROOT / item['path']
        if p.stat().st_size != item['bytes'] or file_hash(p) != item['sha256']:
            raise ValueError('Changed bound artifact: ' + item['path'])


def check_wrapper(value, observer, legal, trace_path):
    """Cheap supervisor checks before the full independent continuation audit."""
    expected = {'decision', 'trace', 'ruleset_id', 'catalogue_sha256', 'profile_id', 'observer_sha256'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Unexpected completed result schema')
    if (value['ruleset_id'] != RULESET_ID or value['catalogue_sha256'] != CATALOGUE_SHA256
            or value['profile_id'] != PROFILE_ID or value['observer_sha256'] != digest(observer)):
        raise ValueError('Completed result identity mismatch')
    decision = value['decision']
    if not isinstance(decision, dict) or set(decision) != {'action', 'record'} or decision['action'] not in legal:
        raise ValueError('Completed action is not in corrected legal support')
    record = decision['record']
    count = len(record['candidates']['actions'])
    if not 1 <= count <= 4 or len(record['worlds']) != 128 or record['planner'] != 'reference':
        raise ValueError('Wrong candidate/world budget or player')
    if any(len(w['outcomes']) != count for w in record['worlds']):
        raise ValueError('Incomplete continuation budget')
    trace = value['trace']
    if set(trace) != {'sha256', 'bytes', 'records'} or trace['records'] != 128 * count:
        raise ValueError('Trace record count differs')
    if Path(trace_path).stat().st_size != trace['bytes'] or file_hash(trace_path) != trace['sha256']:
        raise ValueError('Trace artifact binding differs')
    return True


def check_audit_proof(proof, observer, result):
    if not isinstance(proof, dict):
        raise ValueError('Missing audit proof')
    expected = {'status': 'passed', 'profile_id': PROFILE_ID, 'ruleset_id': RULESET_ID,
                'catalogue_sha256': CATALOGUE_SHA256, 'observer_sha256': digest(observer),
                'result_sha256': digest(result), 'decision_sha256': digest(result['decision']),
                'trace_sha256': result['trace']['sha256'], 'trace_bytes': result['trace']['bytes'],
                'audited_worlds': 128, 'candidate_support': len(result['decision']['record']['candidates']['actions']),
                'audited_continuations': result['trace']['records'], 'proposal_model_reexecuted': True,
                'proposal_selection_independent': True, 'all_continuations_replayed': True,
                'old_outcomes_used_as_ground_truth': False}
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('Audit proof identity or complete-work mismatch')
    if (type(proof.get('audited_transitions')) is not int or proof['audited_transitions'] < result['trace']['records']
            or type(proof.get('raw035_choices_reexecuted')) is not int
            or proof['raw035_choices_reexecuted'] != proof['audited_transitions'] - result['trace']['records']):
        raise ValueError('Audit transition/policy coverage mismatch')
    return True
