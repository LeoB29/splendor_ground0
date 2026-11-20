"""Final receipt audit and fixed budget recommendation; no models or rollout work."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
FITS = (161001, 161002, 161003)
PROFILE = 'four_candidate128_corrected_rules_20261018_v1'


def require(value, message):
    if not value: raise ValueError('Independent corrected qualification audit: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value): return type(value) in (int, float) and math.isfinite(value)


def quantile(values, fraction):
    if not values: return None
    values = sorted(values); position = (len(values)-1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high]-values[low]) * (position-low)


def description(values):
    return {'count': len(values), 'mean': sum(values)/len(values) if values else None,
            'median': quantile(values, .5), 'p95': quantile(values, .95), 'max': max(values) if values else None}


def cases_for(entries):
    require(len(entries) == 23 and {(e['fit'], e['actor'], e['phase']) for e in entries[:18]} ==
            {(f, s, p) for f in FITS for s in (0, 1) for p in (0, 1, 2)}, 'Exact three-fit core coverage')
    rows = [{'index': 23*b+i, 'root_index': i, 'fit': entry['fit'], 'breadth': False,
             'allowance_seconds': allowance, 'kind': 'core' if i < 18 else 'edge', 'hidden_variant': False}
            for b, allowance in enumerate((1, 120)) for i, entry in enumerate(entries)]
    seen = set()
    for i, entry in enumerate(entries[:18]):
        if entry['fit'] not in seen:
            seen.add(entry['fit'])
            rows.append({'index': len(rows), 'root_index': i, 'fit': entry['fit'], 'breadth': False,
                         'allowance_seconds': 120, 'kind': 'repeat', 'hidden_variant': True})
    require(len(rows) == 49, 'Schedule size')
    return rows


def budget_recommendation(rows):
    research = [r for r in rows if r['case']['allowance_seconds'] == 120]
    short = [r for r in rows if r['case']['allowance_seconds'] == 1]
    require(len(rows) == 49 and len(research) == 26 and len(short) == 23, 'Recommendation schedule coverage')
    require({r['fit'] for r in research if r['case']['kind'] == 'core'} == set(FITS), 'Recommendation fit coverage')
    complete = all(r['correct'] is True and r['clock']['status'] == 'ok' for r in research)
    maximum = max((r['clock']['receipt_elapsed_seconds'] for r in research if r['correct']), default=None)
    return {'recommended_research_seconds': 120 if complete else None,
            'research_120_seconds_qualified': complete,
            'research_completed_and_verified': sum(r['correct'] is True for r in research), 'research_prescribed': 26,
            'observed_margin': ('at_least_20_percent' if maximum <= 96 else 'thin_margin') if complete else 'unqualified',
            'maximum_accepted_research_receipt_seconds': maximum,
            'one_second_qualified': all(r['correct'] is True and r['clock']['status'] == 'ok' for r in short),
            'one_second_completed_and_verified': sum(r['correct'] is True for r in short), 'one_second_prescribed': 23,
            'other_budgets_inferred': False, 'strength_or_head_to_head_authorized': False,
            'interpretation': 'Finite corrected-rules saved-position qualification; observed timing margin is not a universal deadline guarantee.'}


def check_completion_flags(result, recommendation=None):
    require(result['completed_without_referee_failure'] is True and result['complete_schedule'] is True and
            result['failure'] is None and result['attempted'] == result['prescribed'] == 49 and
            result['strength_games'] == 0, 'Incomplete or failed workload')
    if recommendation is not None:
        qualified = recommendation['research_120_seconds_qualified']
        require(result['scientifically_accepted'] is qualified and result['research_qualified_120s'] is qualified,
                'Reported research qualification differs from independent completed-case gate')


def analyze(base):
    base = Path(base).resolve()
    completion, result, freeze, launch = [read(base / name) for name in ('completion.json', 'result.json', 'freeze.json', 'launch.json')]
    require(completion['status'] == 'completed' and completion['worker_processes_closed'] is True and
            completion['result'] == artifact(base / 'result.json'), 'Completion/result binding')
    check_completion_flags(result)
    require(launch['freeze'] == artifact(base / 'freeze.json') and launch['maximum_seconds'] == 14400 and
            launch['torch_loaded_in_supervisor'] is False and finite(launch['epoch_seconds']), 'Launch/freeze/time identity')
    started = launch['epoch_seconds']
    def boundary(): require(0 <= time.time()-started < 14400, 'Shared qualification/verification four-hour cap')
    boundary()
    paths, size = set(), 0
    for item in freeze['inputs']:
        boundary(); p = (ROOT / item['path']).resolve()
        require(p.is_relative_to(ROOT) and p not in paths, 'Frozen input path identity')
        require(p.stat().st_size == item['bytes'] and file_hash(p) == item['sha256'], 'Frozen hash/size changed: ' + item['path'])
        paths.add(p); size += item['bytes']
    require(freeze['profile_id'] == PROFILE and freeze['cases'] == 49 and freeze['fits'] == list(FITS) and
            freeze['worlds'] == 128 and freeze['maximum_candidates'] == 4 and freeze['strength_games_authorized'] == 0 and
            freeze['limits']['seconds'] == 14400 and freeze['limits']['audit_seconds_per_completed_root'] == 600 and
            freeze['limits']['initialization_seconds'] == 60, 'Frozen protocol constants')
    final_inputs = read(base / 'final_input_check.json')
    require(final_inputs['inputs_checked'] == len(paths) and final_inputs['all_hashes_and_sizes_match'] is True, 'Final source check')
    prior = read(base / 'prior_preservation.json')
    require(prior['all_hashes_and_sizes_match'] is True and all((ROOT / i['path']).resolve() in paths for i in prior['inputs']), 'Inherited preservation coverage')
    for folder in ('external_strength_screen_20261018_02', 'local_player_qualification_20261018_recovery_01'):
        previous = ROOT / 'outputs' / folder
        require(all(p.resolve() in paths for p in previous.rglob('*') if p.is_file()), 'Prior evidence omitted')
    for p in (ROOT, ROOT / 'src'):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    # Import only after all source identities pass. These modules do no inference.
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    from tools.analyze_external_strength_screen_20261018_02 import check_clock
    require('torch' not in sys.modules, 'Unexpected numerical model import')
    cpu = freeze['topology']['selected_logical_processor']
    selection = read(base / 'selection.json'); entries = selection['core'] + selection['edges']
    schedule = cases_for(entries)
    require(read(base / 'schedule.json') == schedule, 'Independent fixed49 schedule')
    require(read(base / 'selection.json') == read(ROOT / 'outputs/local_player_qualification_20261018_recovery_01/selection.json'), 'Outcome-blind selection changed')
    privacy, fixture_changes = [], []
    for index, entry in enumerate(entries):
        boundary(); fixture = read(base / 'fixtures' / f'{index:02d}.json')
        source = ROOT / entry['path']; require(fixture['source_artifact'] == artifact(source) and entry['sha256'] == file_hash(source), 'Selected source identity')
        original = read(source)
        require(fixture['snapshot'] == rules.convert_legacy_snapshot(original['snapshot']), 'Counterfactual component conversion')
        normal = read(base / 'observations' / f'{index:02d}.json'); hidden = read(base / 'variants' / f'{index:02d}.json')
        require(normal == hidden and fixture['snapshot'] != fixture['hidden_variant'], 'Privacy fixture did not alter hidden state')
        for snapshot, obs in ((fixture['snapshot'], normal), (fixture['hidden_variant'], hidden)):
            clean = {k: v for k, v in obs.items() if k != 'context'}; clean['schema'] = rules.OBSERVATION_SCHEMA
            ledger.check_observer(snapshot, clean)
            require(obs['context'] == entry['context'], 'Public fit/stream context')
        legal = [rules.serialize_action(a) for a in rules.legal_actions(rules.deserialize_snapshot(fixture['snapshot']))]
        require(fixture['legal_actions'] == legal, 'Corrected legal support fixture')
        privacy.append({'root_id': entry['root_id'], 'observer_sha256': digest(normal),
                        'hidden_variant_observer_sha256': digest(hidden), 'hidden_state_actually_changed': True, 'identical': True})
        fixture_changes.append({'root_id': entry['root_id'], 'fit': entry['fit'], 'corrected_legal_count': len(legal),
            'legacy_legal_count': len(original['proposal']['legal_actions']), 'legal_support_changed': legal != original['proposal']['legal_actions']})
    require(read(base / 'privacy_inputs.json') == privacy and read(base / 'fixture_conversion.json') == fixture_changes, 'Privacy/conversion receipts')
    require(sorted(p.name for p in (base / 'rows').glob('*.json')) == [f'{i:03d}.json' for i in range(49)], 'Exact row inventory')
    rows, wrappers, clocks, audit_clocks, initializations, proofs, proof_paths = [], {}, [], [], [], [], []
    def receipt(folder, name, saved, limit):
        value = read(folder / name)
        require({k: v for k, v in value.items() if k != 'result'} == saved, 'Copied clock receipt differs')
        check_clock(value, None if name == 'initialization.json' else 'request-00000001', limit, cpu)
        require(set(p['role'] for p in value['telemetry']['process_tree']) <= {'python', 'console:python'} and
                value['telemetry']['processes_observed'] <= 2, 'Unexpected planner/audit descendant')
        return value
    for case in schedule:
        boundary(); i = case['index']; entry = entries[case['root_index']]
        row = read(base / 'rows' / f'{i:03d}.json'); rows.append(row)
        obs = read(base / ('variants' if case['hidden_variant'] else 'observations') / f'{case["root_index"]:02d}.json')
        require(row['case'] == case and row['root_id'] == entry['root_id'] and row['fit'] == entry['fit'] and
                row['original_sha256'] == entry['sha256'] and row['observer_sha256'] == digest(obs) and type(row['correct']) is bool, 'Row/observer identity')
        folder = base / 'attempts' / f'{i:03d}'
        require({p.name for p in folder.glob('request-*.json')} <= {'request-00000001.json'}, 'Extra deployment requests')
        if row['clock']['status'].startswith('initialization_'):
            init = read(folder / 'initialization.json')
            check_clock(init, None, 60, cpu)
            require(init['status'] == 'timeout' and row['clock']['status'] == 'initialization_timeout' and row['correct'] is False, 'Non-timeout initialization failure')
            initializations.append(init); continue
        init = receipt(folder, 'initialization.json', row['initialization'], 60)
        require(init['status'] == 'ok' and init['result']['threads'] == 1 and all(v == '1' for v in init['thread_environment'].values()), 'Initialization thread contract')
        initializations.append(init)
        clock = receipt(folder, 'request-00000001.json', row['clock'], case['allowance_seconds']); clocks.append(clock)
        if clock['status'] != 'ok':
            require(clock['status'] == 'timeout' and row['correct'] is False and not (base / 'audits' / f'{i:03d}').exists(), 'Unaudited non-timeout failure')
            continue
        wrapper = clock['result']; wrappers[i] = wrapper; decision = wrapper['decision']; record = decision['record']
        require(row['correct'] is True and row['result_sha256'] == digest(wrapper) and row['decision_sha256'] == digest(decision) and
                wrapper['observer_sha256'] == digest(obs) and wrapper['profile_id'] == PROFILE and
                wrapper['ruleset_id'] == rules.RULESET_ID and wrapper['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Completed result identity')
        trace_path = folder / 'continuations.jsonl.gz'
        require(row['trace'] == artifact(trace_path) and wrapper['trace']['sha256'] == row['trace']['sha256'] and wrapper['trace']['bytes'] == row['trace']['bytes'], 'Trace file binding')
        support = len(record['candidates']['actions']); require(1 <= support <= 4 and len(record['worlds']) == 128, 'Delivered root budget')
        require(all(len(w['outcomes']) == support for w in record['worlds']) and wrapper['trace']['records'] == 128 * support and
                row['work'] == {'worlds': 128, 'continuations': 128 * support}, 'Delivered continuation count')
        outcomes = [o for w in record['worlds'] for o in w['outcomes']]
        require(all(o['reason'] in ('completed', 'forced_pass_deadlock') and type(o['turns']) is int and o['turns'] >= 1 and
                    type(o['wdl']) in (int, float) and o['wdl'] in (-1, 0, 1) for o in outcomes), 'Continuation records')
        totals = [sum(w['outcomes'][j]['wdl'] for w in record['worlds']) for j in range(support)]
        choice = max(range(support), key=lambda j: totals[j])
        require(record['means'] == [x/128 for x in totals] and type(record['choice']) is int and record['choice'] == choice and
                decision['action'] == record['candidates']['actions'][choice], 'Independent means/ties')
        audit_folder = base / 'audits' / f'{i:03d}'
        ai = receipt(audit_folder, 'initialization.json', row['audit_initialization'], 60)
        require(ai['status'] == 'ok' and ai['result']['threads'] == 1, 'Audit initialization'); initializations.append(ai)
        ac = receipt(audit_folder, 'request-00000001.json', row['audit_clock'], 600); audit_clocks.append(ac)
        require(ac['status'] == 'ok' and row['verification'] == ac['result'] and row['audit_result_sha256'] == digest(ac['result']), 'Audit proof receipt')
        proof = ac['result']; proofs.append(proof); proof_paths.append(artifact(audit_folder / 'request-00000001.json'))
        expected = {'status': 'passed', 'profile_id': PROFILE, 'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
            'observer_sha256': digest(obs), 'result_sha256': digest(wrapper), 'decision_sha256': digest(decision),
            'trace_sha256': row['trace']['sha256'], 'trace_bytes': row['trace']['bytes'], 'audited_worlds': 128,
            'candidate_support': support, 'audited_continuations': 128 * support, 'audited_transitions': sum(o['turns'] for o in outcomes),
            'raw035_choices_reexecuted': sum(o['turns']-1 for o in outcomes), 'proposal_model_reexecuted': True,
            'proposal_selection_independent': True, 'all_continuations_replayed': True, 'old_outcomes_used_as_ground_truth': False}
        require(all(proof.get(k) == value for k, value in expected.items()), 'Complete replay proof/coverage')
    require(result['verified_completed_decisions'] == len(proofs), 'Completed verification total')
    recommendation = budget_recommendation(rows)
    check_completion_flags(result, recommendation)
    for row in rows[46:]:
        base_row = rows[23 + row['case']['root_index']]
        if row['correct'] and base_row['correct']:
            require(row['decision_sha256'] == base_row['decision_sha256'] and row['trace']['sha256'] == base_row['trace']['sha256'], 'Repeat scientific trace changed')
    require({p.name for p in (base / 'audits').iterdir() if p.is_dir()} == {f'{i:03d}' for i in wrappers} if (base / 'audits').exists() else not wrappers,
            'Audit folder coverage')
    old_rows = [read(p) for p in (ROOT / 'outputs/local_player_qualification_20261018_recovery_01/rows').glob('*.json')]
    old_map = {(r['root_id'], r['fit'], r['case']['kind'], r['case']['hidden_variant']): r for r in old_rows
               if r['case']['breadth'] is False and r['case']['allowance_seconds'] == 120 and r['correct']}
    comparisons = []
    for row in rows[23:]:
        if not row['correct']: continue
        old = old_map[(row['root_id'], row['fit'], row['case']['kind'], row['case']['hidden_variant'])]
        old_path = ROOT / 'outputs/local_player_qualification_20261018_recovery_01/attempts' / f'{old["case"]["index"]:03d}' / 'request-00000001.json'
        previous = read(old_path)['result']; current = wrappers[row['case']['index']]['decision']
        require(digest(previous) == old['result_sha256'], 'Historical four-candidate receipt binding')
        comparisons.append({'case': row['case']['index'], 'root_id': row['root_id'], 'fit': row['fit'],
            'kind': row['case']['kind'], 'action_changed': previous['action'] != current['action'],
            'candidate_set_changed': previous['record']['candidates'] != current['record']['candidates'],
            'historical_receipt': artifact(old_path), 'used_as_acceptance_ground_truth': False})
    boundary(); require('torch' not in sys.modules, 'Analyzer loaded models')
    elapsed = time.time()-started
    utc = (datetime.fromisoformat(result['time']) - datetime.fromisoformat(launch['time'])).total_seconds()
    require(finite(result['elapsed_seconds']) and 0 < result['elapsed_seconds'] <= elapsed and abs(utc-result['elapsed_seconds']) < 3, 'Result elapsed reconciliation')
    telemetry = [r['telemetry'] for r in clocks + audit_clocks + initializations]
    sizes = sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    require(sizes < 256*1024**2, 'Evidence size cap')
    return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
        'freeze_sha256': file_hash(base / 'freeze.json'), 'source_preservation': {'files': len(paths), 'bytes': size, 'all_hashes_and_sizes_match': True},
        'attempted': 49, 'verified_completed_decisions': len(proofs), 'privacy_inputs_checked': 23, 'repeat_cases': 3,
        'recommendation': recommendation,
        'per_allowance': {str(a): {'prescribed': sum(r['case']['allowance_seconds'] == a for r in rows),
            'statuses': dict(Counter(r['clock']['status'] for r in rows if r['case']['allowance_seconds'] == a)),
            'verified_receipt_seconds': description([r['clock']['receipt_elapsed_seconds'] for r in rows if r['case']['allowance_seconds'] == a and r['correct']])} for a in (1, 120)},
        'per_fit': {str(f): {str(a): {'prescribed': sum(r['fit'] == f and r['case']['allowance_seconds'] == a for r in rows),
            'verified': sum(r['fit'] == f and r['case']['allowance_seconds'] == a and r['correct'] for r in rows)} for a in (1, 120)} for f in FITS},
        'coverage': {k: sum(p[k] for p in proofs) for k in ('audited_worlds', 'audited_continuations', 'audited_transitions', 'raw035_choices_reexecuted')},
        'historical_comparison': {'fixture_legal_support_changed': sum(r['legal_support_changed'] for r in fixture_changes),
            'fixtures': fixture_changes, 'completed_four_candidate_comparisons': comparisons, 'acceptance_uses_historical_outcomes': False},
        'resources': {'initializations': description([r['receipt_elapsed_seconds'] for r in initializations if r['status'] == 'ok']),
            'full_replay_audit_seconds': description([r['receipt_elapsed_seconds'] for r in audit_clocks]),
            'peak_sampled_combined_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
            'sum_last_observed_worker_cpu_seconds': sum(r['telemetry']['cpu_seconds'] for r in clocks+audit_clocks),
            'qualification_and_full_replay_seconds': result['elapsed_seconds'], 'including_final_analyzer_seconds': elapsed,
            'evidence_bytes_before_this_receipt': sizes, 'sampled_not_continuous': True},
        'scope': {'no_strength_games': True, 'no_new_training_data': True, 'final_analyzer_reexecutes_models': False,
            'full_replay_audit_workers_reexecute_every_recorded_raw_policy_choice': True,
            'trusted_dependencies': ['Frozen model/feature kernels and corrected legal enumeration', 'Separate frozen dictionary transition ledger', 'Frozen independent request-clock checker'],
            'repeat_cases_add_no_independent_strength_evidence': True, 'timeout_latencies_not_imputed': True},
        'full_replay_proof_receipts': proof_paths}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, default=BASE); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.base / 'independent_analysis.json'
    require(not destination.exists(), 'Never overwrite independent analysis')
    result = analyze(args.base)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    require(sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file()) + len(payload.encode()) < 256*1024**2, 'Final evidence bytes')
    require(time.time() - read(args.base / 'launch.json')['epoch_seconds'] < 14400, 'Shared final verification deadline')
    with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    print(json.dumps({'status': 'passed', 'output': str(destination), 'recommendation': result['recommendation']}, sort_keys=True))


if __name__ == '__main__': main()
