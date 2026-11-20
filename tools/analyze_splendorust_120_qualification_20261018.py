"""Independent timing-only qualification audit; no model imports or inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/splendorust_120_qualification_20261018_01'
LADDER = (256, 1024, 4096, 6400)
CALIBRATION_FIXTURES = (0, 3, 18, 19, 20, 21, 22)
BASE_PROFILE = 'outputs/external_strength_screen_20261018_02/opponent_profile.json'
BASE_PROFILE_SHA256 = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
BASE_PROFILE_BYTES = 15889
MODULE = 'tools.splendorust_120_qualification_20261018.worker'
METADATA = frozenset(('command', 'identity', 'affinity', 'topology', 'thread_environment'))
THREADS = frozenset(('BLIS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
RULESET = 'splendor_base_qualified_rules_20261018_v1'
CATALOGUE = 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'


def require(value, message):
    if not value: raise ValueError('Independent SplendoRust 120-second audit: ' + message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def finite(value): return type(value) in (int, float) and math.isfinite(value)


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_profile(baseline, quota):
    require(type(quota) is int and quota in LADDER, 'Unexpected quota')
    return {**baseline, 'schema': 'splendorust_120_fixed_work_profile_v1',
            'profile_id': f'entity_onehot_puct{quota}_canonical_atomic_prereveal_cpu_120s_v1',
            'iterations': quota, 'internal_budget_seconds': 108, 'external_budget_seconds': 120,
            'base_profile': {'path': BASE_PROFILE, 'sha256': BASE_PROFILE_SHA256, 'bytes': BASE_PROFILE_BYTES}}


def calibration_gate(row):
    clock = row['clock']
    if clock['status'] != 'ok' or row.get('result', {}).get('status') != 'completed': return False
    return row['correct'] is True and finite(clock['receipt_elapsed_seconds']) and 0 <= clock['receipt_elapsed_seconds'] <= 96


def reconstruct_selection(rows):
    """Require the exact ascending ladder, stopping at its first timing rejection."""
    selected, offset, stopped, summaries = None, 0, False, []
    for quota in LADDER:
        attempts = []
        for fixture in CALIBRATION_FIXTURES:
            require(offset < len(rows), 'Calibration ended without rejection or complete ladder')
            row = rows[offset]; offset += 1
            require(row['stage'] == 'calibration' and row['quota'] == quota and row['fixture'] == fixture and
                    row['kind'] == 'original', 'Calibration order or fixture changed')
            attempts.append(row)
            if not calibration_gate(row): stopped = True; break
        passed = not stopped
        summaries.append({'quota': quota, 'attempted': len(attempts), 'passed': passed,
                          'maximum_completed_receipt_seconds': max((r['clock']['receipt_elapsed_seconds'] for r in attempts
                              if r['correct']), default=None)})
        if stopped: break
        selected = quota
    require(offset == len(rows), 'Calibration continued after first rejection or prescribed ladder')
    return {'selected_quota': selected, 'attempted': offset, 'quotas': summaries,
            'stopped_on_first_timing_rejection': stopped}


def final_gate(selected, rows):
    require(selected is None or type(selected) is int and selected in LADDER, 'Final quota outside ladder')
    require(len(rows) == (0 if selected is None else 46), 'Final panel request count')
    pairs = 0
    for index, row in enumerate(rows):
        require(row['stage'] == 'qualification' and row['quota'] == selected and row['fixture'] == index // 2 and
                row['kind'] == ('original' if index % 2 == 0 else 'hidden_variant'), 'Final panel identity/order')
        if index % 2 and rows[index-1]['correct'] and row['correct']:
            require(rows[index-1]['result'] == row['result'], 'Hidden-state result/diagnostic repeat differs')
            pairs += 1
    verified = sum(r['correct'] is True for r in rows)
    return {'qualified': selected is not None and verified == 46 and pairs == 23,
            'verified_decisions': verified, 'privacy_pairs_matched': pairs}


def check_initialization(value, saved, freeze):
    require(METADATA <= value.keys() and
            {k: v for k, v in value.items() if k not in METADATA and k != 'result'} == saved,
            'Initialization raw receipt copy')
    require(value['command'] == [freeze['executable'], '-B', '-u', '-m', MODULE], 'Initialization command')
    require(value['affinity'] == [freeze['topology']['selected_logical_processor']] and
            value['topology'] == freeze['topology'], 'Initialization CPU/topology')
    require(value['thread_environment'] == {k: '1' for k in THREADS}, 'Numerical thread environment')
    identity = value['identity']
    require(type(identity) is dict and set(identity) == {'pid', 'created'} and
            type(identity['pid']) is int and identity['pid'] > 0 and finite(identity['created']) and
            identity['created'] > 0, 'Initialization identity')
    pythons = [p for p in value['telemetry']['process_tree'] if p['role'] == 'python']
    require(len(pythons) == 1 and pythons[0]['pid'] == identity['pid'] and pythons[0]['created'] == identity['created'] and
            Path(pythons[0]['executable']).resolve() == Path(freeze['executable']).resolve(), 'Initialization telemetry identity')
    require(value['status'] == 'ok' and type(value['result']['pid']) is int and
            value['result']['pid'] == identity['pid'] and value['result']['threads'] == 1, 'Initialization ready contract')


def process_identity(receipts, freeze, profile):
    owners = {}
    binary = (ROOT / profile['native_binary']).resolve()
    native = [a for a in profile['artifacts'] if (ROOT / a['path']).resolve() == binary]
    require(len(native) == 1, 'Native binary source identity')
    for receipt in receipts:
        for p in receipt['telemetry']['process_tree']:
            role = p['role']; identity = (p['pid'], p['created'])
            require(type(p['pid']) is int and p['pid'] > 0 and finite(p['created']), 'Process identity types')
            require(role in ('python', 'rust', 'console:python', 'console:rust'), 'Unexpected process role')
            executable = Path(p['executable']).resolve()
            if role == 'python': require(executable == Path(freeze['executable']).resolve(), 'Python executable path')
            elif role == 'rust':
                require(executable == binary and p['executable_sha256'] == native[0]['sha256'], 'Native executable hash/path')
            else:
                require(executable == (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve(), 'Exact OS helper path')
            current = {'identity': identity, 'executable': str(executable), 'sha256': p['executable_sha256'],
                       'startup': {k: v for k, v in p.items() if k.startswith('startup_affinity_')}}
            require(role not in owners or owners[role] == current, 'Worker process or executable replacement')
            owners[role] = current
        require(receipt['telemetry']['processes_observed'] == len(owners) and {'python', 'rust'} <= owners.keys(),
                'Process accounting coverage')


class Audit:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.initializations, self.clocks, self.work = [], [], Counter()
        self.completed_transitions = 0
        self.rows = {'calibration': [], 'qualification': []}

    def boundary(self):
        require(0 <= time.time() - self.launch['epoch_seconds'] < 14400, 'Original four-hour qualification/audit limit')

    def freeze(self):
        self.completion = read(self.base / 'completion.json')
        self.result = read(self.base / 'qualification_result.json')
        self.freeze_doc = read(self.base / 'freeze.json')
        self.launch = read(self.base / 'launch.json')
        require(self.completion['status'] == 'completed' and self.completion['worker_processes_closed'] is True and
                self.completion['result'] == artifact(self.base / 'qualification_result.json') and self.result['failure'] is None,
                'Incomplete or referee-failed run')
        require(self.launch['freeze'] == artifact(self.base / 'freeze.json') and self.launch['maximum_seconds'] == 14400 and
                self.launch['torch_loaded_in_supervisor'] is False and finite(self.launch['epoch_seconds']), 'Launch/freeze binding')
        self.boundary()
        paths, size = set(), 0
        for item in self.freeze_doc['inputs']:
            self.boundary(); p = (ROOT / item['path']).resolve()
            require(p.is_relative_to(ROOT) and p not in paths and p.stat().st_size == item['bytes'] and
                    file_hash(p) == item['sha256'], 'Frozen source identity: ' + item['path'])
            paths.add(p); size += item['bytes']
        self.sources = {'files': len(paths), 'bytes': size, 'all_hashes_and_sizes_match': True}
        constants = {'ruleset_id': RULESET, 'catalogue_sha256': CATALOGUE, 'grid': list(LADDER),
            'calibration_indices': list(CALIBRATION_FIXTURES), 'seconds_per_move': 120,
            'initialization_seconds': 60, 'internal_budget_seconds': 108, 'calibration_margin_seconds': 96,
            'execution_cap_seconds': 14400, 'finalization_reserve_seconds': 600, 'qualification_requests': 46,
            'maximum_requests': 74, 'max_evidence_bytes': 256*1024**2, 'max_process_tree_rss_bytes': 2*1024**3,
            'minimum_free_disk_bytes': 5*1024**3, 'strength_games_authorized': 0, 'larger_data_deferred': True}
        require(all(canonical(self.freeze_doc.get(k)) == canonical(v) for k, v in constants.items()), 'Frozen protocol constants')
        prior = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
        require(all((ROOT / i['path']).resolve() in paths for i in read(prior / 'freeze.json')['inputs']), 'Inherited frozen input missing')
        require(all(p.resolve() in paths for p in prior.rglob('*') if p.is_file()), 'Prior completed qualification evidence missing')
        inherited = read(self.base / 'prior_preservation.json')
        require(inherited['all_hashes_and_sizes_match'] is True and
                all((ROOT / i['path']).resolve() in paths for i in inherited['inputs']), 'Preservation inventory coverage')
        baseline_path = ROOT / BASE_PROFILE
        require(file_hash(baseline_path) == BASE_PROFILE_SHA256 and baseline_path.stat().st_size == BASE_PROFILE_BYTES,
                'Qualified one-second baseline profile changed')
        self.profiles = {}
        baseline = read(baseline_path)
        for quota in LADDER:
            path = self.base / 'profiles' / f'q{quota:05d}.json'
            profile = read(path)
            require(path.resolve() in paths and canonical(profile) == canonical(expected_profile(baseline, quota)), 'Candidate profile changed outside quota/budget')
            require(all((ROOT / item['path']).resolve() in paths for item in profile['artifacts']), 'Profile artifact not frozen')
            self.profiles[quota] = profile
        self.cpu = self.freeze_doc['topology']['selected_logical_processor']
        for p in (ROOT, ROOT / 'src'):
            if str(p) not in sys.path: sys.path.insert(0, str(p))
        from tools.external_strength_screen_20261018 import rules, audit as ledger
        from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics, descriptive
        self.rules, self.ledger = rules, ledger
        self.check_clock, self.check_diagnostics, self.descriptive = check_clock, check_diagnostics, descriptive
        require('torch' not in sys.modules and rules.RULESET_ID == RULESET and rules.CATALOGUE_SHA256 == CATALOGUE, 'Model-free corrected-rule identity')
        self.fixtures = read(self.base / 'fixtures.json')
        self.variants = read(self.base / 'variants.json'); self.observers = read(self.base / 'observers.json')
        require(len(self.fixtures) == len(self.variants) == len(self.observers) == 23 and
                self.fixtures == read(ROOT / 'outputs/external_strength_screen_20261018_02/fixtures.json'), 'Unchanged selected fixtures')
        expected_schedule = [{'index': 2*i+j, 'fixture': i, 'kind': kind} for i in range(23)
                             for j, kind in enumerate(('original', 'hidden_variant'))]
        require(read(self.base / 'qualification_schedule.json') == expected_schedule, 'Independent final pair schedule')
        for fixture, hidden, payloads in zip(self.fixtures, self.variants, self.observers):
            require(fixture['snapshot'] != hidden and payloads['original'] == payloads['hidden_variant'], 'Hidden-world privacy construction')
            ledger.check_observer(fixture['snapshot'], payloads['original'])
            ledger.check_observer(hidden, payloads['hidden_variant'])

    def row(self, stage, name, quota, fixture, kind):
        self.boundary()
        path = self.base / 'rows' / stage / (name + '.json')
        row = read(path); self.rows[stage].append(row)
        require(row['stage'] == stage and type(row['quota']) is int and row['quota'] == quota and
                type(row['fixture']) is int and row['fixture'] == fixture and row['kind'] == kind and
                row['root_id'] == self.fixtures[fixture]['source']['root_id'] and type(row['correct']) is bool, 'Request identity/order')
        observer = self.observers[fixture][kind]
        snapshot = self.fixtures[fixture]['snapshot'] if kind == 'original' else self.variants[fixture]
        profile_path = self.base / 'profiles' / f'q{quota:05d}.json'
        require(row['observer_sha256'] == digest(observer) and row['profile'] == artifact(profile_path), 'Observer/profile binding')
        folder = self.base / 'workers' / stage / name
        require({p.name for p in folder.glob('request-*.json')} == {'request-00000001.json'}, 'One decision per fresh worker')
        init = read(folder / 'initialization.json'); clock = read(folder / 'request-00000001.json')
        check_initialization(init, row['initialization'], self.freeze_doc)
        require({k: v for k, v in clock.items() if k != 'result'} == row['clock'], 'Copied decision receipt')
        self.check_clock(init, None, 60, self.cpu); self.check_clock(clock, 'request-00000001', 120, self.cpu)
        process_identity([init, clock], self.freeze_doc, self.profiles[quota])
        self.initializations.append(init); self.clocks.append(clock)
        require(clock['status'] in ('ok', 'timeout'), 'Non-timing player/resource/referee failure')
        if clock['status'] == 'timeout':
            require(row['correct'] is False and row.get('result') is None and row.get('post_snapshot') is None and
                    row.get('ledger_sha256') is None, 'External timeout was promoted')
            return row
        result = clock['result']
        require(row['result'] == result and row['result_sha256'] == digest(result), 'Completed/exhausted result copy/hash')
        legal = [self.rules.serialize_action(a) for a in self.rules.legal_actions(self.rules.deserialize_snapshot(snapshot))]
        identity = {'player': 'opponent', 'observer_sha256': digest(observer), 'legal_actions_sha256': digest(legal),
                    'legal_count': len(legal), 'ruleset': RULESET, 'catalogue_sha256': CATALOGUE,
                    'profile_sha256': file_hash(profile_path)}
        require(type(result['legal_count']) is int and all(result.get(k) == v for k, v in identity.items()), 'Result legal/rule/profile identity')
        if result['status'] == 'internal_budget_exhausted':
            require(result == {**identity, 'status': 'internal_budget_exhausted', 'internal_budget_seconds': 108,
                    'opponent_id': 'splendorust', 'profile_id': self.profiles[quota]['profile_id'],
                    'reason': 'Internal fixed-work deadline exceeded'} and row['correct'] is False and
                    row.get('post_snapshot') is None and row.get('ledger_sha256') is None, 'Internal exhaustion was promoted or changed')
            return row
        require(set(result) == set(identity) | {'status', 'action', 'diagnostics'} and result['status'] == 'completed' and
                result['action'] in legal and row['correct'] is True, 'Completed legal decision')
        self.work.update(self.check_diagnostics(result['diagnostics'], observer, result['action'], legal, self.profiles[quota]))
        after = self.ledger.transition(snapshot, result['action'])
        require(after == row['post_snapshot'] and row['ledger_sha256'] == digest(after), 'Independent corrected-rule successor')
        self.completed_transitions += 1
        return row

    def run(self):
        self.freeze()
        calibration_files = {p.name for p in (self.base / 'rows/calibration').glob('*.json')}
        expected_files, stopped = set(), False
        for quota in LADDER:
            for ordinal, fixture in enumerate(CALIBRATION_FIXTURES):
                name = f'q{quota:05d}_{ordinal:02d}'; expected_files.add(name + '.json')
                row = self.row('calibration', name, quota, fixture, 'original')
                if not calibration_gate(row): stopped = True; break
            if stopped: break
        require(calibration_files == expected_files, 'Calibration file coverage/extra attempts')
        selection = reconstruct_selection(self.rows['calibration'])
        selected = selection['selected_quota']
        saved_selection = read(self.base / 'calibration_selection.json')
        expected_attempts = [{'quota': v['quota'], 'passed_margin': v['passed'], 'requests': v['attempted'],
            'profile': artifact(self.base / 'profiles' / f'q{v["quota"]:05d}.json'),
            'maximum_completed_receipt_seconds': v['maximum_completed_receipt_seconds']} for v in selection['quotas']]
        expected_selection = {'grid': list(LADDER), 'calibration_indices': list(CALIBRATION_FIXTURES),
            'maximum_accepted_receipt_seconds': 96, 'attempts': expected_attempts, 'selected_quota': selected,
            'no_match_outcomes_used': True, 'profile': artifact(self.base / 'selected_profile.json') if selected is not None else None}
        require({k: v for k, v in saved_selection.items() if k != 'time'} == expected_selection, 'Timing-only selection differs')
        if selected is not None:
            require(read(self.base / 'selected_profile.json') == self.profiles[selected], 'Selected fixed profile differs')
        else:
            require(not (self.base / 'selected_profile.json').exists(), 'Unselected profile was emitted')
        qualification_files = {p.name for p in (self.base / 'rows/qualification').glob('*.json')}
        require(qualification_files == ({f'{i:03d}.json' for i in range(46)} if selected is not None else set()), 'Final fixed panel coverage')
        if selected is not None:
            for fixture in range(23):
                for k, kind in enumerate(('original', 'hidden_variant')):
                    self.row('qualification', f'{2*fixture+k:03d}', selected, fixture, kind)
        final = self.rows['qualification']
        gate = final_gate(selected, final)
        for stage, names in (('calibration', {p[:-5] for p in expected_files}),
                             ('qualification', {p[:-5] for p in qualification_files})):
            folder = self.base / 'workers' / stage
            actual = {p.name for p in folder.iterdir() if p.is_dir()} if folder.exists() else set()
            require(actual == names, 'Fresh worker folder coverage')
        expected = {'complete_schedule': True, **gate, 'selected_quota': selected,
                    'calibration_attempted': len(self.rows['calibration']), 'qualification_attempted': len(final),
                    'failure': None, 'strength_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.result.get(k)) == canonical(v) for k, v in expected.items()), 'Final qualification gate/counts')
        final_sources = read(self.base / 'final_input_check.json')
        require(final_sources['all_hashes_and_sizes_match'] is True and final_sources['inputs_checked'] == self.sources['files'], 'Final preservation receipt')
        self.boundary()
        elapsed = time.time() - self.launch['epoch_seconds']
        utc_elapsed = (datetime.fromisoformat(self.result['time']) - datetime.fromisoformat(self.launch['time'])).total_seconds()
        require(finite(self.result['elapsed_seconds']) and 0 < self.result['elapsed_seconds'] <= elapsed and
                abs(self.result['elapsed_seconds'] - utc_elapsed) < 3, 'Elapsed reconciliation')
        sizes = sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file())
        require(sizes < 256*1024**2 and 'torch' not in sys.modules, 'Evidence cap or model-free audit')
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
            'freeze_sha256': file_hash(self.base / 'freeze.json'), 'source_preservation': self.sources,
            'selection': selection, 'qualification': expected,
            'completed_dictionary_transitions': self.completed_transitions, 'reported_work_validated': dict(self.work),
            'per_stage': {stage: {'attempted': len(rows), 'statuses': dict(Counter(
                r['result']['status'] if r['clock']['status'] == 'ok' else r['clock']['status'] for r in rows)),
                'completed_receipt_seconds': self.descriptive([r['clock']['receipt_elapsed_seconds'] for r in rows if r['correct']])}
                for stage, rows in self.rows.items()},
            'resources': {'initialization_seconds': self.descriptive([r['receipt_elapsed_seconds'] for r in self.initializations]),
                'peak_sampled_combined_rss_bytes': max(r['telemetry']['peak_sampled_rss_bytes'] for r in self.initializations+self.clocks),
                'sum_last_observed_worker_cpu_seconds': sum(r['telemetry']['cpu_seconds'] for r in self.clocks),
                'including_final_analyzer_seconds': elapsed, 'evidence_bytes_before_this_receipt': sizes, 'sampled_not_continuous': True},
            'scope': {'no_models_or_neural_output_reexecution': True, 'no_internal_search_edge_replay': True,
                'corrected_root_action_and_dictionary_successor_audited': True,
                'fixed_three_world_pool_and_depth_64_unchanged': True, 'chance_universes_inactive_on_scalar_canonical_path': True,
                'auxiliary_single_choice_and_turn_124_heuristics_retained': True,
                'finite_panel_only': True, 'timeout_latencies_not_imputed': True, 'playing_strength_not_measured': True}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, default=BASE); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.base / 'independent_analysis.json'
    require(not destination.exists(), 'Never overwrite independent audit')
    audit = Audit(args.base); result = audit.run()
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    audit.boundary()
    require(sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file()) + len(payload.encode()) < 256*1024**2, 'Final evidence cap')
    with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    print(json.dumps({'status': 'passed', 'output': str(destination), 'qualification': result['qualification']}, sort_keys=True))


if __name__ == '__main__': main()
