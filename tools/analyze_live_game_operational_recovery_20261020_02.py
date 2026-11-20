"""Model-free, outcome-closed audit across three preserved operational clock segments."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from tools import analyze_live_game_operational_20261019 as core
from tools import analyze_live_game_operational_recovery_20261020 as first

ROOT = core.ROOT
BASE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
SOURCE = ROOT/'outputs/live_game_operational_recovery_20261020_01'
ORIGINAL = ROOT/'outputs/live_game_operational_20261019_01'
SOURCE_FREEZE_SHA256 = '3af87db0e61a9d014b4ec60e8b031f9a36a2b6f90214b559f2485f60b75ae716'
COMPLETED_GAMES = core.GAME_ORDER[:16]
PARTIAL_GAME, NEXT_PLY = 21, 11
MODULE = 'tools.live_game_operational_recovery_20261020_02.worker'
PRIOR_MODULE = first.MODULE
ORIGINAL_MODULE = 'tools.live_game_operational_20261019.worker'
CAP = 43200
require, read, digest, canonical = core.require, core.read, core.digest, core.canonical
artifact, file_hash, finite = core.artifact, core.file_hash, core.finite


def imported_move(game, ply):
    require(type(game) is int and game in core.GAMES and type(ply) is int and 0 <= ply < 400,
            'Recovery game/ply identity')
    return game in COMPLETED_GAMES or game == PARTIAL_GAME and ply < NEXT_PLY


def segment_name(game, ply):
    if first.original_move(game, ply): return 'original'
    return 'recovery01' if imported_move(game, ply) else 'recovery02'


def check_failure_attempt(row):
    require(set(row) == {'correct', 'failure_category', 'id', 'observer', 'player', 'status', 'total_seconds'} and
            row['correct'] is False and row['failure_category'] == 'ResourceHold' and
            row['id'] == 'games/g021/t0011' and row['player'] == 'opponent' and row['status'] == 'incomplete' and
            finite(row['total_seconds']) and row['total_seconds'] >= 0 and
            row['observer']['schema'] == 'external_screen_observation_v1' and
            row['observer']['observation']['turn_index'] == NEXT_PLY and
            row['observer']['observation']['current_player'] == (PARTIAL_GAME//2+PARTIAL_GAME%2)%2 and
            'context' not in row['observer'],
            'Preserved unstarted resource-hold decision')


def check_source_completion(completion, progress, execution, launch):
    require(completion['status'] == progress['status'] == 'preserved_hold' and
            completion['failure'] == {'category': 'ResourceHold', 'detail_path': 'sealed/failure.json'} and
            completion['completed_games'] == len(COMPLETED_GAMES) and completion['validation_passed'] is True and
            completion['worker_processes_closed'] is True and completion['outcomes_closed'] is True and
            completion['strength_analysis_games'] == 0 and completion['larger_data_deferred'] is True and
            {k: v for k, v in completion.items() if k not in ('progress', 'worker_processes_closed')} == progress,
            'Preserved first-attempt hold')
    require(type(execution['exit_code']) is int and execution['exit_code'] == 1 and
            execution['outcomes_closed'] is True and execution['runner_identity']['pid'] == launch['pid'] and
            finite(progress['elapsed_seconds']) and 0 < progress['elapsed_seconds'] < 86400 and
            progress['validation_reused'] is True and progress['new_validation_requests'] == 0 and
            progress['imported_completed_games'] == 5,
            'First-attempt failed runner receipt')
    utc = (datetime.fromisoformat(progress['time'])-datetime.fromisoformat(launch['time'])).total_seconds()
    require(abs(progress['elapsed_seconds']-utc) < 3, 'First-attempt elapsed reconciliation')


def check_memory_receipt(value, name, launch):
    initial = name == 'initial'
    reserve = 600 if name == 'analysis' or name.startswith('audits/') else 21600
    request = 0 if name == 'analysis' else 600 if name.startswith('audits/') else 120
    constants = {'schema': 'operational_memory_admission_v2', 'name': name, 'status': 'passed',
        'failure_category': None, 'failure_detail': None, 'initial': initial,
        'maximum_wait_seconds': 7200, 'maximum_cumulative_wait_seconds': 14400,
        'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
        'sample_interval_seconds': 10, 'stable_seconds': 120, 'required_consecutive_samples': 13,
        'reserve_seconds': reserve, 'request_seconds': request, 'initialization_seconds': 60,
        'no_worker_during_admission': True, 'outcomes_closed': True}
    require(all(canonical(value.get(k)) == canonical(v) for k, v in constants.items()), 'Fixed memory-wait contract')
    numeric = ('started_epoch_seconds', 'finished_epoch_seconds', 'started_monotonic', 'finished_monotonic',
               'elapsed_seconds', 'charged_seconds', 'cumulative_before_seconds', 'cumulative_after_seconds')
    require(all(finite(value[k]) for k in numeric), 'Finite memory-wait clocks')
    begin, end = value['started_monotonic'], value['finished_monotonic']
    wall_begin, wall_end = value['started_epoch_seconds'], value['finished_epoch_seconds']
    charge = max(end-begin, wall_end-wall_begin, 0.)
    require(value['elapsed_seconds'] == end-begin and value['charged_seconds'] == charge and
            0 <= charge < 7200 and 0 <= value['cumulative_before_seconds'] <= value['cumulative_after_seconds'] < 14400 and
            value['cumulative_after_seconds'] == value['cumulative_before_seconds']+charge and
            0 <= begin-launch['monotonic_start'] <= end-launch['monotonic_start'] < CAP and
            0 <= wall_begin-launch['epoch_seconds'] < CAP and
            0 <= wall_end-launch['epoch_seconds'] < CAP,
            'Unpaused bounded memory-wait clock charges')
    rows = value['readings']; require(type(rows) is list and 1 <= len(rows) <= 721, 'Bounded memory-reading inventory')
    needs = initial; streak = 0; first_good = None; previous = None
    for index, row in enumerate(rows):
        require(set(row) == {'epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'available_physical_bytes',
                'available_commit_bytes', 'admissible', 'consecutive_admissible_samples', 'remaining_seconds'},
                'Memory reading schema')
        mono, epoch = row['monotonic_seconds'], row['epoch_seconds']
        physical, commit = row['available_physical_bytes'], row['available_commit_bytes']
        require(all(finite(row[k]) for k in ('epoch_seconds', 'monotonic_seconds', 'elapsed_seconds', 'remaining_seconds')) and
                begin <= mono <= end and 0 <= epoch-launch['epoch_seconds'] < CAP and
                row['elapsed_seconds'] == mono-begin and
                (previous is None or mono-previous >= 9.95) and type(physical) is int and type(commit) is int and
                min(physical, commit) >= 0 and reserve+60+request <= row['remaining_seconds'] <= CAP,
                'Memory values, spacing and retained work reserve')
        good = physical >= 2*1024**3 and commit >= 4*1024**3
        needs = needs or not good; streak = streak+1 if good else 0
        first_good = (mono if streak == 1 else first_good) if good else None
        require(row['admissible'] is good and type(row['consecutive_admissible_samples']) is int and
                row['consecutive_admissible_samples'] == streak, 'Independent threshold and consecutive-good count')
        eligible = good and (not needs or streak >= 13 and mono-first_good >= 120)
        require(eligible is (index == len(rows)-1), 'First eligible fast/stable memory admission')
        previous = mono
    require(value['stability_required'] is needs and type(value['prior_admission_count']) is int and
            value['prior_admission_count'] >= 0, 'Memory stability and serial-count identity')
    return wall_end


class SegmentAudit(core.Audit):
    """The frozen helpers retain each segment's explicit directory and worker identity."""
    def __init__(self, base, owner, module, label):
        super().__init__(base)
        self.owner, self.module, self.label = owner, module, label

    def boundary(self):
        self.owner.boundary()

    def receipts(self, row, name, player, audit=False):
        folder = self.base/('audits' if audit else 'workers')/name
        require({p.name for p in folder.glob('request-*.json')} == {'request-00000001.json'},
                'Exactly one segment request per worker')
        init = read(self.bound(row['initialization_artifact'], folder/'initialization.json'))
        receipt = read(self.bound(row['receipt'], folder/'request-00000001.json'))
        module = 'tools.splendorust_120_qualification_20261018.worker' if player == 'opponent' else self.module
        core.check_initialization(init, row['initialization'], self.frozen, module)
        if self.label == 'recovery02':
            gate = 'audits/games/'+name.replace('-', '/') if audit else name
            passed = self.owner.memory_gate(row['memory_admission'], gate)
            require(init['identity']['created'] >= passed, 'New worker started only after its memory admission')
        elif self.label == 'recovery01':
            require(init['identity']['created'] >= self.owner.prior_admission_epoch, 'Prior worker started after its original admission')
        require({k: v for k, v in receipt.items() if k != 'result'} == row['clock'], 'Copied segment clock')
        self.check_clock(init, None, 60, self.cpu)
        self.check_clock(receipt, 'request-00000001', 600 if audit else 120, self.cpu)
        self.process_identity([init, receipt], player)
        self.initializations.append(init); self.clocks.append((player, audit, receipt))
        (self.audit_files if audit else self.worker_files).add(folder.resolve())
        return receipt


class Audit:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.old = SegmentAudit(ORIGINAL, self, ORIGINAL_MODULE, 'original')
        self.prior = SegmentAudit(SOURCE, self, PRIOR_MODULE, 'recovery01')
        self.new = SegmentAudit(self.base, self, MODULE, 'recovery02')
        self.segments = {'original': self.old, 'recovery01': self.prior, 'recovery02': self.new}
        self.imports = {}
        self.parent_imports = {}
        self.memory_paths = set()
        self.memory_values = {}
        self.roots = []; self.game_costs = []
        self.actual_transitions = 0
        self.local_recorded_worlds = self.local_recorded_continuations = 0
        self.resume_checked = False
        self.original_resume_checked = False
        self.segment_transitions = Counter()

    def boundary(self):
        elapsed = time.time()-self.launch['epoch_seconds']
        require(0 <= elapsed < CAP and
                0 <= time.monotonic()-self.launch['monotonic_start'] < CAP,
                'Fresh authorized second recovery twelve-hour cap')

    def bound(self, item, expected=None):
        return self.new.bound(item, expected)

    def copied(self, item, destination):
        destination = Path(destination).resolve()
        path = self.bound(item)
        if path != destination:
            allowed = self.imports.get(destination)
            require(allowed is not None, 'Declared immediate source copy')
            aliases = [allowed]
            nested = self.parent_imports.get((ROOT/allowed['path']).resolve())
            if nested is not None: aliases.append(nested)
            require(item in aliases, 'Only authenticated nested imported artifact aliases')
        require(artifact(destination)['sha256'] == item['sha256'] and
                destination.stat().st_size == item['bytes'], 'Byte-identical imported artifact')
        return destination

    def memory_gate(self, descriptor, name):
        path = self.bound(descriptor, self.base/'memory_admissions'/(name+'.json'))
        value = read(path); finished = check_memory_receipt(value, name, self.launch)
        log = path.with_suffix('.jsonl'); raw = log.read_bytes()
        require(value['readings_log'] == {'path': log.relative_to(self.base).as_posix(),
                'sha256': file_hash(log), 'bytes': len(raw)} and
                [json.loads(line) for line in raw.decode('utf-8').splitlines()] == value['readings'],
                'Immutable complete memory-reading log')
        self.memory_paths.add(path)
        require(name not in self.memory_values or self.memory_values[name] == value, 'Stable memory gate reference')
        self.memory_values[name] = value
        return finished

    def memory_inventory(self):
        import psutil
        passed = self.memory_gate(artifact(self.base/'memory_admissions/analysis.json'), 'analysis')
        require(psutil.Process().create_time() >= passed, 'Analyzer launched after analysis memory admission')
        ordered = sorted(self.memory_values.values(), key=lambda value: value['prior_admission_count'])
        require([v['prior_admission_count'] for v in ordered] == list(range(len(ordered))) and
                ordered[0]['name'] == 'initial' and ordered[-1]['name'] == 'analysis',
                'Exact serial memory gate inventory')
        previous = []
        for value in ordered:
            charge = sum(v['charged_seconds'] for v in sorted(previous, key=lambda row: row['name']+'.json'))
            require(math.isclose(value['cumulative_before_seconds'], charge, rel_tol=0., abs_tol=1e-6),
                    'Independent cumulative memory-wait charge')
            if previous:
                require(value['started_monotonic'] >= previous[-1]['finished_monotonic'],
                        'Nonoverlapping memory admissions')
            previous.append(value)
        require({p.resolve() for p in (self.base/'memory_admissions').rglob('*.json')} == self.memory_paths and
                {p.resolve() for p in (self.base/'memory_admissions').rglob('*.jsonl')} ==
                {p.with_suffix('.jsonl') for p in self.memory_paths}, 'No omitted or extra memory gates/logs')
        self.memory_summary = {'admissions': len(ordered), 'stability_required_admissions':
            sum(v['stability_required'] for v in ordered), 'charged_seconds': sum(v['charged_seconds'] for v in ordered)}

    def imports_and_lineage(self, frozen_items):
        plan = self.plan
        previous = first.Audit(SOURCE)
        previous.frozen = read(SOURCE/'freeze.json')
        previous.plan = read(SOURCE/'recovery_plan.json')
        previous.imports_and_lineage({(ROOT/i['path']).resolve(): i for i in previous.frozen['inputs']})
        self.parent_imports = previous.imports
        self.prior_plan = previous.plan
        self.original_lineage = previous.first_lineage
        self.original_progress, self.original_launch, self.original_runner = (
            previous.source_progress, previous.source_launch, previous.source_runner)
        self.old.frozen = previous.old.frozen
        self.original_failed_attempt = previous.failed_attempt
        require(plan['schema'] == 'live_operational_recovery02_plan_v1' and
                plan['source_base'] == SOURCE.relative_to(ROOT).as_posix() and
                plan['completed_games'] == list(COMPLETED_GAMES) and
                plan['partial_game']['game'] == PARTIAL_GAME and plan['partial_game']['next_ply'] == NEXT_PLY and
                plan['validation_reused'] is True, 'Exact authorized preserved prefix')
        for key, name in (('source_freeze', 'freeze.json'), ('source_launch', 'launch.json'),
                          ('source_completion', 'completion.json'), ('source_runner_execution', 'runner_execution.json')):
            self.bound(plan[key], SOURCE/name)
            require(frozen_items.get((SOURCE/name).resolve()) == plan[key], 'Prior control artifact frozen')
        require(file_hash(SOURCE/'freeze.json') == SOURCE_FREEZE_SHA256, 'Original frozen attempt identity')
        require(plan['original_freeze'] == artifact(ORIGINAL/'freeze.json') and
                file_hash(ORIGINAL/'freeze.json') == first.SOURCE_FREEZE_SHA256, 'Underlying original attempt identity')
        source_frozen = read(SOURCE/'freeze.json')
        require(all(frozen_items.get((ROOT/item['path']).resolve()) == item for item in source_frozen['inputs']),
                'All original frozen source identities preserved')
        snapshot_path = self.bound(plan['source_snapshot'], self.base/'source_snapshot.json')
        require(frozen_items.get(snapshot_path) == plan['source_snapshot'], 'Prospective first-attempt snapshot frozen')
        snapshot = read(snapshot_path)
        require(snapshot['schema'] == 'operational_recovery01_snapshot_v1' and
                snapshot['source_base'] == SOURCE.relative_to(ROOT).as_posix() and snapshot['outcomes_closed'] is True,
                'First-attempt snapshot identity')
        source_items = {}
        for item in snapshot['files']:
            path = self.bound(item)
            require(path.is_relative_to(SOURCE) and path not in source_items and frozen_items.get(path) == item,
                    'Unique immutable first-attempt evidence')
            source_items[path] = item
        require(set(source_items) == {p.resolve() for p in SOURCE.rglob('*')
                                     if p.is_file() and '__pycache__' not in p.parts},
                'Complete first-attempt evidence snapshot')
        for pair in plan['imports']:
            require(set(pair) == {'source', 'destination'}, 'Import descriptor schema')
            src = self.bound(pair['source']); dst = self.bound(pair['destination'])
            require(src in source_items and source_items[src] == pair['source'] and
                    dst.is_relative_to(self.base) and dst not in self.imports and
                    pair['source']['sha256'] == pair['destination']['sha256'] and
                    pair['source']['bytes'] == pair['destination']['bytes'] and
                    frozen_items.get(dst) == pair['destination'], 'Exact source-to-destination copy binding')
            self.imports[dst] = pair['source']
        preserved = plan['preserved_failure']
        required_failure = {origin/name for origin, failed_name in (
            (ORIGINAL, 'decision_attempts/games/g011/t0015.json'), (SOURCE, 'decision_attempts/games/g021/t0011.json'))
            for name in ('sealed/failure.json', 'completion.json', 'coordinator_hold.json',
                         'operational_progress.json', 'runner_execution.json', failed_name)}
        failure_paths = {self.bound(item) for item in preserved}
        require(required_failure <= failure_paths and all(p in frozen_items and
                (p.is_relative_to(SOURCE) or p.is_relative_to(ORIGINAL)) for p in failure_paths),
                'Original failures retained without replacement')
        failed = read(SOURCE/'decision_attempts/games/g021/t0011.json'); check_failure_attempt(failed)
        failed_folder = SOURCE/'workers/games/g021/t0011'
        require(not any(p.is_file() for p in failed_folder.rglob('*')),
                'Resumed ply had no original initialization or request')
        self.failed_attempt = failed
        require(self.imports.get((self.base/'preserved/incomplete_g021_t0011.json').resolve()) ==
                artifact(SOURCE/'decision_attempts/games/g021/t0011.json') and
                self.imports.get((self.base/'preserved/recovery01_failure.json').resolve()) ==
                artifact(SOURCE/'sealed/failure.json'), 'Separate preserved copies of the original hold')
        completion = read(SOURCE/'completion.json'); source_progress = read(SOURCE/'operational_progress.json')
        source_launch = read(SOURCE/'launch.json'); execution = read(SOURCE/'runner_execution.json')
        check_source_completion(completion, source_progress, execution, source_launch)
        require(completion['progress'] == artifact(SOURCE/'operational_progress.json') and
                source_launch['freeze'] == artifact(SOURCE/'freeze.json'), 'Original control bindings')
        process = read(SOURCE/'runner_process.json')
        require(execution['runner_identity'] == process['identity'] and execution['command'] == process['command'] ==
                [source_frozen['executable'], '-B', '-u', '-m', 'tools.live_game_operational_recovery_20261020.runner', 'run'] and
                process['freeze_sha256'] == SOURCE_FREEZE_SHA256, 'Original closed runner identity/command')
        final = read(SOURCE/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == len(source_frozen['inputs']),
                'Original final input check retained')
        require(self.frozen['executable'] == source_frozen['executable'] and
                self.frozen['topology'] == source_frozen['topology'], 'Same executable and assigned worker core')
        self.prior.frozen = source_frozen
        self.source_progress, self.source_launch, self.source_runner = source_progress, source_launch, execution
        self.first_lineage = {'source_freeze_sha256': SOURCE_FREEZE_SHA256,
            'source_snapshot_sha256': file_hash(snapshot_path), 'preserved_files': len(source_items),
            'completed_games_reused': list(COMPLETED_GAMES), 'partial_game': PARTIAL_GAME,
            'next_unstarted_ply': NEXT_PLY, 'validation_requests_reused': 12,
            'original_failure_retained': True, 'prior_recovery_status': 'preserved_hold',
            'prior_recovery_source_and_evidence_unchanged': True}

    def freeze(self):
        self.completion = read(self.base/'completion.json'); self.progress = read(self.base/'operational_progress.json')
        self.frozen = read(self.base/'freeze.json'); self.launch = read(self.base/'launch.json')
        self.plan = read(self.base/'recovery_plan.json')
        require(self.completion['status'] == 'completed_pending_independent_analysis' and
                self.completion['failure'] is None and self.completion['worker_processes_closed'] is True and
                self.completion['progress'] == artifact(self.base/'operational_progress.json') and
                {k: v for k, v in self.completion.items() if k not in ('worker_processes_closed', 'progress')} == self.progress,
                'Completed recovery supervisor/progress binding')
        require(self.launch['freeze'] == artifact(self.base/'freeze.json') and self.launch['maximum_seconds'] == CAP and
                self.launch['torch_loaded_in_supervisor'] is False and self.launch['outcomes_closed'] is True and
                finite(self.launch['epoch_seconds']) and finite(self.launch['monotonic_start']),
                'Fresh authorized recovery launch identity')
        self.boundary()
        frozen_items, sizes = {}, 0
        for item in self.frozen['inputs']:
            self.boundary(); path = self.bound(item)
            require(path not in frozen_items, 'Duplicate recovery frozen input')
            frozen_items[path] = item; sizes += item['bytes']
        constants = {'ruleset_id': core.RULESET, 'catalogue_sha256': core.CATALOGUE, 'local_profile_id': core.LOCAL_PROFILE,
            'games': 18, 'pairs': 9, 'seconds_per_move': 120, 'initialization_seconds': 60, 'audit_seconds': 600,
            'execution_cap_seconds': CAP, 'post_game_reserve_seconds': 21600, 'final_reserve_seconds': 600,
            'evidence_cap_bytes': 4*1024**3, 'minimum_free_bytes': 5*1024**3, 'max_process_tree_rss_bytes': 2*1024**3,
            'maximum_plies': 400, 'validation_requests': 12, 'sample_worlds': 8, 'audit_seed': core.AUDIT_SEED,
            'audit_domain': core.AUDIT_DOMAIN, 'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.frozen.get(k)) == canonical(v) for k, v in constants.items()), 'Unchanged operational gates')
        require(self.frozen['schema'] == 'live_operational_recovery02_freeze_v1' and
                self.frozen['source_freeze'] == artifact(SOURCE/'freeze.json') and
                self.frozen['source_snapshot'] == artifact(self.base/'source_snapshot.json') and
                self.frozen['validation_reused'] is True and self.frozen['new_validation_requests'] == 0 and
                self.frozen['imported_completed_games'] == 16 and
                self.frozen['recovery_plan'] == artifact(self.base/'recovery_plan.json') and
                self.frozen['opponent_profile'] == artifact(ROOT/core.OPPONENT_PROFILE) and
                file_hash(ROOT/core.OPPONENT_PROFILE) == core.OPPONENT_SHA256, 'Recovery plan and qualified opponent binding')
        self.imports_and_lineage(frozen_items)
        self.pilot_lineage = core.check_pilot_lineage(self.old.frozen['prior_pilot_lineage'], frozen_items)
        required = [self.base/name for name in ('recovery_plan.json', 'source_snapshot.json', 'local_profile.json',
                    'preflight_tests.json', 'schedule.json', 'validation_cases.json', 'seed_collision_check.json')]
        required += [Path(__file__), ROOT/'tests/test_live_game_operational_recovery_analyzer_20261020_02.py',
                     ROOT/'tools/operational_recovery_completion_20261020_02.py',
                     Path(core.__file__), ROOT/'docs/engineering/live_game_operational_recovery_20261020_02.md']
        required += list((ROOT/'tools/live_game_operational_recovery_20261020_02').glob('*.py'))
        require(all(p.resolve() in frozen_items for p in required), 'Recovery verifier and implementation frozen')
        require(read(self.base/'local_profile.json') == core.local_profile() and
                read(self.base/'schedule.json') == core.schedule(), 'Preserved player and paired schedule')
        core.check_preflight(read(self.base/'preflight_tests.json'))
        self.sources = {'files': len(frozen_items), 'bytes': sizes, 'all_hashes_and_sizes_match': True}
        for path in (ROOT, ROOT/'src'):
            if str(path) not in sys.path: sys.path.insert(0, str(path))
        from tools.external_strength_screen_20261018 import rules, audit as ledger
        from tools.corrected_planner_qualification_20261018 import verify as recorded
        from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics, initial_snapshot, descriptive
        require('torch' not in sys.modules and rules.RULESET_ID == core.RULESET and rules.CATALOGUE_SHA256 == core.CATALOGUE,
                'Model-free frozen corrected rule helpers')
        self.rules, self.ledger, self.initial_snapshot, self.descriptive = rules, ledger, initial_snapshot, descriptive
        self.catalogue = read(ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
        require(digest(self.catalogue) == core.CATALOGUE, 'Corrected catalogue semantic identity')
        for segment in self.segments.values():
            if segment is self.new: segment.frozen = self.frozen
            segment.profile = read(ROOT/core.OPPONENT_PROFILE)
            segment.cpu = self.frozen['topology']['selected_logical_processor']
            segment.rules, segment.ledger, segment.recorded = rules, ledger, recorded
            segment.check_clock, segment.check_diagnostics = check_clock, check_diagnostics
            segment.initial_snapshot, segment.descriptive, segment.catalogue = initial_snapshot, descriptive, self.catalogue
            segment.registry = read(ROOT/'outputs/proposal_breadth_arena_128/models_lock.json')
        self.old.seed_collisions(set(frozen_items)); self.seed_scope = self.old.seed_scope
        self.prior_admission_epoch = first.check_admission(read(SOURCE/'admission.json'), self.source_launch)
        self.memory_gate(artifact(self.base/'memory_admissions/initial.json'), 'initial')

    def validation(self):
        # No new validation policy execution: verify the entire original twelve-request evidence again.
        self.old.validation(); self.validation_counts = self.old.validation_counts
        for name in ('validation_cases.json', 'adapter_validation.json', 'local_profile.json', 'schedule.json', 'seed_collision_check.json'):
            require(self.imports.get((self.base/name).resolve()) == artifact(SOURCE/name), 'Reused validation/preparation copy')
        for index in range(12):
            for prefix in ('decision_attempts/validation/v', 'audit_rows/validation/v'):
                relative = f'{prefix}{index:03d}.json'
                require(self.imports.get((self.base/relative).resolve()) == artifact(SOURCE/relative),
                        'All validation decision/proof copies')
            relative = f'validation/rows/{index:03d}.json'
            require(self.imports.get((self.base/relative).resolve()) == artifact(SOURCE/relative), 'Validation summary copy')
        admission = read(self.base/'games_admission.json')
        require(admission['adapter_validation'] == artifact(self.base/'adapter_validation.json') and
                admission['outcomes_closed'] is True and admission['validation_reused'] is True and
                admission['memory_admission'] == artifact(self.base/'memory_admissions/initial.json') and
                21600 < admission['remaining_seconds'] <= CAP, 'Recovery admission reuses verified validation only')

    def game(self, spec):
        folder = self.base/'sealed/games'/f'{spec["game"]:03d}'
        initial = read(folder/'initial.json')
        snapshot = self.initial_snapshot(spec['seed'], self.catalogue,
            {'ruleset_id': core.RULESET, 'catalogue_sha256': core.CATALOGUE})
        require(initial == {**spec, 'initial': snapshot, 'initial_sha256': digest(snapshot)},
                'Independent unchanged paired initial world')
        self.ledger.validate_ledger(snapshot)
        game = read(folder/'result.json')
        require(all(game.get(k) == v for k, v in spec.items()) and game['initial_sha256'] == digest(snapshot) and
                1 <= len(game['moves']) <= 400, 'Combined schedule/initial/ply binding')
        require({p.name for p in (folder/'moves').glob('*.json')} == {f'{i:04d}.json' for i in range(len(game['moves']))},
                'Complete combined actual-move inventory')
        if spec['game'] in COMPLETED_GAMES:
            require(self.imports.get((folder/'result.json').resolve()) == artifact(SOURCE/'sealed/games'/f'{spec["game"]:03d}'/'result.json'),
                    'Completed games retained byte-for-byte')
        if spec['game'] in (*COMPLETED_GAMES, PARTIAL_GAME):
            require(self.imports.get((folder/'initial.json').resolve()) == artifact(SOURCE/'sealed/games'/f'{spec["game"]:03d}'/'initial.json'),
                    'Started initial worlds retained byte-for-byte')
        failure = None; costs = {label: Counter() for label in self.segments}
        for ply, descriptor in enumerate(game['moves']):
            self.boundary()
            require(not snapshot['state']['terminal'] and failure is None, 'No post-ending continuation')
            path = self.copied(descriptor, folder/'moves'/f'{ply:04d}.json'); row = read(path)
            imported = imported_move(spec['game'], ply); label = segment_name(spec['game'], ply)
            segment = self.segments[label]
            original_path = SOURCE/'sealed/games'/f'{spec["game"]:03d}'/'moves'/f'{ply:04d}.json'
            require((path.resolve() in self.imports) is imported, 'Exact immutable prefix versus new suffix')
            if imported:
                require(self.imports[path.resolve()] == artifact(original_path), 'Imported move source identity')
            actor = snapshot['state']['current_player']; player = 'local' if actor == spec['local_seat'] else 'opponent'
            require(canonical([row[k] for k in ('game', 'pair', 'fit', 'ply', 'actor', 'before_sha256')]) ==
                    canonical([spec['game'], spec['pair'], spec['fit'], ply, actor, digest(snapshot)]),
                    'Combined dictionary chain across segment boundary')
            if spec['game'] == PARTIAL_GAME and ply == NEXT_PLY:
                require(self.plan['partial_game']['state_sha256'] == digest(snapshot) and
                        row['observer'] == self.failed_attempt['observer'], 'Exact unstarted resumed position/observer')
                resume = read(self.base/'resume_check.json')
                require(all(resume.get(k) == v for k, v in {'game': PARTIAL_GAME, 'next_ply': NEXT_PLY,
                        'state_sha256': digest(snapshot), 'prefix_transitions': NEXT_PLY,
                        'no_completed_request_repeated': True, 'outcomes_closed': True}.items()),
                        'Bound independent reconstructed resume receipt')
                self.resume_checked = True
            if spec['game'] == 11 and ply == 15:
                require(label == 'recovery01' and self.prior_plan['partial_game']['state_sha256'] == digest(snapshot) and
                        row['observer'] == self.original_failed_attempt['observer'], 'Exact earlier authorized continuation boundary')
                self.original_resume_checked = True
            name = f'games/g{spec["game"]:03d}/t{ply:04d}'
            if imported:
                require(self.imports.get((self.base/'decision_attempts'/(name+'.json')).resolve()) ==
                        artifact(SOURCE/'decision_attempts'/(name+'.json')), 'Imported decision attempt identity')
            outcome = segment.decision(row, snapshot, name, player, 'games')
            alternate = core.hidden_variant(snapshot, self.catalogue)
            clean = {**{k: deepcopy(v) for k, v in row['observer'].items() if k != 'context'}, 'schema': self.rules.OBSERVATION_SCHEMA}
            self.ledger.check_observer(alternate, clean)
            require(finite(row['total_seconds']) and row['total_seconds'] >=
                    row['initialization']['elapsed_seconds']+row['clock']['elapsed_seconds']-.1,
                    'Segment decision wall-cost accounting')
            costs[label].update(attempted_decisions=1, decision_supervision_seconds=row['clock']['elapsed_seconds'],
                initialization_supervision_seconds=row['initialization']['elapsed_seconds'], decision_attempt_wall_seconds=row['total_seconds'])
            if outcome is None: failure = (row['status'], actor, player)
            else:
                wrapper, snapshot = outcome; self.actual_transitions += 1; self.segment_transitions[label] += 1
                if player == 'local':
                    support = len(wrapper['decision']['record']['candidates']['actions'])
                    self.local_recorded_worlds += 128; self.local_recorded_continuations += 128*support
                    self.roots.append({'id': f'g{spec["game"]:03d}-t{ply:04d}', 'game': spec['game'], 'ply': ply,
                        'observer': row['observer'], 'candidate_support': support, 'row': row, 'wrapper': wrapper, 'path': path})
        core.check_game_ending(game, snapshot, failure)
        combined = sum(costs.values(), Counter())
        self.game_costs.append({'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'],
            **dict(combined), 'segments': {key: dict(value) for key, value in costs.items()}})

    def inventories(self):
        require(self.resume_checked and self.original_resume_checked, 'Both exact authorized continuation boundaries observed')
        for origin, completed, partial, next_ply in ((ORIGINAL, first.COMPLETED_GAMES, 11, 15),
                                                    (SOURCE, COMPLETED_GAMES, PARTIAL_GAME, NEXT_PLY)):
            require({p.name for p in (origin/'sealed/games').iterdir() if p.is_dir()} ==
                    {f'{i:03d}' for i in (*completed, partial)}, 'Exact held-attempt game inventory')
            partial_folder = origin/'sealed/games'/f'{partial:03d}'
            require({p.name for p in (partial_folder/'moves').glob('*.json')} ==
                    {f'{i:04d}.json' for i in range(next_ply)} and not (partial_folder/'result.json').exists(),
                    'Each held partial game has exactly its preserved prefix')
        require({p.name for p in (self.base/'sealed/games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in core.GAMES},
                'Exact eighteen-game combined inventory')
        def at(paths, origin, destination): return {destination/p.relative_to(origin) for p in paths}
        old_at_prior = at(self.old.decision_files, ORIGINAL, SOURCE)
        all_imported = at(old_at_prior | self.prior.decision_files, SOURCE, self.base)
        require({p.resolve() for p in (ORIGINAL/'decision_attempts').rglob('*.json')} ==
                self.old.decision_files | {(ORIGINAL/'decision_attempts/games/g011/t0015.json').resolve()},
                'All original decisions and first unstarted hold retained')
        require({p.resolve() for p in (SOURCE/'decision_attempts').rglob('*.json')} ==
                old_at_prior | self.prior.decision_files | {(SOURCE/'decision_attempts/games/g021/t0011.json').resolve()},
                'All first-recovery decisions and second unstarted hold retained')
        require({p.resolve() for p in (self.base/'decision_attempts').rglob('*.json')} == all_imported | self.new.decision_files,
                'No discarded, replaced or extra combined decisions')
        for segment in self.segments.values():
            require({p.parent.resolve() for p in (segment.base/'workers').rglob('initialization.json')} == segment.worker_files and
                    {p.parent.resolve() for p in (segment.base/'audits').rglob('initialization.json')} == segment.audit_files,
                    'Exact segment worker inventory without reexecuted imports')
        old_names = {name for name, _, _ in self.old.proofs}
        require(not self.prior.proofs, 'No first-recovery game audits silently reused')
        for origin, names in ((ORIGINAL, old_names), (SOURCE, old_names),
                              (self.base, old_names | {name for name, _, _ in self.new.proofs})):
            require({p.resolve() for p in (origin/'audit_rows').rglob('*.json')} ==
                    {origin/'audit_rows'/(name+'.json') for name in names}, 'Complete three-segment proof inventory')
        require({p.name for p in (self.base/'validation/rows').glob('*.json')} == {f'{i:03d}.json' for i in range(12)},
                'Exact reused validation summary inventory')
        self.memory_inventory()

    def run(self):
        self.freeze(); self.validation()
        for spec in core.schedule(): self.game(spec)
        self.new.roots = self.roots
        self.new.local_recorded_worlds, self.new.local_recorded_continuations = self.local_recorded_worlds, self.local_recorded_continuations
        coverage = self.new.manifest(); self.inventories()
        expected = {'validation_passed': True, 'validation_reused': True, 'new_validation_requests': 0,
                    'imported_completed_games': 16, 'completed_games': 18, 'outcomes_closed': True,
                    'strength_analysis_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.progress.get(k)) == canonical(v) for k, v in expected.items()), 'Complete recovered operational coverage')
        final = read(self.base/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == self.sources['files'], 'Recovery final input check')
        identities = set().union(*(segment.identities for segment in self.segments.values()))
        require(sum(len(segment.identities) for segment in self.segments.values()) == len(identities), 'No cross-segment worker reuse')
        identities.update((value['runner_identity']['pid'], value['runner_identity']['created']) for value in (self.original_runner, self.source_runner))
        import psutil
        for pid, created in identities:
            try: require(psutil.Process(pid).create_time() != created, 'Authenticated segment process still alive')
            except psutil.NoSuchProcess: pass
        self.boundary(); elapsed = time.time()-self.launch['epoch_seconds']
        utc = (datetime.fromisoformat(self.progress['time'])-datetime.fromisoformat(self.launch['time'])).total_seconds()
        require(finite(self.progress['elapsed_seconds']) and 0 < self.progress['elapsed_seconds'] <= elapsed and
                abs(self.progress['elapsed_seconds']-utc) < 3, 'Fresh recovery elapsed reconciliation')
        sizes = sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file())
        require(sizes < 4*1024**3 and 'torch' not in sys.modules, 'Recovery evidence ceiling/model-free audit')
        statuses, latencies, work = {}, {}, Counter()
        for segment in self.segments.values():
            work.update(segment.work)
            for key, value in segment.statuses.items(): statuses.setdefault(key, Counter()).update(value)
            for key, value in segment.latencies.items(): latencies.setdefault(key, []).extend(value)
        for segment in (self.old, self.prior):
            require(segment.statuses.get('games:local', Counter()).get('forfeit_timeout', 0) == 2 and
                    sum(n for key, values in segment.statuses.items() if key.startswith('games:')
                        for status, n in values.items() if status != 'completed') == 2, 'All four preserved local timeout forfeits')
        clocks = [r for segment in self.segments.values() for _, _, r in segment.clocks]
        inits = [r for segment in self.segments.values() for r in segment.initializations]
        telemetry = [r['telemetry'] for r in inits+clocks]
        proof_costs = [read(segment.base/'audit_rows'/(name+'.json'))['total_seconds']
                       for segment in self.segments.values() for name, _, _ in segment.proofs]
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
            'prior_pilot_lineage': self.pilot_lineage, 'first_attempt_lineage': self.original_lineage,
            'prior_recovery_lineage': self.first_lineage,
            'memory_waiting': self.memory_summary,
            'freeze_sha256': file_hash(self.base/'freeze.json'), 'source_preservation': self.sources,
            'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True,
            'operational_schedule': {'games': 18, 'pairs': 9, 'fits': list(core.FITS), 'order': list(core.GAME_ORDER),
                'identical_worlds_with_reversed_seats_per_fit': True, 'actual_dictionary_transitions': self.actual_transitions,
                'segment_actual_dictionary_transitions': dict(self.segment_transitions)},
            'validation': {'requests': 12, 'pilot_transfer_requests': 6, 'new_context_requests': 6,
                'exact_hidden_state_repeat_pairs': 6, 'full_replay': self.validation_counts,
                'reused_from_first_attempt': True, 'new_validation_requests': 0},
            'actual_continuation_verification': coverage, 'seed_collision_check': self.seed_scope,
            'per_stage': {key: {'attempted': sum(values.values()), 'statuses': dict(values),
                'completed_receipt_seconds': self.descriptive(latencies.get(key, []))} for key, values in sorted(statuses.items())},
            'per_segment': {name: {key: {'attempted': sum(values.values()), 'statuses': dict(values),
                'completed_receipt_seconds': self.descriptive(segment.latencies.get(key, []))}
                for key, values in sorted(segment.statuses.items())} for name, segment in self.segments.items()},
            'reported_external_work_validated': dict(work), 'game_operational_costs': self.game_costs,
            'resources': {'initializations': len(inits), 'initialization_receipt_seconds':
                self.descriptive([r['receipt_elapsed_seconds'] for r in inits]),
                'audit_attempt_wall_seconds': self.descriptive(proof_costs), 'audit_wall_seconds_total': sum(proof_costs),
                'peak_sampled_combined_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
                'sum_last_observed_worker_cpu_seconds': sum(r['telemetry']['cpu_seconds'] for r in clocks),
                'authenticated_exited_process_identities': len(identities), 'worker_cleanup_checked': True,
                'production_elapsed_seconds': self.progress['elapsed_seconds'], 'including_final_analysis_seconds': elapsed,
                'original_attempt_elapsed_seconds': self.original_progress['elapsed_seconds'],
                'first_recovery_elapsed_seconds': self.source_progress['elapsed_seconds'],
                'sum_attempt_execution_seconds_including_analysis': self.original_progress['elapsed_seconds']+self.source_progress['elapsed_seconds']+elapsed,
                'between_attempt_wall_gap_not_charged_to_fresh_cap': True,
                'evidence_bytes_before_this_receipt': sizes, 'sampled_not_continuous': True},
            'scope': {'sealed_endings_internally_checked_without_outcome_summaries': True,
                'all_actual_actions_and_dictionary_successors_audited': True,
                'imported_prefix_and_new_suffix_audited_as_one_continuation': True,
                'original_timeout_forfeits_preserved': 4, 'unstarted_hold_not_scored_as_player_failure': True,
                'all128_worlds_and_recorded_outcome_arithmetic_checked_per_completed_local_decision': True,
                'raw_model_reexecution_performed_by_bound_audit_workers_not_this_analyzer': True,
                'unsampled_continuations_structurally_checked_by_bound_audit_workers_not_replayed': True,
                'external_neural_outputs_and_internal_search_edges_not_reexecuted': True,
                'finite_operational_exercise_not_strength_evidence': True, 'timeout_latencies_not_imputed': True,
                'no_additional_games_or_data_authorized_by_this_analysis': True}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, default=BASE); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.base/'independent_analysis.json'
    require(not destination.exists(), 'Never overwrite recovery independent analysis')
    try:
        audit = Audit(args.base); result = audit.run()
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
        audit.boundary()
        require(sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file())+len(payload.encode()) < 4*1024**3,
                'Final recovery evidence cap')
        with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(payload)
    except BaseException:
        folder = args.base/'sealed'; folder.mkdir(exist_ok=True)
        with (folder/'independent_analysis_failure_01.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump({'time': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(),
                       'outcomes_closed': True}, stream, sort_keys=True, indent=2)
        raise
    print(json.dumps({'status': 'passed', 'output': str(destination), 'outcomes_closed': True,
                      'operational_games_checked': 18, 'strength_analysis_games': 0}, sort_keys=True))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        print(json.dumps({'status': 'failed', 'category': type(error).__name__, 'outcomes_closed': True}), file=sys.stderr)
        raise SystemExit(1)
