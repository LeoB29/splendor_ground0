"""Model-free audit of one preserved operational prefix and its authorized continuation."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tools import analyze_live_game_operational_20261019 as core

ROOT = core.ROOT
BASE = ROOT/'outputs/live_game_operational_recovery_20261020_01'
SOURCE = ROOT/'outputs/live_game_operational_20261019_01'
SOURCE_FREEZE_SHA256 = '21ca9cb617ca65ad9745eeed76a2f5f91153dd638981c7fbfd76f4e348a27c51'
COMPLETED_GAMES = (6, 8, 10, 7, 9)
PARTIAL_GAME, NEXT_PLY = 11, 15
MODULE = 'tools.live_game_operational_recovery_20261020.worker'
ORIGINAL_MODULE = 'tools.live_game_operational_20261019.worker'
CAP = 86400
require, read, digest, canonical = core.require, core.read, core.digest, core.canonical
artifact, file_hash, finite = core.artifact, core.file_hash, core.finite


def original_move(game, ply):
    require(type(game) is int and game in core.GAMES and type(ply) is int and 0 <= ply < 400,
            'Recovery game/ply identity')
    return game in COMPLETED_GAMES or game == PARTIAL_GAME and ply < NEXT_PLY


def check_failure_attempt(row):
    require(set(row) == {'correct', 'failure_category', 'id', 'observer', 'player', 'status', 'total_seconds'} and
            row['correct'] is False and row['failure_category'] == 'ResourceHold' and
            row['id'] == 'games/g011/t0015' and row['player'] == 'local' and row['status'] == 'incomplete' and
            finite(row['total_seconds']) and row['total_seconds'] >= 0 and
            row['observer']['context'] == core.context_for(PARTIAL_GAME, NEXT_PLY),
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
            finite(progress['elapsed_seconds']) and 0 < progress['elapsed_seconds'] < CAP,
            'First-attempt failed runner receipt')
    utc = (datetime.fromisoformat(progress['time'])-datetime.fromisoformat(launch['time'])).total_seconds()
    require(abs(progress['elapsed_seconds']-utc) < 3, 'First-attempt elapsed reconciliation')


def check_admission(value, launch):
    expected = {'schema': 'operational_recovery_memory_admission_v1', 'status': 'passed', 'failure_category': None,
        'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
        'required_consecutive_samples': 4, 'sample_interval_seconds': 10, 'maximum_wait_seconds': 600,
        'outcomes_closed': True, 'no_player_requests_during_admission': True}
    require(all(canonical(value.get(k)) == canonical(v) for k, v in expected.items()),
            'Unchanged stable recovery memory admission')
    elapsed = value['elapsed_seconds']; rows = value['readings']; streak = 0; first_good = None; previous = None
    require(finite(elapsed) and 30 <= elapsed < 600 and 4 <= len(rows) <= 61, 'Bounded admission elapsed/readings')
    for index, row in enumerate(rows):
        require(set(row) == {'elapsed_seconds', 'available_physical_bytes', 'available_commit_bytes',
                            'admissible', 'consecutive_admissible_samples'}, 'Admission reading schema')
        stamp = row['elapsed_seconds']; physical = row['available_physical_bytes']; commit = row['available_commit_bytes']
        require(finite(stamp) and 0 <= stamp <= elapsed and type(physical) is int and type(commit) is int and
                min(physical, commit) >= 0 and (previous is None or stamp-previous >= 9.95),
                'Admission sample values and ten-second spacing')
        good = physical >= 2*1024**3 and commit >= 4*1024**3
        streak = streak+1 if good else 0
        first_good = (stamp if streak == 1 else first_good) if good else None
        require(row['admissible'] is good and type(row['consecutive_admissible_samples']) is int and
                row['consecutive_admissible_samples'] == streak, 'Independent admission threshold/streak')
        eligible = streak >= 4 and stamp-first_good >= 30
        require(eligible is (index == len(rows)-1), 'First qualifying stable memory window')
        previous = stamp
    passed_epoch = datetime.fromisoformat(value['time']).timestamp()
    require(finite(passed_epoch) and 0 <= passed_epoch-launch['epoch_seconds'] < CAP and
            passed_epoch-elapsed >= launch['epoch_seconds']-.1, 'Admission charged to fresh launch')
    return passed_epoch


class SegmentAudit(core.Audit):
    """The frozen helpers retain each segment's explicit directory and worker identity."""
    def __init__(self, base, owner, module):
        super().__init__(base)
        self.owner, self.module = owner, module

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
        if self.base == self.owner.base:
            require(init['identity']['created'] >= self.owner.admission_epoch, 'New worker started only after memory admission')
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
        self.old = SegmentAudit(SOURCE, self, ORIGINAL_MODULE)
        self.new = SegmentAudit(self.base, self, MODULE)
        self.imports = {}
        self.roots = []; self.game_costs = []
        self.actual_transitions = 0
        self.local_recorded_worlds = self.local_recorded_continuations = 0
        self.resume_checked = False
        self.segment_transitions = Counter()

    def boundary(self):
        elapsed = time.time()-self.launch['epoch_seconds']
        require(0 <= elapsed < CAP and
                0 <= time.monotonic()-self.launch['monotonic_start'] < CAP,
                'Fresh authorized recovery twenty-four-hour cap')

    def bound(self, item, expected=None):
        return self.new.bound(item, expected)

    def copied(self, item, destination):
        destination = Path(destination).resolve()
        path = self.bound(item)
        if path != destination:
            require(self.imports.get(destination) == item, 'Only authenticated imported artifact aliases')
        require(artifact(destination)['sha256'] == item['sha256'] and
                destination.stat().st_size == item['bytes'], 'Byte-identical imported artifact')
        return destination

    def imports_and_lineage(self, frozen_items):
        plan = self.plan
        require(plan['schema'] == 'live_operational_recovery_plan_v1' and
                plan['source_base'] == SOURCE.relative_to(ROOT).as_posix() and
                plan['completed_games'] == list(COMPLETED_GAMES) and
                plan['partial_game']['game'] == PARTIAL_GAME and plan['partial_game']['next_ply'] == NEXT_PLY and
                plan['validation_reused'] is True, 'Exact authorized preserved prefix')
        for key, name in (('source_freeze', 'freeze.json'), ('source_launch', 'launch.json'),
                          ('source_completion', 'completion.json'), ('source_runner_execution', 'runner_execution.json')):
            self.bound(plan[key], SOURCE/name)
            require(frozen_items.get((SOURCE/name).resolve()) == plan[key], 'Prior control artifact frozen')
        require(file_hash(SOURCE/'freeze.json') == SOURCE_FREEZE_SHA256, 'Original frozen attempt identity')
        source_frozen = read(SOURCE/'freeze.json')
        require(all(frozen_items.get((ROOT/item['path']).resolve()) == item for item in source_frozen['inputs']),
                'All original frozen source identities preserved')
        snapshot_path = self.bound(plan['source_snapshot'], self.base/'source_snapshot.json')
        require(frozen_items.get(snapshot_path) == plan['source_snapshot'], 'Prospective first-attempt snapshot frozen')
        snapshot = read(snapshot_path)
        require(snapshot['schema'] == 'operational_first_attempt_snapshot_v1' and
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
        required_failure = {SOURCE/name for name in ('sealed/failure.json', 'completion.json', 'coordinator_hold.json',
            'decision_attempts/games/g011/t0015.json')}
        failure_paths = {self.bound(item) for item in preserved}
        require(required_failure <= failure_paths and all(p in source_items for p in failure_paths),
                'Original failures retained without replacement')
        failed = read(SOURCE/'decision_attempts/games/g011/t0015.json'); check_failure_attempt(failed)
        failed_folder = SOURCE/'workers/games/g011/t0015'
        require(not any(p.is_file() for p in failed_folder.rglob('*')),
                'Resumed ply had no original initialization or request')
        self.failed_attempt = failed
        require(self.imports.get((self.base/'preserved/incomplete_g011_t0015.json').resolve()) ==
                artifact(SOURCE/'decision_attempts/games/g011/t0015.json') and
                self.imports.get((self.base/'preserved/first_attempt_failure.json').resolve()) ==
                artifact(SOURCE/'sealed/failure.json'), 'Separate preserved copies of the original hold')
        completion = read(SOURCE/'completion.json'); source_progress = read(SOURCE/'operational_progress.json')
        source_launch = read(SOURCE/'launch.json'); execution = read(SOURCE/'runner_execution.json')
        check_source_completion(completion, source_progress, execution, source_launch)
        require(completion['progress'] == artifact(SOURCE/'operational_progress.json') and
                source_launch['freeze'] == artifact(SOURCE/'freeze.json'), 'Original control bindings')
        process = read(SOURCE/'runner_process.json')
        require(execution['runner_identity'] == process['identity'] and execution['command'] == process['command'] ==
                [source_frozen['executable'], '-B', '-u', '-m', 'tools.live_game_operational_20261019.runner', 'run'] and
                process['freeze_sha256'] == SOURCE_FREEZE_SHA256, 'Original closed runner identity/command')
        final = read(SOURCE/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == len(source_frozen['inputs']),
                'Original final input check retained')
        require(self.frozen['executable'] == source_frozen['executable'] and
                self.frozen['topology'] == source_frozen['topology'], 'Same executable and assigned worker core')
        self.old.frozen = source_frozen
        self.source_progress, self.source_launch, self.source_runner = source_progress, source_launch, execution
        self.first_lineage = {'source_freeze_sha256': SOURCE_FREEZE_SHA256,
            'source_snapshot_sha256': file_hash(snapshot_path), 'preserved_files': len(source_items),
            'completed_games_reused': list(COMPLETED_GAMES), 'partial_game': PARTIAL_GAME,
            'next_unstarted_ply': NEXT_PLY, 'validation_requests_reused': 12,
            'original_failure_retained': True, 'original_attempt_status': 'preserved_hold',
            'original_source_and_evidence_unchanged': True}

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
        require(self.frozen['schema'] == 'live_operational_recovery_freeze_v1' and
                self.frozen['source_freeze'] == artifact(SOURCE/'freeze.json') and
                self.frozen['source_snapshot'] == artifact(self.base/'source_snapshot.json') and
                self.frozen['validation_reused'] is True and self.frozen['new_validation_requests'] == 0 and
                self.frozen['imported_completed_games'] == 5 and
                self.frozen['recovery_plan'] == artifact(self.base/'recovery_plan.json') and
                self.frozen['opponent_profile'] == artifact(ROOT/core.OPPONENT_PROFILE) and
                file_hash(ROOT/core.OPPONENT_PROFILE) == core.OPPONENT_SHA256, 'Recovery plan and qualified opponent binding')
        self.imports_and_lineage(frozen_items)
        self.pilot_lineage = core.check_pilot_lineage(self.old.frozen['prior_pilot_lineage'], frozen_items)
        required = [self.base/name for name in ('recovery_plan.json', 'source_snapshot.json', 'local_profile.json',
                    'preflight_tests.json', 'schedule.json', 'validation_cases.json', 'seed_collision_check.json')]
        required += [Path(__file__), ROOT/'tests/test_live_game_operational_recovery_analyzer_20261020.py',
                     ROOT/'tools/operational_recovery_completion_20261020.py',
                     Path(core.__file__), ROOT/'docs/engineering/live_game_operational_recovery_20261020.md']
        required += list((ROOT/'tools/live_game_operational_recovery_20261020').glob('*.py'))
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
        for segment in (self.old, self.new):
            if segment is self.new: segment.frozen = self.frozen
            segment.profile = read(ROOT/core.OPPONENT_PROFILE)
            segment.cpu = self.frozen['topology']['selected_logical_processor']
            segment.rules, segment.ledger, segment.recorded = rules, ledger, recorded
            segment.check_clock, segment.check_diagnostics = check_clock, check_diagnostics
            segment.initial_snapshot, segment.descriptive, segment.catalogue = initial_snapshot, descriptive, self.catalogue
            segment.registry = read(ROOT/'outputs/proposal_breadth_arena_128/models_lock.json')
        self.old.seed_collisions(set(frozen_items)); self.seed_scope = self.old.seed_scope
        self.admission = read(self.base/'admission.json')
        self.admission_epoch = check_admission(self.admission, self.launch)

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
        failure = None; costs = {'original': Counter(), 'recovery': Counter()}
        for ply, descriptor in enumerate(game['moves']):
            self.boundary()
            require(not snapshot['state']['terminal'] and failure is None, 'No post-ending continuation')
            path = self.copied(descriptor, folder/'moves'/f'{ply:04d}.json'); row = read(path)
            imported = original_move(spec['game'], ply); segment = self.old if imported else self.new
            label = 'original' if imported else 'recovery'
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
        combined = costs['original']+costs['recovery']
        self.game_costs.append({'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'],
            **dict(combined), 'segments': {key: dict(value) for key, value in costs.items()}})

    def inventories(self):
        require(self.resume_checked, 'Required exact continuation boundary observed')
        require({p.name for p in (SOURCE/'sealed/games').iterdir() if p.is_dir()} ==
                {f'{i:03d}' for i in (*COMPLETED_GAMES, PARTIAL_GAME)}, 'Exact first-attempt started game inventory')
        require({p.name for p in (SOURCE/'sealed/games/011/moves').glob('*.json')} ==
                {f'{i:04d}.json' for i in range(NEXT_PLY)} and not (SOURCE/'sealed/games/011/result.json').exists(),
                'Original partial game ended at the declared unstarted boundary')
        require({p.name for p in (self.base/'sealed/games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in core.GAMES},
                'Exact eighteen-game combined inventory')
        imported_decisions = {self.base/p.relative_to(SOURCE) for p in self.old.decision_files}
        require({p.resolve() for p in (self.base/'decision_attempts').rglob('*.json')} ==
                imported_decisions | self.new.decision_files, 'No discarded/replaced/extra combined decisions')
        require({p.resolve() for p in (SOURCE/'decision_attempts').rglob('*.json')} ==
                self.old.decision_files | {(SOURCE/'decision_attempts/games/g011/t0015.json').resolve()},
                'All original decisions retained including the unstarted hold')
        require({p.parent.resolve() for p in (SOURCE/'workers').rglob('initialization.json')} == self.old.worker_files and
                {p.parent.resolve() for p in (SOURCE/'audits').rglob('initialization.json')} == self.old.audit_files,
                'Original workers and validation audits wholly accounted')
        require({p.parent.resolve() for p in (self.base/'workers').rglob('initialization.json')} == self.new.worker_files and
                {p.parent.resolve() for p in (self.base/'audits').rglob('initialization.json')} == self.new.audit_files,
                'No duplicated imported workers or extra recovery workers')
        expected_proofs = {self.base/'audit_rows'/(name+'.json') for segment in (self.old, self.new) for name, _, _ in segment.proofs}
        require({p.resolve() for p in (self.base/'audit_rows').rglob('*.json')} == expected_proofs,
                'Exact reused-validation and fresh-game audit inventory')
        require({p.resolve() for p in (SOURCE/'audit_rows').rglob('*.json')} ==
                {SOURCE/'audit_rows'/(name+'.json') for name, _, _ in self.old.proofs}, 'No omitted original audits')
        require({p.name for p in (self.base/'validation/rows').glob('*.json')} == {f'{i:03d}.json' for i in range(12)},
                'Exact reused validation summary inventory')

    def run(self):
        self.freeze(); self.validation()
        for spec in core.schedule(): self.game(spec)
        self.new.roots = self.roots
        self.new.local_recorded_worlds, self.new.local_recorded_continuations = self.local_recorded_worlds, self.local_recorded_continuations
        coverage = self.new.manifest(); self.inventories()
        expected = {'validation_passed': True, 'validation_reused': True, 'new_validation_requests': 0,
                    'imported_completed_games': 5, 'completed_games': 18, 'outcomes_closed': True,
                    'strength_analysis_games': 0, 'larger_data_deferred': True}
        require(all(canonical(self.progress.get(k)) == canonical(v) for k, v in expected.items()), 'Complete recovered operational coverage')
        final = read(self.base/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == self.sources['files'], 'Recovery final input check')
        identities = self.old.identities | self.new.identities
        require(not self.old.identities & self.new.identities, 'No cross-segment worker reuse')
        identities.add((self.source_runner['runner_identity']['pid'], self.source_runner['runner_identity']['created']))
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
        for segment in (self.old, self.new):
            work.update(segment.work)
            for key, value in segment.statuses.items(): statuses.setdefault(key, Counter()).update(value)
            for key, value in segment.latencies.items(): latencies.setdefault(key, []).extend(value)
        old_timeouts = self.old.statuses.get('games:local', Counter()).get('forfeit_timeout', 0)
        require(old_timeouts == 2 and sum(n for key, values in self.old.statuses.items() if key.startswith('games:')
                    for status, n in values.items() if status != 'completed') == 2, 'Both original timeout forfeits preserved')
        clocks = [r for segment in (self.old, self.new) for _, _, r in segment.clocks]
        inits = self.old.initializations+self.new.initializations
        telemetry = [r['telemetry'] for r in inits+clocks]
        proof_costs = [read(segment.base/'audit_rows'/(name+'.json'))['total_seconds']
                       for segment in (self.old, self.new) for name, _, _ in segment.proofs]
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(), 'analyzer': artifact(__file__),
            'prior_pilot_lineage': self.pilot_lineage, 'first_attempt_lineage': self.first_lineage,
            'memory_admission': {'status': 'passed', 'readings': len(self.admission['readings']),
                'elapsed_seconds': self.admission['elapsed_seconds'], 'stable_window_seconds': 30,
                'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
                'all_new_workers_created_after_admission': True},
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
                for key, values in sorted(segment.statuses.items())} for name, segment in (('original', self.old), ('recovery', self.new))},
            'reported_external_work_validated': dict(work), 'game_operational_costs': self.game_costs,
            'resources': {'initializations': len(inits), 'initialization_receipt_seconds':
                self.descriptive([r['receipt_elapsed_seconds'] for r in inits]),
                'audit_attempt_wall_seconds': self.descriptive(proof_costs), 'audit_wall_seconds_total': sum(proof_costs),
                'peak_sampled_combined_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
                'sum_last_observed_worker_cpu_seconds': sum(r['telemetry']['cpu_seconds'] for r in clocks),
                'authenticated_exited_process_identities': len(identities), 'worker_cleanup_checked': True,
                'production_elapsed_seconds': self.progress['elapsed_seconds'], 'including_final_analysis_seconds': elapsed,
                'original_attempt_elapsed_seconds': self.source_progress['elapsed_seconds'],
                'sum_attempt_execution_seconds_including_analysis': self.source_progress['elapsed_seconds']+elapsed,
                'between_attempt_wall_gap_not_charged_to_fresh_cap': True,
                'evidence_bytes_before_this_receipt': sizes, 'sampled_not_continuous': True},
            'scope': {'sealed_endings_internally_checked_without_outcome_summaries': True,
                'all_actual_actions_and_dictionary_successors_audited': True,
                'imported_prefix_and_new_suffix_audited_as_one_continuation': True,
                'original_timeout_forfeits_preserved': 2, 'unstarted_hold_not_scored_as_player_failure': True,
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
