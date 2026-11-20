"""Independent amended recovery audit; original fixed-clock primary stays incomplete.

Scientific decision, referee, privacy, continuation and bootstrap checks reuse
the frozen analyzer. Only explicit provenance/orchestration boundaries differ.
No mutation of the original analyzer's globals or original evidence is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tools import analyze_matched_strength_20261020 as old

ROOT = old.ROOT
SOURCE = old.BASE
BASE = ROOT/'outputs/matched_strength_recovery_20261022_01'
SOURCE_FREEZE_SHA256 = '41244628326e23ea6a8685752f367bd4ca5e13809cfb47d983d0cd6f139b248e'
RECOVERY_MODULE = 'tools.matched_strength_recovery_20261022.worker'
OPPONENT_MODULE = 'tools.matched_strength_recovery_20261022.opponent_worker'
MAX_SECONDS = 43200
POST_RESERVE = 21600
FINAL_RESERVE = 600
ANALYSIS_SECONDS = 3600
PARTIAL_GAME, NEXT_PLY = 59, 43
require, canonical, read, artifact, file_hash, digest = old.require, old.canonical, old.read, old.artifact, old.file_hash, old.digest


def source_decision(name):
    """The one explicit retry starts at g059/t0043; no other origin is flexible."""
    if name.startswith('validation/'):
        require(name in {f'validation/v{i:03d}' for i in range(12)}, 'Fixed reused validation identity')
        return True
    import re
    match = re.fullmatch(r'games/g(\d{3})/t(\d{4})', name)
    require(match is not None, 'Canonical game request identity')
    game, ply = map(int, match.groups())
    require(game in range(60) and ply in range(400), 'Game request range')
    return game != PARTIAL_GAME or ply < NEXT_PLY


def missing_original_game_sensitivity(score_rows):
    """Identification bounds, not an imputed winner or confidence interval."""
    from tools.matched_strength_20261020.statistics import validated_games
    rows = validated_games(score_rows, 10)
    known = sum(row['local_score_halfpoints'] for row in rows if row['game'] != PARTIAL_GAME)
    return {'schema': 'missing_original_final_game_point_bounds_v1',
        'original_frozen_status': 'incomplete', 'missing_game': PARTIAL_GAME,
        'verified_original_completed_games': 59, 'known_score_halfpoints': known,
        'denominator_halfpoints': 120, 'lower_mean_score': known/120,
        'upper_mean_score': (known+2)/120, 'maximum_point_estimate_width': 1/60,
        'possible_missing_game_scores': [0, .5, 1],
        'recovered_game_result_used_for_original': False,
        'confidence_interval': None, 'interpretation':
            'Point-score identification range only; the original final game remains unknown.'}


def check_plan(plan):
    require(plan['schema'] == 'matched_strength_amended_recovery_plan_v1' and
            plan['source_base'] == SOURCE.relative_to(ROOT).as_posix() and
            canonical(plan['completed_games']) == canonical(list(old.GAME_ORDER[:-1])) and
            canonical([plan['partial_game']['game'],plan['partial_game']['next_ply']]) == canonical([PARTIAL_GAME,NEXT_PLY]) and
            type(plan['partial_game']['state_sha256']) is str and len(plan['partial_game']['state_sha256']) == 64 and
            canonical(plan['amendment']) == canonical({'retry_identity':'games/g059/t0043','new_allowance_seconds':120,
                'maximum_retries':1,'partial_search_reused':False,'original_frozen_status':'incomplete'}) and
            canonical(plan['preserved_timeout']) == canonical({'game':39,'ply':7,'status':'forfeit_timeout'}) and
            plan['original_primary_complete'] is False and plan['validation_reused'] is True and
            type(plan['new_validation_requests']) is int and plan['new_validation_requests'] == 0 and plan['outcomes_closed'] is True,
            'Exactly one authorized amended retry; original primary remains incomplete')


def check_recovery_gate_reserves(value, source_launch, launch):
    name = value['name']
    reserve = 600 if name in ('initial','analysis') else 4200 if name.startswith('audits/') else POST_RESERVE
    original_reserve = 64800 if reserve == POST_RESERVE else reserve
    request = 0 if name == 'initial' else 3600 if name == 'analysis' else 600 if name.startswith('audits/') else 120
    for origin,cap,protected in ((source_launch,259200,original_reserve),(launch,MAX_SECONDS,reserve)):
        remaining = min(origin['epoch_seconds']+cap-value['finished_epoch_seconds'],
                        origin['monotonic_start']+cap-value['finished_monotonic'])
        require(remaining >= protected+60+request, 'Both original and recovery work reserves survive admission')


def check_retry_receipt(value, plan, snapshot, observer):
    expected = {'amendment':plan['amendment'],'observer_sha256':digest(observer),
        'before_sha256':digest(snapshot),'preserved_original_request':plan['preserved_interrupted_request'],
        'original_primary_complete':False,'outcomes_closed':True}
    require(set(value) == set(expected)|{'time'} and
            canonical({k:v for k,v in value.items() if k != 'time'}) == canonical(expected),
            'Explicit amended retry receipt binds exact state and preserved partial work')


def runtime_source_receipt(sources):
    return {'schema':'corrected_runtime_sources_v1','ruleset_id':old.RULESET,
            'catalogue_sha256':old.CATALOGUE,'sources':sources}


def check_supervisor_runtime_sources(value, sources):
    require(canonical(value) == canonical(runtime_source_receipt(sources)),
            'Recovery supervisor corrected-runtime startup source receipt')


def check_launcher_receipts(launcher, process, launch, frozen, freeze_sha):
    owner, child = launcher['identity'], process['identity']
    require(launcher['freeze_sha256'] == process['freeze_sha256'] == freeze_sha and
            launcher['automatic_retries'] == process['automatic_retries'] == 0 and
            launcher['detachment_method'] == 'WMI_breakaway_no_window' and
            process['supervisor_identity'] == owner and owner['pid'] != child['pid'] and
            launch['identity'] == {'pid':child['pid'],'created':child['created']} and
            launch['pid'] == child['pid'], 'Authenticated amended launcher/runner identities')
    for identity in (owner,child):
        require(type(identity['pid']) is int and identity['pid'] > 0 and
                old.finite(identity['created']) and identity['created'] > 0 and
                identity['in_any_windows_job'] is False and
                identity['affinity'] == frozen['topology']['allowed_logical_processors'] and
                Path(identity['executable']).resolve() == Path(frozen['executable']).resolve() and
                Path(identity['command'][0]).resolve() == Path(frozen['executable']).resolve(),
                'Job-isolated qualified full-affinity supervisor identities')
    require(child['command'][1:] == ['-B','-m','tools.matched_strength_recovery_20261022.runner',
            'run','--freeze-sha256',freeze_sha] and owner['command'][1:2] == ['-B'] and
            Path(owner['command'][2]).resolve() == (ROOT/'tools/matched_strength_recovery_20261022/launch.py').resolve() and
            owner['command'][3:] == ['--freeze-sha256',freeze_sha], 'Exact one-shot amended launch commands')


class Audit(old.Audit):
    def __init__(self, base=BASE):
        super().__init__(base)
        self.output = self.base
        self.source = SOURCE.resolve()
        self.imports = {}
        self.segment_memory = {}
        self.segment_clocks = {'original': [], 'recovery': []}

    @contextmanager
    def at(self, base):
        previous = self.base
        self.base = Path(base).resolve()
        try:
            yield
        finally:
            self.base = previous

    def boundary(self):
        epoch, mono = time.time(), time.monotonic()
        require(all(0 <= now-launch[key] < cap-FINAL_RESERVE
                    for launch, cap in ((self.source_launch, old.CONFIG['execution_cap_seconds']), (self.launch, MAX_SECONDS))
                    for now, key in ((epoch, 'epoch_seconds'), (mono, 'monotonic_start'))),
                'Original72h and recovery12h dual caps with final reserve')
        require(shutil.disk_usage(ROOT).free >= 5*1024**3, 'Five-GiB disk reserve')

    def expected_origin(self, name, audit=False):
        original = name.startswith('validation/') if audit else source_decision(name)
        return self.source if original else self.output

    def bound(self, item, expected=None):
        # Imports are byte-identical metadata, with source and destination both
        # independently frozen. Return the destination when a stored old path
        # names the exact authenticated import expected by inherited game code.
        if expected is not None:
            expected = Path(expected).resolve()
            imported = self.imports.get(expected)
            if imported is not None and item == imported['source']:
                super().bound(imported['source'])
                return super().bound(imported['destination'], expected)
            if expected.is_relative_to(self.output):
                relative = expected.relative_to(self.output)
                parts = relative.parts
                if len(parts) >= 3 and parts[0] in ('workers', 'audits'):
                    name = '/'.join(parts[1:-1])
                    origin = self.expected_origin(name, audit=parts[0] == 'audits')
                    expected = origin/relative
        return super().bound(item, expected)

    def source_inventory(self):
        inventory = read(self.bound(self.plan['source_inventory']))
        require(inventory['schema'] == 'preserved_interrupted_benchmark_inventory_v1' and
                inventory['source'] == self.source.relative_to(ROOT).as_posix() and
                inventory['content_used_only_for_hashing'] is True and type(inventory['files']) is list,
                'Immutable interrupted source inventory')
        paths = set()
        for item in inventory['files']:
            self.boundary(); path = self.bound(item)
            require(path.is_relative_to(self.source) and path not in paths and not path.is_symlink(),
                    'Unique confined interrupted source file')
            require(self.frozen_items.get(path) == item, 'Whole interrupted evidence bound in recovery freeze')
            paths.add(path)
        require(paths == {p.resolve() for p in self.source.rglob('*') if p.is_file()},
                'No missing or added original evidence')
        require(inventory['total_bytes'] == sum(i['bytes'] for i in inventory['files']), 'Source inventory byte total')
        self.source_paths = paths
        self.source_inventory_value = inventory
        return inventory

    def verify_imports(self):
        imports = self.plan['imports']
        require(type(imports) is list, 'Explicit import inventory')
        destinations, sources = set(), set()
        for value in imports:
            require(set(value) == {'source', 'destination'}, 'Exact source/destination import')
            source, destination = super().bound(value['source']), super().bound(value['destination'])
            require(source in self.source_paths and source.is_relative_to(self.source) and
                    destination.is_relative_to(self.output) and
                    source.relative_to(self.source) == destination.relative_to(self.output) and
                    source not in sources and destination not in destinations and
                    value['source']['sha256'] == value['destination']['sha256'] and
                    value['source']['bytes'] == value['destination']['bytes'] and
                    self.frozen_items.get(destination) == value['destination'], 'Immutable exact metadata import')
            sources.add(source); destinations.add(destination); self.imports[destination] = value
        self.import_paths = destinations
        expected = {self.source/name for name in ('config.json','local_profile.json','validation_cases.json',
            'seed_collision_check.json','adapter_validation.json','sealed/schedule.json','sealed/referee_seed_material.json')}
        for directory in ('validation/rows','decision_attempts/validation','audit_rows/validation'):
            expected.update((self.source/directory).glob('*.json'))
        for game in old.GAMES:
            folder = self.source/'sealed/games'/f'{game:03d}'
            expected.add(folder/'initial.json')
            if game != PARTIAL_GAME: expected.add(folder/'result.json')
            for path in (folder/'moves').glob('*.json'):
                expected.add(path)
                expected.add(self.source/'decision_attempts/games'/f'g{game:03d}'/f't{path.stem}.json')
        require(sources == {p.resolve() for p in expected}, 'Exact original metadata import set')

    def freeze(self):
        self.frozen = read(self.output/'freeze.json'); self.launch = read(self.output/'launch.json')
        self.source_launch = read(self.source/'launch.json')
        self.source_frozen = read(self.source/'freeze.json')
        self.config = old.check_config(read(self.output/'config.json'))
        self.plan = read(self.output/'recovery_plan.json'); check_plan(self.plan)
        self.completion = read(self.output/'completion.json'); self.progress = read(self.output/'operational_progress.json')
        require(self.completion['status'] == 'completed_pending_independent_analysis' and
                self.completion['failure'] is None and self.completion['worker_processes_closed'] is True and
                self.completion['progress'] == artifact(self.output/'operational_progress.json') and
                {k:v for k,v in self.completion.items() if k not in ('worker_processes_closed', 'progress')} == self.progress,
                'Completed amended supervisor and progress')
        require(file_hash(self.source/'freeze.json') == SOURCE_FREEZE_SHA256 and
                self.plan['source_freeze'] == artifact(self.source/'freeze.json') and
                self.plan['source_launch'] == artifact(self.source/'launch.json') and
                self.source_launch['freeze'] == artifact(self.source/'freeze.json') and
                self.source_launch['maximum_seconds'] == 259200,
                'Original frozen launch retained')
        require(self.launch['freeze'] == artifact(self.output/'freeze.json') and self.launch['maximum_seconds'] == MAX_SECONDS and
                self.launch['source_launch'] == artifact(self.source/'launch.json') and
                self.launch['recovery_plan'] == artifact(self.output/'recovery_plan.json') and
                self.launch['torch_loaded_in_supervisor'] is False and self.launch['outcomes_closed'] is True and
                self.launch['epoch_seconds'] > self.source_launch['epoch_seconds'] and
                self.launch['monotonic_start'] > self.source_launch['monotonic_start'], 'Separate bounded amended launch')
        check_launcher_receipts(read(self.output/'launcher_identity.json'),read(self.output/'runner_process.json'),
                                self.launch,self.frozen,file_hash(self.output/'freeze.json'))
        self.boundary()
        self.frozen_items = {}
        for item in self.frozen['inputs']:
            self.boundary(); path = super().bound(item)
            require(path not in self.frozen_items, 'Unique recovery frozen input')
            self.frozen_items[path] = item
        require(all(self.frozen_items.get((ROOT/item['path']).resolve()) == item for item in self.source_frozen['inputs']),
                'Every original frozen dependency retained exactly')
        for key in ('topology', 'executable', 'ruleset_id', 'catalogue_sha256', 'local_profile_id', 'opponent_profile',
                    'games', 'pairs', 'seconds_per_move', 'initialization_seconds', 'audit_seconds',
                    'analysis_seconds', 'evidence_cap_bytes', 'minimum_free_bytes', 'max_process_tree_rss_bytes',
                    'maximum_plies', 'sample_worlds', 'audit_seed', 'audit_domain'):
            require(self.frozen[key] == self.source_frozen[key], 'Original players/rules/hardware/work contract preserved')
        require(self.frozen['schema'] == 'matched_strength_amended_recovery_freeze_v1' and
                self.frozen['recovery_plan'] == artifact(self.output/'recovery_plan.json') and
                self.frozen['source_freeze'] == artifact(self.source/'freeze.json') and
                self.frozen['recovery_max_seconds'] == MAX_SECONDS and
                self.frozen['recovery_post_game_reserve_seconds'] == POST_RESERVE and
                self.frozen['original_primary_complete'] is False, 'Explicit bounded recovery freeze')
        self.source_inventory(); self.verify_imports()
        review = read(self.bound(self.plan['source_review']))
        require(review['schema'] == 'matched_strength_interruption_review_v1' and review['status'] == 'inspection_complete' and
                review['freeze'] == artifact(self.source/'freeze.json') and review['original_files_modified'] == 0 and
                review['new_model_or_game_requests'] == 0 and review['matchup_outcomes_opened'] is False and
                review['interruption']['id'] == 'games/g059/t0043' and
                review['interruption']['classification'] == 'unresolved_active_request_infrastructure_interruption' and
                review['interruption']['timeout_forfeit_supported'] is False,
                'Preserved unresolved interruption classification')
        require((ROOT/self.plan['source_review']['path']).resolve() in self.frozen_items,
                'Interruption review prospectively bound')
        self.prior_lineage = self.accepted_history(self.frozen_items)
        self.pilot_lineage = old.check_pilot_lineage(self.frozen['prior_pilot_lineage'], self.frozen_items)
        required = [Path(__file__), self.output/'recovery_plan.json', self.output/'preflight_tests.json',
            ROOT/'tests/test_matched_strength_recovery_analyzer_20261022.py',
            ROOT/'docs/engineering/matched_strength_recovery_20261022.md',
            ROOT/'docs/engineering/matched_strength_recovery_20261022_review.md']
        required += list((ROOT/'tools/matched_strength_recovery_20261022').glob('*.py'))
        required += [ROOT/'tools/future_training_rules_20261021.py', ROOT/'docs/engineering/future_training_rules_20261021.md',
                     self.output/'prior_memory_state.json']
        require(all(p.resolve() in self.frozen_items for p in required), 'Recovery implementation/protocol/test coverage')
        old.check_preflight(read(self.output/'preflight_tests.json'))
        require(read(self.output/'local_profile.json') == old.local_profile() and
                self.frozen['opponent_profile'] == artifact(ROOT/old.OPPONENT_PROFILE) and
                file_hash(ROOT/old.OPPONENT_PROFILE) == old.OPPONENT_SHA256, 'Unchanged qualified entrants')
        self.profile = read(ROOT/old.OPPONENT_PROFILE)
        self.schedule = old.schedule(read(self.output/'sealed/referee_seed_material.json'))
        require(read(self.output/'sealed/schedule.json') == self.schedule == read(self.source/'sealed/schedule.json'),
                'Exact original private paired schedule')
        self.registry = read(ROOT/'outputs/proposal_breadth_arena_128/models_lock.json')
        self.cpu = self.frozen['topology']['selected_logical_processor']
        for path in (ROOT, ROOT/'src'):
            if str(path) not in sys.path: sys.path.insert(0, str(path))
        from tools.external_strength_screen_20261018 import rules, audit as ledger
        from tools.corrected_planner_qualification_20261018 import verify as recorded
        from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics, initial_snapshot, descriptive
        self.rules, self.ledger, self.recorded = rules, ledger, recorded
        self.check_clock, self.check_diagnostics = check_clock, check_diagnostics
        self.initial_snapshot, self.descriptive = initial_snapshot, descriptive
        require('torch' not in sys.modules and rules.RULESET_ID == old.RULESET and rules.CATALOGUE_SHA256 == old.CATALOGUE,
                'Model-free unchanged corrected rules')
        self.catalogue = read(ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
        require(digest(self.catalogue) == old.CATALOGUE, 'Corrected catalogue identity')
        from tools.matched_strength_20261020.statistics import runtime_contract
        from tools.future_training_rules_20261021 import verify_runtime_sources
        self.corrected_runtime_sources = verify_runtime_sources()
        check_supervisor_runtime_sources(read(self.output/'runtime_sources.json'), self.corrected_runtime_sources)
        self.statistics_runtime = read(ROOT/'outputs/matched_strength_20261020_preparation/statistics_runtime.json')
        require(runtime_contract() == self.statistics_runtime, 'Frozen statistical runtime')
        self.sources = {'files': len(self.frozen_items), 'bytes': sum(i['bytes'] for i in self.frozen_items.values()),
                        'all_hashes_and_sizes_match': True}
        self.check_original_structure()
        self.seed_collisions(set(self.frozen_items))

    def seed_collisions(self, paths):
        # Preserve the original prospectively scanned inventory. The sole new
        # schedule is an explicitly identical copy, not a newly sampled seed.
        value = read(self.source/'seed_collision_check.json')
        require(value == read(self.output/'seed_collision_check.json') and value['collisions'] == [] and
                value['skipped_oversize'] == [] and value['distinct_seeds'] == 30 and
                value['namespace'] == old.PRIVATE_NAMESPACE, 'Original no-collision seed receipt retained')
        targets = {int(s['seed']) for s in self.schedule}
        def contains(x):
            if type(x) is int: return x in targets
            if type(x) is str: return x.isdecimal() and int(x) in targets
            if type(x) is list: return any(contains(v) for v in x)
            if type(x) is dict: return any(contains(v) for v in x.values())
            return False
        checked = set()
        for item in value['checked']:
            self.boundary(); path = self.bound(item)
            require(path in paths and path not in checked and item['bytes'] <= 64*1024**2 and not contains(read(path)),
                    'Original independently scanned historical seeds')
            checked.add(path)
        self.seed_scope = {'checked_files':len(checked), 'original_inventory_preserved':True,
                           'new_seeds':0, 'same_original_thirty_pairs':True}

    def memory_gate(self, descriptor, name):
        path = super().bound(descriptor)
        candidates = [(base, label) for base,label in ((self.source,'original'),(self.output,'recovery'))
                      if path == base/'memory_admissions'/(name+'.json')]
        require(len(candidates) == 1, 'Memory gate exact origin')
        base, segment = candidates[0]; value = read(path)
        config = deepcopy(self.config); launch = self.source_launch
        if segment == 'recovery':
            config.update(execution_cap_seconds=MAX_SECONDS, post_game_reserve_seconds=POST_RESERVE)
            launch = self.launch
        finished = old.check_memory_receipt(value, name, launch, config)
        if segment == 'recovery':
            require(all(0 <= value[key]-self.source_launch[start] < 259200
                        for key,start in (('started_epoch_seconds','epoch_seconds'),('finished_epoch_seconds','epoch_seconds'),
                                          ('started_monotonic','monotonic_start'),('finished_monotonic','monotonic_start'))),
                    'Recovery waiting stays inside original clock')
            check_recovery_gate_reserves(value,self.source_launch,self.launch)
        log = path.with_suffix('.jsonl'); raw = log.read_bytes()
        require(value['readings_log'] == {'path':log.relative_to(base).as_posix(), 'sha256':file_hash(log), 'bytes':len(raw)} and
                [json.loads(line) for line in raw.decode().splitlines()] == value['readings'], 'Immutable memory log')
        key = (segment, name)
        require(key not in self.segment_memory or self.segment_memory[key] == value, 'Stable origin-qualified memory reference')
        self.segment_memory[key] = value; self.memory_paths.add(path)
        return finished

    def receipts(self, row, name, player, audit=False):
        base = self.expected_origin(name, audit=audit)
        folder = base/('audits' if audit else 'workers')/name
        require({p.name for p in folder.glob('request-*.json')} == {'request-00000001.json'}, 'Exactly one request per worker')
        init = read(super().bound(row['initialization_artifact'], folder/'initialization.json'))
        receipt = read(super().bound(row['receipt'], folder/'request-00000001.json'))
        module = (('tools.splendorust_120_qualification_20261018.worker' if player == 'opponent' else
                   'tools.matched_strength_20261020.worker') if base == self.source else
                  OPPONENT_MODULE if player == 'opponent' else RECOVERY_MODULE)
        old.check_initialization(init, row['initialization'], self.frozen, module)
        owner = (init['identity']['pid'],init['identity']['created'])
        require(owner not in self.worker_owners or self.worker_owners[owner] == folder.resolve(), 'No worker reuse across segments')
        self.worker_owners[owner] = folder.resolve()
        gate = 'audits/'+name if audit else name
        require(row['memory_admission']['path'] == (base/'memory_admissions'/(gate+'.json')).relative_to(ROOT).as_posix(),
                'Receipt and memory gate share required origin')
        passed = self.memory_gate(row['memory_admission'], gate)
        require(init['identity']['created'] >= passed, 'Worker follows origin-specific admission')
        require({k:v for k,v in receipt.items() if k != 'result'} == row['clock'], 'Unmodified copied clock')
        self.check_clock(init,None,60,self.cpu); self.check_clock(receipt,'request-00000001',600 if audit else 120,self.cpu)
        if base == self.output:
            runtime = read(super().bound(row['runtime_sources'],folder/'runtime_sources.json'))
            require(runtime == {'schema':'corrected_runtime_sources_v1','ruleset_id':old.RULESET,
                'catalogue_sha256':old.CATALOGUE,'sources':self.corrected_runtime_sources}, 'Fresh corrected-runtime source receipt')
        else: require('runtime_sources' not in row, 'Original receipt not relabelled as new startup')
        self.process_identity([init,receipt],player)
        self.initializations.append(init); self.clocks.append((player,audit,receipt))
        self.segment_clocks['original' if base == self.source else 'recovery'].append((player,audit,receipt))
        (self.audit_files if audit else self.worker_files).add(folder.resolve())
        return receipt

    def validation(self):
        with self.at(self.source): super().validation()
        admission = read(self.output/'games_admission.json')
        require(admission['validation_reused'] is True and admission['outcomes_closed'] is True and
                admission['adapter_validation'] == artifact(self.output/'adapter_validation.json') and
                old.finite(admission['remaining_seconds']) and POST_RESERVE < admission['remaining_seconds'] <= MAX_SECONDS,
                'New game admission explicitly reuses verified original validation')

    def game(self, spec):
        # All imported metadata is byte-identical; bound() resolves the old move
        # descriptors to their exact frozen copies in the combined game view.
        super().game(spec)

    def decision(self, row, snapshot, name, player, stage):
        if name == 'games/g059/t0043':
            require(player == 'local' and stage == 'games' and digest(snapshot) == self.plan['partial_game']['state_sha256'],
                    'Only authorized retry resumes exact reconstructed prefix')
            check_retry_receipt(read(self.output/'amended_retry.json'),self.plan,snapshot,row['observer'])
        return super().decision(row,snapshot,name,player,stage)

    def check_original_structure(self):
        require({p.parent.name for p in (self.source/'sealed/games').glob('*/result.json')} ==
                {f'{g:03d}' for g in old.GAME_ORDER[:-1]}, 'Exactly fifty-nine original final games')
        partial = self.source/'sealed/games/059'
        require({p.name for p in (partial/'moves').glob('*.json')} == {f'{i:04d}.json' for i in range(NEXT_PLY)} and
                not (partial/'result.json').exists(), 'Original final game remains incomplete at ply43')
        forbidden = ('completion.json','runner_execution.json','audit_manifest.json','independent_analysis.json',
                     'finalization.json','finalization_validation.json','completion_hold.json')
        require(not any((self.source/name).exists() for name in forbidden), 'Preserve original interrupted closure absence')
        folder = self.source/'workers/games/g059/t0043'
        require({p.name for p in folder.iterdir()} == {'initialization.json','events.jsonl','stdout.log','stderr.log','continuations.jsonl.gz'} and
                (folder/'events.jsonl').stat().st_size == 0 and
                not (self.source/'decision_attempts/games/g059/t0043.json').exists(), 'Unresolved original partial request retained')
        require(self.plan['preserved_interrupted_request'] == [artifact(p) for p in sorted(folder.iterdir())],
                'Every partial search artifact prospectively preserved')
        packets = [json.loads(line) for line in (folder/'stdout.log').read_bytes().splitlines()]
        require(len(packets) == 1 and packets[0]['type'] == 'ready', 'Original interrupted request has no completed response')
        init = read(folder/'initialization.json')
        saved = {k:v for k,v in init.items() if k not in old.METADATA and k != 'result'}
        old.check_initialization(init,saved,self.source_frozen,'tools.matched_strength_20261020.worker')
        self.check_clock(init,None,60,self.cpu); self.process_identity([init],'local')
        passed = self.memory_gate(artifact(self.source/'memory_admissions/games/g059/t0043.json'),'games/g059/t0043')
        require(init['identity']['created'] >= passed, 'Original partial worker followed original admission')
        owner = (init['identity']['pid'],init['identity']['created'])
        self.worker_owners[owner] = folder.resolve()
        self.partial_worker = folder.resolve()
        self.partial_initialization = init
        # No timing inference or completed response is fabricated for this search.

    def memory_inventory(self):
        import psutil
        for base in (self.source,self.output):
            self.memory_gate(artifact(base/'memory_admissions/initial.json'),'initial')
        passed = self.memory_gate(artifact(self.output/'memory_admissions/analysis.json'),'analysis')
        require(psutil.Process().create_time() >= passed, 'Final analyzer follows fresh memory admission')
        ordered = []
        for segment in ('original','recovery'):
            values = sorted((v for (s,_),v in self.segment_memory.items() if s == segment),key=lambda v:v['prior_admission_count'])
            require(values and values[0]['name'] == 'initial', 'Initial admission in both segments')
            if segment == 'recovery': require(values[-1]['name'] == 'analysis', 'Final recovery admission is analyzer')
            for value in values:
                require(value['prior_admission_count'] == len(ordered), 'Inherited serial admission count')
                charge = sum(v['charged_seconds'] for v in ordered)
                require(math.isclose(value['cumulative_before_seconds'],charge,rel_tol=0,abs_tol=1e-5),
                        'Original RAM charges are not refunded')
                if ordered: require(value['started_monotonic'] >= ordered[-1]['finished_monotonic'], 'Serial admission order')
                ordered.append(value)
        prior = read(self.output/'prior_memory_state.json')
        originals = [v for (s,_),v in self.segment_memory.items() if s == 'original']
        original_paths = sorted(p for p in self.memory_paths if p.is_relative_to(self.source))
        require(prior == {'schema':'matched_strength_prior_memory_v1','admission_count':len(originals),
            'charged_seconds':sum(read(p)['charged_seconds'] for p in original_paths),
            'inputs':[artifact(p) for p in original_paths]}, 'Frozen original RAM seed independently reconstructed')
        for base in (self.source,self.output):
            actual = {p.resolve() for p in (base/'memory_admissions').rglob('*.json')}
            require(actual == {p for p in self.memory_paths if p.is_relative_to(base)} and
                    {p.resolve() for p in (base/'memory_admissions').rglob('*.jsonl')} == {p.with_suffix('.jsonl') for p in actual},
                    'Complete original and amended memory inventories')
        self.memory_summary = {'admissions':len(ordered),'stability_required_admissions':sum(v['stability_required'] for v in ordered),
                               'charged_seconds':sum(v['charged_seconds'] for v in ordered),'original_charges_retained':True}

    def check_inventories(self):
        require({p.name for p in (self.output/'sealed/games').iterdir() if p.is_dir()} == {f'{g:03d}' for g in old.GAMES},
                'Combined exact sixty-game inventory')
        expected_decisions = {self.output/p.relative_to(self.source) if p.is_relative_to(self.source) else p for p in self.decision_files}
        require({p.resolve() for p in (self.output/'decision_attempts').rglob('*.json')} == expected_decisions,
                'Exact combined decisions without duplicate amended attempt')
        original_workers = {p for p in self.worker_files if p.is_relative_to(self.source)} | {self.partial_worker}
        new_workers = self.worker_files-original_workers
        for base,expected in ((self.source,original_workers),(self.output,new_workers)):
            require({p.parent.resolve() for p in (base/'workers').rglob('initialization.json')} == expected,
                    'Exact original/new worker inventory including interrupted worker')
        for base in (self.source,self.output):
            require({p.parent.resolve() for p in (base/'audits').rglob('initialization.json')} ==
                    {p for p in self.audit_files if p.is_relative_to(base)}, 'Exact audit worker inventory')
        expected_proofs = {self.output/'audit_rows'/(name+'.json') for name,_,_ in self.proofs}
        require({p.resolve() for p in (self.output/'audit_rows').rglob('*.json')} == expected_proofs,
                'Exact combined proof-row inventory')

    def run(self):
        self.freeze(); self.validation()
        for spec in self.schedule: self.game(spec)
        coverage = self.manifest(); self.memory_inventory(); self.check_inventories()
        require(all(self.progress.get(k) == v for k,v in {'validation_passed':True,'completed_games':60,'outcomes_closed':True,
            'strength_analysis_games':0,'larger_data_deferred':True,'validation_reused':True,'new_validation_requests':0,
            'imported_completed_games':59,'original_primary_complete':False}.items()), 'Completed amended coverage only')
        final = read(self.output/'final_input_check.json')
        require(final['all_hashes_and_sizes_match'] is True and final['files'] == self.sources['files'], 'Final recovery source check')
        import psutil
        outer = read(self.source/'runner_process.json')
        for identity in (self.source_launch['identity'],{'pid':outer['supervisor_pid'],'created':outer['supervisor_created']}):
            self.identities.add((identity['pid'],identity['created']))
        for pid,created in self.identities:
            try: require(psutil.Process(pid).create_time() != created, 'Authenticated old/new worker or original supervisor still alive')
            except psutil.NoSuchProcess: pass
        # The known earlier timeout is mandatory, separate from the interrupted search.
        earlier = read(self.source/'decision_attempts/games/g039/t0007.json')
        require(earlier['status'] == 'forfeit_timeout' and earlier['player'] == 'local', 'Original genuine timeout remains a loss')
        require(read(self.output/'decision_attempts/games/g039/t0007.json') == earlier, 'No timeout replacement')
        old_failures = [r['status'] for player,audit,r in self.segment_clocks['original'] if not audit and r['status'] != 'ok']
        require(old_failures == ['timeout'], 'Exactly the preserved original genuine timeout remains attributed')
        self.boundary()
        sizes = sum(p.stat().st_size for base in (self.source,self.output) for p in base.rglob('*') if p.is_file())
        require(sizes < old.CONFIG['evidence_cap_bytes']-32*1024**2 and 'torch' not in sys.modules, 'Combined evidence/model-free cap')
        elapsed = time.time()-self.launch['epoch_seconds']
        utc_elapsed = (datetime.fromisoformat(self.progress['time'])-datetime.fromisoformat(self.launch['time'])).total_seconds()
        require(old.finite(self.progress['elapsed_seconds']) and 0 < self.progress['elapsed_seconds'] <= elapsed and
                abs(self.progress['elapsed_seconds']-utc_elapsed) < 3, 'Distinct recovery segment accounting')
        from tools.matched_strength_20261020.statistics import statistics, runtime_contract
        proof_costs = [read(self.expected_origin(name,audit=True)/'audit_rows'/(name+'.json'))['total_seconds']
                       for name,_,_ in self.proofs]
        telemetry = [r['telemetry'] for r in self.initializations]+[r['telemetry'] for _,_,r in self.clocks]
        # This is the first statistical opening, after every combined scientific gate.
        result = {'schema':'matched_strength_amended_analysis_v1','status':'passed_amended',
            'study_label':'amended_recovered_benchmark',
            'original_frozen_status':'incomplete','original_primary_complete':False,'final_acceptance':False,
            'outcomes_opened':True,'outcomes_closed':False,'completed_games':60,'completed_pairs':30,
            'strength_analysis_games':60,'larger_data_deferred':True,'freeze_sha256':file_hash(self.output/'freeze.json'),
            'analyzer':artifact(__file__),'recovery_plan':artifact(self.output/'recovery_plan.json'),
            'corrected_runtime_sources':runtime_source_receipt(self.corrected_runtime_sources),
            'supervisor_runtime_sources':artifact(self.output/'runtime_sources.json'),
            'launcher_identity':artifact(self.output/'launcher_identity.json'),
            'runner_process':artifact(self.output/'runner_process.json'),
            'source_preservation':self.sources,'prior_pilot_lineage':self.pilot_lineage,'prior_operational_lineage':self.prior_lineage,
            'amendment':{'retry_identity':'games/g059/t0043','additional_attempts_authorized':1,
                'fresh_seconds':120,'original_partial_preserved':True},
            'interruption':{'game':59,'ply':43,
                'original_interruption_classification':'unresolved_infrastructure_interruption',
                'original_interruption_not_relabelled_timeout':True,'original_genuine_timeout_losses_retained':1},
            'validation':{'requests':12,'reused':True,'new_requests':0,'full_replay':self.validation_counts},
            'actual_continuation_verification':coverage,'seed_collision_check':self.seed_scope,
            'memory_waiting':self.memory_summary,
            'per_stage':{key:{'attempted':sum(values.values()),'statuses':dict(values),
                'completed_receipt_seconds':self.descriptive(self.latencies.get(key,[]))} for key,values in sorted(self.statuses.items())},
            'game_operational_costs':self.game_costs,'reported_external_work_validated':dict(self.work),
            'resources':{'recovery_production_elapsed_seconds':self.progress['elapsed_seconds'],
                'recovery_including_final_analysis_seconds':elapsed,
                'original_elapsed_including_gap_and_recovery_seconds':time.time()-self.source_launch['epoch_seconds'],
                'combined_evidence_bytes_before_report':sizes,'authenticated_exited_process_identities':len(self.identities),
                'worker_cleanup_checked':True,'original_clock_not_reset':True,'sampled_not_continuous':True,
                'audit_wall_seconds_total':sum(proof_costs),'audit_attempt_wall_seconds':self.descriptive(proof_costs),
                'initialization_receipt_seconds':self.descriptive([r['receipt_elapsed_seconds'] for r in self.initializations]),
                'peak_sampled_combined_rss_bytes':max(t['peak_sampled_rss_bytes'] for t in telemetry),
                'sum_last_observed_worker_cpu_seconds':sum(r['telemetry']['cpu_seconds'] for _,_,r in self.clocks),
                'per_segment':{segment:{'request_count':len(values),
                    'request_supervision_seconds':sum(r['elapsed_seconds'] for _,_,r in values)}
                    for segment,values in self.segment_clocks.items()}},
            'statistics':statistics(self.score_rows,pairs_per_fit=10,bootstrap_seed=202610206701,boundary=self.boundary),
            'original_missing_game_sensitivity':missing_original_game_sensitivity(self.score_rows),
            'statistics_runtime':self.statistics_runtime,
            'scope':{'amended_recovered_sample_only':True,'not_original_fixed_clock_primary_acceptance':True,
                'all_actual_moves_and_endings_independently_checked':True,'all_combined_manifest_checks_passed':True,
                'unsampled_continuations_not_replayed':True,'external_neural_search_not_reexecuted':True,
                'no_further_retry_or_sample_extension_authorized':True}}
        for item in self.frozen['inputs']: self.boundary(); self.bound(item)
        self.source_inventory()
        require(runtime_contract() == self.statistics_runtime, 'Statistical runtime preserved after analysis')
        result['source_preservation']['post_analysis_all_hashes_and_sizes_match'] = True
        self.boundary(); result['time'] = datetime.now(timezone.utc).isoformat()
        result['resources']['recovery_including_final_analysis_seconds'] = time.time()-self.launch['epoch_seconds']
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--output',type=Path)
    args=parser.parse_args();destination=args.output or args.base/'independent_analysis.json'
    require(args.base.resolve() == BASE.resolve() and destination.resolve() == (BASE/'independent_analysis.json').resolve(),
            'Fixed separate recovery analysis output')
    require(not destination.exists(),'Never overwrite amended analysis')
    audit=Audit(args.base)
    try:
        result=audit.run();payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n';audit.boundary()
        size=sum(p.stat().st_size for base in (SOURCE,BASE) for p in base.rglob('*') if p.is_file())
        require(size+len(payload.encode()) < old.CONFIG['evidence_cap_bytes']-32*1024**2,'Final combined evidence cap')
        with destination.open('x',encoding='utf-8',newline='\n') as stream:stream.write(payload)
    except BaseException:
        folder=BASE/'sealed';folder.mkdir(exist_ok=True)
        with (folder/'independent_analysis_failure_01.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump({'time':datetime.now(timezone.utc).isoformat(),'traceback':traceback.format_exc(),
                       'outcomes_closed':True,'original_primary_complete':False},stream,sort_keys=True,indent=2)
        raise
    print(json.dumps({'status':'passed_amended','original_frozen_status':'incomplete',
                      'output':str(destination),'outcomes_closed':False,'games_checked':60},sort_keys=True))


if __name__ == '__main__':
    try:main()
    except BaseException as error:
        print(json.dumps({'status':'failed','category':type(error).__name__,'outcomes_closed':True}),file=sys.stderr)
        raise SystemExit(1)
