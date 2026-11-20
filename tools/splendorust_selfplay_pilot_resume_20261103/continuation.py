"""Immutable source-prefix admission for the explicitly requested continuation.

The old job remains incomplete.  Its completed games are checked through its
own frozen readers; no old module, output path or clock is rebound.
"""
import argparse
import sys
import time
from pathlib import Path
import psutil
from . import common as c

OLD_OUT = c.ROOT / 'outputs/splendorust_selfplay_pilot_20261102_02'
OLD_LAUNCH = c.ROOT / 'outputs/splendorust_selfplay_pilot_launch_budget_20261102_01'
NEW_OUT = c.ROOT / 'outputs/splendorust_selfplay_pilot_resume_20261103_01'
OLD_FREEZE_SHA = '1059e7dcb1657cba988c0a40ca6d051f90e31a48b8bbf1b687bc011a8dfee8e6'
QUALIFIED_FREEZE_SHA = '79763d252c700dcd8159b4b71ddd19a93434bc2dff5a3bd54a9f4dac5ba53c80'
ACCEPTANCE_SHA = 'd47a4a087133ba66820d37afc4813f98429c1d6379d7b06d3266acaed77b1352'
COMPLETED_GAMES = 32
PREFIX_NAMES = ('capture_off', 'capture_on', 'collect_001', 'collect_002', 'collect_003')
MANIFEST = NEW_OUT / 'continuation_manifest.json'
PREFIX_REVIEW = NEW_OUT / 'prelaunch/saved_prefix_review.json'


def old_freeze():
    c.require(c.file_hash(OLD_OUT/'freeze.json') == OLD_FREEZE_SHA, 'Exact interrupted pilot freeze')
    value = c.read(OLD_OUT/'freeze.json')
    c.require(value['scope'] == 'pilot' and value['source_games'] == 256 and
              value['teacher_simulations'] == 512, 'Original fixed pilot science')
    return value


def prefix_game_path(index):
    c.require(type(index) is int and 0 <= index < COMPLETED_GAMES, 'Only complete original source prefix')
    return OLD_OUT/'games'/f'g{index:03d}'/'game.json'


def prefix_source_root(name):
    c.require(name in PREFIX_NAMES, 'Only the five complete original source lifecycles')
    return OLD_OUT


def source_continuation():
    freeze = old_freeze()
    return dict(schema='selfplay_saved_source_prefix_v1', completed_games=COMPLETED_GAMES,
                original_freeze=c.artifact(OLD_OUT/'freeze.json'), original_schedule=freeze['game_schedule'],
                games=[c.artifact(prefix_game_path(i)) for i in range(COMPLETED_GAMES)])


def _identities():
    from tools.splendorust_selfplay_pilot_budget_20261102.workflow import identities
    documents = [c.read(OLD_OUT/name) for name in ('guardian_claim.json','watchdog_claim.json','launch.json',
                  'guardian_owner.json','execution.json','watchdog_execution.json')]
    documents += [c.read(OLD_LAUNCH/name) for name in ('claim.json','pilot_guardian/owner.json',
                  'pilot_guardian/exit.json','pilot_admission/owner.json','pilot_admission/exit.json')]
    for name in PREFIX_NAMES:
        documents += [c.read(OLD_OUT/'workers'/name/file) for file in ('initialization.json','cleanup.json')]
    return [dict(pid=pid, created=created) for pid, created in sorted(identities(documents))]


def require_absent(identities):
    for item in identities:
        try:
            process = psutil.Process(item['pid'])
        except psutil.NoSuchProcess:
            continue
        c.require(process.create_time() != item['created'], 'Exact prior process remains alive')


def verify_stop():
    freeze = old_freeze()
    completion = c.read(OLD_OUT/'completion.json')
    failure = c.read(OLD_OUT/'failure.json')
    c.require(completion['status'] == 'incomplete' and completion['completed_games'] == 32 and
              completion['freeze_sha256'] == OLD_FREEZE_SHA and completion['automatic_retries'] == 0 and
              completion['coverage'] is None and failure['stage'] == 'collection' and
              'Remaining fixed source sample and downstream reserves are unaffordable' in failure['error'],
              'Only the clean resource-forecast stop may continue')
    c.require({p.name for p in (OLD_OUT/'games').iterdir()} == {f'g{i:03d}' for i in range(32)} and
              {p.name for p in (OLD_OUT/'workers').iterdir()} == set(PREFIX_NAMES),
              'Exact complete old game and worker prefix; no active next game')
    c.require(not any((OLD_OUT/name).exists() for name in
              ('fit','fit_execution.json','corpus_manifest.json','replays','independent_review.json')),
              'No fit, replay, acceptance or future sample silently retained')
    for name in ('execution.json','watchdog_execution.json'):
        value = c.read(OLD_OUT/name)
        c.require(value['exit_code'] == 1 and not value['cleanup_errors'] and
                  not value['cleanup_receipt']['unconfirmed'] and value['freeze_sha256'] == OLD_FREEZE_SHA,
                  'Actual failed original owner exit and complete cleanup')
    exit_receipt = c.read(OLD_LAUNCH/'pilot_guardian/exit.json')
    c.require(exit_receipt['exit_code'] == 1 and not exit_receipt['cleanup_errors'] and
              exit_receipt['freeze_sha256'] == OLD_FREEZE_SHA, 'Retained actual guardian exit')
    for name in PREFIX_NAMES:
        closure = c.read(OLD_OUT/'closed_workers'/f'{name}.json')
        cleanup = c.read(OLD_OUT/'workers'/name/'cleanup.json')
        c.require(closure['status'] == 'passed' and closure['worker'] == name and
                  not cleanup['alive'] and not cleanup['errors'], 'Complete old worker closure')
    identities = _identities(); require_absent(identities)
    claim = c.read(OLD_OUT/'guardian_claim.json')
    collection = c.read(OLD_OUT/'stages/collection_start.json')
    forecast = c.read(OLD_OUT/'block_forecasts/003.json')
    c.require(forecast['snapshot']['completed_games'] == 32 and forecast['snapshot']['qualified_source_workers'] == 5 and
              forecast['checks']['collection'] is False and all(v for k,v in forecast['checks'].items() if k != 'collection'),
              'Saved forecast-only resource stop')
    for key, origin, span in [('monotonic_deadline','monotonic_origin','monotonic'),('unix_deadline','unix_origin','unix_time')]:
        c.require(collection[key] == min(collection[span] + freeze['stage_seconds']['collection'],
                  claim[origin] + freeze['collection_cutoff_seconds']), 'Original collection deadline unchanged')
    return dict(exact_owner_identities_absent=identities,
                original_monotonic_origin=claim['monotonic_origin'], original_unix_origin=claim['unix_origin'],
                original_collection=collection, prior_ram_wait_seconds=forecast['snapshot']['cumulative_ram_wait_seconds'],
                original_completed_games=32, prior_source_lifecycles=5)


def _tree_inventory(root, boundary):
    items = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            boundary(); items.append(c.artifact(path))
    return dict(root=root.relative_to(c.ROOT).as_posix(), files=len(items),
                bytes=sum(x['bytes'] for x in items), paths=[x['path'] for x in items]), items


def prepare_continuation(*, boundary=lambda: None):
    c.require(c.SCOPE == 'pilot' and c.OUT == NEW_OUT, 'Exclusive new pilot continuation output')
    if MANIFEST.exists():
        return verify_continuation(c.artifact(MANIFEST), boundary=boundary)
    freeze = old_freeze(); stop = verify_stop(); trees=[]; entries=list(freeze['inputs'])
    for root in (OLD_OUT, OLD_LAUNCH):
        tree, items = _tree_inventory(root, boundary); trees.append(tree); entries.extend(items)
    unique = {}
    for item in entries:
        c.require(item['path'] not in unique or unique[item['path']] == item, 'Preservation identity conflict')
        unique[item['path']] = item
    value = dict(schema='selfplay_pilot_continuation_manifest_v1', status='original_incomplete_preserved',
        original_freeze=c.artifact(OLD_OUT/'freeze.json'), original_launch_plan=c.artifact(OLD_LAUNCH/'plan.json'),
        source_continuation=source_continuation(), trees=trees, inputs=sorted(unique.values(),key=lambda x:x['path']),
        prior_evidence_bytes=sum(t['bytes'] for t in trees), **stop,
        next_game_index=32, next_block_index=4, failed_decision_retries=0, new_starting_checkpoint=False,
        original_status_remains_incomplete=True, future_forecasts_advisory=True,
        original_job_seconds=freeze['job_seconds'], original_model_seconds=freeze['collection_cutoff_seconds'])
    c.save(MANIFEST,value)
    return verify_continuation(c.artifact(MANIFEST), boundary=boundary)


def verify_continuation(descriptor, *, boundary=lambda: None):
    c.require(descriptor == c.artifact(MANIFEST), 'Exact continuation manifest')
    value=c.read(MANIFEST); stop=verify_stop()
    c.require(value['schema']=='selfplay_pilot_continuation_manifest_v1' and
              value['original_freeze']==c.artifact(OLD_OUT/'freeze.json') and
              value['source_continuation']==source_continuation() and all(value[k]==v for k,v in stop.items()) and
              value['next_game_index']==32 and value['next_block_index']==4 and
              value['failed_decision_retries']==0 and value['original_status_remains_incomplete'] and
              value['future_forecasts_advisory'] and value['new_starting_checkpoint'] is False,
              'Exact preserved-prefix continuation contract')
    c.require([x['root'] for x in value['trees']]==[p.relative_to(c.ROOT).as_posix() for p in (OLD_OUT,OLD_LAUNCH)],
              'Exactly original pilot and launch evidence trees')
    indexed={x['path']:x for x in value['inputs']}
    c.require(len(indexed)==len(value['inputs']), 'Unique preserved inputs')
    for tree,root in zip(value['trees'],(OLD_OUT,OLD_LAUNCH),strict=True):
        paths=sorted(p.relative_to(c.ROOT).as_posix() for p in root.rglob('*') if p.is_file())
        c.require(paths==tree['paths'] and len(paths)==tree['files'] and
                  sum(indexed[x]['bytes'] for x in paths)==tree['bytes'],'Entire original evidence membership unchanged')
    c.require(value['prior_evidence_bytes']==sum(x['bytes'] for x in value['trees']), 'Cumulative prior evidence charge')
    c.require(all(x in value['inputs'] for x in old_freeze()['inputs']), 'All original frozen dependencies preserved')
    for item in value['inputs']: boundary(); c.verify([item])
    return value


class SavedGuard:
    """Short saved-only review; original player clocks are evidence, never reset."""
    def __init__(self, maximum_seconds=1200):
        claim=c.read(OLD_OUT/'guardian_claim.json')
        self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin']
        self.deadline=min(time.monotonic()+maximum_seconds,self.start+72*3600)
        self.wall_deadline=min(time.time()+maximum_seconds,self.wall_start+72*3600)
        self.placement_sequence=len(list((OLD_OUT/'placement_requests').glob('*.json')))
        self.last=-float('inf');self.peak_rss=0
    def remaining(self):return min(self.deadline-time.monotonic(),self.wall_deadline-time.time())
    def effective_deadline(self):return time.monotonic()+self.remaining()
    def guard(self):
        c.require(self.remaining()>0,'Bounded saved-prefix review and original global deadline')
        now=time.monotonic()
        if now-self.last<1:return
        self.last=now; self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
        c.require(self.peak_rss<=2*c.GiB,'Saved-prefix reader memory cap')
    def close_worker(self,folder,child=None):
        c.require(child is None and folder.parent==OLD_OUT/'workers','Saved-only old worker closure');self.guard()


def saved_prefix_review(maximum_seconds=1200):
    """Actual original reader chain, not a metadata-only corpus declaration."""
    c.require(c.SCOPE=='pilot' and c.OUT==NEW_OUT and not PREFIX_REVIEW.exists(),'New exclusive saved-prefix review')
    from tools.splendorust_selfplay_pilot_budget_20261102 import qualification_control as old,transport,data
    guard=SavedGuard(maximum_seconds);guard.guard();manifest=prepare_continuation(boundary=guard.guard)
    freeze=old_freeze();plan=c.read(c.ROOT/freeze['qualification_plan']['path']);specs=c.read(c.ROOT/freeze['game_schedule']['path'])
    counts={name:0 for name in PREFIX_NAMES};counts['capture_off']=counts['capture_on']=16
    game_summaries=[];live=0;eligible=0
    old.verify_capture_fixtures(guard,freeze,plan,writing=False)
    for index in range(32):
        guard.guard();path=prefix_game_path(index);game=c.read(path);folder=path.parent
        c.require([p.name for p in sorted((folder/'moves').glob('*.json'))]==[f'{i:06d}.json' for i in range(len(game['rows']))],
                  'Every completed prefix game has its exact contiguous row inventory')
        name='capture_on' if index<8 else f'collect_{index//8:03d}'
        for row in game['rows']:
            guard.guard();c.require(c.read(folder/'moves'/f"{row['ply']:06d}.json")==row and
                 Path(row['worker_folder'])==Path((OLD_OUT/'workers'/name).relative_to(c.ROOT)), 'Exact original game/row/worker binding')
            old.validate_row(row,freeze,guard);live+=1;counts[name]+=1
        loaded=data.load_game(game,specs[index],boundary=guard.guard);eligible+=loaded['eligible_roots']
        game_summaries.append(dict(game_index=index,plies=loaded['plies'],eligible_roots=loaded['eligible_roots'],game=c.artifact(path)))
    for name in PREFIX_NAMES:
        old.qualification_evidence(name,guard,freeze,plan,writing=False)
        proof=transport.verify_closed_worker(OLD_OUT/'workers'/name,freeze,guard)
        c.require(proof==c.read(OLD_OUT/'closed_workers'/f'{name}.json'),'Original complete raw/clock/closure recheck')
        c.require(len(list((OLD_OUT/'workers'/name/'inputs').glob('*.json')))==counts[name],
                  'All original ordinary requests belong to fixed fixtures or completed games')
    old.verify_phase_transition(guard);old.verify_exchanges(guard,freeze)
    verify_continuation(c.artifact(MANIFEST),boundary=guard.guard);guard.guard()
    c.require(live==1820 and eligible==1819 and 'torch' not in sys.modules,'Exact complete prefix, no model import')
    return c.save(PREFIX_REVIEW,dict(schema='selfplay_saved_prefix_review_v1',approved=True,
        status='verified_preserved_source_prefix_only',manifest=c.artifact(MANIFEST),original_freeze=c.artifact(OLD_OUT/'freeze.json'),
        completed_games=32,live_decisions=live,eligible_roots=eligible,worker_lifecycles=5,graph_cases=55,
        scientific_qualification_decisions=335,capture_fixture_decisions=32,ordinary_request_counts=counts,
        games=game_summaries,original_attempt_remains_incomplete=True,models=0,new_games=0,
        source_inputs=[c.artifact(Path(__file__))],peak_rss_bytes=guard.peak_rss,maximum_seconds=maximum_seconds))


def verify_prefix_review(descriptor, *, boundary=lambda: None):
    c.require(descriptor==c.artifact(PREFIX_REVIEW),'Exact completed saved-prefix review')
    value=c.read(PREFIX_REVIEW)
    c.require(value['approved'] is True and value['status']=='verified_preserved_source_prefix_only' and
              value['completed_games']==32 and value['live_decisions']==1820 and value['eligible_roots']==1819 and
              value['worker_lifecycles']==5 and value['models']==value['new_games']==0 and
              value['original_attempt_remains_incomplete'] is True,'Complete saved-prefix verification required')
    c.verify(value['source_inputs']);verify_continuation(value['manifest'],boundary=boundary)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['preserve','review'])
    parser.add_argument('--maximum-seconds',type=int,default=1200);args=parser.parse_args()
    c.require(0<args.maximum_seconds<=1200,'Finite saved-only preparation cap')
    result=saved_prefix_review(args.maximum_seconds) if args.operation=='review' else prepare_continuation()
    print(c.canonical(result))
