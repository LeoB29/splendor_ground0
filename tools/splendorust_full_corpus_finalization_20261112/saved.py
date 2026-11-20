"""Two narrowly scoped saved-evidence checks, without model or live-game routes.

The interrupted recovery remains immutable and incomplete. Its sixth independent
check is rerun through the frozen verifier with writing=False. The pooled reader
uses an explicit new execution receipt for that one missing slot; no old exit or
old result is invented, and no historical module's output root is rebound.
"""
from pathlib import Path
import os
import sys
import time

from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent

MATCHUP = 'mixed_terminal'
INDEPENDENT = 'independent_mixed_terminal_r1'
READER = 'reader_mixed_terminal_r0'
SOURCE_FREEZE_SHA256 = 'fa7ce53819e73c57a3de2cdd8d1c0330c00e411bd9f6f0e159c28da2798659c5'


def _path(item):
    return c.ROOT / item['path']


def _same(left, right):
    return c.canonical(left) == c.canonical(right)


def _context(repetition):
    c.require(os.environ.get('FULL_CORPUS_MATCHUP') == MATCHUP and
              os.environ.get('INDEPENDENT_DATA_REPETITION') == str(repetition),
              'Exact saved-only matchup and repetition environment')
    from tools.splendorust_full_corpus_recovery_20261111.strength import common as a
    c.require(a.MATCHUP == MATCHUP and a.REPETITION == repetition and
              a.OUT.resolve() == (c.SOURCE/'strength'/MATCHUP/f'r{repetition}').resolve() and
              a.STUDY_OUT.resolve() == c.SOURCE.resolve() and
              a.parent.ORIGINAL_OUT.resolve() == c.ORIGINAL.resolve(),
              'Frozen verifier keeps its original source roots and cohort')
    c.require(c.OUT.resolve() not in (c.SOURCE.resolve(), c.ORIGINAL.resolve()),
              'Separate finalization output root')
    c.require(c.file_hash(c.SOURCE/'freeze.json') == SOURCE_FREEZE_SHA256,
              'Exact frozen amended recovery source')
    c.require('torch' not in sys.modules, 'Saved-only process has no tensor runtime')
    return a, c.read(c.SOURCE/'freeze.json')


def source_clocks():
    """Authenticate the existing reader, final and global clocks without renewal."""
    claim_item = c.artifact(c.SOURCE/'claim.json')
    phase_item = c.artifact(c.SOURCE/'stages/final_start.json')
    reader_item = c.artifact(c.SOURCE/'reader_clock.json')
    claim, phase, reader = (c.read(_path(x)) for x in (claim_item, phase_item, reader_item))
    c.require(claim['freeze'] == c.artifact(c.SOURCE/'freeze.json') and
              claim['plan_sha256'] == SOURCE_FREEZE_SHA256 and
              claim['total_seconds'] == 120*3600 and claim['model_seconds'] == 96*3600,
              'Original study identity and unchanged global bounds')
    c.require(phase['claim'] == claim_item and phase['stage'] == 'final' and
              type(phase['cap_seconds']) is int and phase['cap_seconds'] == 32*3600 and
              reader['claim'] == claim_item and reader['original_phase'] == phase_item and
              type(reader['cap_seconds']) is int and reader['cap_seconds'] == 12*3600,
              'Exact original final and independent-reader clock provenance')
    for key, origin, deadline in (('monotonic', 'monotonic_origin', 'monotonic_deadline'),
                                  ('unix_time', 'unix_origin', 'unix_deadline')):
        c.require(claim[origin] <= phase[key] <= reader[key] and
                  phase[deadline] == min(phase[key]+phase['cap_seconds'],
                                          claim[origin]+claim['total_seconds']) and
                  reader[deadline] == min(reader[key]+reader['cap_seconds'], phase[deadline]),
                  'Existing final and reader clocks cannot restart or extend')
    return claim, phase, reader


def interrupted_source(boundary):
    """An absent process is not an observed successful exit."""
    boundary()
    folder = c.SOURCE/'segments'/INDEPENDENT
    c.require(not any((folder/name).exists() for name in ('execution.json', 'result.json', 'failure.json')),
              'Only the exact interrupted saved check lacks a terminal receipt')
    c.require(not any((c.SOURCE/name).exists() for name in ('execution.json', 'completion.json', 'failure.json')),
              'Interrupted source root remains without a completion or actual-exit receipt')
    owner = c.read(folder/'owner.json')
    claim = c.read(folder/'claim.json')
    root_claim = c.read(c.SOURCE/'claim.json')
    c.require(claim['mode'] == 'independent' and claim['matchup'] == MATCHUP and
              type(claim['repetition']) is int and claim['repetition'] == 1 and
              claim['block_start'] is None and claim['segment_id'] == INDEPENDENT and
              claim['freeze_sha256'] == SOURCE_FREEZE_SHA256 and
              claim['claim'] == c.artifact(c.SOURCE/'claim.json') and
              claim['phase_clock'] == c.artifact(c.SOURCE/'stages/final_start.json') and
              claim['reader_clock'] == c.artifact(c.SOURCE/'reader_clock.json') and
              owner['segment_claim'] == c.artifact(folder/'claim.json') and
              owner['windows_job_assigned'] is True,
              'Exact interrupted controller, source claim, reader clock and freeze')
    for identity in (owner['identity'], dict(pid=root_claim['pid'], created=root_claim['created'])):
        boundary(); require_absent(identity)
    return dict(status='interrupted_without_observed_exit', owner=c.artifact(folder/'owner.json'),
                claim=c.artifact(folder/'claim.json'), original_root_claim=c.artifact(c.SOURCE/'claim.json'),
                terminal_receipt_present=False, actual_exit_code=None, exact_processes_absent=True,
                completed_scientific_work_repeated=False, saved_check_only=True)


class ReadOnlyEnvelope:
    """Minimal saved-verifier interface; never installs an old-output writer."""
    def __init__(self, boundary):
        self._boundary = boundary
        claim, phase, reader = source_clocks()
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start+claim['total_seconds']
        self.deadline = min(phase['monotonic_deadline'], reader['monotonic_deadline'], self.end)
        self.wall_deadline = min(phase['unix_deadline'], reader['unix_deadline'],
                                 self.wall_start+claim['total_seconds'])
        if hasattr(boundary, 'limits'):
            self.deadline = min(self.deadline, boundary.limits['monotonic_deadline'])
            self.wall_deadline = min(self.wall_deadline, boundary.limits['unix_deadline'])
        self.stage = 'final'
        self.worker_accounting, self.resident = {}, {}
        self.evidence = {'workers': {}}
        self.guard()

    def remaining(self):
        return min(self.deadline-time.monotonic(), self.wall_deadline-time.time())

    def guard(self):
        self._boundary()
        c.require(self.remaining() > 0, 'Unextended original saved-reader deadline')

    def effective_deadline(self):
        self.guard()
        return time.monotonic()+self.remaining()

    def close_worker(self, folder, child=None):
        folder = Path(folder).resolve()
        c.require(child is None and folder.parent ==
                  (c.SOURCE/'strength'/MATCHUP/'r1/workers').resolve(),
                  'Only existing sixth-cohort closed workers may be inspected')
        self.guard()
        # This frozen method's saved-only branch checks the saved closure,
        # cleanup errors, original deadline clamping and every exact identity.
        # No constructor, accounting registration, close action or finish runs.
        from tools.splendorust_full_corpus_recovery_20261111.strength.envelope import SegmentEnvelope
        return SegmentEnvelope.close_worker(self, folder, child=None)


def run_independent(boundary):
    """Return the frozen sixth independent check; writing=False is mandatory."""
    _, freeze = _context(1)
    interrupted_source(boundary)
    from tools.splendorust_full_corpus_recovery_20261111.strength import control
    result = control.verify_saved(ReadOnlyEnvelope(boundary), freeze, writing=False)
    expected = c.read(c.SOURCE/'strength'/MATCHUP/'r1/verification_summary.json')
    c.require(_same(result, expected) and result['status'] == 'verified_amended_full_corpus_assessment' and
              result['matchup'] == MATCHUP and result['repetition'] == 1 and
              result['main_games'] == 200 and result['operational_games'] == 6,
              'Unchanged full saved verification and fixed sixth-cohort coverage')
    boundary()
    c.require('torch' not in sys.modules, 'Independent saved audit loaded no model runtime')
    return result


def validate_passed_execution(execution, owner, result, *, mode, new):
    """Require a real clean exit as well as a successful worker result."""
    c.require(execution['status'] == result['status'] == 'passed' and
              type(execution['worker_exit_code']) is int and execution['worker_exit_code'] == 0 and
              execution['cleanup_errors'] == [] and execution['error'] is result['error'] is None and
              execution['identity'] == owner['identity'] and owner['windows_job_assigned'] is True and
              execution['mode'] == result['mode'] == mode,
              'Completed source-local worker, actual exit zero and clean ownership required')
    identity = result['identity'] if new else dict(pid=result['pid'], created=result['created'])
    c.require(identity == owner['identity'] and type(identity['pid']) is int and
              type(identity['created']) in (int, float), 'Exact actual worker identity')
    c.require(type(execution['observed_owned_identities']) is list,
              'Explicit complete observed ownership inventory')
    if new:
        c.require(execution['retries'] == 0 and type(execution['retries']) is int and
                  all(type(result[k]) is int and result[k] == 0
                      for k in ('model_calls', 'new_games', 'optimizer_calls')),
                  'Finalization cannot retry scientific work or load models')
    return True


def completed_source(folder, boundary):
    """Read one successful historical segment, preserving its true source."""
    folder = Path(folder).resolve()
    source = next((base for base in (c.ORIGINAL, c.SOURCE)
                   if folder.parent == (base/'segments').resolve()), None)
    c.require(source is not None, 'Historical receipt must be under an exact preserved source')
    boundary()
    execution, owner, result, claim = (c.read(folder/name) for name in
                                       ('execution.json', 'owner.json', 'result.json', 'claim.json'))
    validate_passed_execution(execution, owner, result, mode=claim['mode'], new=False)
    c.require(execution['segment_claim'] == owner['segment_claim'] == c.artifact(folder/'claim.json') and
              claim['segment_id'] == folder.name and claim['matchup'] == MATCHUP and
              execution['matchup'] == result['matchup'] == claim['matchup'] and
              execution['repetition'] == result['repetition'] == claim['repetition'] and
              claim['claim'] == c.artifact(source/'claim.json') and
              claim['freeze_sha256'] == c.file_hash(source/'freeze.json'),
              'Source-local role, claim, freeze and exact logical segment')
    c.verify([claim['phase_clock']])
    phase = c.read(_path(claim['phase_clock']))
    c.require(phase['claim'] == claim['claim'], 'Historical phase belongs to its actual source owner')
    for key, origin, end, deadline in (('monotonic', 'created_monotonic', 'ended_monotonic', 'monotonic_deadline'),
                                       ('unix_time', 'created_unix', 'ended_unix', 'unix_deadline')):
        c.require(phase[key] <= owner[origin] <= execution[end] < phase[deadline],
                  'Actual historical execution lies within its original phase clock')
        if claim.get('reader_clock') is not None:
            c.verify([claim['reader_clock']]); reader = c.read(_path(claim['reader_clock']))
            c.require(reader['claim'] == claim['claim'] and reader['original_phase'] == claim['phase_clock'] and
                      owner[origin] >= reader[key] and execution[end] < reader[deadline],
                      'Historical independent execution also respects its original reader clock')
    for identity in execution['observed_owned_identities']+[execution['identity']]:
        boundary(); require_absent(identity)
    return dict(logical_segment=folder.name, source='original' if source == c.ORIGINAL else 'amended_recovery',
                execution=c.artifact(folder/'execution.json'), owner=c.artifact(folder/'owner.json'),
                result=c.artifact(folder/'result.json'), started_monotonic=owner['created_monotonic'],
                ended_monotonic=execution['ended_monotonic'])


def completed_new_independent(boundary):
    """The sole allowed substitution is a separately completed saved check."""
    interrupted_source(boundary)
    folder = c.OUT/'jobs'/INDEPENDENT
    execution, owner, result = (c.read(folder/name) for name in
                                 ('execution.json', 'owner.json', 'worker/result.json'))
    validate_passed_execution(execution, owner, result, mode=INDEPENDENT, new=True)
    plan = c.artifact(c.OUT/'freeze.json')
    c.require(execution['plan_sha256'] == owner['plan_sha256'] == result['plan_sha256'] == plan['sha256'],
              'New saved-check owner and actual exit bind the finalization freeze')
    root_claim = c.read(c.OUT/'claim.json')
    c.require(owner['global_claim'] == c.artifact(c.OUT/'claim.json') and
              owner['source_reader_clock'] == c.artifact(c.SOURCE/'reader_clock.json') and
              owner['model_work'] is False and root_claim['freeze'] == plan and
              root_claim['source_claim'] == c.artifact(c.SOURCE/'claim.json') and
              root_claim['original_reader'] == c.artifact(c.SOURCE/'reader_clock.json'),
              'Explicit new owner binds the unchanged source claim and reader clock')
    expected = c.read(c.SOURCE/'strength'/MATCHUP/'r1/verification_summary.json')
    c.require(_same(result['result'], expected), 'Only the unchanged sixth saved verification substitutes its slot')
    c.verify([result['runtime_sources']])
    runtime = c.read(_path(result['runtime_sources']))
    c.require(runtime['sources'] == c.read(c.SOURCE/'freeze.json')['corrected_source_receipt'] and
              runtime['identity'] == result['identity'] and runtime['mode'] == INDEPENDENT and
              runtime['model_calls'] == runtime['new_games'] == 0,
              'Actual saved worker checked the frozen corrected sources at startup')
    _, phase, reader = source_clocks()
    for start, end, deadline in (('monotonic_origin', 'monotonic_end', 'monotonic_deadline'),
                                  ('unix_origin', 'unix_end', 'unix_deadline')):
        c.require(execution[start] <= execution[end] < execution[deadline] <=
                  min(phase[deadline], reader[deadline]) and owner[deadline] == execution[deadline] and
                  execution[deadline] == min(execution[start]+c.JOB_CAPS[INDEPENDENT], root_claim[deadline]),
                  'New saved-only actual exit respects every original clipped deadline')
    started = owner.get('created_monotonic', owner.get('monotonic_origin'))
    c.require(type(started) in (int, float) and
              execution['monotonic_origin'] <= started <= execution['monotonic_end'],
              'Explicit new controller creation time within its actual execution')
    for identity in execution['observed_owned_identities']+[execution['identity']]:
        boundary(); require_absent(identity)
    return dict(logical_segment=INDEPENDENT, source='saved_only_finalization',
                execution=c.artifact(folder/'execution.json'), owner=c.artifact(folder/'owner.json'),
                result=c.artifact(folder/'worker/result.json'), started_monotonic=started,
                ended_monotonic=execution['monotonic_end'],
                replaced_historical_exit=False, interrupted_source_exit_still_unknown=True)


def receipt_mapping(expected_names, boundary):
    c.require(len(expected_names) == len(set(expected_names)) and INDEPENDENT in expected_names,
              'One exact logical independent-check slot')
    actual = {path.parent.name for path in (c.SOURCE/'segments').glob('*/execution.json')
              if '_'+MATCHUP+'_r' in path.parent.name}
    c.require(actual == set(expected_names)-{INDEPENDENT},
              'All other planned source segments completed; only the disclosed saved check is absent')
    return [completed_new_independent(boundary) if name == INDEPENDENT else
            completed_source(c.SOURCE/'segments'/name, boundary) for name in expected_names]


def run_reader(boundary):
    """Complete only the missing pooled reader, retaining the frozen arithmetic."""
    a, freeze = _context(0)
    source_clocks()
    interrupted = interrupted_source(boundary)
    from tools.splendorust_full_corpus_recovery_20261111.strength import report, analysis
    from tools.splendorust_full_corpus_recovery_20261111.owner import segment_id
    from tools.splendorust_full_corpus_recovery_20261111.boundary import check_source_boundary, verify_retry_consumption
    c.require(not (c.SOURCE/'segments'/READER).exists() and
              not (c.SOURCE/'strength'/MATCHUP/'report/independent_acceptance.json').exists() and
              not (c.SOURCE/'strength'/MATCHUP/'final_inventory.json').exists(),
              'The pooled reader is genuinely unstarted and has no acceptance to replace')
    mappings = receipt_mapping(report.expected_new_segments(), boundary)
    original_passed = c.read(c.ORIGINAL/'execution.json')['segments']
    predecessors = [_path(item).parent for item in original_passed
                    if '/segments/' in item['path'] and '_'+MATCHUP+'_r' in Path(item['path']).parent.name]
    expected_original = []
    for rep in (0, 1):
        expected_original += [c.ORIGINAL/'segments'/segment_id(mode, rep, matchup=MATCHUP)
                              for mode in ('qualification', 'operational')]
        expected_original += [c.ORIGINAL/'segments'/segment_id('main_block', rep, start, matchup=MATCHUP)
                              for start in range(0, (a.parent.COMPLETED_PREFIXES[MATCHUP][rep]//10)*10, 10)]
    c.require(len(predecessors) == len(set(predecessors)) and set(predecessors) == set(expected_original),
              'Exact successful original strength prefix')
    for item in original_passed:
        if _path(item).parent in predecessors: c.verify([item])
    mappings += [completed_source(folder, boundary) for folder in predecessors]
    ordered = sorted(mappings, key=lambda value: value['started_monotonic'])
    for previous, current in zip(ordered, ordered[1:]):
        c.require(previous['ended_monotonic'] <= current['started_monotonic'],
                  'No overlapping accepted original, recovery or new saved controllers')
    preserved_failure = check_source_boundary(boundary=boundary)
    retry = verify_retry_consumption(boundary=boundary)
    source_report = c.SOURCE/'strength'/MATCHUP/'report'
    value = c.read(source_report/'analysis.json')
    arithmetic = analysis.verify_pooled(report.main_games(), value, boundary=boundary)
    for item in freeze['inputs']:
        boundary(); c.verify([item])
    inventory = []
    for base in (c.ORIGINAL/'strength'/MATCHUP/'r0', c.ORIGINAL/'strength'/MATCHUP/'r1',
                 c.SOURCE/'strength'/MATCHUP/'r0', c.SOURCE/'strength'/MATCHUP/'r1', source_report):
        for path in sorted(base.rglob('*')):
            if path.is_file(): boundary(); inventory.append(c.artifact(path))
    inventory_item = c.save(c.OUT/'strength'/MATCHUP/'final_inventory.json',
                           dict(schema='full_corpus_strength_completed_inventory_v1', artifacts=inventory,
                                files=len(inventory), bytes=sum(x['bytes'] for x in inventory)))
    for item in inventory:
        boundary(); c.verify([item])
    workers = list((c.SOURCE/'strength'/MATCHUP).glob('r*/workers/*/initialization.json')) + \
              list((c.ORIGINAL/'strength'/MATCHUP).glob('r*/workers/*/initialization.json'))
    c.require(len(workers) <= 812 <= freeze['maximum_live_lifecycles'],
              'Unchanged complete lifecycle accounting')
    for path in workers:
        boundary(); require_absent(c.read(path)['identity'])
    reviews = {row['logical_segment']: row['result'] for row in mappings}
    value = dict(schema='full_corpus_strength_independent_acceptance_v1', approved=True, blocking_findings=[],
                 matchup=MATCHUP, candidate_arm=a.ROUTE_ARMS['candidate'], reference_arm=a.ROUTE_ARMS['control'],
                 analysis=c.artifact(source_report/'analysis.json'), fixed_main_games=400, operational_games=12,
                 fixed_candidate_checkpoints=2, fixed_control_checkpoints=2, distinct_compared_checkpoints=4,
                 route_arms=dict(a.ROUTE_ARMS), independent_per_repetition_saved_reviews=[
                     reviews[segment_id('independent', rep, matchup=MATCHUP)] for rep in (0, 1)],
                 independent_pooled_arithmetic=arithmetic, completed_segments=len(mappings),
                 exact_worker_lifecycles=len(workers), all_completed_segment_exits_observed=True,
                 saved_inventory=inventory_item, frozen_inputs_unchanged=len(freeze['inputs']),
                 original_reference_retained=True, parent_is_opponent=False, automatic_promotion=False,
                 actual_outer_owner_exit_pending=True, charts_require_final_visual_inspection=True,
                 model_calls=0, new_games=0, amended_recovered_study=True, original_attempt_status='incomplete',
                 original_failed_attempt_preserved=True, authorized_potentially_interrupted_retry_count=0,
                 retry_consumption=retry, preserved_failure_binding=preserved_failure,
                 saved_only_finalization=True, interrupted_recovery_attempt_status='incomplete',
                 interrupted_recovery_actual_exit_code=None, interrupted_saved_check=interrupted,
                 source_aware_completed_receipts=mappings, historical_exits_relabelled=False,
                 source_analysis_unchanged=True, new_optimizer_calls=0, new_decisions=0)
    boundary()
    c.require('torch' not in sys.modules, 'Pooled saved reader loaded no model runtime')
    return c.save(c.OUT/'strength'/MATCHUP/'report/independent_acceptance.json', value)
