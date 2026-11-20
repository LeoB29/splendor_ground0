"""One retained owner: fixed phases, inherited clocks and exact child ownership."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent, finish_owned_process


def child_environment(repetition):
    c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
    env = os.environ.copy()
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(ONEPASS_REPETITION=str(repetition), SELFPLAY_JOB_SCOPE='pilot',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', RAYON_NUM_THREADS='1')
    return env


def contract():
    return dict(schema=c.SCHEMA, scope='pilot', phase='pilot', total_seconds=c.JOB_SECONDS,
                model_seconds=c.MODEL_CUTOFF_SECONDS, stage_seconds=c.STAGES,
                independent_reader_seconds=c.INDEPENDENT_REVIEW_SECONDS, main_games=400,
                operational_games=12, new_source_games=0, optimizer_calls=0, candidate_repetitions=[0, 1],
                matched_simulations=128, worlds=3, cpuct=.4, depth=64, precision='float32',
                internal_seconds=9., complete_response_seconds=10., available_ram_gate=False,
                resource_forecasts_may_stop=False, maximum_inference_rss=c.MAX_RSS_BYTES,
                maximum_evidence=c.MAX_EVIDENCE_BYTES, prefinal_evidence=c.PREFINAL_EVIDENCE_BYTES,
                actual_disk_floor=c.MINIMUM_FREE_DISK_BYTES, maximum_live_lifecycles=850,
                nominal_live_lifecycles=92, reachable_live_lifecycle_maximum=812,
                maximum_model_reproductions=192, maximum_plies=2000, automatic_retries=0,
                failed_decision_retries=0, replacement_games=0, automatic_promotion=False,
                larger_data_deferred=True, scheduled_monitor=False, only_weights_differ=True)


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources = python_dependencies((c.ROOT/'tools/splendorust_onepass_strength_20261106').rglob('*.py'))
    tests = sorted((c.ROOT/'tests').glob('test*onepass*20261106*.py'))
    return [c.artifact(path) for path in sources + tests]


def refresh_seed_inventory():
    """Model-free finite registry recheck; own exact reservations are not reuse."""
    import json
    prepared = c.read(c.PREPARATION/'seed_inventory.json')
    targets = {int(x['seed_decimal']) for x in c.read(c.PREPARATION/'schedule.json')}
    targets.update(prepared['bootstrap_seeds'].values())
    found = subprocess.run(['rg', '--files', '--no-ignore', 'outputs', '-g', '*manifest*.json',
        '-g', '*schedule*.json', '-g', '*seed_registry*.json', '-g', '*reserved_panel*.json', '-g', '*freeze*.json'],
        cwd=c.ROOT, capture_output=True, text=True, check=True, timeout=120)
    checked, skipped, collisions = [], [], []
    def visit(value, name):
        if type(value) is int or type(value) is str and value.isdecimal():
            if int(value) in targets:
                collisions.append(name)
        elif type(value) is list:
            for child in value: visit(child, name)
        elif type(value) is dict:
            for child in value.values(): visit(child, name)
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT/name).resolve()
        if path.is_relative_to(c.PREPARATION) or path.is_relative_to(c.OUT): continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size)); continue
        visit(c.read(path), name)
        checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh reserved seed collision; never replace or retry a seed')
    return c.save(c.OUT/'seed_freshness.json', dict(status='passed', checked=checked,
        skipped_oversized=skipped, collisions=collisions, reserved=c.artifact(c.PREPARATION/'seed_inventory.json'),
        own_exclusions=[str(c.PREPARATION), str(c.OUT)], setup_seeds=206, analysis_seeds=3,
        scope='Registered files up to8MiB; not arbitrary raw logs'))


def prepare():
    c.require(c.file_hash(c.PREPARED_PLAN) == c.PREPARED_PLAN_SHA, 'Accepted prepared plan unchanged')
    c.verify(list(c.nested_artifacts(c.read(c.PREPARED_PLAN))))
    c.OUT.mkdir(parents=True, exist_ok=True)
    for repetition in (0, 1):
        path = c.OUT/f'r{repetition}'/'assessment'/'prepared.json'
        c.require(not path.exists(), 'Never implicitly rerun preparation')
        result = subprocess.run([sys.executable, '-B', '-u', '-m', c.PACKAGE+'.segment', '--mode', 'prepare'],
            cwd=c.ROOT, env=child_environment(repetition), capture_output=True, text=True, timeout=600,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        c.save(c.OUT/'prelaunch'/f'prepare_r{repetition}_exit.json', dict(exit_code=result.returncode,
               stdout=result.stdout, stderr=result.stderr, models=0, games=0))
        c.require(result.returncode == 0 and c.read(path)['status'] == 'prepared', 'Preparation failed and preserved')
    refresh_seed_inventory()
    return dict(status='prepared', models=0, games=0)


def freeze(tests, reviews):
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
    from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS
    c.require(not (c.OUT/'freeze.json').exists(), 'One execution freeze')
    proof = c.read(tests)
    c.require(proof['exit_code'] == 0 and proof['models'] == proof['games'] == 0, 'Passed model-free tests')
    c.verify(proof['inputs'] + [proof['log']])
    c.require(len(reviews) >= 3, 'Independent import, controller and scientific reviews')
    c.require(len({str(Path(path).resolve()).casefold() for path in reviews}) == len(reviews), 'Distinct review receipts')
    reviewers = []
    for path in reviews:
        review = c.read(path)
        c.require(review['approved'] is True and not review['blocking_findings'], 'Unresolved independent review')
        c.require(type(review.get('reviewer')) is str and review['reviewer'], 'Explicit independent reviewer')
        reviewers.append(review['reviewer'])
        c.verify(review['inputs'])
    c.require(len(set(reviewers)) >= 3, 'Three distinct independent review authors')
    prior_path = c.ROOT/'outputs/splendorust_policy_head_20261104_01/freeze.json'
    c.require(c.file_hash(prior_path) == '52641e03dda1a14cb7f4675f97a3c1b1412226679b000db7a78354a6071b81b2',
              'Exact accepted runtime predecessor')
    prior = c.read(prior_path)
    value = {key: deepcopy(prior[key]) for key in RUNTIME_KEYS}
    value.update(contract())
    value.update(protocol=c.artifact(c.PROTOCOL), prepared_plan=c.artifact(c.PREPARED_PLAN),
        user_authorization='Implement and qualify checkpoint import and adapter, then freeze and run400games if checks pass; both one-pass models/matched128/larger-data deferred.',
        tests=c.artifact(tests), reviews=[c.artifact(path) for path in reviews],
        repetitions={str(rep):c.read(c.OUT/f'r{rep}'/'assessment'/'prepared.json')['prepared'] for rep in (0, 1)},
        seed_freshness=c.artifact(c.OUT/'seed_freshness.json'),
        corrected_source_receipt=verify_runtime_sources(), ruleset_id=RULESET_ID,
        catalogue_sha256=CATALOGUE_SHA256, turn_contract=TURN_CONTRACT,
        historical_parent_checkpoint=prior['historical_parent_checkpoint'],
        unchanged_model_definition=prior['unchanged_model_definition'],
        external_runtime_inputs=prior['external_runtime_inputs'], time=c.now())
    code_inputs = source_inputs()
    current_by_path = {item['path']:item for item in code_inputs}
    inherited_code = [item for item in prior['inputs'] if item['path'] in current_by_path]
    for item in inherited_code:
        c.require(item == current_by_path[item['path']], 'Inherited accepted implementation changed')
    value['inherited_accepted_code'] = inherited_code
    value['inherited_runtime_freeze'] = c.artifact(prior_path)
    entries = code_inputs + list(c.nested_artifacts(value)) + list(c.nested_artifacts(c.read(c.PREPARED_PLAN)))
    for path in [tests, *reviews]: entries.extend(c.nested_artifacts(c.read(path)))
    # Pin the complete CPU tensor runtime without importing it.
    entries += [dict(path=x['absolute_path'], bytes=x['bytes'], sha256=x['sha256']) for x in prior['torch_runtime']]
    unique = {}
    for item in entries:
        key = str((c.ROOT/item['path']).resolve()).casefold()
        old = unique.setdefault(key, item)
        c.require(old['bytes'] == item['bytes'] and old['sha256'] == item['sha256'], 'Conflicting frozen input')
    value['inputs'] = sorted(unique.values(), key=lambda x:x['path'])
    c.verify(value['inputs'])
    c.require('torch' not in sys.modules, 'Freeze does not load models')
    return c.save(c.OUT/'freeze.json', value)


def guard(claim, phase):
    mono, wall = time.monotonic(), time.time()
    c.require(claim['monotonic_origin'] <= mono < phase['monotonic_deadline'] and
              claim['unix_origin'] <= wall < phase['unix_deadline'], 'Original phase/global clock expired')
    c.require(shutil.disk_usage(c.ROOT).free >= c.MINIMUM_FREE_DISK_BYTES, 'Actual disk floor')


def begin_phase(claim, name):
    c.require(name in ('qualification', 'operational', 'main', 'replay', 'final'), 'Fixed phase')
    mono, wall = time.monotonic(), time.time()
    cap = c.STAGES[name]
    cutoff = c.JOB_SECONDS if name == 'final' else c.MODEL_CUTOFF_SECONDS
    phase = dict(stage=name, claim=c.artifact(c.OUT/'claim.json'), monotonic=mono, unix_time=wall,
        cap_seconds=cap, monotonic_deadline=min(mono+cap, claim['monotonic_origin']+cutoff,
                                               claim['monotonic_origin']+c.JOB_SECONDS),
        unix_deadline=min(wall+cap, claim['unix_origin']+cutoff, claim['unix_origin']+c.JOB_SECONDS))
    guard(claim, phase)
    return c.save(c.OUT/'stages'/f'{name}_start.json', phase)


def end_phase(claim, descriptor):
    phase = c.read(c.ROOT/descriptor['path']); guard(claim, phase)
    return c.save(c.OUT/'stages'/f"{phase['stage']}_end.json", dict(stage=phase['stage'],
        start=descriptor, monotonic=time.monotonic(), unix_time=time.time()))


def segment_id(mode, repetition, block_start=None):
    c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
    c.require(mode in ('qualification','operational','main_block','replay','saved','independent','aggregate','reader'), 'Fixed segment')
    if mode == 'main_block':
        c.require(type(block_start) is int and block_start in range(0,200,10), 'Fixed ten-game block')
    else: c.require(block_start is None, 'Only main blocks have an offset')
    return f'{mode}_r{repetition}' + (f'_g{block_start:03d}' if block_start is not None else '')


def validate_segment_claim(claim, folder, mode, repetition, block_start, expected):
    ident = segment_id(mode, repetition, block_start)
    c.require(type(claim) is dict and set(claim) == {'segment_id','mode','repetition','block_start',
              'claim','phase_clock','freeze_sha256','reader_clock'}, 'Exact segment claim fields')
    expected_fields = dict(segment_id=ident, mode=mode, repetition=repetition, block_start=block_start,
                           freeze_sha256=expected)
    c.require(c.canonical({k:claim[k] for k in expected_fields}) == c.canonical(expected_fields) and
              Path(folder).resolve() == c.OUT/'segments'/ident, 'Strict segment role, types and path')
    c.verify([claim['claim'],claim['phase_clock']])
    c.require(claim['claim'] == c.artifact(c.OUT/'claim.json') and
              c.read(c.OUT/'claim.json')['freeze'] == c.artifact(c.OUT/'freeze.json') and
              c.file_hash(c.OUT/'freeze.json') == expected, 'Original owner and exact freeze binding')
    return claim


def run_segment(claim, phase_descriptor, expected, mode, repetition, block_start=None, reader_clock=None):
    from tools.local_player_qualification_128.clock import _WindowsJob
    ident = segment_id(mode, repetition, block_start)
    folder = c.OUT/'segments'/ident
    folder.mkdir(parents=True, exist_ok=False)
    local = dict(segment_id=ident, mode=mode, repetition=repetition, block_start=block_start,
        claim=c.artifact(c.OUT/'claim.json'), phase_clock=phase_descriptor, freeze_sha256=expected,
        reader_clock=reader_clock)
    c.save(folder/'claim.json', local)
    phase = c.read(c.ROOT/phase_descriptor['path'])
    module = c.PACKAGE + ('.report' if mode in ('aggregate','reader') else '.segment')
    args = [sys.executable, '-B', '-u', '-m', module, '--mode', mode, '--segment', str(folder), '--freeze-sha256', expected]
    if block_start is not None: args += ['--block-start', str(block_start)]
    proc = job = identity = None
    observed, cleanup, error, status = {}, [], None, 'failed'
    c.status(c.OUT/'stage.json', dict(time=c.now(), stage=phase['stage'], segment=ident,
                                    completed_main_games=completed_games(), status='starting'))
    try:
        guard(claim, phase)
        with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(args, cwd=c.ROOT, env=child_environment(repetition), stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity = dict(pid=proc.pid, created=psutil.Process(proc.pid).create_time())
            job = _WindowsJob(proc.pid)
            c.save(folder/'owner.json', dict(identity=identity, args=args, segment_claim=c.artifact(folder/'claim.json'),
                created_monotonic=time.monotonic(), created_unix=time.time(), windows_job_assigned=True))
            while proc.poll() is None:
                guard(claim, phase)
                if reader_clock is not None:
                    guard(claim, c.read(c.ROOT/reader_clock['path']))
                try:
                    parent = psutil.Process(proc.pid)
                    c.require(parent.create_time() == identity['created'], 'Exact segment owner identity')
                    for child in [parent]+parent.children(recursive=True):
                        try: observed[(child.pid,child.create_time())] = dict(pid=child.pid,created=child.create_time())
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                time.sleep(.5)
            guard(claim, phase)
            if reader_clock is not None: guard(claim,c.read(c.ROOT/reader_clock['path']))
            c.require(proc.returncode == 0, 'Segment failed; preserve without retry: '+ident)
            result = c.read(folder/'result.json')
            c.require(result['status'] == 'passed' and result['mode'] == mode and result['repetition'] == repetition,
                      'Exact completed segment result')
            c.require(result['pid'] == identity['pid'] and result['created'] == identity['created'], 'Actual result owner')
            for item in observed.values():
                require_absent(item)  # A successful segment must clean itself up before our Job closes.
            require_absent(identity)
            status = 'passed'
    except BaseException:
        error = traceback.format_exc()
    finally:
        if job:
            try: job.close()
            except BaseException as exc: cleanup.append(repr(exc))
        if proc:
            try: finish_owned_process(proc, identity)
            except BaseException as exc: cleanup.append(repr(exc))
        for item in [*observed.values(), *([identity] if identity else [])]:
            try: require_absent(item)
            except BaseException as exc: cleanup.append(repr(exc))
        if cleanup: status = 'failed'
    receipt = c.save(folder/'execution.json', dict(status=status, mode=mode, repetition=repetition,
        block_start=block_start, identity=identity, worker_exit_code=None if proc is None else proc.returncode,
        observed_owned_identities=list(observed.values()), cleanup_errors=cleanup, error=error,
        segment_claim=c.artifact(folder/'claim.json'), ended_monotonic=time.monotonic(), ended_unix=time.time()))
    c.require(status == 'passed', 'Failed segment preserved: '+ident)
    return receipt


def completed_games():
    return sum(1 for rep in (0,1) for _ in (c.OUT/f'r{rep}'/'assessment/games/main').glob('g*/result.json'))


def run(expected):
    c.require(c.file_hash(c.OUT/'freeze.json') == expected, 'Exact execution freeze')
    freeze_doc = c.read(c.OUT/'freeze.json')
    c.require(c.canonical({k:freeze_doc.get(k) for k in contract()}) == c.canonical(contract()), 'Exact execution contract')
    c.verify(freeze_doc['inputs'])
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed = placement.configure_execution(freeze_doc,'isolated')
    claim = dict(time=c.now(), pid=os.getpid(), created=psutil.Process().create_time(),
                 monotonic_origin=time.monotonic(), unix_origin=time.time(), freeze=c.artifact(c.OUT/'freeze.json'),
                 placement=placed, automatic_retries=0)
    c.save(c.OUT/'claim.json', claim)  # Exclusive create is the launch lock.
    receipts, error, status = [], None, 'failed'
    try:
        for phase_name, mode in [('qualification','qualification'),('operational','operational')]:
            phase = begin_phase(claim,phase_name)
            for rep in (0,1): receipts.append(run_segment(claim,phase,expected,mode,rep))
            end_phase(claim,phase)
            c.save(c.OUT/f'{phase_name}_gate.json', dict(status='passed', both_repetitions=True,
                phase=phase, actual_exits=receipts[-2:]))
        phase = begin_phase(claim,'main')
        for block in range(0,200,10):
            for rep in (0,1): receipts.append(run_segment(claim,phase,expected,'main_block',rep,block))
        end_phase(claim,phase)
        phase = begin_phase(claim,'replay')
        for rep in (0,1): receipts.append(run_segment(claim,phase,expected,'replay',rep))
        end_phase(claim,phase)
        phase = begin_phase(claim,'final')
        for rep in (0,1): receipts.append(run_segment(claim,phase,expected,'saved',rep))
        receipts.append(run_segment(claim,phase,expected,'aggregate',0))
        final = c.read(c.ROOT/phase['path'])
        reader_mono, reader_wall = time.monotonic(), time.time()
        reader = c.save(c.OUT/'reader_clock.json',dict(monotonic=reader_mono,unix_time=reader_wall,
            cap_seconds=c.INDEPENDENT_REVIEW_SECONDS,
            monotonic_deadline=min(reader_mono+c.INDEPENDENT_REVIEW_SECONDS,final['monotonic_deadline']),
            unix_deadline=min(reader_wall+c.INDEPENDENT_REVIEW_SECONDS,final['unix_deadline']),
            original_phase=phase, claim=c.artifact(c.OUT/'claim.json')))
        for rep in (0,1): receipts.append(run_segment(claim,phase,expected,'independent',rep,reader_clock=reader))
        receipts.append(run_segment(claim,phase,expected,'reader',0,reader_clock=reader))
        c.verify(freeze_doc['inputs'])
        end_phase(claim,phase)
        status = 'verified_pending_actual_owner_exit'
    except BaseException:
        error = traceback.format_exc()
    execution = c.save(c.OUT/'execution.json',dict(status=status, freeze_sha256=expected,
        segments=receipts, error=error, original_claim=c.artifact(c.OUT/'claim.json'),
        ended_monotonic=time.monotonic(), ended_unix=time.time(), automatic_promotion=False))
    c.status(c.OUT/'stage.json',dict(time=c.now(),status=status,stage=status, completed_segments=len(receipts),
                                   completed_main_games=completed_games()))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error))
        return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_reader=c.artifact(c.OUT/'report/independent_acceptance.json'),
        owner_exit_still_requires_observation=True, no_automatic_followup=True))
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('prepare','freeze','run'))
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare': print(c.canonical(prepare()));return 0
    if args.operation=='freeze': print(c.canonical(freeze(Path(args.tests),[Path(x) for x in args.reviews])));return 0
    return run(args.freeze_sha256)


if __name__=='__main__': raise SystemExit(main())
