"""Single retained owner, sequential workers and a prospective immutable freeze."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent, finish_owned_process


def frozen_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    files = python_dependencies((c.ROOT / 'tools/splendorust_learning_diagnostic_20261106').glob('*.py'))
    files += sorted((c.ROOT / 'tests').glob('test_splendorust_learning_diagnostic*20261106.py'))
    return [c.artifact(path) for path in files]


def contract():
    return dict(total_seconds=c.TOTAL_SECONDS, model_seconds=c.MODEL_SECONDS,
        qualification_seconds=c.QUALIFICATION_SECONDS, fit_seconds=c.FIT_SECONDS,
        evaluation_seconds=c.EVALUATION_SECONDS, audit_seconds=c.AUDIT_SECONDS,
        maximum_rss_bytes=c.MAX_RSS, maximum_output_bytes=c.MAX_OUTPUT,
        minimum_free_disk_bytes=c.MIN_DISK, available_ram_reserve_bytes=0,
        resource_forecasts_may_stop=False, optimizer_steps_per_fit=668,
        qualification_optimizer_steps=2, reproduction_optimizer_steps=8,
        scientific_optimizer_steps=5344, maximum_optimizer_steps=5354,
        per_loss_gradient_probes=144, new_games=0, new_source_games=0,
        automatic_retries=0, automatic_promotion=False, larger_data_deferred=True,
        scheduled_monitor=False, fits=c.fits(), supervisor_affinity=[2], fitter_affinity=[0],
        checkpoints=[0, 334, 668], saved_reproduction_origin_step=667,
        dtype='float32', execution_device='cpu', seed=20261106,
        existing_heldouts_already_opened=True, inference_search_calls=0)


def runtime_bindings(previous):
    entries = list(c.nested_artifacts(previous['external_runtime_inputs']))
    entries += [{'path': item['absolute_path'], 'bytes': item['bytes'], 'sha256': item['sha256']}
                for item in previous['torch_runtime']]
    c.require(bool(previous['torch_runtime']) and len(entries) >= len(previous['torch_runtime']),
              'Complete CPU tensor runtime binding')
    return entries


def freeze(tests_path, review_paths):
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
    from . import data
    c.require(not c.PLAN.exists(), 'One prospective execution freeze')
    tests = c.read(tests_path)
    c.require(tests['exit_code'] == 0 and tests['models'] == tests['games'] == 0, 'Passed model-free tests')
    c.verify(tests['inputs'] + [tests['log']])
    c.require(len(review_paths) >= 2, 'Independent implementation reviews')
    for path in review_paths:
        review = c.read(path)
        c.require(review['approved'] and not review['blocking_findings'], 'Unresolved review findings')
        c.verify(review['inputs'])
    selection = c.read(c.SELECTION)
    data.validate_selection(selection)
    previous = c.read(c.ROOT / 'outputs/splendorust_policy_head_20261104_01/freeze.json')
    runtime = runtime_bindings(previous)
    parents = [previous['historical_parent_checkpoint'], previous['unchanged_model_definition']]
    value = dict(schema='saved_learning_breadth_scope_diagnostic_v1', time=c.now(),
        user_authorization='User: ok do it, following the reviewed saved-corpus diagnostic design',
        protocol=c.artifact(c.PROTOCOL), preparation=c.artifact(c.PREPARATION),
        selection=c.artifact(c.SELECTION), tests=c.artifact(tests_path),
        reviews=[c.artifact(path) for path in review_paths], parent_checkpoint=parents[0],
        source_guard=verify_runtime_sources(), ruleset_id=RULESET_ID,
        catalogue_sha256=CATALOGUE_SHA256, turn_contract=TURN_CONTRACT,
        **contract())
    code = frozen_inputs()
    entries = code + runtime + parents + list(c.nested_artifacts(value)) + list(c.nested_artifacts(selection))
    for path in [tests_path, *review_paths]: entries.extend(c.nested_artifacts(c.read(path)))
    unique = {}
    for item in entries:
        key = str(c.resolve(item)).casefold()
        old = unique.setdefault(key, item)
        c.require((old['bytes'], old['sha256']) == (item['bytes'], item['sha256']), 'Conflicting frozen artifact')
    value['inputs'] = sorted(unique.values(), key=lambda x: x['path'])
    value['worker_inputs'] = code + runtime + parents
    c.verify(value['inputs'])
    c.require('torch' not in sys.modules, 'Freeze performs no model work')
    return c.save(c.PLAN, value)


def check_clock(origin, wall_origin, cap, now, wall, label):
    c.require(origin <= now < origin + cap and wall_origin <= wall < wall_origin + cap,
              'Actual dual-clock deadline: ' + label)


class Envelope:
    def __init__(self, claim):
        self.claim = claim
        self.last_deep = -float('inf')
        self.peak_rss = 0
        self.evaluation_clock = None

    def guard(self, deep=False, model=False):
        mono, wall = time.monotonic(), time.time()
        check_clock(self.claim['monotonic_origin'], self.claim['unix_origin'], c.TOTAL_SECONDS, mono, wall, 'overall')
        if model:
            check_clock(self.claim['monotonic_origin'], self.claim['unix_origin'], c.MODEL_SECONDS, mono, wall, 'model work')
            if self.evaluation_clock is not None:
                check_clock(self.evaluation_clock['monotonic_origin'], self.evaluation_clock['unix_origin'],
                            c.EVALUATION_SECONDS, mono, wall, 'original aggregate evaluation')
        if deep or mono - self.last_deep >= 5:
            self.last_deep = mono
            used = sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())
            c.require(used < c.MAX_OUTPUT, 'Actual evidence-byte cap')
            c.require(shutil.disk_usage(c.ROOT).free >= c.MIN_DISK, 'Actual free-disk floor')
            rss = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                try: rss += child.memory_info().rss
                except psutil.NoSuchProcess: pass
            self.peak_rss = max(self.peak_rss, rss)
            c.require(rss < c.MAX_RSS, 'Actual aggregate owned RSS cap')


def run_stage(plan, expected, mode, spec, container, seconds, envelope):
    from tools.local_player_qualification_128.clock import _WindowsJob
    container.mkdir(parents=True, exist_ok=False)
    folder = container / 'worker'
    folder.mkdir()
    model = mode != 'audit'
    begun, wall = time.monotonic(), time.time()
    args = [sys.executable, '-B', '-u', '-m', c.PACKAGE + '.worker', '--mode', mode,
            '--plan-sha256', expected, '--output', str(folder), '--claim-sha256', c.sha(c.OUT/'claim.json')]
    if spec: args += ['--fit-id', spec['fit_id']]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(c.ROOT/'src') + os.pathsep + str(c.ROOT),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    proc = job = identity = None
    seen, cleanup = {}, []
    error, status = None, 'failed'
    c.atomic_json(c.OUT/'stage.json', dict(time=c.now(), mode=mode, fit=spec,
        stage='starting', output=c.artifact(c.PLAN), monotonic=begun))
    try:
        envelope.guard(True, model=model)
        with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(args, cwd=c.ROOT, env=env, stdout=stdout, stderr=stderr,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            identity = dict(pid=proc.pid, created=psutil.Process(proc.pid).create_time())
            job = _WindowsJob(proc.pid)
            c.save(container/'owner.json', dict(time=c.now(), identity=identity, args=args, mode=mode,
                fit=spec, monotonic_origin=begun, unix_origin=wall, maximum_seconds=seconds,
                plan_sha256=expected, global_claim=c.artifact(c.OUT/'claim.json'), windows_job_assigned=True,
                evaluation_clock=envelope.evaluation_clock if mode == 'diagnostics' else None))
            while proc.poll() is None:
                envelope.guard(model=model)
                check_clock(begun, wall, seconds, time.monotonic(), time.time(), mode)
                try:
                    parent = psutil.Process(proc.pid)
                    c.require(parent.create_time() == identity['created'], 'Exact worker identity')
                    for member in [parent] + parent.children(recursive=True):
                        try:
                            item = dict(pid=member.pid, created=member.create_time())
                            seen[(item['pid'], item['created'])] = item
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                time.sleep(.25)
            envelope.guard(True, model=model)
            check_clock(begun, wall, seconds, time.monotonic(), time.time(), mode + ' exit')
            c.require(proc.returncode == 0, 'Worker failed; no implicit retry')
            receipt = c.read(folder/'result.json')
            c.require(receipt['status'] == 'passed' and receipt['mode'] == mode and receipt['fit'] == spec,
                      'Exact completed stage result')
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
        for item in [*seen.values(), *([identity] if identity else [])]:
            try: require_absent(item)
            except BaseException as exc: cleanup.append(repr(exc))
        if cleanup: status = 'failed'
    result = dict(time=c.now(), status=status, mode=mode, fit=spec, identity=identity,
        worker_exit_code=None if proc is None else proc.returncode, observed_owned_identities=list(seen.values()),
        cleanup_errors=cleanup, error=error, monotonic_origin=begun, unix_origin=wall,
        monotonic_end=time.monotonic(), unix_end=time.time(), maximum_seconds=seconds,
        plan_sha256=expected, retries=0)
    c.save(container/'execution.json', result)
    c.require(status == 'passed', 'Failed stage preserved: ' + str(container))
    return result


def run(expected):
    c.require(c.sha(c.PLAN) == expected, 'Exact frozen plan')
    plan = c.read(c.PLAN)
    c.require(all(plan.get(key) == value for key, value in contract().items()), 'Frozen execution contract')
    c.verify(plan['inputs'])
    for name in ('claim.json', 'execution.json', 'failure.json', 'completion.json'):
        c.require(not (c.OUT/name).exists(), 'No duplicate launch or clock reset')
    psutil.Process().cpu_affinity(plan['supervisor_affinity'])
    claim = dict(time=c.now(), pid=os.getpid(), created=psutil.Process().create_time(),
        monotonic_origin=time.monotonic(), unix_origin=time.time(), plan=c.artifact(c.PLAN),
        automatic_retries=0, model_seconds=c.MODEL_SECONDS, total_seconds=c.TOTAL_SECONDS)
    c.save(c.OUT/'claim.json', claim)
    envelope = Envelope(claim)
    stages, error, status = [], None, 'failed'
    evaluation_start = None
    try:
        stages.append(run_stage(plan, expected, 'qualification', None, c.OUT/'qualification', c.QUALIFICATION_SECONDS, envelope))
        c.save(c.OUT/'qualification_gate.json', dict(status='passed', result=c.artifact(c.OUT/'qualification/worker/result.json'),
            actual_exit=c.artifact(c.OUT/'qualification/execution.json'), plan=c.artifact(c.PLAN)))
        for spec in plan['fits']:
            stages.append(run_stage(plan, expected, 'fit', spec, c.OUT/'fits'/spec['fit_id'], c.FIT_SECONDS, envelope))
        evaluation_start = (time.monotonic(), time.time())
        envelope.evaluation_clock = dict(monotonic_origin=evaluation_start[0], unix_origin=evaluation_start[1], maximum_seconds=c.EVALUATION_SECONDS)
        c.save(c.OUT/'evaluation_start.json', envelope.evaluation_clock)
        for spec in plan['fits']:
            remaining = min(evaluation_start[0]+c.EVALUATION_SECONDS-time.monotonic(),
                            evaluation_start[1]+c.EVALUATION_SECONDS-time.time())
            c.require(remaining > 0, 'Actual aggregate evaluation cap')
            stages.append(run_stage(plan, expected, 'diagnostics', spec, c.OUT/'diagnostics'/spec['fit_id'], remaining, envelope))
        envelope.guard(True, model=True)
        stages.append(run_stage(plan, expected, 'audit', None, c.OUT/'audit', c.AUDIT_SECONDS, envelope))
        envelope.guard(True)
        c.verify(plan['inputs'], boundary=envelope.guard)
        status = 'completed_pending_independent_acceptance'
    except BaseException:
        error = traceback.format_exc()
    result = dict(time=c.now(), status=status, plan_sha256=expected, stages=stages, error=error,
        monotonic_origin=claim['monotonic_origin'], unix_origin=claim['unix_origin'],
        monotonic_end=time.monotonic(), unix_end=time.time(), peak_owned_rss_bytes=envelope.peak_rss,
        new_games=0, retries=0, automatic_promotion=False)
    c.save(c.OUT/'execution.json', result)
    c.atomic_json(c.OUT/'stage.json', dict(time=c.now(), stage=status, completed_stages=len(stages)))
    if status == 'failed':
        c.save(c.OUT/'failure.json', result)
        return 1
    c.save(c.OUT/'completion.json', dict(time=c.now(), status=status, plan=c.artifact(c.PLAN),
        execution=c.artifact(c.OUT/'execution.json'), audit=c.artifact(c.OUT/'audit/worker/result.json'),
        all_worker_exits_observed=True, scientific_fits=8, scheduled_followup=False))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['freeze', 'run'])
    parser.add_argument('--tests')
    parser.add_argument('--reviews', nargs='*')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.operation == 'freeze':
        print(c.canonical(freeze(Path(args.tests), [Path(p) for p in args.reviews])))
        return 0
    return run(args.plan_sha256)


if __name__ == '__main__': raise SystemExit(main())
