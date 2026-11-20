"""One exact owned process per qualification, fit, diagnostic or saved audit."""
import argparse
import os
import shutil
import time
import traceback
from pathlib import Path
import psutil
from . import common as c
from .control import check_clock


class Boundary:
    def __init__(self, claim, owner, folder, model):
        self.claim, self.owner, self.folder, self.model = claim, owner, folder, model
        self.last_deep = -float('inf')
        self.calls = 0

    def __call__(self):
        self.calls += 1
        mono, wall = time.monotonic(), time.time()
        check_clock(self.claim['monotonic_origin'], self.claim['unix_origin'], c.TOTAL_SECONDS, mono, wall, 'overall')
        if self.model:
            check_clock(self.claim['monotonic_origin'], self.claim['unix_origin'], c.MODEL_SECONDS, mono, wall, 'model')
        check_clock(self.owner['monotonic_origin'], self.owner['unix_origin'], self.owner['maximum_seconds'], mono, wall, 'worker')
        if self.owner.get('evaluation_clock') is not None:
            evaluation = self.owner['evaluation_clock']
            check_clock(evaluation['monotonic_origin'], evaluation['unix_origin'], c.EVALUATION_SECONDS,
                        mono, wall, 'original aggregate evaluation')
        if mono - self.last_deep >= 5:
            self.last_deep = mono
            parent = psutil.Process(self.claim['pid'])
            c.require(parent.create_time() == self.claim['created'] and parent.is_running(), 'Exact owner remains alive')
            c.require(psutil.Process().memory_info().rss < c.MAX_RSS, 'Actual worker RSS cap')
            c.require(shutil.disk_usage(c.ROOT).free >= c.MIN_DISK, 'Actual disk floor')
            c.atomic_json(self.folder/'boundary/progress.json', dict(time=c.now(), mode=self.owner['mode'],
                fit=self.owner['fit'], boundary_calls=self.calls,
                elapsed_seconds=mono-self.owner['monotonic_origin'], model_work=self.model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['qualification', 'fit', 'diagnostics', 'audit'], required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--claim-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-id')
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    c.require(c.sha(c.PLAN) == args.plan_sha256 and c.sha(c.OUT/'claim.json') == args.claim_sha256, 'Exact plan and original clock')
    plan, claim = c.read(c.PLAN), c.read(c.OUT/'claim.json')
    c.require(claim['plan'] == c.artifact(c.PLAN), 'Claim binds execution freeze')
    matches = [spec for spec in plan['fits'] if spec['fit_id'] == args.fit_id]
    spec = matches[0] if len(matches) == 1 else None
    c.require((args.mode in ('fit', 'diagnostics')) == (spec is not None), 'Declared fit identity')
    expected = c.OUT / (('fits' if args.mode == 'fit' else 'diagnostics') if spec else args.mode)
    if spec: expected /= spec['fit_id']
    expected /= 'worker'
    c.require(folder == expected.resolve(), 'Exact fresh worker evidence folder')
    wait_start = time.monotonic()
    owner_path = folder.parent/'owner.json'
    while not owner_path.exists():
        c.require(time.monotonic()-wait_start < 10, 'Bounded owning-job handshake')
        time.sleep(.01)
    owner = c.read(owner_path)
    me = dict(pid=os.getpid(), created=psutil.Process().create_time())
    c.require(owner['identity'] == me and owner['windows_job_assigned'] and owner['fit'] == spec and
              owner['mode'] == args.mode and owner['plan_sha256'] == args.plan_sha256 and
              owner['global_claim'] == c.artifact(c.OUT/'claim.json'), 'Exact Windows job and worker owner')
    if args.mode == 'diagnostics':
        c.require(owner['evaluation_clock'] == c.read(c.OUT/'evaluation_start.json'), 'Original shared evaluator clock')
    else:
        c.require(owner.get('evaluation_clock') is None, 'Evaluation clock only applies to evaluators')
    boundary = Boundary(claim, owner, folder, args.mode != 'audit')
    boundary()
    status, result, error = 'failed', None, None
    try:
        c.verify(plan['worker_inputs'], boundary=boundary)
        if args.mode == 'audit':
            from tools.future_training_rules_20261021 import verify_runtime_sources
            runtime = verify_runtime_sources()
            c.save(folder/'runtime_sources.json', runtime)
            from . import audit
            result = audit.run(plan, folder, boundary=boundary)
        else:
            from . import training
            c.save(folder/'runtime_sources.json', training.startup())
            c.verify([plan['selection']], boundary=boundary)
            selection = c.read(c.resolve(plan['selection']))
            if args.mode == 'qualification':
                result = training.run_qualification(selection, folder, boundary=boundary)
            else:
                gate = c.read(c.OUT/'qualification_gate.json')
                c.require(gate['status'] == 'passed' and gate['plan'] == c.artifact(c.PLAN), 'Passed qualification before science')
                c.verify([gate['result'], gate['actual_exit']], boundary=boundary)
                if args.mode == 'fit':
                    result = training.run_fit(selection, spec['repetition'], spec['breadth'], spec['scope'], folder, boundary=boundary)
                else:
                    fit_folder = c.OUT/'fits'/spec['fit_id']/'worker'
                    fit_execution = c.read(fit_folder.parent/'execution.json')
                    c.require(fit_execution['status'] == 'passed' and fit_execution['worker_exit_code'] == 0,
                              'Completed original fit before evaluation')
                    result = training.run_diagnostics(selection, spec['repetition'], spec['breadth'], spec['scope'],
                        fit_folder, folder, boundary=boundary)
        boundary()
        status = 'passed'
    except BaseException:
        error = traceback.format_exc()
    receipt = dict(status=status, time=c.now(), mode=args.mode, fit=spec, identity=me,
        result=result, error=error, plan_sha256=args.plan_sha256, boundary_calls=boundary.calls,
        new_games=0, automatic_promotion=False)
    c.save(folder/'result.json', receipt)
    if status != 'passed':
        c.save(folder/'failure.json', receipt)
        print(error, flush=True)
        return 1
    print(c.canonical({'status': status, 'mode': args.mode, 'fit': spec}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
