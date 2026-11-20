"""Complete, source-bound model-free qualification for the separate recovery."""
import argparse
import json
import subprocess
import sys

from . import common as c


SCHEMA = 'independent_data_recovery_complete_pytest_v1'
MINIMUM_TESTS = 284


def test_paths():
    """Keep the accepted predecessor suite and include every recovery test file."""
    predecessor = sorted((c.ROOT / 'tests').glob('*independent_data*20261107*.py'))
    recovery = sorted((c.ROOT / 'tests').glob('*independent_data*recovery*20261109*.py'))
    required = c.ROOT / 'tests/test_independent_data_recovery_20261109.py'
    c.require(len(predecessor) == 10 and required in recovery,
              'Complete predecessor suite and independent recovery counterexamples')
    return predecessor + recovery


def complete_summary(summary):
    """Reject partial, skipped, model-loading and type-aliased test proofs."""
    keys = {'tests_collected', 'tests_passed', 'failures', 'skipped',
            'tensor_runtime_loaded', 'models', 'games'}
    c.require(type(summary) is dict and set(summary) == keys,
              'Exact complete pytest summary schema')
    for name in ('tests_collected', 'tests_passed', 'failures', 'skipped', 'models', 'games'):
        c.require(type(summary[name]) is int, 'Exact integer test accounting')
    c.require(summary['tests_collected'] == summary['tests_passed'] >= MINIMUM_TESTS
              and summary['failures'] == summary['skipped'] == summary['models'] == summary['games'] == 0
              and summary['tensor_runtime_loaded'] is False,
              'Every collected model-free test passes without skips')
    return True


def suite():
    import pytest

    class Counts:
        collected = passed = failed = skipped = 0

        def pytest_collection_finish(self, session):
            self.collected = len(session.items)

        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.failed += 1
            if report.skipped:
                self.skipped += 1
            if report.when == 'call' and report.passed:
                self.passed += 1

    counts = Counts()
    result = pytest.main(['-q', *map(str, test_paths())], plugins=[counts])
    summary = dict(tests_collected=counts.collected, tests_passed=counts.passed,
                   failures=counts.failed, skipped=counts.skipped,
                   tensor_runtime_loaded='torch' in sys.modules, models=0, games=0)
    print(c.canonical(summary), flush=True)
    try:
        complete_summary(summary)
    except ValueError:
        return 1
    return 0 if result == 0 else 1


def record(name):
    from .owner import child_environment, source_inputs
    c.require(type(name) is str and name.isalnum(), 'Simple exclusive preflight identity')
    folder = c.OUT / 'prelaunch'
    folder.mkdir(parents=True, exist_ok=True)
    inputs = source_inputs()
    paths = test_paths()
    log = folder / (name + '.log')
    env = child_environment(0)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    with log.open('xb') as stream:
        process = subprocess.run([sys.executable, '-B', '-u', '-m', c.PACKAGE + '.preflight', 'suite'],
                                 cwd=c.ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=600)
    c.verify(inputs)
    lines = log.read_text(encoding='utf-8').splitlines()
    try:
        summary = json.loads(lines[-1])
    except (IndexError, ValueError, TypeError):
        summary = None
    try:
        complete_summary(summary)
        summary_valid = True
    except (ValueError, TypeError):
        summary_valid = False
    exit_code = process.returncode if process.returncode != 0 else (0 if summary_valid else 1)
    proof = c.save(folder / (name + '.json'), dict(
        schema=SCHEMA, exit_code=exit_code, observed_process_exit_code=process.returncode,
        inputs=inputs, runner=c.artifact(__file__), full_suite=True,
        test_files=[c.artifact(p) for p in paths], suite_summary=summary,
        log=c.artifact(log), models=0, games=0, time=c.now(),
        model_free_assertion='Torch absent after complete predecessor and recovery suites'))
    print(c.canonical(proof), flush=True)
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('suite', 'record'))
    parser.add_argument('--name', default='tests01')
    args = parser.parse_args()
    raise SystemExit(suite() if args.mode == 'suite' else record(args.name))
