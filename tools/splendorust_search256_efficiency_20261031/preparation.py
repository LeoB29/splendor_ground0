"""Model-free exact freeze for one separately bounded diagnostic attempt."""
import argparse
from pathlib import Path
from . import common as c, runner, panel, analysis, qualification

INHERITED = (
    'baseline_profiles', 'profiles', 'research_profiles',
    'native_artifacts', 'native_artifacts_by_route', 'native_roles_by_route',
    'research_native_artifacts', 'research_native_roles', 'research_native_sha256',
    'graph_parity_fixtures', 'graph_parity_plan', 'unchanged_graph_backend_identity',
    'python', 'python_executable_sha256', 'torch_runtime', 'external_runtime',
    'numpy_version', 'torch_package_version', 'torch_runtime_version',
    'compression_runtime', 'statistics_runtime', 'sampler_api_modules',
    'topology', 'placement', 'source_guard', 'ruleset_id', 'catalogue_sha256',
    'turn_contract', 'request_max_bytes', 'frozen_response_max_bytes',
    'frozen_log_max_bytes', 'research_response_max_bytes', 'research_log_max_bytes',
    'build_receipt',
)


def verify_panel_extraction():
    extraction_path = c.OUT/'prelaunch/panel_extraction_03.json'
    extraction = c.read(extraction_path)
    execution = c.read(c.OUT/'prelaunch/panel_extraction_execution_03.json')
    c.require(extraction['status'] == execution['status'] == 'passed' and
              execution['actual_exit_code'] == 0 and
              execution['receipt'] == c.artifact(extraction_path),
              'Successful actual bounded panel extraction')
    c.require(extraction['eligible_decisions'] == 11066 and extraction['roots'] == 18 and
              extraction['new_models'] == extraction['new_games'] ==
              execution['models'] == execution['games'] == 0 and
              extraction['source_modified'] is False, 'Saved-only complete panel extraction')
    c.verify([extraction['selection_precommit']])
    precommit = c.read(c.ROOT/extraction['selection_precommit']['path'])
    c.require(precommit['selection_started_after_rule_saved'] is True and
              precommit['new_models'] == precommit['new_games'] == 0 and
              precommit['implementation'] == c.artifact(Path(panel.__file__)) and
              precommit['protocol'] == c.artifact(c.PROTOCOL) and
              precommit['selection_rule'] == c.artifact(c.OUT/'selection_rule.json'),
              'Exact implementation and rule committed before extraction')
    c.require(extraction['panel'] == c.artifact(c.OUT/'panel.json') and
              extraction['matrices'] == {phase: c.artifact(c.OUT/(phase+'_matrix.json'))
                                        for phase in ('overhead', 'diagnostic')},
              'Panel and matrices bind completed extraction')
    return c.artifact(extraction_path)


def accepted_suite_reference(old, relative='entrypoint_checks/research.json'):
    """Bind the comparison suite through its already accepted historical seal."""
    c.verify([old['recovery_manifest']])
    manifest = c.read(c.ROOT/old['recovery_manifest']['path'])
    c.verify([manifest['original_tree']])
    inventory = c.read(c.ROOT/manifest['original_tree']['path'])
    c.require(relative in ('entrypoint_checks/research.json', 'qualification_plan.json'),
              'Only declared original qualification references')
    wanted = (c.OLD_OUT/relative).relative_to(c.ROOT).as_posix()
    matches = [item for item in inventory['files'] if item['path'] == wanted]
    c.require(len(matches) == 1, 'Exactly one accepted original complete256 suite')
    c.verify(matches)
    return matches[0]


def freeze():
    c.require(not (c.OUT/'freeze.json').exists(), 'Never replace an existing freeze')
    old_path = c.ACCEPTED_OUT/'freeze.json'
    c.require(c.file_hash(old_path) == c.ACCEPTED_FREEZE_SHA, 'Exact accepted predecessor')
    old = c.read(old_path)
    extraction = verify_panel_extraction()
    suite_reference = accepted_suite_reference(old)
    original_plan_reference = accepted_suite_reference(old, 'qualification_plan.json')
    selected = c.read(c.OUT/'panel.json')
    analysis.panel_roots(selected)
    c.require(c.read(c.OUT/'selection_rule.json') == panel.selection_rule(),
              'Selection committed before extraction')
    for phase in ('overhead', 'diagnostic'):
        c.require(c.read(c.OUT/(phase+'_matrix.json')) == panel.request_matrix(selected, phase),
                  'Exact prospective request matrix')
    test_path = c.OUT/'prelaunch/model_free_tests.json'
    tests = c.read(test_path)
    c.require(tests['exit_code'] == 0 and tests['models'] == tests['games'] == 0,
              'Passed model-free tests before freezing')
    c.verify(tests['inputs'] + [tests['log']])
    review_paths = [c.OUT/'prelaunch'/name for name in
                    ('independent_review_worker.json', 'independent_review_runner.json',
                     'independent_review_design.json')]
    for path in review_paths:
        review = c.read(path)
        c.require(review['approved'] is True and not review.get('blocking_findings', []),
                  'Independent implementation review passed')
        c.verify(review['inputs'])
    plan = qualification.build_plan()
    c.require(plan['accepted_suite'] == suite_reference and
              plan['accepted_plan'] == original_plan_reference,
              'New qualification plan cites authenticated original suite and plan')
    plan_path = c.OUT/'qualification_plan.json'
    if plan_path.exists():
        c.require(c.read(plan_path) == plan, 'Do not replace qualification plan')
    else:
        c.save(plan_path, plan)
    value = {name: old[name] for name in INHERITED}
    value.update(runner.contract(), time=c.now(), protocol=c.artifact(c.PROTOCOL),
        study_kind='saved_position_efficiency_diagnostic',
        accepted_predecessor=c.artifact(old_path),
        accepted_predecessor_receipt=c.artifact(c.ROOT/
            'outputs/splendorust_search256_recovery_20261030_02_closeout/independent_acceptance_01.json'),
        panel=c.artifact(c.OUT/'panel.json'), qualification_plan=c.artifact(plan_path),
        overhead_matrix=c.artifact(c.OUT/'overhead_matrix.json'),
        diagnostic_matrix=c.artifact(c.OUT/'diagnostic_matrix.json'),
        selection_rule=c.artifact(c.OUT/'selection_rule.json'),
        overhead_limits=analysis.OVERHEAD_LIMITS,
        bootstrap=dict(seed=analysis.BOOTSTRAP_SEED, draws=analysis.BOOTSTRAP_DRAWS),
        source_preservation_scope='All inherited frozen inputs retained; new version only',
        weights_unchanged=True, precision_unchanged=True, main_games=0,
        operational_games=0, max_graph_builds=2, original_failed_attempts_preserved=True,
        model_free_tests=c.artifact(test_path),
        panel_extraction=extraction, original256_suite_reference=suite_reference,
        implementation_reviews=[c.artifact(path) for path in review_paths])
    pins = {}
    def add(item):
        previous = pins.setdefault(item['path'], item)
        c.require(previous == item, 'Conflicting input descriptor: '+item['path'])
    for item in old['inputs'] + selected['inputs']:
        add(item)
    add(suite_reference)
    add(original_plan_reference)
    add(value['accepted_predecessor_receipt'])
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((c.ROOT/'tests').glob('test_splendorust_search256_efficiency*_20261031.py'))
    paths += [p for p in c.OUT.rglob('*') if p.is_file()]
    paths += [c.PROTOCOL, old_path]
    for path in paths:
        add(c.artifact(path))
    value['inputs'] = sorted(pins.values(), key=lambda x: x['path'])
    runner.validate_freeze_contract(value)
    c.verify(value['inputs'])
    return dict(status='frozen_not_launched', freeze=c.save(c.OUT/'freeze.json', value),
                inputs=len(pins), new_models=0, new_games=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze'])
    parser.parse_args()
    print(c.canonical(freeze()), flush=True)
