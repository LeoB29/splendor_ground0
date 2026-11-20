"""Model-free binding of the accepted design and one additive execution attempt."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from tools.future_training_rules_20261021 import verify_runtime_sources
from . import common as c

DESIGN = c.ROOT / 'outputs/decision_quality_design_20261027_01'
PANEL = c.ROOT / 'outputs/decision_quality_preparation_20261027_01/panel.json'
PRIOR = c.ROOT / 'outputs/splendorust_isolated_reliability_20261026_01'
RECOVERY = c.ROOT / 'outputs/splendorust_reliability_finalization_20261026_01'
BUILD = c.ROOT / 'outputs/splendorust_decision_quality_build_20261027_01'
FAILED = c.ROOT / 'outputs/splendorust_decision_quality_20261027_01'
FAILED_FREEZE_SHA = 'a07c1d54dbbf9a67eb6a202e5c6d08717949ef50de8bcfee5413581e8d4d20b0'
PRESERVATION_BEFORE_SHA = 'b6bfefa4acf99917a1cbed86b78c488edb3e47ca5fdd58c923c3f18c7e0e70af'
DESIGN_SEAL_SHA = '3fa5c6d3841f8fa2de02056398945e199127127dae38c579c4cae19fd3d22fc6'
COMPLETION_SHA = '2731be515c4ad2ece8cb3a7286dc52df84c9218a18c5c04ea5cd7c232233731c'
PRIOR_SHA = 'a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497'
RECOVERY_REVIEW_SHA = '54f0d7a4babb1970a2d47118988f66683f33f3dbc1bd8878c837f07ca4836c9a'
RUNTIME_KEYS = ('compression_runtime', 'external_runtime', 'native_artifacts',
    'numpy_version', 'placement', 'sampler_api_modules', 'source_guard', 'topology',
    'torch_package_version', 'torch_runtime', 'torch_runtime_version', 'python',
    'python_executable_sha256', 'unchanged_graph_backend_identity')


def require(condition, message):
    if not condition:
        raise ValueError('DECISION_QUALITY_PREPARATION: ' + message)


def load_design():
    require(c.file_hash(DESIGN/'design_seal.json') == DESIGN_SEAL_SHA, 'Design seal changed')
    require(c.file_hash(DESIGN/'preparation_complete.json') == COMPLETION_SHA, 'Preparation acceptance changed')
    seal = c.read(DESIGN/'design_seal.json')
    done = c.read(DESIGN/'preparation_complete.json')
    require(seal['status'] == 'sealed_preparation_only' and not seal['execution_freeze']
            and done['status'] == 'accepted_preparation_only', 'Prepared acceptance status')
    c.verify(seal['artifacts'])
    plan, matrix, panel = c.read(DESIGN/'plan.json'), c.read(DESIGN/'prepared_request_matrix.json'), c.read(PANEL)
    require(panel['positions'] == 24 and panel['unique_paired_worlds'] == 24,
            'Fixed24 independent paired-world roots')
    require(len(matrix['requests']) == 156 and matrix['request_counts'] == {
        'frozen_reference_reproduction': 24, 'research_pilot': 30,
        'research_main': 90, 'research_reproduction': 12}, 'Fixed156 request matrix')
    require(plan['resources']['job_seconds'] == 7200 and
            plan['resources']['stage_seconds'] == {'qualification': 900, 'diagnostic': 4500, 'finalization': 1800},
            'Prepared stage limits differ')
    return plan, matrix, panel, seal


def preserved_attempt_inputs():
    """Bind the failed attempt without replacing its completion or evidence."""
    require(c.file_hash(FAILED/'freeze.json') == FAILED_FREEZE_SHA, 'Failed-attempt freeze changed')
    before_path = c.OUT/'prelaunch/prior_attempt_preservation_before.json'
    require(c.file_hash(before_path) == PRESERVATION_BEFORE_SHA, 'Before-repair preservation inventory changed')
    before, failed = c.read(before_path), c.read(FAILED/'freeze.json')
    require(before['old_freeze'] == c.artifact(FAILED/'freeze.json') and
            before['frozen_input_count'] == len(failed['inputs']) == 1776,
            'Original failed-attempt binding')
    require(c.read(FAILED/'completion.json')['status'] == 'incomplete',
            'The failed attempt must remain incomplete')
    items = before['prior_attempt_artifacts']
    require(len(items) == 145 and len({x['path'] for x in items}) == 145,
            'Exact original attempt and postrun inventory')
    c.verify(failed['inputs'])
    c.verify(items)
    return failed, failed['inputs'] + items + [c.artifact(before_path)]


def source_inventory(prior, panel, seal):
    """Finite provenance reuse; no recursive historical evidence audit."""
    pins = {}
    def add(item):
        require(c.artifact(c.ROOT/item['path']) == item, 'Changed input: ' + item['path'])
        if item['path'] in pins:
            require(pins[item['path']] == item, 'Conflicting input hash')
        pins[item['path']] = item
    _, preserved = preserved_attempt_inputs()
    for item in preserved:
        add(item)
    for item in prior['inputs']:
        if item['path'].startswith(('tools/', 'src/', 'outputs/external_opponent_search_20261018_02/',
                'outputs/external_strength_screen_20261018_02/',
                'outputs/splendorust_search_scale_20261024_01/profiles/')):
            add(item)
    for item in prior['native_artifacts']:
        add(item)
    for item in seal['artifacts']:
        add(item)
    recovered = c.read(RECOVERY/'independent_receipt_review.json')
    require(c.file_hash(RECOVERY/'independent_receipt_review.json') == RECOVERY_REVIEW_SHA
            and recovered['accepted'] is True, 'Accepted recovery binding')
    recovery_pins = {x['path']: x for x in recovered['consumed_receipts_and_source_artifacts']}
    original_seal = PRIOR/'completed_evidence_seal.json'
    add(recovery_pins[original_seal.relative_to(c.ROOT).as_posix()])
    old_items = {x['path']: x for x in c.read(original_seal)['artifacts']}
    for root in panel['roots']:
        require(old_items[root['source']['path']] == root['source'], 'Selected source seal binding')
        add(root['source'])
        add(old_items[root['source_worker_clock_file']])
    # Timing/evidence metadata is used for an explicit heuristic cost forecast only.
    performance = PRIOR/'performance.json'
    rel = performance.relative_to(c.ROOT).as_posix()
    add(recovery_pins.get(rel, old_items.get(rel)))
    for path in (PRIOR/'freeze.json', RECOVERY/'independent_receipt_review.json',
                 DESIGN/'design_seal.json', DESIGN/'preparation_complete.json',
                 c.ROOT/'docs/engineering/splendorust_decision_quality_execution_20261027_v2.md',
                 c.ROOT/'docs/engineering/future_training_rules_20261021.md'):
        add(c.artifact(path))
    for folder in (c.ROOT/'tools/splendorust_decision_quality_20261027_v2',):
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                add(c.artifact(path))
    for path in (c.ROOT/'tests').glob('test_splendorust_decision_quality*20261027_v2*.py'):
        add(c.artifact(path))
    # The controller regression imports this existing model-free profile fixture.
    add(c.artifact(c.ROOT/'tests/test_splendorust_boundary_timing_analyze.py'))
    for folder in (c.OUT/'prelaunch', c.OUT/'profiles'):
        if folder.exists():
            for path in folder.rglob('*'):
                if path.is_file(): add(c.artifact(path))
    for name in ('qualification_plan.json', 'resource_forecast.json', 'runtime_sources.json'):
        path = c.OUT/name
        if path.exists(): add(c.artifact(path))
    return pins


def resource_forecast(matrix, qualification_plan):
    """Cost-only planning extrapolation; the six-root measured gate remains mandatory."""
    perf = c.read(PRIOR/'performance.json')
    maximum64 = perf['players']['challenger']['completed_arrival_seconds']['max']
    old_seal = c.read(PRIOR/'completed_evidence_seal.json')
    sealed_bytes = sum(x['bytes'] for x in old_seal['artifacts'])
    per_move_bytes = sealed_bytes / 11020
    equivalents = sum(row['quota']/64 for row in matrix['requests'])
    #104 suite decisions are conservatively charged at64; parity has6of each quota plus1frozen64.
    qual_equivalents = 104 + 6*(1+4+16) + 1
    diagnostic = 1.5*maximum64*equivalents + 300 + 120 + 300
    qualification = 1.5*maximum64*qual_equivalents + 180 + 60
    prefinal = int(1.5*per_move_bytes*(equivalents+qual_equivalents) + 64*1024**2)
    final = 256*1024**2
    require(diagnostic < 4500 and qualification < 900 and prefinal < 512*1024**2
            and final < 512*1024**2, 'Model-free projection does not fit prepared envelope')
    return dict(schema='decision_quality_model_free_forecast_v1', status='passed_projection_only',
        source_performance=c.artifact(PRIOR/'performance.json'),
        source_seal=c.artifact(PRIOR/'completed_evidence_seal.json'),
        source_maximum64_seconds=maximum64, source_sealed_bytes=sealed_bytes,
        source_live_decisions=11020, source_physical_bytes_per_live_decision=per_move_bytes,
        diagnostic64_equivalents=equivalents, qualification64_equivalents=qual_equivalents,
        multiplier=1.5, qualification_seconds=qualification, diagnostic_seconds=diagnostic,
        finalization_seconds=1200, projected_prefinal_bytes=prefinal, projected_final_bytes=final,
        projected_total_bytes=prefinal+final, qualification_plan_sha256=c.digest(qualification_plan),
        caveats=['Linear quota extrapolation is a planning assumption, not measured higher-quota performance.',
                 'Historical physical bytes include all original worker/evidence/audit costs, averaged over live moves.',
                 'Extra64MiB covers new startup/telemetry/metadata; final reserve remains512MiB.',
                 'Fixed six-root measured1.5x time/evidence gate must pass before expansion.',
                 'Hard clocks and byte/RAM guards override every projection.'],
        builds_counted_separately_before_freeze=True, no_models_run=True)


def save_fixed(path, value):
    if Path(path).exists():
        require(c.read(path) == value, 'Never replace prepared execution input: '+str(path))
    else:
        c.save(path, value)
    return c.artifact(path)


def nested_artifacts(value):
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= set(value) and isinstance(value['path'],str):
            yield {key:value[key] for key in ('path','sha256','bytes')}
        for child in value.values():
            yield from nested_artifacts(child)
    elif isinstance(value,list):
        for child in value:
            yield from nested_artifacts(child)


def freeze():
    """Called after build, model-free tests and implementation review; never launches."""
    from . import adapters, qualification
    require(not (c.OUT/'freeze.json').exists(), 'Never replace an execution freeze')
    sources = verify_runtime_sources()
    failed, _ = preserved_attempt_inputs()
    plan, matrix, panel, seal = load_design()
    require(c.file_hash(PRIOR/'freeze.json') == PRIOR_SHA, 'Original reliability freeze changed')
    prior = c.read(PRIOR/'freeze.json')
    require(shutil.disk_usage(c.ROOT).free >= 7*1024**3, 'Initial7GiB disk reserve')
    require((c.OUT/'prelaunch/model_free_tests.json').is_file(), 'Model-free test receipt missing')
    tests = c.read(c.OUT/'prelaunch/model_free_tests.json')
    require(tests['passed'] is True and tests['exit_code'] == 0, 'Model-free tests did not pass')
    review = c.read(c.OUT/'prelaunch/implementation_review.json')
    require(review['approved'] is True, 'Independent implementation review did not approve')
    c.verify(review['inputs'])
    profile_items = {}
    for quota in (16,64,256,1024):
        profile = adapters.profile(quota)
        require(profile == c.read(c.ROOT/failed['research_profiles'][str(quota)]['path']),
                'Research player profile changed during checker-only repair')
        profile_items[str(quota)] = save_fixed(c.OUT/f'profiles/research{quota}.json', profile)
    qplan = qualification.build_plan()
    require(qplan == c.read(c.ROOT/failed['qualification_plan']['path']),
            'Qualification workload changed during checker-only repair')
    qitem = save_fixed(c.OUT/'qualification_plan.json', qplan)
    forecast = resource_forecast(matrix, qplan)
    save_fixed(c.OUT/'resource_forecast.json', forecast)
    save_fixed(c.OUT/'runtime_sources.json', sources)
    #Builder supplies the exact finite source/binary manifest; no target-cache scan.
    build_receipt_path = BUILD/'build_receipt.json'
    build = c.read(build_receipt_path)
    require(build['status'] == 'compiled_not_model_qualified' and
            build['copied_core_agents_unchanged'] is True and
            build['original_sources_modified'] is False and
            build['model_inference_executed'] is False and build['search_executed'] is False,
            'New diagnostic bridge not safely built')
    pins = source_inventory(prior, panel, seal)
    build_inputs = list(nested_artifacts(build))
    postbuild, changed_cache = [], []
    for item in c.read(c.ROOT/build['source_inventory']['path']):
        observed = c.artifact(c.ROOT/item['path'])
        if observed != item:
            require(item['path'] == (BUILD/'cargo_home/.global-cache').relative_to(c.ROOT).as_posix(),
                    'Native source changed during build: '+item['path'])
            changed_cache.append(dict(before=item,after=observed,
                explanation='Cargo build cache metadata; not loaded by deployed worker'))
        postbuild.append(observed)
    postbuild_receipt = save_fixed(c.OUT/'prelaunch/build_freeze_inventory.json', dict(
        status='postbuild_inputs_authenticated', prebuild_inventory=build['source_inventory'],
        artifacts=postbuild, declared_build_metadata_changes=changed_cache,
        source_or_dependency_code_changes=False))
    pins[postbuild_receipt['path']] = postbuild_receipt
    build_inputs += postbuild
    parity = prior['graph_parity_plan']
    extra_inputs = [parity] + list(nested_artifacts(c.read(c.ROOT/parity['path']))) + list(nested_artifacts(qplan))
    for item in build_inputs + extra_inputs:
        require(c.artifact(c.ROOT/item['path']) == item, 'New build artifact changed')
        pins[item['path']] = item
    pins[build_receipt_path.relative_to(c.ROOT).as_posix()] = c.artifact(build_receipt_path)
    frozen = {key:deepcopy(prior[key]) for key in RUNTIME_KEYS}
    frozen.update(schema='splendorust_decision_quality_execution_v2', time=c.now(), attempt=2,
        prior_failed_attempt=dict(freeze=c.artifact(FAILED/'freeze.json'),
            preservation_before=c.artifact(c.OUT/'prelaunch/prior_attempt_preservation_before.json'),
            preserved_frozen_inputs=1776, preserved_attempt_files=145, status_remains='incomplete'),
        checker_repair_only=True, native_recompiled=False,
        protocol=c.artifact(c.ROOT/'docs/engineering/splendorust_decision_quality_execution_20261027_v2.md'),
        prepared_design_seal=c.artifact(DESIGN/'design_seal.json'),
        panel=c.artifact(PANEL), request_matrix=c.artifact(DESIGN/'prepared_request_matrix.json'),
        qualification_plan=qitem, research_profiles=profile_items,
        research_native_sha256=adapters.profile(64)['native_binary_sha256'],
        research_native_artifacts=list(build['natives'].values()),
        research_native_roles={item['path']:role for role,item in build['natives'].items()},
        graph_parity_plan=parity, graph_parity_fixtures=c.read(c.ROOT/parity['path'])['fixtures'],
        build_receipt=c.artifact(build_receipt_path),
        ruleset_id=plan['rules']['ruleset_id'], catalogue_sha256=plan['rules']['catalogue_sha256'],
        turn_contract=plan['rules']['turn_contract'], source_guard=sources,
        job_seconds=7200, collection_cutoff_seconds=5400,
        stage_seconds={'qualification':900,'diagnostic':4500,'finalization':1800},
        max_evidence_bytes=1024**3, final_reserve_bytes=512*1024**2,
        prefinal_evidence_bytes=512*1024**2, max_rss_bytes=2*1024**3,
        minimum_available_ram_bytes=2*1024**3, initial_ram_stability_seconds=60,
        maximum_ram_wait_seconds=300, initial_free_disk_bytes=7*1024**3,
        minimum_free_disk_bytes=5*1024**3, seconds_per_move=1., internal_seconds=.9,
        research_seconds_per_move=30., research_internal_seconds=29.,
        research_log_max_bytes=128*1024**2, frozen_log_max_bytes=16*1024**2,
        research_response_max_bytes=4*1024**2, frozen_response_max_bytes=256*1024,
        request_max_bytes=1024**2,
        placement_mode='isolated', max_workers=4, max_graph_builds=4,
        total_requests=156, original_reproductions=24, research_requests=120,
        research_reproductions=12, qualification_decisions=123,
        external_candidate_testing='paused_by_user', larger_data_deferred=True,
        automatic_retries=0, failed_decision_retries=0, new_games=0, training=False,
        source_inventory_scope='finite frozen source/runtime chain plus24selected moves/clocks and prepared receipts',
        resource_forecast=c.artifact(c.OUT/'resource_forecast.json'),
        python=sys.version, inputs=sorted(pins.values(), key=lambda x:x['path']))
    from .runner import validate_freeze_contract
    unchanged_contract = validate_freeze_contract(frozen)
    require(all(frozen[key] == failed[key] for key in unchanged_contract),
            'Players, clocks or resource bounds changed')
    require(all(frozen[key] == failed[key] for key in RUNTIME_KEYS if key != 'python') and
            all(frozen[key] == failed[key] for key in ('panel','request_matrix','research_native_sha256',
                'research_native_artifacts','research_native_roles','build_receipt','graph_parity_plan')),
            'Frozen player/runtime/panel identity changed')
    c.save(c.OUT/'freeze.json', frozen)
    print(json.dumps(dict(status='frozen_not_launched', freeze=c.artifact(c.OUT/'freeze.json'),
                          inputs=len(frozen['inputs']))))
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze'])
    parser.parse_args()
    freeze()
