"""Finite immutable input binding before any actual policy qualification."""
from copy import deepcopy
import secrets
import shutil
import sys
from pathlib import Path
from importlib.metadata import version

from . import common as c
from tools import future_training_rules_20261021 as rules_guard

PRIOR = c.ROOT/'outputs/splendorust_isolated_reliability_20261026_01'
FINAL = c.ROOT/'outputs/splendorust_reliability_finalization_20261026_01'
BRECK = c.ROOT/'outputs/breck_screen_20261026_01'
BRECK_ACCEPTED_SHA = 'f43c7a92e1dbb53b1835615870a4508d3c7313f2840f5d1f5134b5193b33aa08'
PRIOR_SHA = 'a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497'
ACCEPTED_REVIEW_SHA = '54f0d7a4babb1970a2d47118988f66683f33f3dbc1bd8878c837f07ca4836c9a'
RUNTIME_KEYS = ('compression_runtime','external_runtime','native_artifacts','numpy_version',
    'placement','sampler_api_modules','source_guard','topology','torch_package_version',
    'torch_runtime','torch_runtime_version','python','python_executable_sha256')


def profile():
    from . import adapter, network
    return dict(profile_id=adapter.PROFILE_ID, repository='domdomegg/splendor',
        upstream_commit='426fa50e06fe99b40620131ca962f418824bb73a',
        weight_git_blob='73153f137bd98c23cb1fb36300efde98757c600a',
        weight_sha256=network.WEIGHT_SHA256, historical_weights=True,
        ruleset_id=rules_guard.RULESET_ID,catalogue_sha256=rules_guard.CATALOGUE_SHA256,
        turn_contract=rules_guard.TURN_CONTRACT, no_search=True, no_training=True,
        precision=dict(weights='float32',accumulation='float64',hidden_storage='float32',final_logits='float64'),
        encoder_reference='independently ported pinned Rust append-order semantics; source engine not executed',
        numerical_reference='literal published browser sequential double loops',
        dimensions=list(network.DIMENSIONS),activation='ReLU',
        source_output='PPO policy logits only; value weights omitted by upstream export',critic=False,
        completion_domain=adapter.COMPLETION_DOMAIN,
        completion='one deterministic observation-hash-ordered unseen corrected card per unknown reserve; no referee seeds',
        returns='policy-selected one-token discards in a lawful pre-reveal shadow; all calls share the whole-turn clock',
        payment='source colored-first, gold shortfall',
        noble='source noble-index policy on corrected eligible support in the pre-reveal shadow',
        forced_pass='source index76 only when corrected PASS is legal',
        maximum_policy_calls_per_decision=adapter.MAX_POLICY_CALLS,
        auxiliary_convention=adapter.AUXILIARY_CONVENTION,
        action_table=adapter.ACTION_TABLE,action_table_sha256=adapter.ACTION_TABLE_SHA256,
        tie_rule='first source index attaining maximum among corrected admissible outputs',
        feature_tolerance={'atol':1e-7,'rtol':0.0},logit_tolerance={'atol':1e-6,'rtol':1e-6},
        timing_seconds=1.,qualification_max_seconds=.7,
        never_claimed_identical_to_historical_strongest_player=True)


def finite_inputs(prior):
    """Preserve reviewed sources without re-expanding old evidence trees."""
    paths={}
    for item in prior['inputs']:
        p=item['path']
        if p.startswith(('tools/','src/','outputs/external_opponent_search_20261018_02/',
                         'outputs/external_strength_screen_20261018_02/',
                         'outputs/splendorust_search_scale_20261024_01/profiles/')):
            c.require(c.artifact(c.ROOT/p)==item,'Changed inherited source/runtime input: '+p)
            paths[p]=item
    for item in prior['native_artifacts']:
        c.require(c.artifact(c.ROOT/item['path'])==item,'Native binary identity changed')
        paths[item['path']]=item
    for folder in (c.ROOT/'tools/dom_screen_20261027', c.ROOT/'external/dom_splendor_426fa50'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                paths[p.relative_to(c.ROOT).as_posix()]=c.artifact(p)
    for p in (PRIOR/'freeze.json',FINAL/'freeze.json',FINAL/'independent_receipt_review.json',
              FINAL/'recovered_acceptance.json',FINAL/'completion.json',FINAL/'execution.json',
              BRECK/'freeze.json',BRECK/'independent_results_review.json',BRECK/'completion.json',
              BRECK/'screen_assessment.json',BRECK/'execution.json',BRECK/'watchdog_execution.json',BRECK/'guardian_os_exit.json',
              c.PROTOCOL,c.ROOT/'docs/engineering/independent_one_second_comparison_20261026.md',
              c.ROOT/'docs/engineering/future_training_rules_20261021.md',
              c.ROOT/'tests/test_splendorust_live_boundary_adapters.py'):
        c.require(p.is_file(),'Required finite acceptance/protocol missing: '+str(p))
        paths[p.relative_to(c.ROOT).as_posix()]=c.artifact(p)
    for p in c.OUT.glob('*'):
        if p.is_file() and p.name not in ('freeze.json','freeze.sha256'):
            paths[p.relative_to(c.ROOT).as_posix()]=c.artifact(p)
    for p in (c.OUT/'prelaunch').rglob('*'):
        if p.is_file():
            paths[p.relative_to(c.ROOT).as_posix()]=c.artifact(p)
    for p in (c.ROOT/'tests').glob('test_dom*20261027.py'):
        paths[p.relative_to(c.ROOT).as_posix()]=c.artifact(p)
    return sorted(paths.values(),key=lambda x:x['path'])


def check_known_schedules(planned):
    """Bounded schedule-only collision check, never a historical outcome scan."""
    files=sorted(set((c.ROOT/'outputs').glob('splendorust*/schedule.json')) | set((c.ROOT/'outputs').glob('breck_screen*/schedule.json')))
    c.require(len(files)<=64,'Known schedule inventory exceeds64file bound')
    c.require(sum(p.stat().st_size for p in files)<=32*c.MiB,'Known schedules exceed32MiB bound')
    old=set()
    def collect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('seed_decimal','world_seed') and isinstance(item,(str,int)):
                    old.add(str(item))
                else:collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    for p in files:
        collect(c.read(p))
    new={row['seed_decimal'] for row in planned}
    c.require(not new.intersection(old),'Fresh-world seed overlaps known schedule')
    return dict(status='passed',new_unique_worlds=len(new),known_unique_seeds=len(old),
        schedules=[c.artifact(p) for p in files],exhaustive_history_scan=False,
        scope='top-level splendorust*/schedule.json and breck_screen*/schedule.json only; fresh256bit namespace prevents designed seed reuse')


def freeze():
    from . import qualification, schedule, numerical_runtime
    c.require(not (c.OUT/'freeze.json').exists(),'Never replace an execution freeze')
    c.require(c.file_hash(PRIOR/'freeze.json')==PRIOR_SHA,'Prior isolated identity')
    c.require(c.file_hash(FINAL/'independent_receipt_review.json')==ACCEPTED_REVIEW_SHA,'Accepted recovery receipt identity')
    c.require(c.file_hash(BRECK/'independent_results_review.json')==BRECK_ACCEPTED_SHA,'Accepted first-opponent negative-screen identity')
    first=c.read(BRECK/'independent_results_review.json')
    c.require(first.get('accepted_completed_attempt') is True and first.get('accepted_competitive_opponent') is False,'Reserve requires completed rejected first opponent')
    first_bound={item['path']:item for item in first['consumed_inputs']}
    for name in ('freeze.json','completion.json','screen_assessment.json','execution.json',
                 'watchdog_execution.json','guardian_os_exit.json'):
        path=BRECK/name
        c.require(c.artifact(path)==first_bound.get(path.relative_to(c.ROOT).as_posix()),
                  'Accepted first-opponent summary changed: '+name)
    c.require(c.file_hash(BRECK/'freeze.json')==first['freeze_sha256'],
              'Accepted first-opponent freeze binding differs')
    accepted=c.read(FINAL/'independent_receipt_review.json')
    c.require(accepted.get('accepted') is True,'Accepted recovered verification required')
    c.require(shutil.disk_usage(c.ROOT).free>=13*c.GiB,'Initial13GiBdisk reserve')
    asset_folders=(c.ROOT/'external/breck_webapp_6ec66ef',c.ROOT/'external/dom_splendor_426fa50')
    acquired_bytes=sum(p.stat().st_size for folder in asset_folders for p in folder.rglob('*') if p.is_file())
    c.require(acquired_bytes<=256*c.MiB,'Combined new opponent asset cap exceeded')
    sources=rules_guard.verify_runtime_sources()
    prior=c.read(PRIOR/'freeze.json')
    p=profile()
    c.save(c.OUT/'opponent_profile.json',p)
    panel=qualification.build_plan()
    c.save(c.OUT/'qualification_plan.json',panel)
    namespace=secrets.token_hex(32)
    planned=schedule.generate_schedule(namespace)
    c.save(c.OUT/'schedule.json',planned)
    c.save(c.OUT/'seed_collision_check.json',check_known_schedules(planned))
    c.save(c.OUT/'seed_namespace.json',dict(namespace=namespace,cryptographic_bits=256,
        reuse=False,history_scan='Fresh random namespace; bounded known-seed registry check recorded separately, not an exhaustive filesystem scan'))
    ready=dict(player='dom',profile_id=p['profile_id'],profile_sha256=c.file_hash(c.OUT/'opponent_profile.json'),
        threads=1,sources=sources,ruleset_id=rules_guard.RULESET_ID,catalogue_sha256=rules_guard.CATALOGUE_SHA256,
        turn_contract=rules_guard.TURN_CONTRACT,weight_sha256=p['weight_sha256'],
        no_search=True,no_training=True,historical_weights=True)
    numerical_contract,_=numerical_runtime.inspect_runtime()
    ready['numpy_blas_runtime_sha256']=c.digest(numerical_contract)
    f={k:deepcopy(prior[k]) for k in RUNTIME_KEYS}
    f.update(schema='dom_screen_20261027_v1',time=c.now(),job_seconds=c.JOB_SECONDS,
        collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES,
        seed_namespace=namespace,
        seconds_per_move=1.,internal_seconds=.9,dom_internal_limit=None,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        max_rss_bytes=2*c.GiB,external_asset_cap_bytes=256*c.MiB,combined_acquired_asset_bytes=acquired_bytes,
        ruleset_id=rules_guard.RULESET_ID,catalogue_sha256=rules_guard.CATALOGUE_SHA256,
        turn_contract=rules_guard.TURN_CONTRACT,placement_mode='isolated',
        dom_ready_contract=ready,opponent_profile_sha256=ready['profile_sha256'],
        numpy_blas_runtime=numerical_contract,
        control32_profile_sha256=c.profile_sha256(32),
        operational_games=6,main_games=60,main_pairs=30,total_games=66,
        external_timed_requests=256,external_startup_decisions=64,external_reference_decisions=64,
        external_maximum_forwards_per_decision=p['maximum_policy_calls_per_decision'],
        external_reference_forwards_maximum=64*p['maximum_policy_calls_per_decision'],
        external_handler_forwards_maximum=64*p['maximum_policy_calls_per_decision'],
        local_startup_decisions=52,bundled_checks_max_seconds=300,
        final_reproductions_maximum=48,reproductions_per_role=24,
        margin_seconds=.7,margin_is_admission_gate=True,margin_is_forfeit=False,
        competition_gate_natural_points=15.,automatic_retries=0,failed_decision_retries=0,
        larger_data_deferred=True,confirmation_authorized=False,additional_opponent_authorized=False,
        reserve_profile_is_current_authorized_opponent=True,
        ancestry=dict(original_status='incomplete',recovered_verification='accepted_separately',
            accepted_review_sha256=ACCEPTED_REVIEW_SHA,first_opponent_accepted_review_sha256=BRECK_ACCEPTED_SHA,recursive_old_evidence_reaudit=False),
        python=sys.version,numpy_version=version('numpy'))
    f['inputs']=finite_inputs(prior)
    c.save(c.OUT/'freeze.json',f)
    print(c.file_hash(c.OUT/'freeze.json'))
    return f

if __name__ == '__main__':
    freeze()
