"""Model-free receipt semantics; no game readers, runtime reruns or tree pass."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.splendorust_reliability_finalization_20261026 import receipts as r


class Envelope:
    def __init__(self, limit=None):
        self.calls = 0
        self.limit = limit

    def guard(self):
        self.calls += 1
        if self.limit is not None and self.calls > self.limit:
            raise TimeoutError('same shared deadline')


def artifact(path='tools/example.py', value=b'{}'):
    return dict(path=path, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


@pytest.mark.parametrize('path', ['', '.', '..', '../x', 'a/../b', './a', '/a', 'a//b',
                                  'C:/a', 'a\\b', 'a/'])
def test_unsafe_or_aliased_manifest_paths_rejected(path):
    with pytest.raises(ValueError, match='artifact path'):
        r.index([artifact(path)])


@pytest.mark.parametrize('field,value', [('bytes', True), ('bytes', -1), ('bytes', 1.5),
    ('sha256', 'A'*64), ('sha256', 'a'*63), ('sha256', None)])
def test_invalid_manifest_hash_and_size_rejected(field, value):
    item=artifact(); item[field]=value
    with pytest.raises(ValueError, match='size/hash'):
        r.index([item])


@pytest.mark.parametrize('path', ['tools/example.py', 'TOOLS/EXAMPLE.py'])
def test_duplicate_and_case_alias_rejected(path):
    with pytest.raises(ValueError, match='Duplicate'):
        r.index([artifact(), artifact(path)])


def test_manifest_guard_preserves_shared_deadline():
    with pytest.raises(TimeoutError):
        r.index([artifact('a'),artifact('b')],Envelope(limit=1).guard)


def test_pinned_json_byte_binding_and_no_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(r.c,'ROOT',tmp_path)
    content=b'{"status":"component_only"}\n'
    (tmp_path/'receipt.json').write_bytes(content)
    item=artifact('receipt.json',content)
    envelope=Envelope()
    assert r.load_pinned('receipt.json',{'receipt.json':item},envelope) == {'status':'component_only'}
    assert envelope.calls >= 3
    (tmp_path/'receipt.json').write_bytes(content.replace(b'component_only',b'component_FAIL'))
    with pytest.raises(ValueError, match='Pinned receipt changed'):
        r.load_pinned('receipt.json',{'receipt.json':item},Envelope())


def test_receipt_read_deadline_missing_pin_size_and_known_sha(tmp_path,monkeypatch):
    monkeypatch.setattr(r.c,'ROOT',tmp_path)
    path=r.REQUIRED_RECEIPTS[0]
    with pytest.raises(ValueError,match='not frozen'):
        r.load_pinned(path,{},Envelope())
    item=artifact(path); item['bytes']=32*1024**2+1
    with pytest.raises(ValueError,match='bounded metadata'):
        r.load_pinned(path,{path:item},Envelope())
    item=artifact(path)
    with pytest.raises(ValueError,match='Known successful/failure'):
        r.load_pinned(path,{path:item},Envelope())
    with pytest.raises(TimeoutError):
        r.load_pinned(path,{},Envelope(limit=0))


def failure_fixture():
    docs={}
    for root in (r.ORIGINAL,r.V1,r.V2):
        docs[root+'/execution.json']=dict(status='failed',exit_code=0,freeze_sha256=r.FREEZE_SHAS[root],
            cleanup_errors=[],cleanup_receipt=dict(errors=[],unconfirmed=[]),automatic_retries=0,
            monotonic_origin=297762.765,unix_origin=1793041905.7336407)
    old=docs[r.ORIGINAL+'/execution.json'];old.update(status='runner_failed',exit_code=1)
    del old['automatic_retries']  # The actual original schema puts this in completion.
    old_error='line 199, in verify_runtime\nValueError: Stage/global real-clock deadline\n'
    docs[r.ORIGINAL+'/completion.json']=dict(status='incomplete',stage='final',coverage=None,
        failure=old_error,freeze_sha256=r.FREEZE_SHAS[r.ORIGINAL],automatic_retries=0)
    docs[r.ORIGINAL+'/failure.json']=dict(stage='final',partial_evidence_not_accepted=True,error=old_error)
    docs[r.V1+'/watchdog_error.json']=dict(error='in resource_guard\nValueError: RECOVERY: Auditor launched an unexpected child\n')
    docs[r.V2+'/execution.json'].update(status='watchdog_deadline',bootstrap_relaunches=1,
        model_or_decision_retries=0,charged_job_seconds=1800.03)
    docs[r.V2+'/completion.json']=dict(status='incomplete',original_status='incomplete')
    frames=[('runner.py',85,'main'),('runner.py',49,'execute'),('runner.py',17,'finish_snapshot'),
            ('preparation.py',15,'inventory'),('common.py',75,'guard'),('common.py',28,'require')]
    package=r.c.ROOT/'tools/splendorust_reliability_recovery_v2_20261026'
    trace='Traceback (most recent call last):\n'+''.join(f'  File "{package/file}", line {line}, in {name}\n' for file,line,name in frames)
    docs[r.V2+'/failure.json']=dict(phase='final_original_snapshot_readback_begin',original_attempt_preserved=True,
        error=trace+'ValueError: RECOVERY: Recovery UTC/monotonic deadline\n')
    for root in (r.V1,r.V2):
        docs[root+'/watchdog_claim.json']=dict(freeze_sha256=r.FREEZE_SHAS[root],
            monotonic_origin=297762.765,unix_origin=1793041905.7336407)
    return docs


def test_three_disclosed_failures_remain_incomplete():
    out=r.validate_failures(failure_fixture())
    assert out['original_status']=='incomplete'
    assert out['second_recovery_status']=='incomplete'
    assert out['exit_zero_after_job_cleanup_is_not_acceptance'] is True


@pytest.mark.parametrize('root,name,key,value', [
    (r.ORIGINAL,'completion.json','status','passed'),
    (r.ORIGINAL,'completion.json','automatic_retries',1),
    (r.ORIGINAL,'execution.json','exit_code',0),
    (r.ORIGINAL,'failure.json','stage','main'),
    (r.V1,'execution.json','status','completed'),
    (r.V1,'execution.json','automatic_retries',1),
    (r.V1,'watchdog_error.json','error','unexpected data failure'),
    (r.V2,'completion.json','status','passed'),
    (r.V2,'execution.json','exit_code',1),
    (r.V2,'execution.json','charged_job_seconds',1799),
    (r.V2,'execution.json','charged_job_seconds',1801),
    (r.V2,'execution.json','bootstrap_relaunches',2),
    (r.V2,'execution.json','model_or_decision_retries',1),
    (r.V2,'execution.json','cleanup_errors',['unknown child']),
    (r.V2,'failure.json','phase','saved_review'),
    (r.V2,'failure.json','original_attempt_preserved',False),
    (r.V2,'watchdog_claim.json','monotonic_origin',298000),
    (r.V2,'watchdog_claim.json','unix_origin',1793042000),
])
def test_other_failure_retry_reset_or_unclean_exit_rejected(root,name,key,value):
    docs=failure_fixture();docs[root+'/'+name][key]=value
    with pytest.raises(ValueError):r.validate_failures(docs)


def test_matching_error_message_at_wrong_preservation_locus_rejected():
    docs=failure_fixture()
    docs[r.V2+'/failure.json']['error']=docs[r.V2+'/failure.json']['error'].replace('line 17, in finish_snapshot','line 19, in finish_snapshot')
    with pytest.raises(ValueError,match='first final inventory'):
        r.validate_failures(docs)


def science_fixture():
    performance=dict(games=200,natural_games=200,forfeits=[],stage_counts={'main':194,'operational':6},
        engineering_margin_is_descriptive_only=True,engineering_margin_changes_completion_or_stopping=False,no_strength_claim=True,players={})
    for role,quota,slow,maximum in [('challenger',64,1,.8542834),('opponent',32,0,.5134891)]:
        performance['players'][role]=dict(quota=quota,attempted_decisions=5510,completed_decisions=5510,
            statuses={'ok':5510},completed_arrival_seconds=dict(count=5510,max=maximum),completed_at_or_above_engineering_margin=slow)
    coverage=dict(status='full_workload_verified',planned_games=200,verified_games=200,unplayed_games=0,
        verified_games_by_stage={'main':194,'operational':6},complete_planned_workload=True,failure=None,
        stopped_on_first_attributable_failure=False,placement_mode='isolated',engineering_margin_is_descriptive_only=True,no_strength_claim=True)
    saved=dict(status='passed_saved_evidence_review',freeze_sha256=r.FREEZE_SHAS[r.ORIGINAL],**r.SCIENCE_COUNTS,
        original_completion_status='incomplete',original_exit_code=1,original_snapshot_byte_authentication='passed',
        boundary_failures=[],no_overall_recovered_acceptance_claim=True,no_causal_or_strength_claim=True,
        final_snapshot_readback='required_from_owning_recovery_pipeline_before_acceptance',coverage=coverage,performance=performance,
        replay_cap=48,replay_populated_cells=48,replay_empty_cells=0)
    sample=dict(decisions=[dict(game_index=i,ply=2*i) for i in range(48)],empty_cells=[],selection_uses_outcomes=False,selection_uses_timing=False)
    repeated=dict(status='passed',requests=48,prefix_only=False,identities=deepcopy(sample['decisions']))
    independent=dict(passed_components=dict(saved_scientific_evidence=dict(r.SCIENCE_COUNTS)))
    return saved,deepcopy(performance),repeated,sample,independent


def test_reused_science_full200_and48_is_component_only():
    args=science_fixture()
    coverage,performance=r.validate_science(*args)
    assert coverage['verified_games']==200
    assert performance['players']['challenger']['completed_at_or_above_engineering_margin']==1
    assert args[0]['no_overall_recovered_acceptance_claim'] is True


@pytest.mark.parametrize('key,value',[('games',199),('live_decisions',11019),('actual_entrypoint_decisions',103),
    ('reproduced_completed_decisions',47),('worker_count',449),('graph_workers',224),('worker_requests',11294),
    ('original_snapshot_artifacts',74578),('sealed_artifacts',74527),('sealed_bytes',700262088),
    ('preserved_player_faults',1),('no_overall_recovered_acceptance_claim',False),('boundary_failures',['fault']),
    ('replay_populated_cells',47),('replay_empty_cells',1),('original_exit_code',0)])
def test_partial_counts_or_scope_cannot_reuse_success(key,value):
    args=science_fixture();args[0][key]=value
    with pytest.raises(ValueError):r.validate_science(*args)


def test_reproduction_identity_and_independent_component_must_agree():
    args=science_fixture();args[2]['identities'][0]['ply']=1000
    with pytest.raises(ValueError,match='reproductions'):r.validate_science(*args)
    args=science_fixture();args[4]['passed_components']['saved_scientific_evidence']['games']=199
    with pytest.raises(ValueError,match='Independent'):r.validate_science(*args)


def test_no_failure_is_hidden_by_conditional_performance_or_descriptive_threshold():
    args=list(science_fixture());args[0]['performance']['players']['challenger']['statuses']={'ok':5509,'timeout':1}
    args[1]=deepcopy(args[0]['performance'])
    with pytest.raises(ValueError,match='per-player'):r.validate_science(*args)
    args=list(science_fixture());args[0]['performance']['engineering_margin_changes_completion_or_stopping']=True
    args[1]=deepcopy(args[0]['performance'])
    with pytest.raises(ValueError,match='performance scope'):r.validate_science(*args)


def test_actual_small_metadata_failure_and_science_schema_without_old_reader():
    # This fixture reads only small successful/failure summaries. It does not
    # execute the receipt pipeline, open any game/move, hash a tree or rerun any
    # old validator. It catches interfaces such as original execution lacking
    # automatic_retries while original completion owns that field.
    docs={path:json.loads((r.c.ROOT/path).read_text(encoding='utf-8')) for path in failure_fixture()}
    assert r.validate_failures(docs)['all_recorded_failures_preserved'] is True
    load=lambda root,name:json.loads((r.c.ROOT/root/name).read_text(encoding='utf-8'))
    coverage,_=r.validate_science(load(r.V2,'saved_review.json'),load(r.ORIGINAL,'performance.json'),
        load(r.ORIGINAL,'repeated_decision_summary.json'),load(r.ORIGINAL,'replay_sample_manifest.json'),
        load(r.V2,'independent_receipt_review.json'))
    assert coverage['verified_games']==200


def test_module_cannot_import_old_science_runtime_or_launch_workers():
    tree=ast.parse(Path(r.__file__).read_text(encoding='utf-8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module or '')
            if node.level:assert node.module is None and [x.name for x in node.names]==['common']
    assert set(imports)=={'hashlib','json','pathlib','re',''}
    assert len(r.REQUIRED_RECEIPTS)==len(set(r.REQUIRED_RECEIPTS))==32
