"""No models: fixed selection, exactness, clustered gate and component arithmetic."""
from copy import deepcopy
import json
import math
import pytest

from tools.splendorust_search256_efficiency_20261031 import panel as p, analysis as a


def candidates():
    return [dict(game_index=game, seat=game % 2, game_plies=60, ply=ply,
                 progress=p.progress(ply, 60), original_receipt_seconds=1.+game/1000+ply/10000)
            for game in range(60) for ply in range(game % 2, 60, 2)]


def fixture_panel():
    roots = []
    for index, selected in enumerate(p.select_positions(candidates())):
        observer = {'fixture': index}
        trace = dict(observer_seed=int(p.digest(observer)[:16], 16), all_arrays=[1., 2., 3.])
        action = {'move': index}
        result = dict(simulations_quota=256, observer_sha256=p.digest(observer),
                      action=action, diagnostics=trace)
        roots.append(dict(**selected, root_id=f'r{index}', panel_id=f'r{index}', observer=observer,
                          observer_seed=trace['observer_seed'], action=action, trace=trace,
                          before={'state': index}, after={'state': index+1}, original_result256=result))
    return dict(schema=p.SCHEMA, selection=p.selection_rule(), roots=roots)


def fixture_rows(panel, phase, *, ratio=1.0, off_seconds=1.4):
    roots = {r['root_id']: r for r in panel['roots']}
    rows = []
    for spec in p.request_matrix(panel, phase):
        root = roots[spec['root_id']]
        result = deepcopy(root['original_result256'])
        row = dict(spec=deepcopy(spec), observer=deepcopy(root['observer']),
                   before=deepcopy(root['before']), after=deepcopy(root['after']),
                   action=deepcopy(root['action']), trace=deepcopy(root['trace']), completed=True,
                   clock=dict(status='ok', accepted=True, received_before_deadline=True,
                       deadline_seconds=10., request_id=f"request-{spec['index']+3:08d}",
                       receipt_elapsed_seconds=off_seconds*(ratio if spec['mode']=='on' else 1), result=result))
        if spec['mode'] == 'on':
            totals = {label: dict(calls=1, failures=0, inclusive_wall_ns=100,
                exclusive_wall_ns=50, inclusive_thread_cpu_ns=50, exclusive_thread_cpu_ns=20)
                      for label in a.COMPONENT_LABELS}
            profile = dict(handler_wall_ns=1000, handler_thread_cpu_ns=500, native_cpu_ns=100,
                           handler_unattributed_wall_ns=650, handler_unattributed_thread_cpu_ns=360, totals=totals)
            row['component_profile'] = dict(schema='splendorust_search256_efficiency_v1',
                observer_mode='on', profile_enabled=True, quota=256, route='research',
                request_id=row['clock']['request_id'], result_sha256=p.digest(result), profiles=[profile])
        rows.append(row)
    return rows


def test_selection_is_fixed_and_separate():
    chosen = p.select_positions(candidates())
    assert len(chosen) == len({r['game_index'] for r in chosen}) == 18
    assert sum(r['group']=='systematic' for r in chosen) == 12
    assert sum(r['group']=='tail' for r in chosen) == 6
    assert p.select_positions(reversed(candidates())) == chosen


@pytest.mark.parametrize('game',[0,235,294,300])
def test_actual_saved_initial_and_compact_proof_schema(game):
    """The failed preparation fixture: initial.json wraps the snapshot."""
    from tools.splendorust_search256_recovery_20261030_02 import verification
    schedule=p.read(p.ACCEPTED/'schedule.json')['main']
    spec=schedule[game];folder=verification.source_game_folder(spec)
    initial=p.read(folder/'initial.json');proof=p.read(folder/'online_replay.json')
    assert all(initial[k]==v for k,v in spec.items())
    first=initial['initial']['state']['current_player']
    indices=list(range((spec['challenger_seat']-first)%2,proof['transitions'],2))
    assert len(indices)==len(proof['latencies']['challenger'])
    seam=30 if game==235 else 50 if game==294 else -1
    selected={0,len(indices)-1}
    selected.update(i for i,ply in enumerate(indices) if abs(ply-seam)<=2)
    for index in sorted(selected):
        row=p.read(verification.source_move_path(spec,indices[index]))
        assert row['clock']['receipt_elapsed_seconds']==proof['latencies']['challenger'][index]
        assert row['seat']==spec['challenger_seat'] and row['quota']==256


def test_materialized_panel_exact_rows_match_actual_runner_schema():
    from tools.splendorust_search256_efficiency_20261031 import common as c
    panel=p.read(c.OUT/'panel.json');roots=a.panel_roots(panel)
    assert panel['census']['decisions']==11066
    for spec in p.request_matrix(panel,'diagnostic')[:36]:
        root=roots[spec['root_id']]
        row=p.read(p.ROOT/root['source_row']['path'])
        row['spec']=deepcopy(spec)
        # The actual runner retains the scientific mode and carries observer
        # mode separately in the frozen spec/owning worker identity.
        assert row['mode']=='research'
        assert a.validate_row(row,spec,panel)['full_exact_equality']


def test_actual_worker_profiler_snapshot_matches_analysis_schema():
    from tools.splendorust_search256_efficiency_20261031 import worker
    class Clock:
        value=0
        def __call__(self):self.value+=10;return self.value
    class Probe:
        pid=123;created=10.
        def read(self):return 0
    profile=worker.EfficiencyProfiler(Probe(),wall_clock=Clock(),cpu_clock=Clock())
    outer=profile.begin('native_exchange');inner=profile.begin('compiled_forward')
    profile.end(inner);profile.end(outer);profile.finish()
    panel=fixture_panel();row=next(r for r in fixture_rows(panel,'diagnostic') if r['spec']['mode']=='on')
    row['component_profile']['profiles']=[profile.snapshot()]
    values=a.component_metrics(row)
    assert values['native_cpu_ns']==0
    assert sum(values[label+'_exclusive_wall_ns'] for label in a.COMPONENT_LABELS)+values['handler_unattributed_wall_ns']==values['handler_wall_ns']


def test_actual_preparation_binds_successful_extraction_and_original_suite():
    from tools.splendorust_search256_efficiency_20261031 import common as c,preparation
    assert preparation.verify_panel_extraction()==c.artifact(c.OUT/'prelaunch/panel_extraction_03.json')
    old=c.read(c.ACCEPTED_OUT/'freeze.json')
    assert preparation.accepted_suite_reference(old)==c.artifact(c.OLD_OUT/'entrypoint_checks/research.json')


@pytest.mark.parametrize('mutation',['failed_exit','wrong_count','changed_panel','changed_source','uncommitted','changed_matrix'])
def test_preparation_rejects_extraction_binding_changes(monkeypatch,mutation):
    from tools.splendorust_search256_efficiency_20261031 import common as c,preparation
    original=c.read
    def altered(path):
        value=original(path);name=str(path).replace('\\','/')
        if name.endswith('panel_extraction_execution_03.json') and mutation=='failed_exit':value['actual_exit_code']=1
        if name.endswith('panel_extraction_03.json'):
            if mutation=='wrong_count':value['eligible_decisions']-=1
            if mutation=='changed_panel':value['panel']['sha256']='0'*64
            if mutation=='changed_matrix':value['matrices']['overhead']['sha256']='0'*64
        if name.endswith('selection_precommit_03.json'):
            if mutation=='changed_source':value['implementation']['sha256']='0'*64
            if mutation=='uncommitted':value['selection_started_after_rule_saved']=False
        return value
    monkeypatch.setattr(c,'read',altered)
    with pytest.raises(ValueError):preparation.verify_panel_extraction()


@pytest.mark.parametrize('mutation',['missing','duplicate','changed_bytes'])
def test_preparation_rejects_unaccepted_suite_reference(monkeypatch,mutation):
    from tools.splendorust_search256_efficiency_20261031 import common as c,preparation
    original=c.read;old=original(c.ACCEPTED_OUT/'freeze.json')
    manifest=original(c.ROOT/old['recovery_manifest']['path'])
    inventory_path=(c.ROOT/manifest['original_tree']['path']).resolve()
    target=(c.OLD_OUT/'entrypoint_checks/research.json').relative_to(c.ROOT).as_posix()
    def altered(path):
        value=original(path)
        if path.resolve()==inventory_path:
            selected=next(x for x in value['files'] if x['path']==target)
            if mutation=='missing':value['files'].remove(selected)
            elif mutation=='duplicate':value['files'].append(deepcopy(selected))
            else:selected['sha256']='0'*64
        return value
    monkeypatch.setattr(c,'read',altered)
    with pytest.raises(ValueError):preparation.accepted_suite_reference(old)


def test_source_context_ignores_unconsumed_external_runtime_pins():
    outside=dict(path='C:/external_runtime/python.exe',bytes=1,sha256='a'*64)
    inside=dict(path='tools/future_training_rules_20261021.py',bytes=1,sha256='b'*64)
    class Reader:
        added=[]
        def add(self,item):self.added.append(item)
        def bound(self,path,expected_sha=None):
            name=str(path).replace('\\','/')
            if name.endswith('/freeze.json'):
                return dict(inputs=[outside,inside],recovery_manifest=dict(path='manifest.json',bytes=1,sha256='c'*64))
            if name.endswith('/independent_acceptance_01.json'):
                return dict(approved=True,freeze_sha256=p.FREEZE_SHA,actual_runner_watchdog_reader_exit_codes=[0,0,0],
                    coverage=dict(main_games=400,model_reproductions=48),guardian_actual_exit=dict(exit_code=0),inputs=[])
            if name.endswith('/actual_guardian_exit_01.json'):return {}
            if name.endswith('/manifest.json'):
                return {k:dict(path=k+'.json',bytes=1,sha256='d'*64) for k in ('original_tree','previous_tree')}
            if name.endswith('_tree.json'):return dict(files=[])
            if name.endswith('/completed_evidence_seal.json'):return dict(artifacts=[])
            if name.endswith('/main_analysis.json'):return dict(games=400)
            if name.endswith('/schedule.json'):return dict(main=[dict(game_index=i) for i in range(400)])
            raise AssertionError(path)
    reader=Reader();p._source_context(reader)
    assert inside in reader.added and outside not in reader.added


def test_outcomes_do_not_affect_selection():
    source = candidates()
    expected = [(r['game_index'], r['ply']) for r in p.select_positions(source)]
    for row in source:
        row.update(winner=1, score=999, final_value=-1000)
    assert [(r['game_index'], r['ply']) for r in p.select_positions(source)] == expected


def test_latency_cannot_affect_systematic_controls():
    source = candidates()
    expected = [(r['game_index'], r['ply']) for r in p.select_positions(source) if r['group']=='systematic']
    for row in source:
        row['original_receipt_seconds'] = 9 - row['original_receipt_seconds']
    assert [(r['game_index'], r['ply']) for r in p.select_positions(source) if r['group']=='systematic'] == expected


def test_tail_is_cell_maximum_excluding_preselected_games():
    source = candidates(); selected = p.select_positions(source)
    used = {r['game_index'] for r in selected if r['group']=='systematic'}
    for row in selected[12:]:
        eligible = [r for r in source if r['progress']==row['progress'] and r['seat']==row['seat'] and r['game_index'] not in used]
        expected = min(eligible, key=lambda r: (-r['original_receipt_seconds'],r['game_index'],r['ply']))
        assert row['game_index']==expected['game_index'] and row['ply']==expected['ply']
        used.add(row['game_index'])


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-1.,0.,10.01,True])
def test_selection_rejects_invalid_arrival(bad):
    rows=candidates();rows[0]['original_receipt_seconds']=bad
    with pytest.raises(ValueError):p.select_positions(rows)


def test_selection_rejects_duplicate_decision():
    rows=candidates()
    with pytest.raises(ValueError):p.select_positions(rows+[rows[0]])


def test_selection_rejects_changed_progress():
    rows=candidates();rows[0]['progress']='late'
    with pytest.raises(ValueError):p.select_positions(rows)


@pytest.mark.parametrize('phase,total', [('overhead',48),('diagnostic',144)])
def test_matrix_complete_counterbalanced(phase,total):
    panel=fixture_panel();matrix=p.request_matrix(panel,phase)
    assert len(matrix)==total
    a.validate_layout(panel,matrix,phase)
    for root in {s['root_id'] for s in matrix}:
        matches=[s for s in matrix if s['root_id']==root]
        assert len(matches)==8
        assert sum(s['mode']=='on' and s['index']%2==0 for s in matches)==2


@pytest.mark.parametrize('mutation', ['missing','reorder','duplicate','wrong_quota'])
def test_matrix_tampering_rejected(mutation):
    panel=fixture_panel();matrix=p.request_matrix(panel,'overhead')
    if mutation=='missing':matrix.pop()
    elif mutation=='reorder':matrix[0],matrix[1]=matrix[1],matrix[0]
    elif mutation=='duplicate':matrix[0]=matrix[1]
    else:matrix[0]['quota']=128
    with pytest.raises(ValueError):a.validate_layout(panel,matrix,'overhead')


def test_overhead_exact_degenerate_equal_timings_pass():
    panel=fixture_panel();rows=fixture_rows(panel,'overhead')
    result=a.analyze_overhead(rows,panel)
    assert result['passed'] and result['all_exact']
    assert result['paired_median_ratio']==result['bootstrap']['upper_one_sided_95']==1.
    assert result['bootstrap']['positions']==6 and result['bootstrap']['pairs_per_position']==4


@pytest.mark.parametrize('ratio,base,gate', [(1.06,1.0,'paired_median_ratio'),(1.04,2.,'paired_median_extra_seconds')])
def test_overhead_actual_prespecified_limits(ratio,base,gate):
    panel=fixture_panel();result=a.analyze_overhead(fixture_rows(panel,'overhead',ratio=ratio,off_seconds=base),panel)
    assert not result['passed'] and not result['gates'][gate]


def test_cluster_bootstrap_preserves_within_position_repeats():
    panel=fixture_panel();rows=fixture_rows(panel,'overhead')
    roots=sorted({r['spec']['root_id'] for r in rows})
    for row in rows:
        if row['spec']['root_id'] in roots[:2] and row['spec']['mode']=='on':
            row['clock']['receipt_elapsed_seconds']*=1.2
    result=a.analyze_overhead(rows,panel)
    assert result['paired_median_ratio']==1.
    assert result['bootstrap']['upper_one_sided_95']>1.10
    assert not result['passed']
    assert result==a.analyze_overhead(rows,panel)


@pytest.mark.parametrize('mutation', ['array','action','observer','seed','after','deadline','identity'])
def test_exactness_covers_full_results(mutation):
    panel=fixture_panel();rows=fixture_rows(panel,'overhead');row=rows[0]
    if mutation=='array':row['clock']['result']['diagnostics']['all_arrays'][0]=1.001
    elif mutation=='action':row['action']={'move':999}
    elif mutation=='observer':row['observer']['fixture']=999
    elif mutation=='seed':row['clock']['result']['diagnostics']['observer_seed']+=1
    elif mutation=='after':row['after']['state']=999
    elif mutation=='deadline':row['clock']['deadline_seconds']=1.
    else:row['spec']['mode']='on' if row['spec']['mode']=='off' else 'off'
    with pytest.raises(ValueError):a.analyze_overhead(rows,panel)


def test_failed_request_is_preserved_and_cannot_pass_gate():
    panel=fixture_panel();rows=fixture_rows(panel,'overhead');row=rows[0]
    row.update(completed=False,after=None)
    row['clock'].update(status='timeout',accepted=False,received_before_deadline=False,result=None)
    result=a.analyze_overhead(rows,panel)
    assert not result['passed'] and not result['all_exact']
    assert len(result['failed_requests'])==1 and result['gates'] is None


def test_diagnostic_separates_groups_and_disjoint_spans():
    panel=fixture_panel();result=a.analyze_diagnostic(fixture_rows(panel,'diagnostic'),panel)
    assert result['passed'] and result['full_historical_output_comparisons']==144
    assert set(result['groups'])=={'systematic','tail'}
    assert result['groups']['systematic']['timings']['on']['count']==48
    assert result['groups']['tail']['timings']['on']['count']==24
    assert not result['deployment_qualified'] and not result['causal_host_bottleneck_proven']


@pytest.mark.parametrize('mutation',['sum','negative','missing','identity','result','duplicate','calls'])
def test_component_corruption_rejected(mutation):
    panel=fixture_panel();rows=fixture_rows(panel,'diagnostic')
    row=next(r for r in rows if r['spec']['mode']=='on');sidecar=row['component_profile'];profile=sidecar['profiles'][0]
    if mutation=='sum':profile['handler_unattributed_wall_ns']+=1
    elif mutation=='negative':profile['native_cpu_ns']=-1
    elif mutation=='missing':del profile['totals']['native_read_wait']
    elif mutation=='identity':sidecar['request_id']='wrong'
    elif mutation=='result':sidecar['result_sha256']='0'*64
    elif mutation=='duplicate':sidecar['profiles'].append(deepcopy(profile))
    else:profile['totals']['compiled_forward']['calls']=True
    with pytest.raises(ValueError):a.analyze_diagnostic(rows,panel)
