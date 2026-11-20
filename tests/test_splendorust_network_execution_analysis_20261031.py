"""Model-free fixed-panel, exactness and independently recomputed timing tests."""
from copy import deepcopy
import statistics

import numpy as np
import pytest

from tools.splendorust_network_execution_20261031 import analysis as a, common as c
from tools.splendorust_search256_efficiency_20261031 import panel as p


def panel_fixture():
    roots = []
    for group, per_cell in (('systematic', 2), ('tail', 1)):
        for phase, ply in (('early', 6), ('middle', 26), ('late', 46)):
            for seat in (0, 1):
                for _ in range(per_cell):
                    i = len(roots)
                    observer = {'fixture': i}
                    trace = dict(observer_seed=int(c.digest(observer)[:16], 16),
                                 scientific_arrays=[.1, -.2, .3], search_work=256)
                    action = {'move': i}
                    result = dict(simulations_quota=256, observer_sha256=c.digest(observer),
                                  action=action, diagnostics=trace)
                    roots.append(dict(root_id=f'r{i:02d}', panel_id=f'r{i:02d}',
                        game_index=i, game_plies=60, ply=ply, progress=phase, seat=seat, group=group,
                        observer=observer, observer_seed=trace['observer_seed'], before={'state': i},
                        after={'state': i+1}, action=action, trace=trace, original_result256=result,
                        original_receipt_seconds=1.2))
    return dict(schema=p.SCHEMA, selection=p.selection_rule(), roots=roots)


def rows_fixture(panel, matrix, *, systematic=.90, tail=.95, control=1.):
    rows = []
    for spec in matrix:
        root = panel['roots'][spec['panel_index']]
        ratio = systematic if spec['group'] == 'systematic' else tail
        ratio = ratio(root, spec['repeat']) if callable(ratio) else ratio
        seconds = control*(ratio if spec['variant'] == 'candidate' else 1)
        rows.append(dict(spec=deepcopy(spec), variant=spec['variant'], root_id=spec['root_id'],
            route='research', completed=True,
            **{name: deepcopy(root[name]) for name in ('observer', 'before', 'after', 'action', 'trace')},
            clock=dict(status='ok', accepted=True, received_before_deadline=True, deadline_seconds=10.,
                       receipt_elapsed_seconds=seconds, result=deepcopy(root['original_result256']))))
    return rows


def run(*, systematic=.9, tail=.95, control=1.):
    panel=panel_fixture();matrix=a.evaluation_matrix(panel)
    rows=rows_fixture(panel,matrix,systematic=systematic,tail=tail,control=control)
    return a.analyze_evaluation(rows,panel,matrix)


def test_matrix_rotation_pairs_and_balanced_order_independent_assertions():
    panel=panel_fixture();matrix=a.evaluation_matrix(panel)
    assert len(matrix)==144
    assert [matrix[36*r]['panel_index'] for r in range(4)]==[0,1,2,3]
    assert [matrix[2*i]['panel_index'] for i in range(18)]==list(range(18))
    assert [matrix[36+2*i]['panel_index'] for i in range(18)]==list(range(1,18))+[0]
    assert matrix[0]==dict(index=0,pair_index=0,root_id='r00',panel_id='r00',variant='control',
                          repeat=0,panel_index=0,group='systematic',order=0)
    assert matrix[2]['variant']=='candidate' and matrix[3]['variant']=='control'
    for root in panel['roots']:
        matches=[x for x in matrix if x['root_id']==root['root_id']]
        assert len(matches)==8
        assert {x['repeat'] for x in matches}=={0,1,2,3}
        assert sum(x['variant']=='control' and x['order']==0 for x in matches)==2
        assert sum(x['variant']=='candidate' and x['order']==0 for x in matches)==2
    for offset in range(0,144,2):
        left,right=matrix[offset:offset+2]
        assert left['pair_index']==right['pair_index']==offset//2
        assert left['root_id']==right['root_id'] and left['repeat']==right['repeat']
        assert {left['variant'],right['variant']}=={'control','candidate'}


def test_useful_screen_is_separate_from_one_second_deployment_and_strength():
    result=run(control=2.)
    assert result['status']=='accepted_speedup' and result['passed'] and result['all_exact']
    assert result['groups']['systematic']['timings']['candidate']['completed_over_one_second']==48
    assert result['groups']['tail']['timings']['candidate']['count']==24
    assert result['full_historical_output_comparisons']==144
    assert not result['strength_test'] and not result['deployment_qualified'] and not result['reference_promoted']
    assert not result['overhead_subtracted']


@pytest.mark.parametrize('systematic,tail,status',[
    (.95,1.05,'accepted_speedup'),(.95000001,.9,'valid_no_useful_speedup'),
    (.9,1.05000001,'valid_no_useful_speedup'),(1.,.5,'valid_no_useful_speedup'),
    (1.1,.5,'valid_no_useful_speedup')])
def test_exact_inclusive_ratio_boundaries_and_negative_result(systematic,tail,status):
    result=run(systematic=systematic,tail=tail)
    assert result['status']==status and result['comparison_complete'] and result['all_exact']


def test_cluster_upper_gate_is_strictly_below_one():
    # Seven equally weighted root clusters improve; five do not. Median passes,
    # but the upper bootstrap quantile is exactly1, which must reject.
    result=run(systematic=lambda root,repeat:.9 if root['game_index']<7 else 1.)
    assert result['groups']['systematic']['paired_median_ratio']==.9
    assert result['bootstrap']['upper_one_sided_95']==1.
    assert result['gates']['systematic_paired_median']
    assert not result['gates']['systematic_clustered_upper']
    assert result['status']=='valid_no_useful_speedup'


def test_bootstrap_matches_independent_position_resampling():
    ratios=np.array([[.77+.018*i+.001*j for j in range(4)] for i in range(12)])
    result=run(systematic=lambda root,repeat:float(ratios[root['game_index'],repeat]))
    generator=np.random.Generator(np.random.PCG64(20261031257))
    indices=generator.integers(0,12,size=(10000,12),dtype=np.int64)
    sampled=[statistics.median(float(x) for i in draw for x in ratios[i]) for draw in indices]
    independent=float(np.quantile(sampled,.95,method='linear'))
    assert result['bootstrap']['upper_one_sided_95']==independent
    assert result['bootstrap']['positions']==12 and result['bootstrap']['pairs_per_position']==4
    assert result['bootstrap']['replicates']==10000
    assert result['bootstrap']['degenerate_resamples_retained']
    assert result['groups']['systematic']['paired_median_ratio']==statistics.median(ratios.flatten())


def test_degenerate_clusters_stay_exact_and_tail_never_enters_bootstrap():
    first=run(systematic=.9,tail=.5);second=run(systematic=.9,tail=2.)
    assert first['bootstrap']==second['bootstrap']
    assert first['bootstrap']['upper_one_sided_95']==.9
    assert first['groups']['systematic']==second['groups']['systematic']
    assert first['status']=='accepted_speedup' and second['status']=='valid_no_useful_speedup'


@pytest.mark.parametrize('mutation',['missing_row','extra_row','swapped_rows','duplicated_row',
    'missing_matrix','matrix_reorder','matrix_variant','matrix_pair','matrix_order','matrix_extra_field'])
def test_no_omitted_replaced_or_counterbalance_changed_samples(mutation):
    panel=panel_fixture();matrix=a.evaluation_matrix(panel);rows=rows_fixture(panel,matrix)
    if mutation=='missing_row':rows.pop()
    elif mutation=='extra_row':rows.append(deepcopy(rows[-1]))
    elif mutation=='swapped_rows':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='duplicated_row':rows[1]=deepcopy(rows[0])
    elif mutation=='missing_matrix':matrix.pop()
    elif mutation=='matrix_reorder':matrix[0],matrix[1]=matrix[1],matrix[0]
    elif mutation=='matrix_variant':matrix[0]['variant']='candidate'
    elif mutation=='matrix_pair':matrix[0]['pair_index']=1
    elif mutation=='matrix_order':matrix[0]['order']=1
    elif mutation=='matrix_extra_field':matrix[0]['accepted_extra']=True
    with pytest.raises(ValueError):a.analyze_evaluation(rows,panel,matrix)


@pytest.mark.parametrize('mutation',['result_array','result_work','action','trace','observer','before',
    'after','variant','root','route','completed','clock_status','accepted','deadline','late'])
def test_full_scientific_identity_and_research_guards_fail_closed(mutation):
    panel=panel_fixture();matrix=a.evaluation_matrix(panel);rows=rows_fixture(panel,matrix);row=rows[0]
    if mutation=='result_array':row['clock']['result']['diagnostics']['scientific_arrays'][0]+=.0000001
    elif mutation=='result_work':row['clock']['result']['diagnostics']['search_work']=128
    elif mutation in ('action','trace','observer','before','after'):row[mutation]={'changed':True}
    elif mutation=='variant':row['variant']='candidate'
    elif mutation=='root':row['root_id']='r01'
    elif mutation=='route':row['route']='frozen'
    elif mutation=='completed':row['completed']=False
    elif mutation=='clock_status':row['clock']['status']='timeout'
    elif mutation=='accepted':row['clock']['accepted']=False
    elif mutation=='deadline':row['clock']['deadline_seconds']=11
    elif mutation=='late':row['clock']['received_before_deadline']=False
    with pytest.raises(ValueError):a.analyze_evaluation(rows,panel,matrix)


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf'),True,10.00001,'1.0',None])
def test_invalid_timing_cannot_enter_speed_screen(bad):
    panel=panel_fixture();matrix=a.evaluation_matrix(panel);rows=rows_fixture(panel,matrix)
    rows[0]['clock']['receipt_elapsed_seconds']=bad
    with pytest.raises(ValueError):a.analyze_evaluation(rows,panel,matrix)


def test_coherently_changed_both_arms_still_rejected_against_historical_output():
    panel=panel_fixture();matrix=a.evaluation_matrix(panel);rows=rows_fixture(panel,matrix)
    for row in rows[:2]:
        row['clock']['result']['diagnostics']['scientific_arrays'][0]=.1234
        row['trace']=deepcopy(row['clock']['result']['diagnostics'])
    with pytest.raises(ValueError):a.analyze_evaluation(rows,panel,matrix)


@pytest.mark.parametrize('mutation',['missing_root','duplicate_game','group','quota','seed'])
def test_changed_prescribed_panel_rejected(mutation):
    panel=panel_fixture()
    if mutation=='missing_root':panel['roots'].pop()
    elif mutation=='duplicate_game':panel['roots'][1]['game_index']=0
    elif mutation=='group':panel['roots'][0]['group']='tail'
    elif mutation=='quota':panel['roots'][0]['original_result256']['simulations_quota']=128
    else:panel['roots'][0]['observer_seed']+=1
    with pytest.raises(ValueError):a.evaluation_matrix(panel)


def test_actual_accepted_panel_schema_and_saved_full_rows_without_models():
    panel=c.read(c.PREVIOUS_OUT/'panel.json');matrix=a.evaluation_matrix(panel)
    assert len(matrix)==144 and len(panel['roots'])==18
    for spec in matrix[:36]:
        root=panel['roots'][spec['panel_index']]
        row=c.read(c.ROOT/root['source_row']['path'])
        row.update(spec=deepcopy(spec),variant=spec['variant'],root_id=spec['root_id'],route='research')
        elapsed=a._validate_row(row,spec,root)
        assert elapsed==root['original_receipt_seconds']


def test_row_iterator_and_serializable_recomputation_digest():
    panel=panel_fixture();matrix=a.evaluation_matrix(panel);rows=rows_fixture(panel,matrix)
    result=a.analyze_evaluation(iter(rows),panel,matrix)
    assert result['full_observations_sha256']==c.digest(rows)
    assert result['matrix_sha256']==c.digest(matrix) and result['panel_sha256']==c.digest(panel)
    assert a.analyze_evaluation(rows,panel,matrix)==result
