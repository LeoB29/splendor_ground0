import copy
import math
import sys
import pytest
from tools.splendorust_selfplay_postmortem_20261104 import targets as t


def row(*,target=(.75,.25),logits=(0.,0.),values=(0.,0.)):
    return dict(game_index=192,ply=0,split='validation',root_input_sha256='0'*64,
        legal_mask81=[True]*len(logits)+[False]*(81-len(logits)),
        policy_target81=list(target)+[0.]*(81-len(target)),value_target2=[1.,-1.],
        output83=list(logits)+[0.]*(81-len(logits))+list(values))


def test_closed_form_teacher_entropy_kl_and_effective_support():
    parent=row();m=t.row_metrics(parent,copy.deepcopy(parent))
    entropy=-.75*math.log(.75)-.25*math.log(.25)
    assert m['teacher_entropy_nats']==pytest.approx(entropy)
    assert m['teacher_effective_support']==pytest.approx(math.exp(entropy))
    assert m['parent_policy_crossentropy_nats']==pytest.approx(math.log(2))
    assert m['parent_teacher_to_policy_kl_nats']==pytest.approx(math.log(2)-entropy)
    assert m['teacher_inverse_simpson_support']==pytest.approx(1/(.75**2+.25**2))
    assert m['policy_total_variation']==m['parent_to_candidate_kl_nats']==m['value_mean_absolute_shift']==0


def test_illegal_high_logits_have_exactly_zero_probability():
    parent=row();parent['output83'][50]=1000.
    m=t.row_metrics(parent,copy.deepcopy(parent))
    assert m['parent_policy_crossentropy_nats']==pytest.approx(math.log(2))
    assert m['parent_teacher_visited_support_probability']==pytest.approx(1.)
    assert m['parent_teacher_mode_agreement']==1


def test_extreme_legal_logits_preserve_finite_kl_without_smoothing():
    parent=row(target=(0.,1.),logits=(0.,-1000.))
    candidate=row(target=(0.,1.),logits=(0.,0.))
    m=t.row_metrics(parent,candidate)
    assert m['parent_policy_crossentropy_nats']==pytest.approx(1000.)
    assert m['candidate_to_parent_kl_nats']==pytest.approx(500.-math.log(2))
    assert m['parent_to_candidate_kl_nats']==pytest.approx(math.log(2))
    assert m['policy_total_variation']==pytest.approx(.5)


def test_teacher_ties_are_all_accepted_modes():
    parent=row(target=(.5,.5),logits=(0.,1.))
    candidate=row(target=(.5,.5),logits=(1.,0.))
    m=t.row_metrics(parent,candidate)
    assert m['teacher_top_tie_count']==2
    assert m['parent_teacher_mode_agreement']==m['candidate_teacher_mode_agreement']==1
    assert m['policy_argmax_changed']==1
    assert m['parent_teacher_top_set_probability']==pytest.approx(1)


def test_network_tie_break_is_distinct_from_top_set_overlap():
    parent=row(target=(0.,.5,.5),logits=(0.,0.,-1.))
    m=t.row_metrics(parent,copy.deepcopy(parent))
    assert m['parent_teacher_mode_agreement']==0
    assert m['parent_teacher_model_top_set_overlap']==1
    assert m['parent_policy_top_tie_count']==2


def test_zero_visit_legal_actions_count_as_legal_not_visited():
    parent=row(target=(1.,0.,0.),logits=(0.,0.,0.))
    m=t.row_metrics(parent,copy.deepcopy(parent))
    assert m['legal_support_count']==3 and m['visited_support_count']==1
    assert m['visited_fraction_of_legal']==pytest.approx(1/3)
    assert m['teacher_entropy_nats']==0 and m['teacher_effective_support']==1
    assert m['parent_teacher_visited_support_probability']==pytest.approx(1/3)


def test_one_legal_action_has_finite_normalized_entropy():
    parent=row(target=(1.,),logits=(0.,))
    m=t.row_metrics(parent,copy.deepcopy(parent))
    assert m['teacher_entropy_fraction_of_log_legal']==0
    assert m['parent_policy_crossentropy_nats']==m['parent_teacher_to_policy_kl_nats']==0


def test_two_value_slots_and_signed_shifts_are_not_collapsed():
    parent=row(values=(.25,-.5));candidate=row(values=(.5,-.25))
    m=t.row_metrics(parent,candidate)
    assert m['value_slot0_signed_shift']==m['value_slot1_signed_shift']==.25
    assert m['value_mean_absolute_shift']==m['value_root_rms_shift']==.25


@pytest.mark.parametrize('field,value',[
    ('root_input_sha256','1'*64),('ply',1),('value_target2',[-1.,1.]),
    ('policy_target81',[.5,.5]+[0.]*79)])
def test_same_input_binding_rejects_mismatched_rows(field,value):
    parent=row();candidate=copy.deepcopy(parent);candidate[field]=value
    with pytest.raises(ValueError,match='Exact same input'):t.row_metrics(parent,candidate)


@pytest.mark.parametrize('mutation',[
    lambda r:r['legal_mask81'].__setitem__(0,1),
    lambda r:r['policy_target81'].__setitem__(0,-.25),
    lambda r:r['policy_target81'].__setitem__(2,.5),
    lambda r:r['output83'].__setitem__(0,float('nan')),
    lambda r:r['output83'].__setitem__(0,.1),
    lambda r:r['output83'].__setitem__(81,2.),
    lambda r:r.update(value_target2=[1.,1.]),
    lambda r:r.update(game_index=0),
])
def test_saved_data_contract_is_strict(mutation):
    parent=row();mutation(parent)
    with pytest.raises(ValueError):t.validate_row(parent)


def test_whole_games_have_equal_weight_even_with_different_lengths():
    result=t.aggregate_games([(192,{'signal':0.}),(193,{'signal':1.}),
                              (193,{'signal':1.}),(193,{'signal':1.})])
    assert result['root_count']==4 and result['game_count']==2
    assert result['mean_game_metrics']['signal']==.5
    assert result['mean_game_metrics']['signal']!=.75


def test_aggregate_rejects_missing_metric_in_one_game():
    with pytest.raises(ValueError,match='complete metrics'):
        t.aggregate_games([(192,{'signal':0.}),(193,{'other':1.})])


def test_pure_saved_metric_module_never_imports_torch():
    assert 'torch' not in sys.modules
