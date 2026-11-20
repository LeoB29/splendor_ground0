"""Independent acceptance, contrast-role and multiplicity counterexamples."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.splendorust_full_corpus_20261110 import common as c, family


def score(wins=224, ci95=(.51, .61), ci975=(.505, .615)):
    mean = wins / 400
    result = dict(games=400, pairs=200, wins=wins, draws=0, losses=400-wins,
        numerator=2*wins, denominator=800, mean_score=mean,
        seat_denominator=400, seat_numerators={'0':wins, '1':wins},
        by_candidate_seat={'0':mean, '1':mean},
        pair_numerators=[4]*(wins//2) + [2]*(wins%2) + [0]*(200-(wins+1)//2),
        paired_ci95=list(ci95), paired_ci975=list(ci975), bootstrap_degenerate=ci95[0] == ci95[1])
    for level, interval, factor, criteria, useful in (
        ('95', ci95, 40, 'useful_gain_criteria', 'useful_gain'),
        ('975', ci975, 80, 'familywise_parent_criteria', 'familywise_parent_useful_gain'),
    ):
        radius=math.sqrt(math.log(factor)/400)
        bounds=[max(0.,mean-radius), min(1.,mean+radius)]
        gates=dict(score_at_least_55pct=wins>=220, each_seat_score_at_least_50pct=wins>=200,
            paired_lower_bound_above_50pct=interval[0]>.5,
            degenerate_interval_safeguard=interval[0]!=interval[1] or bounds[0]>.5)
        result['bounded_pair_hoeffding_'+level]=bounds
        result[criteria]=gates;result[useful]=all(gates.values())
    return result


def record(group, *, official=None, conservative=None):
    candidate, reference=family.ARMS[group]
    a=official or score();b=conservative or deepcopy(a);primary=group=='mixed_terminal'
    raw=a['useful_gain'] and b['useful_gain']
    adjusted=a['familywise_parent_useful_gain'] and b['familywise_parent_useful_gain']
    analysis=dict(schema='full_corpus_pooled_strength_analysis_v1', status='complete_verified_both_repetitions',
        matchup=group, candidate_arm=candidate, control_arm=reference, reference_arm=reference,
        primary_estimand='equal_mean_of_two_fixed_seed_pairs_for_'+group,
        checkpoint_names=[f'r{r}_{candidate}_joint' for r in (0,1)],
        reference_checkpoints=['original_ef8a' if reference=='parent' else f'r{r}_{reference}_joint' for r in (0,1)],
        games=400, pairs=200, allocation_each=dict(games=200,pairs=100),
        candidate_simulations=128,reference_simulations=128,world_pool_each=3,
        research_guard_seconds=dict(internal=9.,complete_response=10.),
        complete_verification=True,evidence_gates_passed=True,automatic_promotion=False,
        individual_checkpoint_selection=False,operational_games_pooled=False,other_study_games_pooled=False,
        no_sample_extension=True,official=a,conservative=b,unadjusted_95_criteria_met=raw,
        useful_gain_established=raw if primary else False,
        familywise_parent_useful_gain=False if primary else adjusted,
        uncertainty_role='primary_target_contrast95' if primary else 'secondary_parent_family_bonferroni97.5',
        administrative_sensitivity=dict(administrative_games=0))
    acceptance=dict(schema='full_corpus_strength_independent_acceptance_v1',matchup=group,
        candidate_arm=candidate,reference_arm=reference,approved=True,blocking_findings=[],
        fixed_main_games=400,operational_games=12,all_completed_segment_exits_observed=True,
        automatic_promotion=False,model_calls=0,new_games=0,
        independent_pooled_arithmetic=dict(status='passed',verified_result_sha256=c.digest(analysis)))
    return dict(analysis=analysis,acceptance=acceptance)


def rebind(item):
    item['acceptance']['independent_pooled_arithmetic']['verified_result_sha256']=c.digest(item['analysis'])
    return item


class FamilyTests(unittest.TestCase):
    def test_three_different_opponents_not_combined_into_one_win_rate(self):
        rows={g:record(g) for g in family.GROUPS}
        result=family.synthesize(rows)
        self.assertEqual(result['main_games'],1200)
        self.assertEqual(result['operational_games'],36)
        self.assertIsNone(result['combined_win_rate_across_opponents'])
        self.assertTrue(result['no_best_seed_selection'])
        self.assertTrue(result['no_automatic_promotion'])
        self.assertEqual([r['interval_confidence'] for r in result['comparisons']],[.95,.975,.975])

    def test_parent_must_pass_adjusted_not_only_unadjusted_interval(self):
        item=record('mixed_parent',official=score(ci975=(.49,.63)))
        result=family.validate_group('mixed_parent',**item)
        self.assertTrue(item['analysis']['unadjusted_95_criteria_met'])
        self.assertFalse(result['useful_gain_criteria_met'])
        self.assertEqual(result['official']['interval'],[.49,.63])

    def test_primary_keeps_its_single_95_percent_interval(self):
        item=record('mixed_terminal',official=score(ci975=(.49,.63)))
        result=family.validate_group('mixed_terminal',**item)
        self.assertTrue(result['useful_gain_criteria_met'])
        self.assertEqual(result['official']['interval'],[.51,.61])

    def test_conservative_failure_prevents_gain_claim(self):
        item=record('mixed_parent',conservative=score(wins=216,ci95=(.49,.59),ci975=(.48,.60)))
        self.assertFalse(family.validate_group('mixed_parent',**item)['useful_gain_criteria_met'])

    def test_missing_or_extra_matchup_rejected(self):
        for groups in (family.GROUPS[:2],(*family.GROUPS,'extra')):
            rows={g:record(g if g!='extra' else 'mixed_parent') for g in groups}
            with self.assertRaises(ValueError):family.synthesize(rows)

    def test_stale_independent_arithmetic_cannot_accept_changed_score(self):
        item=record('mixed_terminal');item['analysis']['official']['mean_score']=.57
        with self.assertRaisesRegex(ValueError,'authenticates'):family.validate_group('mixed_terminal',**item)

    def test_independent_acceptance_rejects_wrong_role_partial_or_unapproved(self):
        for key,value in [('matchup','mixed_parent'),('candidate_arm','terminal'),('reference_arm','parent'),
                          ('approved',False),('fixed_main_games',399),('operational_games',11),
                          ('blocking_findings',['unresolved']),('all_completed_segment_exits_observed',False)]:
            item=record('mixed_terminal');item['acceptance'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):family.validate_group('mixed_terminal',**item)

    def test_best_seed_checkpoint_substitution_rejected(self):
        item=record('mixed_terminal');item['analysis']['checkpoint_names'][1]='r0_mixed_joint';rebind(item)
        with self.assertRaisesRegex(ValueError,'best-seed'):family.validate_group('mixed_terminal',**item)

    def test_parent_group_cannot_claim_unadjusted_primary_success(self):
        item=record('mixed_parent');item['analysis']['useful_gain_established']=True;rebind(item)
        with self.assertRaisesRegex(ValueError,'roles'):family.validate_group('mixed_parent',**item)

    def test_score_threshold_and_degenerate_safeguard_cannot_be_weakened(self):
        for newscore in (score(wins=218,ci95=(.51,.58),ci975=(.505,.59)),
                         score(wins=224,ci95=(.56,.56),ci975=(.56,.56))):
            item=record('mixed_terminal',official=newscore)
            self.assertFalse(family.validate_group('mixed_terminal',**item)['useful_gain_criteria_met'])
            item['analysis']['official']['useful_gain_criteria']['score_at_least_55pct']=True
            item['analysis']['official']['useful_gain_criteria']['degenerate_interval_safeguard']=True
            item['analysis']['official']['useful_gain']=True;rebind(item)
            with self.assertRaises(ValueError):family.validate_group('mixed_terminal',**item)

    def test_bad_integer_nonfinite_or_too_narrow_adjusted_interval_rejected(self):
        for mutate in (lambda s:s.update(games=True),lambda s:s.update(paired_ci95=[float('nan'),.6]),
                       lambda s:s.update(paired_ci975=[.52,.6]),lambda s:s.update(numerator=450)):
            item=record('mixed_terminal');mutate(item['analysis']['official'])
            with self.assertRaises((ValueError,TypeError)):
                rebind(item);family.validate_group('mixed_terminal',**item)

    def test_physical_analysis_descriptor_must_match_acceptance(self):
        item=record('mixed_terminal');item['acceptance']['analysis']={'path':'wrong','bytes':0,'sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'physical'):
            family.validate_group('mixed_terminal',**item,analysis_descriptor={'path':'right','bytes':0,'sha256':'0'*64})

    def test_finalize_requires_all_acceptances_and_writes_only_new_results(self):
        with tempfile.TemporaryDirectory(prefix='splendor-family-test-') as td:
            out=Path(td).resolve()
            with patch.object(c,'OUT',out):
                for group in family.GROUPS:
                    item=record(group);folder=out/'strength'/group/'report'
                    analysis=c.save(folder/'analysis.json',item['analysis'])
                    item['acceptance']['analysis']=analysis
                    c.save(folder/'independent_acceptance.json',item['acceptance'])
                def draw(summary,path):Path(path).write_bytes(b'fixture-chart-not-a-scientific-result')
                with patch.object(family,'chart',side_effect=draw):
                    result=family.finalize(out/'family')
                    self.assertEqual(result['main_games'],1200)
                    self.assertEqual(result['model_calls'],0)
                    self.assertIn('does not make every contrast precise',(out/'family/results.md').read_text(encoding='utf-8'))
                    with self.assertRaisesRegex(ValueError,'replacement'):family.finalize(out/'family')


if __name__=='__main__':unittest.main()
