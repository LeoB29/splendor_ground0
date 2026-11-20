"""Model-free counterexamples for paired, shared-mask proxy arithmetic."""
from copy import deepcopy
import math
import unittest
from . import analysis as a


def fixture(games=128,roots_per_game=4):
    panel=dict(roots=[],host_metadata={});records=[]
    indices=list(range(1024,1024+games//2))+list(range(1088,1088+games-games//2))
    for g in indices:
        for j in range(roots_per_game):
            rid=f'g{g}_p{j:06d}';meta=dict(game_index=g,split='validation' if g<1088 else 'offline')
            panel['roots'].append(dict(root_id=rid,observer={'unused_lawful_payload':True}))
            panel['host_metadata'][rid]=meta
            shift=((g-1024)%7-3)*.012+(j-1.5)*.004
            for quota,seed in a.CONDITIONS:
                action=a.QUOTAS.index(quota) if quota!=2048 else 3
                visits=[0]*67;qs=[None]*67
                if quota==2048:
                    for k in range(4):
                        visits[k]=16;qs[k]=.40+k*.025+shift+(.003 if seed=='S' else -.003)
                    visits[3]=quota-48
                else:visits[action]=quota;qs[action]=.5
                records.append(dict(root_id=rid,**meta,quota=quota,seed_label=seed,
                    main_action_index=action,visits67=visits,q67=qs,elapsed_seconds=quota/1000+j/10000))
    return records,panel


def fake_contrasts(adjacent=(.02,.02,.02),bounds=(.01,.01,.01),end_lower=.02):
    rows=[]
    for pair,difference,lower in zip(a.ADJACENT,adjacent,bounds):
        rows.append(dict(low=pair[0],high=pair[1],difference=difference,family_ci=[lower,lower+.02],ci95=[lower,lower+.02]))
    rows.append(dict(low=64,high=512,difference=sum(adjacent),ci95=[end_lower,end_lower+.04],family_ci=None))
    return rows


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records,cls.panel=fixture()
        cls.result=a.analyze(cls.records,cls.panel)

    def small(self):
        records,panel=fixture(games=2,roots_per_game=1)
        return records,panel,a._inventory(records,panel,strict=False)

    def test_full_production_inventory_and_independent_reader(self):
        receipt=a.verify_arithmetic(self.records,self.panel,self.result)
        self.assertEqual(receipt['decisions'],5120)
        self.assertEqual(receipt['games'],128)
        self.assertEqual(receipt['status'],'independent_proxy_arithmetic_verified')
        self.assertEqual(self.result['calibration']['status'],'strong_positive_control_support')
        self.assertAlmostEqual(self.result['primary_contrasts'][3]['difference'],.075)

    def test_rejects_wrong_inventory_and_source_split(self):
        with self.assertRaises(ValueError):a._inventory(self.records[:-1],self.panel)
        rows=self.records.copy();rows[-1]=rows[0]
        with self.assertRaises(ValueError):a._inventory(rows,self.panel)
        panel=deepcopy(self.panel)
        for root in panel['roots']:
            host=panel['host_metadata'][root['root_id']]
            host['split']='offline' if host['split']=='validation' else 'validation'
        with self.assertRaises(ValueError):a._inventory(self.records,panel)

    def test_strict_native67_values_and_bindings(self):
        records,panel,(meta,_,_)=self.small();original=records[0];host=meta[original['root_id']]
        modifications=[{'quota':True},{'seed_label':'R'},{'main_action_index':True},
            {'main_action_index':67},{'game_index':1025},{'split':'offline'},
            {'visits67':[0]*81},{'q67':[None]*81},{'elapsed_seconds':math.nan}]
        for change in modifications:
            row=deepcopy(original);row.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):a.validate_record(row,host)
        for changed in ('missing_visited','invented_unvisited','wrong_total','nan_q','negative_visit','bool_visit'):
            row=deepcopy(original)
            if changed=='missing_visited':row['q67'][0]=None
            if changed=='invented_unvisited':row['q67'][1]=0.
            if changed=='wrong_total':row['visits67'][0]-=1
            if changed=='nan_q':row['q67'][0]=math.nan
            if changed=='negative_visit':row['visits67'][1]=-1
            if changed=='bool_visit':row['visits67'][1]=False
            with self.subTest(changed=changed),self.assertRaises(ValueError):a.validate_record(row,host)

    def test_one_missing_edge_excludes_root_for_every_budget(self):
        records,panel,(meta,book,_)=self.small();rid=sorted(meta)[0];ref=book[rid][2048,'R']
        ref['visits67'][3]+=ref['visits67'][0];ref['visits67'][0]=0;ref['q67'][0]=None
        roots=a._root_scores(meta,book);row=next(x for x in roots if x['root_id']==rid)
        self.assertFalse(row['joint_supported'])
        self.assertEqual(row['missing_action_indices'],{'R':[0],'S':[]})
        self.assertTrue(all(value is None for value in row['q_scores'].values()))
        self.assertEqual(row['agreement']['512'],1.)
        self.assertEqual(len(a._games(roots,'agreement')),2)
        self.assertEqual(len(a._games(roots,'q_scores',supported=True)),1)

    def test_candidate_and_reference_seed_means_are_symmetric(self):
        _,_,(meta,book,_)=self.small();before=a._root_scores(meta,book)
        for cell in book.values():
            for quota in a.QUOTAS:cell[quota,'A'],cell[quota,'B']=cell[quota,'B'],cell[quota,'A']
            cell[2048,'R'],cell[2048,'S']=cell[2048,'S'],cell[2048,'R']
        after=a._root_scores(meta,book)
        self.assertEqual([r['q_scores'] for r in before],[r['q_scores'] for r in after])

    def test_reference_selected_gap_is_signed_not_true_regret(self):
        _,_,(meta,book,_)=self.small()
        for cell in book.values():
            for label in a.REFERENCE_SEEDS:cell[2048,label]['q67'][0]=.95
        rows=a._root_scores(meta,book)
        self.assertTrue(all(r['reference_selected_q_gap']['64']<0 for r in rows))

    def test_equal_game_weight_and_whole_game_pairing(self):
        roots=[]
        for game,split,values in ((1024,'validation',[0.,0.,0.]),(1025,'validation',[1.]),
                                  (1088,'offline',[.5,.5])):
            for value in values:roots.append(dict(game_index=game,split=split,joint_supported=True,
                q_scores={str(q):value for q in a.QUOTAS}))
        games=a._games(roots,'q_scores',supported=True)
        self.assertAlmostEqual(sum(r['values'][0] for r in games)/len(games),.5)
        draws,_=a._bootstrap(games,draws=99,seed=a.BOOTSTRAP_SEED)
        self.assertTrue((draws[:,0]==draws[:,3]).all())

    def test_bootstrap_retains_the_same_game_draws_across_seeds_and_budgets(self):
        games=self.result['game_summaries']['primary']
        one,identity=a._bootstrap(games,draws=83,seed=a.BOOTSTRAP_SEED)
        shifted=[{**r,'values':[x+2 for x in r['values']]} for r in games]
        two,again=a._bootstrap(shifted,draws=83,seed=a.BOOTSTRAP_SEED)
        self.assertEqual(identity,again)
        self.assertTrue(abs(two-one-2).max()<1e-12)

    def test_reading_rule_strong_broad_contradicted_unresolved(self):
        coverage=dict(sufficient=True);rows=fake_contrasts();refs={'R':deepcopy(rows),'S':deepcopy(rows)}
        self.assertEqual(a.classification(coverage,rows,refs)['status'],'strong_positive_control_support')
        rows=fake_contrasts(bounds=(-.01,.01,.01))
        self.assertEqual(a.classification(coverage,rows,refs)['status'],'provisional_broad_support')
        rows=fake_contrasts(adjacent=(0.,.02,.02),bounds=(-.01,.01,.01))
        self.assertEqual(a.classification(coverage,rows,refs)['status'],'provisional_broad_support')
        rows=fake_contrasts(bounds=(-.04,.01,.01))
        self.assertEqual(a.classification(coverage,rows,refs)['status'],'contradicted')
        self.assertEqual(a.classification({'sufficient':False},rows,refs)['status'],'unresolved_coverage')
        rows=fake_contrasts(adjacent=(-.001,.02,.02),bounds=(-.01,.01,.01))
        self.assertEqual(a.classification(coverage,rows,refs)['status'],'unresolved')

    def test_opposite_reference_direction_blocks_broad_support(self):
        rows=fake_contrasts(bounds=(-.01,.01,.01));refs={'R':deepcopy(rows),'S':deepcopy(rows)}
        refs['S'][3]['difference']=-.001
        self.assertEqual(a.classification(dict(sufficient=True),rows,refs)['status'],'unresolved')

    def test_ninety_percent_coverage_threshold_is_not_rounded_up(self):
        records=deepcopy(self.records)
        omitted={x['root_id'] for x in self.panel['roots'][:52]}
        for row in records:
            if row['root_id'] in omitted and row['quota']==2048 and row['seed_label']=='R':
                row['visits67'][3]+=row['visits67'][0];row['visits67'][0]=0;row['q67'][0]=None
        meta,book,_=a._inventory(records,self.panel)
        summary=a._summarize(a._root_scores(meta,book),records,draws=31,seed=a.BOOTSTRAP_SEED)
        self.assertEqual(summary['coverage']['supported_roots'],460)
        self.assertEqual(summary['calibration']['status'],'unresolved_coverage')
        self.assertEqual(summary['agreement_estimates']['512']['games'],128)

    def test_no_reference_support_has_no_invented_primary_scores(self):
        records,panel,(meta,book,_)=self.small()
        for cell in book.values():
            row=cell[2048,'S'];row['visits67'][3]+=row['visits67'][0];row['visits67'][0]=0;row['q67'][0]=None
        result=a._summarize(a._root_scores(meta,book),records,draws=17,seed=a.BOOTSTRAP_SEED)
        self.assertIsNone(result['primary_estimates']['64']['mean'])
        self.assertEqual(result['primary_estimates']['64']['games'],0)
        self.assertEqual(result['agreement_estimates']['64']['games'],2)
        self.assertEqual(result['calibration']['status'],'unresolved_coverage')

    def test_independent_reader_rejects_corrupted_arithmetic(self):
        report=deepcopy(self.result);report['roots'][0]['q_scores']['64']+=.001
        with self.assertRaisesRegex(ValueError,'Independent root arithmetic'):
            a.verify_arithmetic(self.records,self.panel,report)
        report=deepcopy(self.result);report['support']['supported_minimum_visits']['16']-=1
        with self.assertRaisesRegex(ValueError,'Independent root-edge support'):
            a.verify_arithmetic(self.records,self.panel,report)
        report=deepcopy(self.result);report['primary_contrasts'][0]['family_ci'][0]+=.001
        with self.assertRaisesRegex(ValueError,'Independent paired contrast'):
            a.verify_arithmetic(self.records,self.panel,report)

    def test_fixed_uncertainty_settings_cannot_drift(self):
        with self.assertRaises(ValueError):a.analyze(self.records,self.panel,draws=999)
        report=deepcopy(self.result);report['bootstrap']['seed']+=1
        with self.assertRaisesRegex(ValueError,'uncertainty specification'):
            a.verify_arithmetic(self.records,self.panel,report)

    def test_one_second_count_is_descriptive_and_exact_boundary_is_not_late(self):
        records,panel,(meta,book,_)=self.small()
        chosen=[r for r in records if r['quota']==64]
        for row,value in zip(chosen,(.999,1.,1.001,1.2),strict=True):row['elapsed_seconds']=value
        report=a._summarize(a._root_scores(meta,book),records,draws=19,seed=a.BOOTSTRAP_SEED)
        self.assertEqual(report['timing']['64']['over_one_second'],2)
        self.assertEqual(report['timing']['64']['requests'],4)
        self.assertEqual(report['agreement_estimates']['64']['games'],2)


if __name__=='__main__':unittest.main()
