"""Model-free numerical, sampling and explicit-checker-projection counterexamples."""
from copy import deepcopy
import hashlib
import json
import unittest

from . import adapters, qualification as q


def sampling(pool=3, mode='split', world_seed=1, encoding_seed=2, heuristic=False):
    worlds=[] if heuristic else [f'GameState {{ synthetic_world: {world_seed}_{i} }}' for i in range(pool)]
    return adapters.enrich_sampling(dict(schema='teacher_variability_sampling_v1',mode=mode,world_pool=pool,
        world_seed=world_seed,encoding_seed=encoding_seed,root_world_debug=worlds,
        encoding_rng_before_worlds=None if heuristic else encoding_seed,
        encoding_rng_after_worlds=None if heuristic else encoding_seed))


class SamplingTests(unittest.TestCase):
    def test_independent_sampling_packet_and_control(self):
        q.validate_sampling(sampling(),pool=3,mode='split',world_seed=1,encoding_seed=2,heuristic=False)
        q.validate_sampling(None,pool=3,mode='legacy',world_seed=1,encoding_seed=1,heuristic=False,control=True)
        q.validate_sampling(sampling(heuristic=True),pool=3,mode='split',world_seed=1,encoding_seed=2,heuristic=True)

    def test_sampling_hash_identity_rng_and_inventory_mutations_rejected(self):
        for field,value in [('schema','other'),('mode','legacy'),('world_pool',6),('world_seed',2),('encoding_seed',1),
                            ('root_world_sha256',['0'*64]*3),('root_worlds_sha256','0'*64),
                            ('root_world_prefix3_sha256','0'*64),('encoding_rng_after_worlds',3),
                            ('root_world_debug',[]),('encoding_rng_before_worlds',True)]:
            packet=sampling();packet[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                q.validate_sampling(packet,pool=3,mode='split',world_seed=1,encoding_seed=2,heuristic=False)

    def crossed(self):
        result={}
        for quota in (512,2048):
            for pool in (3,6):
                for mode in ('split','legacy'):
                    pairs=[(w,e) for w in 'AB' for e in 'AB'] if mode=='split' else [(s,s) for s in 'ABCD']
                    for world,encoding in pairs:
                        packet=sampling(pool,mode,ord(world),ord(encoding))
                        result['search',quota,pool,mode,world,encoding,True]={
                            'research':{'phases':[{'sampling':packet,'callback_bits':[{'input':encoding,'output':encoding}]}]}}
        return result

    def test_full_crossed_coverage_accepts_nested_worlds(self):
        checks=q.validate_crossed_sampling(self.crossed());self.assertEqual(len(checks),2)
        self.assertEqual(checks[0]['legacy_nested_labels'],list('ABCD'))

    def test_crossed_world_encoding_nested_and_budget_mutations_rejected(self):
        for change in ('encoding_changes_world','world_changes_input','pool_changes_input','legacy_prefix','budget'):
            rows=self.crossed()
            if change=='encoding_changes_world':rows['search',512,3,'split','A','B',True]['research']['phases'][0]['sampling']['root_world_debug'][0]='different'
            if change=='world_changes_input':rows['search',512,3,'split','B','A',True]['research']['phases'][0]['callback_bits'][0]['input']='different'
            if change=='pool_changes_input':rows['search',512,6,'split','A','A',True]['research']['phases'][0]['callback_bits'][0]['input']='different'
            if change=='legacy_prefix':rows['search',512,6,'legacy','D','D',True]['research']['phases'][0]['sampling']['root_world_debug'][0]='different'
            if change=='budget':
                for pool in (3,6):
                    rows['search',2048,pool,'legacy','D','D',True]['research']['phases'][0]['sampling']['encoding_rng_after_worlds']=999
            with self.subTest(change=change),self.assertRaises(ValueError):q.validate_crossed_sampling(rows)


class SavedScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=q.ROOT/'outputs/splendorust_proxy_calibration_20261113_01/jobs/diagnostic/worker/transport/request-00000005.json'
        cls.saved=json.loads(path.read_text())['result']
        panel=json.loads((q.ROOT/'outputs/splendorust_proxy_calibration_20261113_01/panel.json').read_text())
        cls.observer=next(r['observer'] for r in panel['roots'] if r['observer_sha256']==cls.saved['observer_sha256'])
        assert cls.saved['simulations_quota']==512 and cls.saved['research']['seed_label']=='A'

    def adapted_fixture(self):
        # Numerical work/callbacks come unchanged from accepted evidence. Added
        # sampling packets below are synthetic checker fixtures, not new evidence.
        value=deepcopy(self.saved);old=value['research'];seed=adapters.seed_pair(self.observer,'legacy','A','A')[0]
        value['research']=dict(schema=adapters.SCHEMA,quota=512,mode='legacy',pool=3,
            world_seed_label='A',encoding_seed_label='A',world_seed=seed,encoding_seed=seed,
            telemetry_enabled=True,internal_seconds=119.,native_binary_sha256=adapters.NATIVE_SHA256,phases=old['phases'])
        for phase,annotation in zip(value['diagnostics']['phases'],value['research']['phases']):
            annotation['sampling']=sampling(3,'legacy',seed,seed,phase['heuristic_phase'])
        return value

    def validate(self,value):
        return q.validate_result(value,self.observer,quota=512,world_seed_label='A',encoding_seed_label='A',
            profile_id=value['profile_id'],profile_sha256=value['profile_sha256'],wrapper_id=value['wrapper_id'],
            native_binary_sha256=adapters.NATIVE_SHA256,pool=3,mode='legacy')

    def test_preserved_full_numerical_checker_accepts_saved_science(self):
        checked=self.validate(self.adapted_fixture());self.assertEqual(checked['simulations'],512)
        self.assertEqual(len(checked['root_statistic_checks']),1)

    def test_wrong_actual_seed_or_root_statistic_rejected(self):
        for change in ('seed','pool','sampling_missing','visits','callback'):
            value=self.adapted_fixture()
            if change=='seed':value['research']['encoding_seed']+=1
            elif change=='pool':value['research']['pool']=6
            elif change=='sampling_missing':value['research']['phases'][0].pop('sampling')
            elif change=='visits':value['research']['phases'][0]['telemetry']['visits'][0]+=1
            elif change=='callback':value['research']['phases'][0]['callback_bits'][0]['input_float32_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):self.validate(value)

    def test_finite_qualification_matrix_covers_all_routes(self):
        plan=q.build_plan(self.observer)
        self.assertEqual(len(plan['parity_requests']),45);self.assertEqual(plan['complete_decisions'],253)
        self.assertEqual({(x['mode'],x['pool']) for x in plan['entrypoint_routes']},
                         {('legacy',3),('legacy',6),('split',3),('split',6)})
        for spec in plan['parity_requests']:
            self.assertEqual(set(q.ordinary_payload(spec)),{'player','observer','quota','mode','pool',
                'world_seed_label','encoding_seed_label','telemetry'})

    def test_projection_changes_only_explicit_annotations(self):
        value=self.adapted_fixture();value['simulations_quota']=64
        value['diagnostics']['observer_seed']=12345
        projected=q.frozen_projection(value)
        self.assertNotIn('research',projected)
        self.assertEqual(projected['diagnostics']['observer_seed'],int(value['observer_sha256'][:16],16))
        self.assertEqual(projected['action'],value['action'])
        self.assertEqual(projected['diagnostics']['phases'],value['diagnostics']['phases'])
        self.assertEqual(value['diagnostics']['observer_seed'],12345)


if __name__=='__main__':unittest.main()
