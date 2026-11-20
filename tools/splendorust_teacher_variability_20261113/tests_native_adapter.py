"""Model-free contract and source-transformation counterexamples."""
import copy
import hashlib
import unittest
from . import adapters as a, native_build as n


class NativeTransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_search = (n.SOURCE/'crates/splendor-agents/src/neural_search.rs').read_text(encoding='utf-8')
        cls.original_bridge = (n.SOURCE/'bridge/src/main.rs').read_text(encoding='utf-8')

    def test_source_bytes_pinned(self):
        self.assertEqual(n.artifact(n.SOURCE/'crates/splendor-agents/src/neural_search.rs')['sha256'],n.ORIGINAL_SEARCH_SHA)
        self.assertEqual(n.artifact(n.SOURCE/'bridge/src/main.rs')['sha256'],n.ORIGINAL_BRIDGE_SHA)

    def test_only_canonical_root_world_rng_changes(self):
        transformed=n.search_source(self.original_search)
        # Leaf encoding, inference and every transition draw retain identical text.
        for line in self.original_search.splitlines():
            if '&mut self.rng' in line and '.map(|_| o.determinize' not in line:
                self.assertEqual(transformed.count(line), self.original_search.count(line))
        self.assertEqual(transformed.count('self.diagnostic_world_seed {'),1)
        self.assertIn('let mut world_rng = Rng::new(seed);', transformed)
        self.assertIn('.map(|_| o.determinize(&mut self.rng).expect("valid observation"))',transformed)

    def test_new_fields_no_rng_draws(self):
        text=n.search_source(self.original_search)
        extra=[line for line in text.splitlines() if 'diagnostic_' in line]
        self.assertFalse(any('next_u64' in line or '.index(' in line for line in extra))
        self.assertIn('diagnostic_world_seed: None,',text)

    def test_root_pool_nested_draws(self):
        text=n.search_source(self.original_search)
        split=text.split('let worlds: Vec<_> = if let Some(seed) = self.diagnostic_world_seed {')[1].split('} else {')[0]
        self.assertEqual(split.count('Rng::new(seed)'),1)
        self.assertEqual(split.count('(0..self.world_pool)'),1)
        self.assertEqual(split.count('o.determinize(&mut world_rng)'),1)

    def test_fails_closed_on_ambiguous_source(self):
        for old,fn in ((self.original_search,n.search_source),(self.original_bridge,n.bridge_source)):
            with self.assertRaises(ValueError): fn(old+old)
            with self.assertRaises(ValueError): fn('')

    def test_bridge_explicit_modes_seeds_and_pool(self):
        text=n.bridge_source(self.original_bridge)
        self.assertIn('assert_eq!(world_seed,seed,"legacy has one shared seed")',text)
        self.assertIn('mode=="legacy" || mode=="split"',text)
        self.assertIn('Some(3)=>3,Some(6)=>6',text)
        self.assertIn('"encoding_rng_after_worlds"',text)
        self.assertIn('"root_world_debug"',text)


class AdapterContractTests(unittest.TestCase):
    observer={'fixture':'observer-only','phase':'Main'}

    def test_legacy_a_retains_original_seed(self):
        wanted=int(a.digest(self.observer)[:16],16)
        self.assertEqual(a.seed_pair(self.observer,'legacy','A','A'),(wanted,wanted))

    def test_legacy_cannot_mix_stream_labels(self):
        with self.assertRaises(ValueError): a.seed_pair(self.observer,'legacy','A','B')

    def test_split_world_stream_independent_of_encoding_label(self):
        aa=a.seed_pair(self.observer,'split','A','A')
        ab=a.seed_pair(self.observer,'split','A','B')
        ba=a.seed_pair(self.observer,'split','B','A')
        self.assertEqual(aa[0],ab[0]); self.assertEqual(aa[1],ba[1])
        self.assertNotEqual(aa[0],ba[0]); self.assertNotEqual(aa[1],ab[1])
        self.assertNotEqual(aa[0],aa[1])

    def test_seeds_deterministic_and_u64(self):
        for mode in a.MODES:
            for label in a.SEED_LABELS:
                first=a.seed_pair(self.observer,mode,label,label)
                self.assertEqual(first,a.seed_pair(copy.deepcopy(self.observer),mode,label,label))
                self.assertTrue(all(0<=x<2**64 for x in first))

    def test_invalid_seed_inputs(self):
        for label in ('E',1,None,True):
            with self.assertRaises(ValueError): a.observer_seed(self.observer,label)
        with self.assertRaises(ValueError): a.observer_seed(self.observer,'A','unfrozen')
        with self.assertRaises(ValueError): a.seed_pair(self.observer,'invented','A','A')

    def test_profile_configuration_allowlist(self):
        self.assertEqual(a.profile_key(2048,6,'split'),'split_p6_q2048')
        for quota,pool,mode in ((True,3,'legacy'),(32,3,'legacy'),(16,True,'legacy'),(16,9,'split'),(16,3,'other')):
            with self.assertRaises(ValueError): a.profile_key(quota,pool,mode)

    def sample(self, worlds=None):
        return dict(schema='teacher_variability_sampling_v1', mode='split', world_pool=3,
            world_seed=1, encoding_seed=2, root_world_debug=worlds if worlds is not None else
            ['GameState { a: 1 }','GameState { a: 2 }','GameState { a: 3 }'],
            encoding_rng_before_worlds=2,encoding_rng_after_worlds=2)

    def test_full_world_sha_reproducible_and_preserved(self):
        raw=self.sample();result=a.enrich_sampling(raw)
        self.assertEqual(result['root_world_debug'],raw['root_world_debug'])
        self.assertEqual(result['root_world_sha256'],[hashlib.sha256(x.encode()).hexdigest() for x in raw['root_world_debug']])
        self.assertNotIn('root_world_sha256',raw)

    def test_nested_prefix_hash(self):
        raw=self.sample();larger=copy.deepcopy(raw);larger['world_pool']=6
        larger['root_world_debug'] += ['GameState { a: 4 }','GameState { a: 5 }','GameState { a: 6 }']
        self.assertEqual(a.enrich_sampling(raw)['root_world_prefix3_sha256'],a.enrich_sampling(larger)['root_world_prefix3_sha256'])
        self.assertNotEqual(a.enrich_sampling(raw)['root_worlds_sha256'],a.enrich_sampling(larger)['root_worlds_sha256'])

    def test_world_mutation_changes_hash(self):
        raw=self.sample();changed=copy.deepcopy(raw);changed['root_world_debug'][0]+=' '
        self.assertNotEqual(a.enrich_sampling(raw)['root_worlds_sha256'],a.enrich_sampling(changed)['root_worlds_sha256'])

    def test_invalid_sampling_telemetry(self):
        for worlds in (['not a world']*3,['GameState {}'],None):
            raw=self.sample();raw['root_world_debug']=worlds
            with self.assertRaises(ValueError): a.enrich_sampling(raw)
        raw=self.sample();raw['invented']=True
        with self.assertRaises(ValueError): a.enrich_sampling(raw)

    def test_heuristic_phase_can_have_zero_worlds(self):
        self.assertEqual(a.enrich_sampling(self.sample([]))['root_world_sha256'],[])

    def test_unqualified_decisions_rejected_before_model(self):
        context=dict(adapters={'reference':None,'search':None,'control':None},qualification_active=False)
        base=dict(player='search',observer={},quota=512,pool=3,mode='legacy',
            world_seed_label='A',encoding_seed_label='A',telemetry=True)
        for player,quota in (('reference',16),('control',512),('search',64),('search',16)):
            request=dict(base,player=player,quota=quota)
            with self.assertRaisesRegex(ValueError,'qualification-only'): a.ordinary(context,request)


if __name__=='__main__':
    unittest.main()
