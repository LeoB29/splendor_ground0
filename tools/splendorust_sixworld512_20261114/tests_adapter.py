"""Model-free regression tests; no network, games or native processes run."""
from copy import deepcopy
import base64,gzip,hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import common as c,adapters as a,qualification as q


def envelope(raw,compressed=None):
    packed=gzip.compress(raw,mtime=0) if compressed is None else compressed
    return dict(schema=a.PACK_SCHEMA,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(packed),compressed_sha256=hashlib.sha256(packed).hexdigest(),
        data_b64=base64.b64encode(packed).decode())


class CodecTests(unittest.TestCase):
    def test_full_lossless_json_and_signed_zero(self):
        value={'nested':[None,True,False,2**64-1,-0.0,{'bits':'0000000080000000'}]}
        got=a.unpack_result(a.pack_result(value))
        self.assertEqual(c.canonical(value),c.canonical(got))
        self.assertEqual(math.copysign(1,got['nested'][4]),-1)

    def test_hashes_sizes_and_envelope_are_strict(self):
        source=a.pack_result({'x':[1,2,3]})
        for key,value in [('raw_bytes',True),('raw_bytes',0),('compressed_bytes',source['compressed_bytes']+1),
            ('raw_sha256','0'*64),('compressed_sha256','f'*64),('data_b64','!'),('schema','old')]:
            with self.subTest(key=key,value=value),self.assertRaises((ValueError,TypeError)):
                bad=dict(source);bad[key]=value;a.unpack_result(bad)
        with self.assertRaises(ValueError):a.unpack_result(dict(source,extra=1))

    def test_duplicate_nonfinite_noncanonical_rejected(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{ "x": 1 }',b'{"x":Infinity}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):a.unpack_result(envelope(raw))
        with self.assertRaises(ValueError):a.pack_result({'x':float('nan')})

    def test_truncated_appended_and_multi_member_rejected(self):
        raw=b'{"x":1}';encoded=gzip.compress(raw,mtime=0)
        for encoded_bad in (encoded[:-1],encoded+b'x',encoded+gzip.compress(b'{}')):
            with self.assertRaises(ValueError):a.unpack_result(envelope(raw,encoded_bad))

    def test_expansion_bound_before_json_allocation(self):
        with patch.object(a,'MAX_RAW_BYTES',32):
            with self.assertRaises(ValueError):a.unpack_result(envelope(b'"'+b'x'*100+b'"'))
            with self.assertRaises(ValueError):a.pack_result('x'*100)


class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=q.build_plan();cls.freeze=c.read(c.PRIOR_FREEZE)

    def test_exact_saved_root_results_validate_and_compress(self):
        for root in self.plan['saved_parity_roots']:
            for pool in (3,6):
                result=c.read(c.resolve(root['expected'][str(pool)]))['result']
                self.assertEqual(q.validate_live_result(result,root['observer'],pool,self.freeze)['status'],'passed')
                packed=a.pack_result(result)
                self.assertEqual(c.canonical(a.unpack_result(packed)),c.canonical(result))
                self.assertLess(packed['compressed_bytes'],packed['raw_bytes'])

    def test_plan_covers_complete_live_edges_and_original_costs(self):
        self.assertEqual(self.plan['complete_decisions'],120)
        self.assertEqual(self.plan['corrected_entrypoint_decisions'],104)
        self.assertEqual([r['case_id'] for r in self.plan['edge_cases']],['forced_pass','turn123','turn124'])
        self.assertEqual(len(self.plan['original_suite']['counterexamples']),15)
        self.assertEqual(len([r for r in self.plan['original_suite']['counterexamples'] if 'corrected_face' in r['tags']]),6)
        self.assertEqual([r['root_id'] for r in self.plan['saved_parity_roots']],['g1143_p000010','g1103_p000025'])

    def fake_context(self,root,pool):
        value=c.read(c.resolve(root['expected'][str(pool)]))['result']
        class Replay:
            last_research_phases=deepcopy(value['research']['phases'])
            request_deadline=None
            def choose(self,observer,**kwargs):
                return a.rules.deserialize_action(value['action']),deepcopy(value['diagnostics'])
        fake=Replay()
        return dict(graph_parity_passed=True,adapters=dict(search=fake,control=fake,reference=fake),
            profiles={k:c.read(c.resolve(v)) for k,v in self.freeze['research_profiles'].items()},
            profile_artifacts=self.freeze['research_profiles'],control_profiles={},control_profile_artifacts={}),value

    def test_new_ordinary_reproduces_saved_full_envelope_without_model(self):
        for pool in (3,6):
            root=self.plan['saved_parity_roots'][0];context,expected=self.fake_context(root,pool)
            events=[];context['evidence_callback']=events.append
            got=a.handle(context,q.payload(root['observer'],pool))
            self.assertEqual(c.canonical(got),c.canonical(expected))
            self.assertEqual(len(events),1)
            self.assertIsNone(context['adapters']['search'].request_deadline)

    def test_live_configuration_and_qualification_bypass_rejected(self):
        root=self.plan['saved_parity_roots'][0];context,_=self.fake_context(root,3)
        for change in ({'quota':64},{'quota':2048},{'pool':4},{'mode':'split'},
            {'world_seed_label':'B','encoding_seed_label':'B'},{'telemetry':False},{'player':'control'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                a.handle(context,dict(q.payload(root['observer'],3),**change))
        with self.assertRaises(ValueError):a.handle(context,{'operation':'qualification_decision','request':q.payload(root['observer'],3)})
        context['graph_parity_passed']=False
        with self.assertRaises(ValueError):a.handle(context,q.payload(root['observer'],3))

    def test_qualification_scope_restored_after_exception(self):
        context={'graph_parity_passed':True,'qualification_enabled':True}
        with patch.object(a,'ordinary',side_effect=ValueError('saved failure')):
            with self.assertRaises(ValueError):a.handle(context,{'operation':'qualification_decision','request':{}})
        self.assertFalse(context['qualification_active'])

    def test_live_turn124_not_rejected_by_saved_panel_restriction(self):
        root=self.plan['saved_parity_roots'][0];context,_=self.fake_context(root,3)
        edge=next(r for r in self.plan['edge_cases'] if r['case_id']=='turn124')
        class Reached(Exception):pass
        with patch.object(context['adapters']['search'],'choose',side_effect=Reached('native reached')):
            with self.assertRaises(Reached):a.handle(context,q.payload(edge['observer'],3))


class ForensicTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.request={'request_id':'request-00000002','payload':q.payload({'test':1},3),
            'result':{'action':{'kind':'Pass'},'diagnostics':{'calls':0},'research':{'phases':[]}}}

    def tearDown(self):self.temp.cleanup()

    def row(self,request,sequence=0,previous=None):
        r=request['result'];p=request['payload']
        return dict(sequence=sequence,previous_sha256=previous,event=dict(transport_request_id=request['request_id'],
            event=dict(event='complete_result_before_worker_validation',arm=p['player'],observer=p['observer'],
                action=r['action'],diagnostics=r['diagnostics'],research=r['research'])))

    def write(self,rows,*,truncated=False,summary=True):
        data=(''.join(c.canonical(r)+'\n' for r in rows)).encode();packed=gzip.compress(data,mtime=0)
        path=self.folder/'ordinary_forensics.jsonl.gz';path.write_bytes(packed[:-8] if truncated else packed)
        if summary:
            (self.folder/'forensic_summary.json').write_text(c.canonical(dict(events=len(rows),
                terminal_sha256=c.digest(rows[-1]) if rows else None,artifact=c.artifact(path),lossless=True)),encoding='utf-8')

    def test_successful_generator_consumed_without_materialization(self):
        self.write([self.row(self.request)])
        proof=q.verify_service_forensics(self.folder,(r for r in [self.request]))
        self.assertEqual(proof['accepted_events'],1);self.assertTrue(proof['gzip_complete'])

    def test_missing_extra_or_tampered_events_fail(self):
        self.write([self.row(self.request)])
        with self.assertRaises(ValueError):q.verify_service_forensics(self.folder,iter([]))
        with self.assertRaises(ValueError):q.verify_service_forensics(self.folder,iter([self.request,self.request]))
        bad=deepcopy(self.request);bad['result']['action']={'kind':'Buy'}
        with self.assertRaises(ValueError):q.verify_service_forensics(self.folder,iter([bad]))

    def test_clean_service_cannot_accept_truncated_gzip(self):
        self.write([self.row(self.request)],truncated=True)
        with self.assertRaises(ValueError):q.verify_service_forensics(self.folder,iter([self.request]))

    def test_failed_service_prefix_is_verified_without_claiming_complete_stream(self):
        self.write([self.row(self.request)],truncated=True,summary=False)
        failed={'request_id':'request-00000003','payload':q.payload({'test':2},6)}
        proof=q.verify_service_forensics(self.folder,iter([self.request]),failed_request=failed)
        self.assertEqual(proof['accepted_events'],1);self.assertFalse(proof['gzip_complete'])
        self.assertTrue(proof['failed_tail_is_not_an_accepted_decision'])

    def test_failed_tail_binds_exact_request_and_preserves_complete_error(self):
        first=self.row(self.request);failed={'request_id':'request-00000003','payload':q.payload({'test':2},6)}
        second=dict(sequence=1,previous_sha256=c.digest(first),event=dict(transport_request_id=failed['request_id'],
            event=dict(event='worker_research_decision_failed',arm='search',observer={'test':2},error='deadline')))
        self.write([first,second]);proof=q.verify_service_forensics(self.folder,iter([self.request]),failed_request=failed)
        self.assertEqual(proof['failed_tail_events'],['worker_research_decision_failed'])
        failed['request_id']='request-00000004'
        with self.assertRaises(ValueError):q.verify_service_forensics(self.folder,iter([self.request]),failed_request=failed)


if __name__=='__main__':unittest.main()
