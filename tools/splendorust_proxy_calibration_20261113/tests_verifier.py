"""Model-free malformed-evidence counterexamples for the new saved verifier."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import adapters, panel, verifier


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='proxy_verifier_test_', dir=verifier.ROOT/'outputs')
        self.folder=Path(self.temp.name); (self.folder/'transport').mkdir(); (self.folder/'service').mkdir()
        self.spec=dict(request_id='g1024_p000001_q0064_A',root_id='g1024_p000001',quota=64,seed_label='A',role='control')
        self.observer={'saved_lawful_fixture':'verifier orchestration only; scientific checker tested separately'}
        self.root=dict(root_id=self.spec['root_id'],observer=self.observer,observer_sha256=panel.digest(self.observer))
        self.panel=dict(roots=[self.root],host_metadata={self.spec['root_id']:dict(game_index=1024,split='validation')})
        self.profile=adapters.profile(64); self.profile_item=self.write(self.folder/'profile64.json',self.profile)
        visits=[0]*67;values=[None]*67;visits[1]=64;values[1]=.5
        self.result=dict(player='search',observer_sha256=panel.digest(self.observer),action={'synthetic_action':True},
            diagnostics={'forced_pass':False,'phases':[{'heuristic_phase':False}]},
            research={'phases':[dict(phase_index=0,phase_kind='Main',telemetry={'visits':visits,'q':values})]})
        self.receipt=dict(status='ok',accepted=True,request_id='request-00000002',result=self.result,detail=None,
            deadline_seconds=120.,elapsed_seconds=.3,receipt_elapsed_seconds=.2,received_before_deadline=True,
            clock='perf_counter_ns',clock_implementation='unit fixture',telemetry={})
        self.write(self.folder/'transport'/'request-00000001.json',{'qualification_receipt_fixture':True})
        self.persist()
        self.science=dict(status='passed',main_action={'kind':'Take','tokens':[1,0,0,0,0]},root_statistic_checks=[{'selected_index':1}])

    def tearDown(self): self.temp.cleanup()

    def write(self,path,value):
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();path.write_bytes(raw)
        return dict(path=path.relative_to(verifier.ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

    def persist(self):
        item=self.write(self.folder/'transport'/'request-00000002.json',self.receipt)
        self.row=dict(spec=deepcopy(self.spec),receipt=item)
        self.write(self.folder/'index.json',{'requests':[self.row]})
        event=dict(event='complete_result_before_worker_validation',arm=self.result['player'],observer=self.observer,
                   **{key:deepcopy(self.result[key]) for key in ('action','diagnostics','research')})
        chain=dict(sequence=0,previous_sha256=None,event=dict(transport_request_id=self.receipt['request_id'],event=event))
        (self.folder/'service/ordinary_forensics.jsonl').write_text(panel.canonical(chain)+'\n',encoding='utf-8')

    def verify(self):
        with patch.object(verifier,'_profiles',return_value={64:(self.profile,self.profile_item)}), \
             patch.object(verifier.qualification,'validate_result',return_value=self.science) as validator:
            result=verifier.verify_records(self.folder,self.panel,[self.spec],freeze={})
            self.assertEqual(validator.call_args.kwargs['player'],'search')
            self.assertEqual(validator.call_args.kwargs['profile_sha256'],self.profile_item['sha256'])
            self.assertTrue(validator.call_args.kwargs['telemetry_required'])
            return result

    def test_valid_receipt_remains_native67(self):
        records,proof=self.verify();self.assertEqual(proof['status'],'passed')
        self.assertEqual(records[0]['main_action_index'],1);self.assertEqual(len(records[0]['visits67']),67)
        self.assertIsNone(records[0]['q67'][0]);self.assertEqual(records[0]['elapsed_seconds'],.2)
        self.assertNotIn('visits81',records[0])

    def test_failed_or_late_clock_cannot_be_accepted(self):
        for change in ({'status':'timeout','accepted':False}, {'received_before_deadline':False},
                       {'receipt_elapsed_seconds':121.,'elapsed_seconds':122.}, {'deadline_seconds':119.},
                       {'clock':'time.time'}, {'receipt_elapsed_seconds':.4}):
            original=deepcopy(self.receipt);self.receipt.update(change);self.persist()
            with self.assertRaises(ValueError):self.verify()
            self.receipt=original

    def test_wrong_hash_or_bytes_rejected(self):
        for key,value in [('sha256','0'*64),('bytes',1)]:
            doc={'requests':[deepcopy(self.row)]};doc['requests'][0]['receipt'][key]=value
            self.write(self.folder/'index.json',doc)
            with self.assertRaises(ValueError):self.verify()

    def test_missing_extra_duplicate_and_reordered_specs_rejected(self):
        for rows in ([],[self.row,self.row],[dict(self.row,spec=dict(self.spec,seed_label='B'))]):
            self.write(self.folder/'index.json',{'requests':rows})
            with self.assertRaises(ValueError):self.verify()

    def test_extra_hidden_transport_request_rejected(self):
        self.write(self.folder/'transport'/'request-00000003.json',self.receipt)
        with self.assertRaises(ValueError):self.verify()

    def test_receipt_outside_worker_or_wrong_request_id_rejected(self):
        self.receipt['request_id']='request-00000003';self.persist()
        with self.assertRaises(ValueError):self.verify()
        self.receipt['request_id']='request-00000002';self.persist()
        doc={'requests':[deepcopy(self.row)]};doc['requests'][0]['receipt']['path']=self.profile_item['path']
        self.write(self.folder/'index.json',doc)
        with self.assertRaises(ValueError):self.verify()

    def test_original_native_control_excluded(self):
        self.result['player']='control';self.persist()
        with self.assertRaises(ValueError):self.verify()

    def test_heuristic_missing_main_and_bad_root_join_rejected(self):
        self.result['diagnostics']['phases'][0]['heuristic_phase']=True;self.persist()
        with self.assertRaises(ValueError):self.verify()
        self.result['diagnostics']['phases'][0]['heuristic_phase']=False;self.persist()
        self.panel['host_metadata']={}
        with self.assertRaises(ValueError):self.verify()

    def test_profile_artifact_authentication(self):
        profiles={}
        for quota in (16,*adapters.QUOTAS):
            profiles[str(quota)]=self.write(self.folder/f'profile{quota}.json',adapters.profile(quota))
        frozen={'research_profiles':profiles}
        self.assertEqual(set(verifier._profiles(frozen,boundary=lambda:None)),set(adapters.QUOTAS))
        changed=adapters.profile(64);changed['cpuct']=.5
        frozen['research_profiles']['64']=self.write(self.folder/'profile64.json',changed)
        with self.assertRaises(ValueError):verifier._profiles(frozen,boundary=lambda:None)

    def test_exact_reproduction_includes_callback_evidence(self):
        # Inventory/clock/science paths have their separate rejection cases.
        # Here a changed callback bit must fail despite the same selected move.
        roots=[dict(self.root,root_id=f'fixture_{i}') for i in range(16)]
        packet=dict(roots=roots,host_metadata={r['root_id']:dict(game_index=1024,split='validation') for r in roots})
        originals=[{'spec':s} for s in panel.request_matrix(packet)]
        repeats=[{'spec':s} for s in panel.reproduction_matrix(packet)]
        def verify_one(row,*args,**kwargs):
            bit=100 if changed and 'original_request_id' in row['spec'] else 99
            return {},{'request_id':row['spec']['request_id']},{'action':'unchanged','callback_bits':[bit]}
        for changed in (False,True):
            with patch.object(verifier,'_profiles',return_value={}), \
                 patch.object(verifier,'_index',side_effect=[(originals,{}),(repeats,{})]), \
                 patch.object(verifier,'verify_forensics',return_value={'status':'passed'}), \
                 patch.object(verifier,'_verify_one',side_effect=verify_one):
                if changed:
                    with self.assertRaises(ValueError):
                        verifier.verify_reproductions(self.folder,self.folder,packet,freeze={})
                else:
                    proof=verifier.verify_reproductions(self.folder,self.folder,packet,freeze={})
                    self.assertEqual(proof['repetitions'],32)
                    self.assertTrue(all(x['full_result_exact'] for x in proof['checks']))

    def test_forensic_changed_raw_action_missing_extra_and_failed_event_rejected(self):
        path=self.folder/'service/ordinary_forensics.jsonl'; original=path.read_text()
        for change in ('action','observer','transport','failure','sequence','previous','missing','extra','truncated'):
            row=json.loads(original)
            if change=='action':row['event']['event']['action']={'different':True}
            elif change=='observer':row['event']['event']['observer']={'different':True}
            elif change=='transport':row['event']['transport_request_id']='request-00000003'
            elif change=='failure':row['event']['event']['event']='worker_research_decision_failed'
            elif change=='sequence':row['sequence']=1
            elif change=='previous':row['previous_sha256']='0'*64
            text=panel.canonical(row)+'\n'
            if change=='missing':text=''
            elif change=='extra':text+=original
            elif change=='truncated':text=text[:-1]
            path.write_text(text,encoding='utf-8')
            with self.subTest(change=change),self.assertRaises(ValueError):self.verify()
        path.write_text(original,encoding='utf-8');self.assertEqual(self.verify()[1]['prevalidation_forensics']['events'],1)


if __name__=='__main__': unittest.main()
