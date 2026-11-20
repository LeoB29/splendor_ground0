"""Model-free adversarial tests for source-aware recovery auditing."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import verifier as v


class SegmentedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='variability_recovery_verifier_',dir=v.ROOT/'outputs')
        self.base=Path(self.temp.name);self.original=self.base/'original';self.oldfolder=self.original/'jobs/diagnostic/worker'
        self.newfolder=self.base/'recovery/jobs/diagnostic/worker'
        for folder in (self.oldfolder,self.newfolder):
            for part in ('completed','transport','service'):(folder/part).mkdir(parents=True,exist_ok=True)
        self.specs=[dict(request_id='r0_'+s,root_id='r0',quota=512,pool=3,mode='legacy',replicate=s+'_'+s,
            world_seed_label=s,encoding_seed_label=s) for s in 'ABCD']
        self.observer={'synthetic_transport_fixture':True}
        self.panel=dict(roots=[dict(root_id='r0',observer=self.observer,observer_sha256=v.digest(self.observer))],
            host_metadata={'r0':dict(game_index=1024,split='validation')},primary_root_ids=['r0'],deep_root_ids=[])
        panel_item=self.write(self.original/'panel.json',self.panel);matrix_item=self.write(self.original/'requests.json',self.specs)
        self.freeze={'panel':panel_item,'request_matrix':matrix_item}
        freeze_item=self.write(self.original/'freeze.json',self.freeze)
        self.stacks=ExitStack();self.stacks.enter_context(patch.object(v,'ORIGINAL_OUTPUT',self.original))
        self.stacks.enter_context(patch.object(v,'ORIGINAL_FREEZE_SHA',freeze_item['sha256']))
        self.stacks.enter_context(patch.object(v,'PREFIX_COUNT',2));self.stacks.enter_context(patch.object(v,'TOTAL_COUNT',4))
        self.prefix=[];self.suffix=[]
        for folder in (self.oldfolder,self.newfolder):self.write(folder/'transport/request-00000001.json',{'synthetic_graph_receipt':True})
        for index,spec in enumerate(self.specs):
            folder=self.oldfolder if index<2 else self.newfolder;ordinal=index if index<2 else index-2
            request_id=f'request-{ordinal+2:08d}'
            result=dict(player='search',observer_sha256=v.digest(self.observer),action={'synthetic':spec['request_id']},
                diagnostics={'synthetic':True},research={'phases':[{'sampling':{'seed':spec['replicate']},'callback_bits':[{'bits':spec['replicate']}]}]})
            clock=dict(status='ok',accepted=True,request_id=request_id,result=result,detail=None,deadline_seconds=120.,
                elapsed_seconds=.3,receipt_elapsed_seconds=.2,received_before_deadline=True,
                clock='perf_counter_ns',clock_implementation='synthetic',telemetry={})
            item=self.write(folder/'transport'/f'{request_id}.json',clock);row=dict(spec=spec,receipt=item)
            if index<2:
                completed=self.write(folder/'completed'/f'{ordinal:05d}.json',row)
                self.prefix.append(dict(matrix_index=index,completed=completed,**row))
            else:self.suffix.append(row)
        self.write(self.newfolder/'index.json',{'requests':self.suffix})
        self.forensics(self.oldfolder,[{k:r[k] for k in ('spec','receipt')} for r in self.prefix])
        self.forensics(self.newfolder,self.suffix)
        (self.oldfolder/'transport/events.jsonl').write_bytes(b'event1\nevent2\nevent3\n')
        (self.oldfolder/'transport/stdout.log').write_bytes(b'ready\ngraph\nresponse1\nresponse2\n')
        proof=v.old.verify_forensics(self.oldfolder,[{k:r[k] for k in ('spec','receipt')} for r in self.prefix])
        tails={name:dict(artifact=self.artifact(self.oldfolder/('service' if name=='ordinary_forensics.jsonl' else 'transport')/name),
            lines=count,partial_tail=False,last_request_id='request-00000003') for name,count in
            (('events.jsonl',3),('stdout.log',4),('ordinary_forensics.jsonl',2))}
        tails['ordinary_forensics.jsonl']['terminal_chain_sha256']=proof['terminal_chain_sha256']
        self.audit=dict(status='passed_read_only_prefix_and_tail_audit',freeze=freeze_item,completed_decisions=2,total_planned=4,
            source_matrix=matrix_item,complete_prefix_receipts=[r['receipt'] for r in self.prefix],checks=tails)
        audit_item=self.write(self.base/'prefix_audit.json',self.audit)
        self.manifest=dict(schema='teacher_variability_crash_boundary_v1',status='original_attempt_incomplete_after_machine_restart',
            original_output=self.original.relative_to(v.ROOT).as_posix(),original_freeze=freeze_item,prefix_count=2,total_decisions=4,
            next_matrix_index=2,next_spec=self.specs[2],next_transport_request_id='request-00000004',dispatch_state='unknown',
            next_decision_original_clock_resumable=False,completed_result_for_next_request_found=False,
            original_interrupted_exit_code=None,prefix_rows=self.prefix,prefix_integrity_audit=audit_item,tail_checks=tails)

    def tearDown(self):self.stacks.close();self.temp.cleanup()

    def artifact(self,path):
        raw=path.read_bytes();return dict(path=path.relative_to(v.ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(v.canonical(value),encoding='utf-8');return self.artifact(path)

    def forensics(self,folder,rows):
        previous=None;lines=[]
        for i,row in enumerate(rows):
            receipt=v.old._read_bound(row['receipt']);result=receipt['result']
            event=dict(event='complete_result_before_worker_validation',arm='search',observer=self.observer,
                **{k:result[k] for k in ('action','diagnostics','research')})
            chain=dict(sequence=i,previous_sha256=previous,event=dict(transport_request_id=receipt['request_id'],event=event))
            lines.append(v.canonical(chain)+'\n');previous=v.digest(chain)
        (folder/'service/ordinary_forensics.jsonl').write_text(''.join(lines),encoding='utf-8')

    def read(self):return v.read_prefix(self.manifest)

    def join(self):
        with patch.object(v.original_panel,'request_matrix',return_value=self.specs):
            return v._joined_inventory(self.oldfolder,self.newfolder,self.panel,self.freeze,self.manifest,boundary=lambda:None)

    def test_valid_prefix_is_preserved_and_remains_incomplete(self):
        rows,proof=self.read();self.assertEqual(len(rows),2);self.assertEqual(proof['original_attempt_status'],'incomplete')
        self.assertIsNone(proof['original_exit_code']);self.assertFalse(proof['original_receipts_relabelled'])
        self.assertFalse((self.oldfolder/'index.json').exists())
        self.assertEqual([r['spec'] for r in self.join()[0]],self.specs)

    def test_missing_duplicate_or_reordered_prefix_rejected(self):
        original=deepcopy(self.manifest)
        for kind in ('missing','duplicate','swapped','bad_ordinal','pending_changed'):
            self.manifest=deepcopy(original)
            if kind=='missing':self.manifest['prefix_rows'].pop()
            elif kind=='duplicate':self.manifest['prefix_rows'][1]=deepcopy(self.manifest['prefix_rows'][0])
            elif kind=='swapped':self.manifest['prefix_rows'].reverse()
            elif kind=='bad_ordinal':self.manifest['prefix_rows'][1]['matrix_index']=0
            else:self.manifest['next_spec']=deepcopy(self.specs[3])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.read()
        self.manifest=original

    def test_prefix_receipt_changed_or_extra_transport_rejected(self):
        path=v.ROOT/self.prefix[0]['receipt']['path'];original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaises(ValueError):self.read()
        path.write_bytes(original)
        self.write(self.oldfolder/'transport/request-00000004.json',{})
        with self.assertRaises(ValueError):self.read()

    def test_prefix_cannot_be_relabelled_clean_exit_or_closed(self):
        self.manifest['original_interrupted_exit_code']=0
        with self.assertRaises(ValueError):self.read()
        self.manifest['original_interrupted_exit_code']=None
        self.write(self.oldfolder/'index.json',{'requests':[]})
        with self.assertRaises(ValueError):self.read()

    def test_late_failed_or_changed_clock_cannot_be_adopted(self):
        path=v.ROOT/self.prefix[0]['receipt']['path'];initial=json.loads(path.read_text())
        for update in ({'status':'timeout','accepted':False},{'received_before_deadline':False},
                       {'receipt_elapsed_seconds':121.,'elapsed_seconds':122.},{'deadline_seconds':119.},
                       {'clock':'time.time'}):
            value=deepcopy(initial);value.update(update);item=self.write(path,value)
            self.manifest['prefix_rows'][0]['receipt']=item
            row={'spec':self.specs[0],'receipt':item}
            self.manifest['prefix_rows'][0]['completed']=self.write(self.oldfolder/'completed/00000.json',row)
            with self.subTest(update=update),self.assertRaises(ValueError):self.read()

    def test_partial_extra_tampered_forensics_and_stdout_rejected(self):
        path=self.oldfolder/'service/ordinary_forensics.jsonl';initial=path.read_bytes()
        for data in (initial[:-1],initial+initial.splitlines(keepends=True)[0],initial.replace(b'complete_result_before_worker_validation',b'worker_research_decision_failed',1)):
            path.write_bytes(data)
            with self.assertRaises(ValueError):self.read()
        path.write_bytes(initial)
        log=self.oldfolder/'transport/stdout.log';log.write_bytes(log.read_bytes()+b'incomplete')
        with self.assertRaises(ValueError):self.read()

    def test_missing_duplicate_swapped_and_tampered_suffix_rejected(self):
        for rows in ([],self.suffix[::-1],[self.suffix[0],self.suffix[0]],
                     [dict(self.suffix[0],spec=self.specs[0]),self.suffix[1]]):
            self.write(self.newfolder/'index.json',{'requests':rows})
            with self.assertRaises(ValueError):self.join()
        self.write(self.newfolder/'index.json',{'requests':self.suffix})
        self.write(self.newfolder/'transport/request-00000004.json',{})
        with self.assertRaises(ValueError):self.join()

    def test_complete_sampling_checked_only_after_prefix_suffix_join(self):
        def scientific(row,*args,**kwargs):
            receipt=v.old._read_bound(row['receipt']);result=receipt['result']
            return {'request_id':row['spec']['request_id']},{'request_id':row['spec']['request_id']},result
        with patch.object(v.original_panel,'request_matrix',return_value=self.specs), \
             patch.object(v.old,'_profiles',return_value={}),patch.object(v.old,'_verify_one',side_effect=scientific), \
             patch.object(v.original_panel,'CONDITIONS',tuple((3,'legacy',s,s) for s in 'ABCD')), \
             patch.object(v.qualification,'validate_crossed_sampling',return_value=['allfour']) as cross:
            records,proof=v.verify_segments(self.oldfolder,self.newfolder,self.panel,self.freeze,self.manifest)
            self.assertEqual(len(records),4);self.assertEqual(len(cross.call_args.args[0]),4)
            self.assertEqual([r['evidence_segment'] for r in proof['request_checks']],['original_prefix']*2+['recovery_suffix']*2)
            self.assertTrue(proof['amended_recovered_study'])


class ReproductionTests(unittest.TestCase):
    def test_exact_replays_bind_both_segments_and_reject_changed_callback(self):
        roots=[dict(root_id=f'r{i}',observer={},observer_sha256=v.digest({})) for i in range(16)]
        panel={'roots':roots,'host_metadata':{r['root_id']:{'game_index':i,'split':'validation'} for i,r in enumerate(roots)}}
        specs=[dict(request_id=f'{r["root_id"]}_{q}',root_id=r['root_id'],quota=q) for r in roots for q in (512,2048)]
        rows=[{'spec':s} for s in specs]
        repeat_specs=[dict(s,request_id='repeat_'+s['request_id'],original_request_id=s['request_id']) for s in specs]
        repeats=[{'spec':s} for s in repeat_specs]
        for changed in (False,True):
            def scientific(row,*args,**kwargs):
                bits=1 if changed and 'original_request_id'in row['spec'] else 0
                return {},{'request_id':row['spec']['request_id']},{'action':'same','callback_bits':[bits]}
            with patch.object(v,'_joined_inventory',return_value=(rows,{},{})), \
                 patch.object(v.original_panel,'reproduction_matrix',return_value=repeat_specs), \
                 patch.object(v.old,'_index',return_value=(repeats,{})),patch.object(v.old,'_profiles',return_value={}), \
                 patch.object(v.old,'_verify_one',side_effect=scientific),patch.object(v.old,'verify_forensics',return_value={}), \
                 patch.object(v,'PREFIX_COUNT',16):
                if changed:
                    with self.assertRaises(ValueError):v.verify_reproductions('old','new','repeat',panel,{}, {})
                else:
                    proof=v.verify_reproductions('old','new','repeat',panel,{}, {})
                    self.assertEqual(proof['repetitions'],32)
                    self.assertEqual({r['evidence_segment'] for r in proof['checks']},{'original_prefix','recovery_suffix'})


if __name__=='__main__':unittest.main()
