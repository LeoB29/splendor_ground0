"""Model-free role placement, immutable topology and real Windows inheritance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch, Mock

from tools.splendorust_supervisor_separation_20261025 import placement as p


def fixture():
    host = dict(api='GetLogicalProcessorInformationEx(RelationProcessorCore)',
        cores=[dict(core_index=i, flags=1, efficiency_class=1, logical_processors=[2*i,2*i+1]) for i in range(6)] +
              [dict(core_index=6+i, flags=0, efficiency_class=0, logical_processors=[12+i]) for i in range(8)],
        allowed_logical_processors=list(range(20)), selected_logical_processor=0, selected_core_index=0,
        selection_rule='highest efficiency class; then SMT width; then lowest allowed logical ID',
        distinct_efficiency_classes=[0,1])
    return dict(topology=host, placement=p.plan(host))


class PlacementTests(unittest.TestCase):
    def test_plan_keeps_host_and_excludes_entire_physical_core(self):
        freeze=fixture();original=deepcopy(freeze['topology'])
        self.assertEqual(freeze['placement']['role_masks']['isolated'],list(range(2,20)))
        self.assertEqual(freeze['placement']['player_mask'],[0])
        self.assertEqual(freeze['topology'],original)
        altered=deepcopy(freeze);altered['placement']['physical_core_mask']=[0]
        with self.assertRaisesRegex(ValueError,'differs'):p._planned(altered,'isolated')

    def test_role_selection_does_not_replace_host_selection(self):
        freeze=fixture();current=deepcopy(freeze['topology'])
        current.update(allowed_logical_processors=list(range(2,20)),selected_logical_processor=2,selected_core_index=1)
        with patch.object(p.clock,'processor_topology',return_value=current),patch.object(p,'_process_masks',return_value=(list(range(2,20)),list(range(20)))):
            self.assertEqual(p.verify_host(freeze),freeze['topology'])
            current['cores'][0]['efficiency_class']=0
            with self.assertRaisesRegex(ValueError,'Physical host'):p.verify_host(freeze)

    def test_system_mask_cannot_be_invented_from_frozen_eligibility(self):
        freeze=fixture()
        with patch.object(p.clock,'processor_topology',return_value=freeze['topology']),patch.object(p,'_process_masks',return_value=(list(range(2,20)),list(range(2,20)))):
            with self.assertRaisesRegex(ValueError,'system processor'):p.verify_host(freeze)

    def test_runtime_rejects_process_mask_and_priority_drift(self):
        freeze=fixture();proc=Mock(pid=123);proc.cpu_affinity.return_value=list(range(2,20));proc.nice.return_value=p.NORMAL_PRIORITY
        with patch.object(p.psutil,'Process',return_value=proc),patch.object(p,'_process_masks',return_value=(list(range(20)),list(range(20)))):
            with self.assertRaisesRegex(ValueError,'affinity changed'):p.verify_runtime(freeze,'isolated',full=False)
        proc.nice.return_value=0x80
        with patch.object(p.psutil,'Process',return_value=proc),patch.object(p,'_process_masks',return_value=(list(range(2,20)),list(range(20)))):
            with self.assertRaisesRegex(ValueError,'priority changed'):p.verify_runtime(freeze,'isolated',full=False)

    def test_configure_refuses_live_worker(self):
        worker=Mock();worker._closed=False
        with patch.object(p,'_workers',[worker]):
            with self.assertRaisesRegex(ValueError,'resident workers'):p.configure_execution(fixture(),'isolated')

    def test_configure_rejects_unexpected_inherited_mask(self):
        proc=Mock();proc.nice.return_value=p.NORMAL_PRIORITY
        with patch.object(p,'verify_host'),patch.object(p.psutil,'Process',return_value=proc),patch.object(p,'_process_masks',return_value=([2],list(range(20)))):
            with self.assertRaisesRegex(ValueError,'inherited'):p.configure_execution(fixture(),'isolated')
        proc.cpu_affinity.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows single-group placement fixture')
    def test_actual_new_threads_and_process_inherit_isolated_mask(self):
        # The fresh fixture subprocess owns every affinity mutation. No model,
        # worker or change to the pytest/Codex process is involved.
        script=r'''
import json,sys,threading,subprocess
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_supervisor_separation_20261025 import placement as p
host=processor_topology();freeze={'topology':host,'placement':p.plan(host)}
original=p.configure_execution(freeze,'original')
isolated=p.configure_execution(freeze,'isolated')
gate=threading.Event();ready=threading.Event();records=[]
def reader():
 records.append(p.verify_runtime(freeze,'isolated'));ready.set();gate.wait(5)
t=threading.Thread(target=reader);t.start()
assert ready.wait(5)
while_live=p.verify_runtime(freeze,'isolated')
inherited=json.loads(subprocess.check_output([sys.executable,'-B','-c','import json,psutil;print(json.dumps(psutil.Process().cpu_affinity()))'],text=True))
gate.set();t.join(5);assert not t.is_alive()
restored=p.configure_execution(freeze,'original')
print(json.dumps({'original':original['process_mask'],'isolated':isolated['process_mask'],'inherited':inherited,'thread':records[0]['threads'],'live':while_live['threads'],'restored':restored['process_mask']}))
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertEqual(data['original'],list(range(20)))
        self.assertEqual(data['isolated'],list(range(2,20)))
        self.assertEqual(data['inherited'],list(range(2,20)))
        self.assertEqual(data['restored'],list(range(20)))
        self.assertGreaterEqual(len(data['live']),2)
        for row in data['live']:
            if row['state']=='observed':self.assertEqual(row['mask'],list(range(2,20)))

    @unittest.skipUnless(os.name=='nt','Windows isolated parent / pinned player fixture')
    def test_real_constructor_keeps_player_zero_without_broadening_parent(self):
        script=r'''
import json,os,sys,tempfile
from pathlib import Path
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_supervisor_separation_20261025 import placement as p
from tools.splendorust_supervisor_separation_20261025.adapters import _PlacementConstructor
host=processor_topology();freeze={'topology':host,'placement':p.plan(host)}
p.configure_execution(freeze,'isolated')
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 (root/'placement_arithmetic_fixture.py').write_text('import psutil\nfrom tools.local_player_qualification_128.clock import serve\nraise SystemExit(serve(lambda:None,lambda context,payload:dict(sum=sum(payload),affinity=psutil.Process().cpu_affinity())))\n')
 child=object.__new__(_PlacementConstructor)
 child._placement_freeze=freeze;child.placement_mode='isolated'
 child.__init__('placement_arithmetic_fixture',root/'worker',affinity=[0],initialization_timeout=10,
  env={'PYTHONPATH':str(root)+os.pathsep+str(Path.cwd())})
 try:
  before=p.verify_runtime(freeze,'isolated')
  receipt=child.request([2,3],1.)
  after=p.verify_runtime(freeze,'isolated')
  assert receipt['status']=='ok',receipt
  assert receipt['result']=={'sum':5,'affinity':[0]},receipt
  assert child.topology==host
  print(json.dumps({'before':before['process_mask'],'after':after['process_mask'],'player':child.affinity,'status':receipt['status'],'parent_threads':after['threads']}))
 finally:child.close()
p.configure_execution(freeze,'original')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertEqual(data['before'],list(range(2,20)))
        self.assertEqual(data['after'],list(range(2,20)))
        self.assertEqual(data['player'],[0])
        self.assertEqual(data['status'],'ok')
        self.assertGreaterEqual(sum(row['state']=='observed' for row in data['parent_threads']),3)


if __name__=='__main__':unittest.main()
