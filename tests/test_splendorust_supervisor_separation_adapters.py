"""Model-free constructor equivalence, inherited clocks and failure preservation."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

from tools.local_player_qualification_128 import clock
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from tools.splendorust_search_scale_20261024.common import StudyWorker
from tools.splendorust_supervisor_separation_20261025 import adapters as a, placement as p
from test_splendorust_supervisor_separation_placement import fixture


def runtime(freeze,mode='isolated',stamp=100):
    return dict(schema=p.SCHEMA,role='runner',placement_mode=mode,pid=234,created=1234.5,
        monotonic_ns=stamp,current_thread_id=567,process_mask=freeze['placement']['role_masks'][mode],
        system_mask=list(range(20)),priority_class=p.NORMAL_PRIORITY,physical_core_mask=[0,1],player_mask=[0],
        host_topology_sha256=freeze['placement']['host_topology_sha256'],full_thread_inventory=True,
        threads=[dict(thread_id=567,state='observed',group=0,mask=freeze['placement']['role_masks'][mode],relative_priority=0)],
        short_lived_threads_constrained_by_process_mask=True)


class AdapterTests(unittest.TestCase):
    def test_constructor_is_exact_except_authenticated_topology_call(self):
        trees=[]
        for method in (clock.PersistentWorker.__init__,a._PlacementConstructor.__init__):
            tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
            assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign) and len(node.targets)==1 and
                isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr=='topology']
            self.assertEqual(len(assignments),1)
            trees.append((tree,assignments[0]))
        self.assertEqual(ast.unparse(trees[0][1].value),'processor_topology()')
        self.assertEqual(ast.unparse(trees[1][1].value),'placement.admit_worker_topology(self._placement_freeze, self.placement_mode)')
        for tree,node in trees:node.value=ast.Constant(value='only intentional constructor substitution')
        self.assertEqual(ast.dump(trees[0][0]),ast.dump(trees[1][0]))

    def test_mro_preserves_scientific_transport_and_uses_new_admission(self):
        cls=a.PlacedProfiledStudyWorker
        order=cls.__mro__
        self.assertIs(order[order.index(StudyWorker)+1],a._PlacementConstructor)
        self.assertIs(cls.request,ProfiledStudyWorker.request)
        self.assertIs(cls._receive,ProfiledStudyWorker._receive)
        self.assertIs(cls._stdout_reader,clock.PersistentWorker._stdout_reader)
        self.assertIs(cls._stderr_reader,clock.PersistentWorker._stderr_reader)
        self.assertIs(cls._log,StudyWorker._log)
        self.assertIs(cls.close,StudyWorker.close)
        self.assertIs(cls._sample,ProfiledStudyWorker._sample)

    def test_constructor_dispatch_and_original_aggregate_guard_are_preserved(self):
        freeze=fixture();called=[];original_guard=Mock()
        def fake_constructor(child,*args,**kwargs):
            called.append((args,kwargs));child.output_dir=Path('unused')
            child.process=Mock(pid=99);child._ps=Mock();child._ps.cpu_affinity.return_value=[0]
            child._ps.nice.return_value=p.NORMAL_PRIORITY;child.identity={'pid':99,'created':10.}
            child._closed=False
        with patch.object(p,'verify_runtime',return_value=runtime(freeze)) as verify,patch.object(p,'register_worker'),\
             patch.object(p,'verify_worker_placement'),patch.object(a.c,'save'),\
             patch('tools.splendorust_search_scale_20261024.common.file_hash',return_value='x'),\
             patch.object(a._PlacementConstructor,'__init__',fake_constructor):
            child=a.PlacedProfiledStudyWorker('fixture','unused',native_artifacts=[],placement_freeze=freeze,
                placement_mode='isolated',aggregate_sample_guard=original_guard,affinity=[0])
            self.assertEqual(len(called),1)
            self.assertEqual(called[0][1]['affinity'],[0])
            child.aggregate_sample_guard(child,{'rss_bytes':20})
            original_guard.assert_called_once_with(child,{'rss_bytes':20})
            self.assertEqual(verify.call_args.kwargs,{'full':False})

    def test_request_keeps_original_failure_when_after_guard_fails(self):
        freeze=fixture();child=Mock();child._placement_freeze=freeze;child.placement_mode='isolated'
        child.output_dir=Path('unused');child.process.pid=99
        receipt={'request_id':'request-00000003','status':'worker_error','received_before_deadline':True,'receipt_elapsed_seconds':.98}
        child.request.return_value=receipt
        with patch.object(p,'verify_runtime',side_effect=[runtime(freeze),ValueError('mask drift')]),patch.object(a.c,'save') as save:
            actual=a.request(child,{'player':'search'},1.)
        self.assertIs(actual,receipt)
        child.request.assert_called_once_with({'player':'search'},1.,validate=None)
        self.assertIn('mask drift',child.last_placement_error)
        self.assertIsNone(child.last_placement_evidence['after'])
        self.assertEqual(save.call_args.args[1]['request_id'],receipt['request_id'])

    def test_request_keeps_original_clock_when_evidence_save_fails(self):
        freeze=fixture();child=Mock();child._placement_freeze=freeze;child.placement_mode='isolated'
        child.output_dir=Path('unused');child.process.pid=99
        receipt={'request_id':'request-00000003','status':'ok'};child.request.return_value=receipt
        with patch.object(p,'verify_runtime',side_effect=[runtime(freeze),runtime(freeze,stamp=200)]),patch.object(a.c,'save',side_effect=OSError('disk')):
            self.assertIs(a.request(child,{},1.),receipt)
        self.assertIn('disk',child.last_placement_error)

    def test_bad_precheck_never_dispatches(self):
        child=Mock();child._placement_freeze=fixture();child.placement_mode='isolated'
        with patch.object(p,'verify_runtime',side_effect=ValueError('wrong role')):
            with self.assertRaisesRegex(ValueError,'wrong role'):a.request(child,{},1.)
        child.request.assert_not_called()

    def test_saved_request_binds_role_threads_identity_and_clock(self):
        freeze=fixture();before=runtime(freeze);after=runtime(freeze,stamp=300)
        proof=dict(schema=p.SCHEMA,placement_mode='isolated',request_id='r1',player_pid=99,before=before,after=after,error=None,
            checks_outside_original_request_clock=True,sample_checks_inside_original_clock=True,
            original_clock_and_failure_receipt_preserved=True)
        profile=dict(pid=234,child_pid=99,request_id='r1',request_start_ns=150,capture_snapshot_ns=250)
        kwargs=dict(request_id='r1',player_pid=99,supervisor_identity={'pid':234,'created':1234.5},parent_profile=profile)
        p.verify_saved_request(proof,freeze,'isolated',**kwargs)
        corruptions=[('role',lambda x:x['before'].update(role='watchdog')),
            ('ownership',lambda x:x['after'].update(pid=235)),
            ('current',lambda x:x['before']['threads'][0].update(state='exited_before_query')),
            ('precheck',lambda x:x['before'].update(monotonic_ns=151)),
            ('aftermath',lambda x:x['after'].update(monotonic_ns=249))]
        for label,mutate in corruptions:
            with self.subTest(label=label):
                bad=deepcopy(proof);mutate(bad)
                with self.assertRaises(ValueError):p.verify_saved_request(bad,freeze,'isolated',**kwargs)


if __name__=='__main__':unittest.main()
