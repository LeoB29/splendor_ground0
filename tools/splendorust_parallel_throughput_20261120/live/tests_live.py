"""Model-free tests exercise the real lane, queue, packing and audit pathways."""
import contextlib,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from concurrent.futures import ThreadPoolExecutor
from . import common as c,worker,queue,schedule,audit,transport,ownership,prepare,service,adapters
from .. import common as parent

class FakeServices:
    lives=[]
    def __init__(self,folder,plan,owner,boundary,role,maximum,**pin):
        self.folder=Path(folder);self.child=None;self.role=role;self.pin=pin;self.boundary=boundary
        self.service_folder=self.folder/'service';self.__class__.lives.append(self)
    def start(self):self.boundary();self.child=object()
    def finish(self):self.child=None;return []
    def close(self):self.child=None
    def request(self,payload,timeout=120):return {'payload':payload,'timeout':timeout}
    def decision(self,observer,pool,*,context):
        self.boundary();assert pool==3
        return dict(clock=dict(status='ok'),source=dict(path='fake-replay',bytes=1,sha256='b'*64))

class LiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.object(parent,'OUT',self.root))
        self.stack.enter_context(patch.object(parent,'PREP',self.root))
        self.stack.enter_context(patch.object(parent,'PLAN',self.root/'freeze.json'))
        self.plan={};parent.save(parent.PLAN,self.plan)
        self.request=dict(kind='lane',mode='lane',lane=0,model_cpu=4,controller_cpus=[12],original_allowed=[0,2,4,6,12])
        FakeServices.lives=[]
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def shared_queue(self,phase=0):
        return queue.prepare(self.root/f'q{phase}',schedule.phase_specs(phase),schedule.phase_arm(phase),phase)
    def fake_game(self,folder,plan,spec,decision,boundary,*,initials):
        boundary();item=c.save(Path(folder)/'result.json',dict(spec=spec))
        candidates=[dict(spec=spec,ply=p,pool=3,source=item,observer={'ply':p},result_sha256='x') for p in (0,7,24,40)]
        return dict(spec=spec,game=item,failed_decisions=0,replay_candidates=candidates)
    def fake_checks(self):
        self.stack.enter_context(patch('tools.splendorust_parallel_throughput_20261120.live.arena.play_game',side_effect=self.fake_game))
        self.stack.enter_context(patch('tools.splendorust_parallel_throughput_20261120.live.verifier.read_bound',return_value={}))
        self.stack.enter_context(patch('tools.splendorust_parallel_throughput_20261120.live.verifier.verify_replays',
            side_effect=lambda a,b,p,**k:dict(status='passed',repetitions=len(a))))
        self.stack.enter_context(patch.object(transport,'verify_services',return_value=[]))
        self.plan['initials']={'fake':True}
    def test_fixed_counterbalance_and_seed_identity(self):
        self.assertEqual([schedule.phase_arm(i) for i in range(4)],['A1','A4','A4','A1'])
        self.assertEqual(schedule.phase_specs(0),schedule.phase_specs(1))
        self.assertEqual(schedule.phase_specs(2),schedule.phase_specs(3))
        for i in range(16):
            s=schedule.expected_spec(i);self.assertEqual(s['control_pool'],s['candidate_pool']);schedule.validate_spec(s)
            s['total_simulations_each']=64
            with self.assertRaises(ValueError):schedule.validate_spec(s)
    def test_real_lane_uses_new_service_for_replays_same_pin(self):
        self.fake_checks();self.request['queue']=self.shared_queue()
        result=worker.run_lane(self.root/'lane',self.plan,{},lambda:None,self.request,service_factory=FakeServices)
        self.assertEqual(len(result['games']),8)
        self.assertEqual(len(result['replayed']),16)
        self.assertEqual([s.role for s in FakeServices.lives],['main','replay'])
        self.assertEqual(FakeServices.lives[0].pin,FakeServices.lives[1].pin)
        self.assertEqual(len(queue.verify(self.request['queue'],[result])),8)
    def test_real_lane_failure_prevents_later_leases_and_retries(self):
        self.fake_checks();self.request['queue']=self.shared_queue()
        with patch('tools.splendorust_parallel_throughput_20261120.live.arena.play_game',side_effect=RuntimeError('saved failure')):
            with self.assertRaisesRegex(RuntimeError,'saved failure'):
                worker.run_lane(self.root/'lane',self.plan,{},lambda:None,self.request,service_factory=FakeServices)
        self.assertTrue((self.root/'lane/failure.json').exists())
        self.assertTrue((c.resolve(self.request['queue']).parent/'abort.json').exists())
        with self.assertRaises(ValueError):queue.lease(self.request['queue'],0)
        self.assertIsNone(FakeServices.lives[0].child)
        self.assertEqual(len(FakeServices.lives),1)
    def test_shared_queue_never_duplicates_under_race(self):
        item=self.shared_queue(1)
        def consume(lane):
            seen=[]
            while True:
                leased=queue.lease(item,lane)
                if leased is None:return seen
                seen.append(leased[0]['game_index'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs=list(pool.map(consume,range(4)))
        all_indices=[i for row in runs for i in row]
        self.assertEqual(sorted(all_indices),list(range(8)))
        self.assertEqual(len(all_indices),len(set(all_indices)))
    def test_no_retry_after_unfinished_lease(self):
        item=self.shared_queue();first=queue.lease(item,0)
        # A later owner may not reclaim the first slot merely because it lacks a result.
        second=queue.lease(item,0)
        self.assertEqual(first[0]['game_index'],0);self.assertEqual(second[0]['game_index'],1)
        with self.assertRaises(ValueError):queue.verify(item,[])
    def test_empty_slow_start_lane_still_closes_service_and_audits(self):
        self.fake_checks();self.request['queue']=self.shared_queue(1)
        while queue.lease(self.request['queue'],0):pass
        self.request['lane']=3
        result=worker.run_lane(self.root/'empty',self.plan,{},lambda:None,self.request,service_factory=FakeServices)
        self.assertEqual(result['games'],[]);self.assertEqual(result['replayed'],[])
        self.assertEqual(len(FakeServices.lives),1);self.assertIsNone(FakeServices.lives[0].child)
    def test_replay_selection_is_independent_of_results(self):
        s=schedule.expected_spec(0)
        rows=[dict(spec=s,ply=i,pool=3,observer={},source={},result_sha256='a') for i in range(60)]
        chosen=worker.selected_replays([dict(replay_candidates=rows)])
        self.assertEqual(len(chosen),2);self.assertTrue(any(x['ply']<24 for x in chosen));self.assertTrue(any(x['ply']>=24 for x in chosen))
        for row in rows:row['result_sha256']='b'
        other=worker.selected_replays([dict(replay_candidates=list(reversed(rows)))])
        self.assertEqual([r['ply'] for r in chosen],[r['ply'] for r in other])
    def test_scientific_identity_catches_float_sign_bit_and_other_fields(self):
        base=dict(action={'move':1},research={'phases':[{'telemetry':{'visits':[1,2],'root_value':0.0}}]},hidden={'raw':'a'})
        changed=json.loads(json.dumps(base));changed['research']['phases'][0]['telemetry']['root_value']=-0.0
        self.assertNotEqual(audit.science(base),audit.science(changed))
        changed=json.loads(json.dumps(base));changed['hidden']['raw']='b'
        self.assertNotEqual(audit.science(base),audit.science(changed))
    def test_first_eight_identity_check_fails_before_later_phase(self):
        items=[dict(path=str(i)) for i in range(8)]
        def bound(item):return dict(spec={'game_index':int(item['path'])})
        with patch('tools.splendorust_parallel_throughput_20261120.live.verifier.read_bound',side_effect=bound), \
             patch.object(audit,'trajectory',side_effect=lambda i:dict(moves=[i['path']],path=i['path'])):
            self.assertEqual(audit.compare_subset(items,items,list(range(8)))['setups'],8)
        calls=[dict(moves=[0]),dict(moves=[1])]
        with patch('tools.splendorust_parallel_throughput_20261120.live.verifier.read_bound',side_effect=bound), \
             patch.object(audit,'trajectory',side_effect=calls) as trajectory:
            with self.assertRaisesRegex(ValueError,'mismatch at setup 0'):
                audit.compare_subset(items,items,list(range(8)))
        self.assertEqual(trajectory.call_count,2)
    def test_qualification_runtime_calls_actual_suite_route(self):
        self.request.update(kind='qualification',mode='qualification')
        with patch('tools.splendorust_parallel_throughput_20261120.live.qualification.run_live_qualification',return_value={'status':'passed'}) as check, \
             patch.object(transport,'verify_services',return_value=[]):
            result=worker.run_lane(self.root/'qualification',{}, {},lambda:None,self.request,service_factory=FakeServices)
        check.assert_called_once();self.assertIn('qualification',result)
        self.assertEqual(FakeServices.lives[0].role,'qualification')
    def test_new_worker_source_guard_once_and_original_owner_clock(self):
        sources={'source':'hash'};placement=dict(model_cpus=[0,2,4,6],controller_cpus=[12],original_allowed=[0,2,4,6,12])
        owner=dict(monotonic_origin=1,monotonic_deadline=2,unix_origin=3,unix_deadline=4)
        plan=dict(corrected_source_receipt=sources,placement=placement,topology={})
        with patch('tools.future_training_rules_20261021.verify_runtime_sources',return_value=sources) as guard, \
             patch('tools.splendorust_parallel_throughput_20261120.topology.verify_startup') as topology, \
             patch('psutil.Process') as process,patch.object(parent,'identity',return_value={'pid':1,'created':2}), \
             patch.object(worker,'run_lane',return_value={'passed':True}) as run:
            worker.run(self.root/'actualentry',plan,owner,lambda:None,self.request)
        guard.assert_called_once();topology.assert_called_once()
        self.assertEqual(run.call_args.args[2]['limits'],owner)
        process.return_value.cpu_affinity.assert_called_with([12])
    def test_actual_service_constructor_restores_controller_on_failure_for_live_and_replay(self):
        import os
        class FakeProcess:
            mask=[0,2,4,6,12]
            pid=os.getpid()
            def cpu_affinity(self,value=None):
                if value is not None:self.__class__.mask=list(value)
                return list(self.__class__.mask)
        def failing_constructor(module,folder,**kwargs):
            self.assertEqual(FakeProcess.mask,[0,2,4,6,12])
            self.assertEqual(kwargs['affinity'],[4])
            self.assertEqual(kwargs['absolute_deadline'],12345)
            raise RuntimeError('admission fixture stopped before any model')
        with patch('psutil.Process',FakeProcess), \
             patch('tools.local_player_qualification_128.clock.PersistentWorker',side_effect=failing_constructor) as constructor:
            for role in ('main','replay'):
                svc=transport.Services(self.root/role,{},dict(limits={'monotonic_deadline':12345}),lambda:None,role,1,
                    model_cpu=4,controller_cpus=[12],original_allowed=[0,2,4,6,12])
                with self.assertRaisesRegex(RuntimeError,'admission fixture'):
                    svc.start()
                self.assertEqual(FakeProcess.mask,[12]);self.assertIsNone(svc.child)
        self.assertEqual(constructor.call_count,2)
    def test_saved_ownership_book_rejects_unplanned_affinity(self):
        fields=dict(schema='x',service={},native_pids={},startup=[],end=[],affinity=[4],all_owned_identities=[],
            native_identities={},expected_binaries={},console_artifact={})
        with patch.object(ownership,'make_book',return_value=fields):
            self.assertEqual(ownership.validate_book(fields,{}, {},affinity=[4])['status'],'passed')
            with self.assertRaises(ValueError):ownership.validate_book(fields,{}, {},affinity=[0])
    def service_fixture(self,late=False):
        folder=self.root/'modelservice';sources={'fixture':'hash'}
        # Exclusive tests rewrite only their temporary fake freeze before entry.
        parent.PLAN.write_text(json.dumps(dict(worker_inputs=[],corrected_source_receipt=sources,
            research_profiles={},control_profiles={})),encoding='utf-8')
        payload=dict(player='search',observer={},quota=512,pool=3,mode='legacy',
                     world_seed_label='A',encoding_seed_label='A',telemetry=True)
        envelope=dict(type='request',request_id='request-00000001',payload=payload)
        stdin=type('Input',(),{'buffer':io.BytesIO((json.dumps(envelope)+'\n').encode())})()
        stdout=io.StringIO();identity={'pid':123,'created':456}
        def handle(context,request):
            context['evidence_callback']({'fixture':'complete saved result'})
            return {'exact':'float32 fixture','value':-0.0}
        with patch.dict(os.environ,dict(SIXWORLD_SERVICE_FOLDER=str(folder),SIXWORLD_FREEZE_SHA256=c.sha(c.PLAN),SIXWORLD_ROLE='main')), \
             patch('tools.future_training_rules_20261021.verify_runtime_sources',return_value=sources) as guard, \
             patch.object(parent,'identity',return_value=identity), \
             patch.object(adapters,'initialize',return_value={}),patch.object(adapters,'handle',side_effect=handle), \
             patch.object(adapters,'close_context'),patch.object(service.sys,'stdin',stdin),patch.object(service.sys,'stdout',stdout), \
             patch.object(service.time,'perf_counter',side_effect=[0,120] if late else [0,1]):
            code=service.serve()
        guard.assert_called_once()
        self.assertTrue((folder/'runtime_sources.json').exists())
        summary=c.read(folder/'forensic_summary.json');self.assertEqual(summary['events'],1)
        return code,[json.loads(row) for row in stdout.getvalue().splitlines()],c.read(folder/'service_exit.json')
    def test_actual_service_loop_source_guard_once_and_lossless_packet(self):
        code,wire,exit=self.service_fixture()
        self.assertEqual(code,0);self.assertEqual(exit['status'],'passed')
        self.assertEqual([p['type'] for p in wire],['ready','response'])
        self.assertEqual(adapters.unpack_result(wire[1]['result']),{'exact':'float32 fixture','value':-0.0})
    def test_actual_service_loop_retains_original_internal119_failure(self):
        code,wire,exit=self.service_fixture(late=True)
        self.assertEqual(code,1);self.assertEqual(exit['status'],'failed')
        self.assertEqual([p['type'] for p in wire],['ready','error'])
        self.assertIn('VARIABILITY_INTERNAL_DEADLINE',exit['error'])
        self.assertTrue((self.root/'modelservice/failure.json').exists())

if __name__=='__main__':unittest.main()
