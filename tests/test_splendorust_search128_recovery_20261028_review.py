"""Bounded model-free checks of the new segmented acceptance boundaries."""
from copy import deepcopy
from contextlib import ExitStack
from itertools import product
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.splendorust_search128_recovery_20261028 import independent_review as v


class SegmentedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=v.OUT/'prelaunch/saved_response_bridge.json'
        cls.bundle=json.loads(p.read_text())
        game=v.ORIGINAL/'games/main/p019_g38'
        cls.initial=json.loads((game/'initial.json').read_text())
        cls.prefix=[dict(kind='ordinary',value=json.loads((game/'moves'/f'{i:04d}.json').read_text())) for i in range(39)]

    def scenario(self):
        b=deepcopy(self.bundle);events=deepcopy(self.prefix)+[dict(kind='recovery_bridge',value=b)]
        # A synthetic subsequent ordinary envelope tests reader chain semantics;
        # the actual collector's independent ledger proof supplies game legality.
        row=deepcopy(self.prefix[-1]['value']);row.update(before=deepcopy(b['after']),ply=40)
        final=deepcopy(b['after']);final['state'].update(terminal=True,winner=0)
        row['after']=final;events.append(dict(kind='ordinary',value=row))
        spec=v.expected_game(38)
        ending=dict(spec,complete=True,plies=41,reason='natural',winner=0,final=final)
        return spec,deepcopy(self.initial),ending,events,b

    def test_bridge_is_distinct_event_and_not_timed_row(self):
        value=v.validate_game_chain(*self.scenario())
        self.assertEqual(value['bridges'],1);self.assertEqual(len(value['ordinary']),40)
        self.assertEqual(value['transitions'],41);self.assertIsNone(value['failed'])

    def test_second_bridge_fails(self):
        args=list(self.scenario());args[3][-1]=deepcopy(args[3][-2])
        with self.assertRaisesRegex(ValueError,'recovered ply39'):v.validate_game_chain(*args)

    def test_bridge_in_another_game_fails(self):
        args=list(self.scenario());args[0]['game_index']=39
        with self.assertRaises(ValueError):v.validate_game_chain(*args)

    def test_missing_bridge_never_counts_as_completion(self):
        args=list(self.scenario());args[3][39]=dict(kind='ordinary',value=deepcopy(self.bundle['source_row']))
        with self.assertRaisesRegex(ValueError,'Only the explicit research hard guard'):v.validate_game_chain(*args)

    def test_recovered_successor_tamper_fails(self):
        args=list(self.scenario());args[3][39]['value']['after']['state']['turn_index']+=2
        with self.assertRaises(ValueError):v.validate_game_chain(*args)

    def test_skip_or_repeat_ply_fails(self):
        args=list(self.scenario());args[3][-1]['value']['ply']=39
        with self.assertRaisesRegex(ValueError,'contiguous chain'):v.validate_game_chain(*args)

    def test_original_rejected_receipt_never_rewritten(self):
        args=self.scenario();before=deepcopy(args);v.validate_game_chain(*args)
        self.assertEqual(args,before);self.assertFalse(args[-1]['source_row']['clock']['accepted'])

    def test_no_game_continuation_after_hard_guard(self):
        args=list(self.scenario());row=args[3][-1]['value'];row['completed']=False;row.update(action=None,trace=None,after=None)
        row['clock'].update(status='timeout',accepted=False,result=None,elapsed_seconds=10.,receipt_elapsed_seconds=None,received_before_deadline=False)
        args[3].append(deepcopy(args[3][-1]));args[2]['plies']+=1
        with self.assertRaisesRegex(ValueError,'Hard guard ends'):v.validate_game_chain(*args)

    def clock_fixture(self):
        final=dict(cap_seconds=14400-131,monotonic=v.ORIGIN_MONO+5000,unix_time=v.ORIGIN_UNIX+5000,
                   monotonic_deadline=v.ORIGIN_MONO+5000+14400-131,unix_deadline=v.ORIGIN_UNIX+5000+14400-131)
        clock=dict(freeze_sha256='x',review_monotonic_origin=final['monotonic']+100,
                   review_unix_origin=final['unix_time']+100,guardian_claim_sha256='g',completion_sha256='c')
        clock.update(deadline_monotonic=clock['review_monotonic_origin']+3600,deadline_unix=clock['review_unix_origin']+3600)
        rows={'guardian_claim.json':dict(freeze_sha256='x',monotonic_origin=v.ORIGIN_MONO,unix_origin=v.ORIGIN_UNIX),
              'stages/final_start.json':final,'postrun/review_owner_clock.json':clock,
              'completion.json':dict(status='amended_recovered_verified',independent_acceptance_deadline_monotonic=final['monotonic_deadline'],
                  independent_acceptance_deadline_unix=final['unix_deadline'])}
        class Stub:
            def read(self,name):return rows[name]
            def pin(self,path):return dict(sha256='g' if path.name=='guardian_claim.json' else 'c')
            def guard(self):pass
        return rows,Stub()

    def test_shared_deadline_carries_prior_final_seconds(self):
        rows,r=self.clock_fixture();v.bind_shared_clock(r,'x')
        self.assertEqual(r.shared_deadlines,(rows['postrun/review_owner_clock.json']['deadline_monotonic'],
                                            rows['postrun/review_owner_clock.json']['deadline_unix']))

    def test_reset_global_origin_rejected(self):
        rows,r=self.clock_fixture();rows['guardian_claim.json']['monotonic_origin']+=1
        with self.assertRaisesRegex(ValueError,'origins'):v.bind_shared_clock(r,'x')

    def test_reset_final_budget_rejected(self):
        rows,r=self.clock_fixture();rows['stages/final_start.json']['cap_seconds']=14400
        with self.assertRaisesRegex(ValueError,'131'):v.bind_shared_clock(r,'x')

    def test_acceptance_does_not_get_fresh_3600(self):
        rows,r=self.clock_fixture();rows['postrun/review_owner_clock.json']['deadline_monotonic']+=1
        with self.assertRaisesRegex(ValueError,'Shared review'):v.bind_shared_clock(r,'x')

    def test_guard_before_publication(self):
        class Stub:
            def guard(self):raise TimeoutError('expired')
        with patch.object(v,'save_new') as saved:
            with self.assertRaises(TimeoutError):v.guarded_publish(Stub(),Path('never'),{})
            saved.assert_not_called()

    def test_guard_after_publication(self):
        class Stub:
            calls=0
            def guard(self):
                self.calls+=1
                if self.calls==2:raise TimeoutError('publication overran')
        with patch.object(v,'save_new') as saved:
            with self.assertRaisesRegex(TimeoutError,'overran'):v.guarded_publish(Stub(),Path('never'),{})
            saved.assert_called_once()

    def test_combined_evidence_reserve_checked_before_save(self):
        class Stub:
            output_evidence_bytes=16*v.GiB-1024
            def guard(self):pass
        with patch.object(v,'save_new') as saved:
            with self.assertRaisesRegex(ValueError,'evidence cap'):v.guarded_publish(Stub(),Path('never'),{})
            saved.assert_not_called()

    def test_unchanged_pure_score_schedule_clock_helpers(self):
        self.assertIs(v.check_statistics,v.old.check_statistics)
        self.assertIs(v.check_clock,v.old.check_clock)
        self.assertIs(v.selection,v.old.selection)
        self.assertEqual(len([v.expected_game(i) for i in range(400)]),400)

    def test_scalar_timing_accumulator_does_not_retain_row_or_clock(self):
        row=deepcopy(self.prefix[0]['value']);acc=v.TimingAccumulator();acc.add(row)
        stamp=row['clock']['receipt_elapsed_seconds'];row['clock']['receipt_elapsed_seconds']=999
        self.assertEqual(acc.times[row['player_role']],[stamp])
        self.assertFalse(any(isinstance(x,dict) and 'clock' in x for x in vars(acc).values()))


class FinalSchemaIntegrationTests(unittest.TestCase):
    """Full406-entry reader flow with explicit non-scientific component stubs.

    Real saved prefix/bridge and raw clock logic exercise segmented schema,
    selection, timing denominators and all final-summary relationships. File
    hashing/owner observation, scientific proofs and bootstrap are separately
    authenticated production components, deliberately stubbed here; no invented
    fixture is claimed as a legal game or experimental result.
    """
    @classmethod
    def setUpClass(cls):SegmentedReviewTests.setUpClass.__func__(cls)
    scenario=SegmentedReviewTests.scenario

    def fixture(self):
        data={};out=v.ROOT/'__review_fixture_new';prior=v.ROOT/'__review_fixture_old';sha='new-freeze'
        def put(path,value):
            path=Path(path);data[path]=deepcopy(value)
            return dict(path=path.relative_to(v.ROOT).as_posix(),sha256=v.digest(value),bytes=len(v.canonical(value).encode()))
        def save(name,value):return put(out/name,value)
        class Reader:
            counts={}
            def __init__(self):self.counts={};self.exact=[]
            def physical(self,path):return v.ROOT/path
            def read(self,name):return deepcopy(data[out/name])
            def read_path(self,path):return deepcopy(data[Path(path)])
            def pin(self,path):
                value=data[Path(path)]
                return dict(path=Path(path).relative_to(v.ROOT).as_posix(),sha256=sha if Path(path)==out/'freeze.json' else v.digest(value),bytes=len(v.canonical(value).encode()))
            def verify(self,item):
                value=data[self.physical(item['path'])]
                if item['path']!=(out/'freeze.json').relative_to(v.ROOT).as_posix():
                    v.require(item['sha256']==v.digest(value),'Fixture descriptor binding')
            def exact_tree(self,path,items):self.exact.append((path,items))
            def guard(self):pass
        r=Reader();bundle=deepcopy(self.bundle);bundle['source_row']['worker_folder']=(prior/'workers/main_g38_challenger').relative_to(v.ROOT).as_posix()
        bundle_pin=save('recovered_bridge.json',bundle)
        oldfreeze=put(prior/'freeze.json',{})
        oldfreeze['sha256']=v.ORIGINAL_SHA
        # The freeze's historical SHA is asserted independently; fixture verifier
        # explicitly delegates this one immutable source pin to the source stub.
        original_verify=r.verify
        r.verify=lambda item:None if item==oldfreeze else original_verify(item)
        failed=put(prior/'games/main/p019_g38/moves/0039.json',bundle['source_row'])
        original_item=put(prior/'completion.json',dict(status='incomplete'))
        for name in ('execution.json','watchdog_execution.json','guardian_os_exit.json'):
            put(prior/name,dict(exit_code=1));save(name,dict(exit_code=0,freeze_sha256=sha))
        manifest=dict(old_freeze=oldfreeze,original_attempt_status='incomplete',original_failed_receipt_preserved=True,
            new_call_for_failed_move=False,move_clock_reset=False,completed_main_games=38,continuation_ply=40,
            original_tree=[original_item],original_tree_files=1,original_tree_bytes=original_item['bytes'],bridge=bundle_pin)
        manifest_pin=save('recovery_manifest.json',manifest)
        frozen=dict(schema='splendorust_search128_disclosed_recovery_v1',inputs=[original_item],original_freeze=oldfreeze,
            recovery_manifest=manifest_pin,main_games=400,operational_games=6,seconds_per_move=10,internal_seconds=9,
            job_seconds=86400,collection_cutoff_seconds=72000,max_workers=864,max_evidence_bytes=16*v.GiB,
            final_reserve_bytes=4*v.GiB,prefinal_evidence_bytes=12*v.GiB,prior_final_seconds=131,prior_ram_wait_seconds=130,
            automatic_retries=0,failed_decision_retries=0,source_attempt_remains_incomplete=True,
            original_failed_receipt_overwritten=False,training=False,max_rss_bytes=2*v.GiB)
        save('freeze.json',frozen)
        schedule=[v.expected_game(i,stage) for stage,n in (('operational',6),('main',400)) for i in range(n)]
        put(prior/'schedule.json',schedule)
        entries=[];ends=[];allrows=[];selected={};selected_rows={};replays=[]
        for spec in schedule:
            recovered=spec['stage']=='main' and spec['game_index']==38
            if recovered:
                _,initial,ending,events,_=self.scenario()
                events[39]['value']=bundle
            else:
                seat=spec['challenger_seat'];before=dict(state=dict(current_player=seat,terminal=False))
                after=dict(state=dict(current_player=seat,terminal=True,winner=seat))
                result=dict(action={'fixture':True},diagnostics={})
                clock=dict(clock='perf_counter_ns',request_id='request-00000001',status='ok',accepted=True,deadline_seconds=10.,
                    elapsed_seconds=.3,receipt_elapsed_seconds=.2,received_before_deadline=True,result=result)
                row=dict({k:spec[k] for k in ('stage','pair_index','game_index')},ply=0,seat=seat,player_role='challenger',
                    mode='research',route='research',quota=128,before=before,after=after,arm='search',placement_mode='isolated',
                    observation_mode='profiled',completed=True,clock=clock,action=result['action'],trace=result['diagnostics'],
                    observer=dict(observation=dict(current_player=seat,players=[dict(reserves=[]),dict(reserves=[])])))
                initial=dict(spec,initial=before);ending=dict(spec,complete=True,plies=1,reason='natural',winner=seat,final=after)
                events=[dict(kind='ordinary',value=row)]
            base=prior if spec['stage']=='operational' or spec['game_index']<38 else out
            folder=base/'games'/spec['stage']/f"p{spec['pair_index']:03d}_g{spec['game_index']}"
            descriptors=[]
            for ply,event in enumerate(events):
                if event['kind']=='recovery_bridge':
                    descriptors.append(dict(kind='recovery_bridge',bridge=bundle_pin,source_freeze_sha256=v.ORIGINAL_SHA));continue
                row=event['value'];rowbase=prior if recovered and ply<39 else base
                worker=rowbase/'workers'/f"{spec['stage']}_g{spec['game_index']}_{row['player_role']}"
                row['worker_folder']=worker.relative_to(v.ROOT).as_posix()
                row['worker_clock_file']=(worker/f"request-{ply:08d}.json").relative_to(v.ROOT).as_posix()
                put(v.ROOT/row['worker_clock_file'],row['clock'])
                pin=put(rowbase/'games'/spec['stage']/folder.name/'moves'/f'{ply:04d}.json',row)
                descriptors.append(dict(kind='ordinary',row=pin,source_freeze_sha256=v.ORIGINAL_SHA if rowbase==prior else sha))
                allrows.append(row)
                if spec['stage']=='main':
                    cell,item=v.selection(row)
                    if cell not in selected or item['selection_sha256']<selected[cell]['selection_sha256']:
                        selected[cell]=item;selected_rows[cell]=row
            checked=v.validate_game_chain(spec,initial,ending,events,bundle)
            proof=put(folder/'online_replay.json',dict(status='passed',transitions=checked['transitions'],failed_moves=0,final_sha256=checked['final_sha256']))
            entries.append(dict(spec=spec,initial=put(folder/'initial.json',initial),result=put(folder/'result.json',ending),online_replay=proof,events=descriptors))
            ends.append(ending);replays.append(proof)
        save('game_evidence_manifest.json',dict(games=entries))
        save('segments_manifest.json',dict(original_freeze=oldfreeze,recovery_freeze=r.pin(out/'freeze.json'),bridge=bundle_pin,
            original_failed_request=failed,ordinary_timing_excludes_bridge=True))
        save('all_game_replay.json',dict(status='passed',games=406,receipts=replays))
        save('all_player_faults.json',dict(count=0,faults=[],failed_move_retries=0))
        cells=list(product(v.ROLES,('0..19','20..39','40+'),(0,1),('neither','own','opponent','both')))
        chosen=[selected[x] for x in cells if x in selected]
        empty=[dict(player_role=x[0],phase=x[1],seat=x[2],private_category=x[3]) for x in cells if x not in selected]
        save('replay_sample_manifest.json',dict(decisions=chosen,empty_cells=empty,maximum_decisions=48,domain=v.old.SAMPLE_DOMAIN,
            scope='main_completed_cells',all_available_cells_selected=True,failed_moves_excluded=True,selection_uses_outcomes=False,selection_uses_timing=False))
        save('repeated_decision_summary.json',dict(status='passed',requests=len(chosen),identities=chosen))
        for i,cell in enumerate(x for x in cells if x in selected):
            row=deepcopy(selected_rows[cell]);row['worker_folder']=(out/'workers'/f'audit_{i:02d}').relative_to(v.ROOT).as_posix()
            save(f'repeated_decisions/{i:02d}.json',row)
        coverage=dict(complete_planned_workload=True,planned_games=406,verified_games=406,operational_games=6,main_games=400,unplayed_games=0,
            live_decisions=len(allrows),live_game_events=len(allrows)+1,recovered_events=1,original_infrastructure_failures_preserved=1,
            qualification_ordinary_requests=84,qualification_scientific_decisions=322,recovery_actual_entrypoint_decisions=119,
            failed_decision_retries=0,original_attempt_status='incomplete',forfeits=0,reproductions=len(chosen))
        save('coverage.json',coverage);save('completion.json',dict(status='amended_recovered_verified',failure=None,freeze_sha256=sha,automatic_retries=0,coverage=coverage))
        workers=dict(workers=816+len(chosen),parent_requests=999,graph_cases=11*(816+len(chosen)),identities=[])
        save('verification_summary.json',dict(status='passed',all_raw_placement_logs_graph_science_sources_verified=True,source_inputs=1,
            coverage=coverage,peak_combined_rss_bytes=100,**{k:val for k,val in workers.items() if k!='identities'}))
        analysis=save('analysis.json',{});statistics=save('analysis_verification.json',{})
        save('analysis_seal.json',dict(artifacts=[analysis,statistics]));save('completed_evidence_seal.json',dict(artifacts=[]))
        # Use actual final-summary schema, independently exercised below.
        self.final_contract_data(save,put,prior,bundle,allrows,len(chosen))
        return r,data,out,prior,sha,bundle,workers,len(allrows)

    def final_contract_data(self,save,put,prior,bundle,rows,n):
        gate=put(prior/'assessment_gate.json',dict(passed=True))
        save('recovery_timing.json',dict(ordinary_metrics_exclude_bridge=True,source_receipt_not_rewritten=True,
            excluded_original_clock=bundle['source_row']['clock'],recovered_raw_arrival_evidence=bundle['bridge']['timing_evidence'],
            ordinary_timed_live_requests=len(rows),logical_live_events=len(rows)+1))
        save('prior_qualification_verification.json',dict(status='passed',ordinary_requests=84,scientific_decisions=203,
            old_freeze_sha256=v.ORIGINAL_SHA,original_suites=[dict(status='passed')]*2,prior_operational_gate=gate))
        receipts={};proofs={}
        for route,count in [('research',67),('frozen',52)]:
            receipt=dict(status='ok',accepted=True,result=dict(execution_route=route,counts=dict(complete_decisions=count)))
            forensic=save('fixture_'+route+'_forensic.json',{})
            proof=dict(status='passed',route=route,complete_decisions=count,original_complete_decisions=52,direct_same_quota=count-52,
                ordinary_handlers=37,negative_handler_cases=26,independent_saved_transitions=37,model_calls=0,qualification_rerun=False,forensic_artifact=forensic)
            receipts[route]=receipt;proofs[route]=proof
        # Execute the actual new runner's summary builder with only worker/model
        # boundaries mocked. This catches route ordering and real writer shape.
        from types import SimpleNamespace
        from tools.splendorust_search128_recovery_20261028 import runner,adapters,qualification
        pins={}
        def capture(path,value):pins[str(path)]=save(str(Path(path).relative_to(runner.c.OUT)),value)
        def child(name,*args,route,**kwargs):return SimpleNamespace(output_dir=runner.c.OUT/'workers'/name,route=route)
        with patch.object(runner,'make_worker',side_effect=child),patch.object(adapters,'actual_entrypoint',side_effect=lambda child,*a,**k:receipts[child.route]),\
             patch.object(qualification,'verify_entrypoint',side_effect=lambda folder,receipt,*a,**k:proofs[receipt['result']['execution_route']]),\
             patch.object(runner.c,'save',side_effect=capture),patch.object(runner.c,'artifact',side_effect=lambda path:pins[str(path)]):
            runner.qualify_resumed_workers(SimpleNamespace(guard=lambda:None),{}, {},{})
        save('recovery_resource_gate.json',dict(passed=True,remaining_scheduled_games=362,outcomes_used=False,move_timing_subtracted_seconds=0,
            prior_final_seconds_charged=131,prior_ram_wait_allocation_seconds=130,original_monotonic_origin=v.ORIGIN_MONO,
            original_unix_origin=v.ORIGIN_UNIX,maximum_combined_workers=864,qualification_extra_scientific_decisions=119,original_gate=gate,
            projected_collection_seconds=1,available_collection_seconds=100,projected_final_seconds=4000,available_global_seconds=9000,
            aggregate_current_evidence_bytes=1,projected_additional_collection_bytes=1,projected_final_bytes=1))
        save('reproduction_resource_gate.json',dict(passed=True,selected_reproductions=n,prior_final_seconds_charged=131,
            independent_review_reserved_seconds=3600,projected_remaining_seconds=4000,available_seconds=5000,projected_final_bytes=1,current_evidence_bytes=1))

    def run_fixture(self,mutate=None):
        r,data,out,prior,sha,bundle,workers,count=self.fixture()
        if mutate:mutate(data,out)
        with ExitStack() as stack:
            stack.enter_context(patch.object(v,'OUT',out));stack.enter_context(patch.object(v,'ORIGINAL',prior))
            for name in ('source_authentication','authenticate_all'):stack.enter_context(patch.object(v,name))
            stack.enter_context(patch.object(v,'verify_bridge_bundle',return_value=bundle))
            stack.enter_context(patch.object(v.old,'qualification_review',return_value=dict(scientific_decisions=203)))
            stack.enter_context(patch.object(v,'review_workers',return_value=workers))
            timing=stack.enter_context(patch.object(v,'review_timing',side_effect=lambda reader,acc,admin:dict(count=acc.count)))
            stats=stack.enter_context(patch.object(v,'check_statistics',side_effect=lambda games,*args:dict(games=len(games))))
            stack.enter_context(patch.object(v,'owner_checks',return_value=dict(actual_exits=3)))
            stack.enter_context(patch.object(v,'evidence_size',return_value=100))
            result=v.review_all(r,sha)
            self.assertEqual(timing.call_count,1);self.assertEqual(stats.call_count,1)
            return result,count

    def test_full406_schema_flow_keeps_one_bridge_and_exact_reproductions(self):
        result,count=self.run_fixture()
        self.assertEqual(result['counts']['games'],406);self.assertEqual(result['counts']['main_games'],400)
        self.assertEqual(result['counts']['ordinary_timed_live_decisions'],count)
        self.assertEqual(result['counts']['live_game_events'],count+1)
        self.assertEqual(result['counts']['recovered_events'],1);self.assertEqual(result['statistics']['games'],400)
        self.assertEqual(result['final_contract']['combined_scientific_decisions'],322)

    def test_original_failed_manifest_must_point_to_full_row_not_clock(self):
        def corrupt(data,out):
            item=data[out/'segments_manifest.json']['original_failed_request'];path=v.ROOT/item['path']
            data[path]=data[path]['clock'];item.update(sha256=v.digest(data[path]))
        with self.assertRaisesRegex(ValueError,'Original rejected row/clock'):self.run_fixture(corrupt)

    def test_replay_manifest_requires_artifact_descriptor_not_inline_proof(self):
        def corrupt(data,out):data[out/'all_game_replay.json']['receipts'][0]=dict(status='passed')
        with self.assertRaisesRegex(ValueError,'full-game independent ledger'):self.run_fixture(corrupt)

    def test_ordinary_coverage_cannot_include_adopted_bridge(self):
        def corrupt(data,out):
            for name in ('coverage.json','completion.json','verification_summary.json'):
                value=data[out/name] if name=='coverage.json' else data[out/name]['coverage'];value['live_decisions']+=1
        with self.assertRaisesRegex(ValueError,'One adoption'):self.run_fixture(corrupt)

    def test_failed_reproduction_reservation_cannot_be_accepted(self):
        def corrupt(data,out):data[out/'reproduction_resource_gate.json']['passed']=False
        with self.assertRaisesRegex(ValueError,'resource reservation'):self.run_fixture(corrupt)

    def test_unbound_resumed_entrypoint_counterproof_fails(self):
        def corrupt(data,out):data[out/'recovery_qualification_summary.json']['receipts'][0]['verification']['direct_same_quota']=0
        with self.assertRaisesRegex(ValueError,'genuine quota'):self.run_fixture(corrupt)


if __name__=='__main__':unittest.main()
