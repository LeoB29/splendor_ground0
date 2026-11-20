"""Saved c135 failure regression and genuine no-model continuation counterexamples."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
import uuid
from unittest.mock import patch

from tools.splendorust_collection4096_resume_20261118_v2 import partial135 as p


class SavedFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=p.c.read(p.root.ORIGINAL_OUT/'collection/plan.json')
        cls.binding=p.build_boundary(cls.plan)

    def test_saved_frontier_is_exact(self):
        self.assertEqual([x['game_id'] for x in self.binding['games']],list(p.COMPLETED+p.UNFINISHED))
        self.assertEqual([x['prefix_count'] for x in self.binding['games'][8:]],[27,23,22,22,22,21,20,17])
        self.assertEqual(self.binding['completed_games'],8)
        self.assertEqual(self.binding['unfinished_games'],8)
        for x in self.binding['games'][8:]:
            clock=p.c.read(p.c.resolve(x['failed_clock']))
            self.assertEqual((clock['internal_seconds'],clock['outer_seconds']),(60.,75.))
            self.assertFalse(clock['completed'])
        timeout=p.c.read(p.c.resolve(self.binding['games'][14]['failed_clock']))
        self.assertEqual(timeout['status'],'timeout')
        self.assertAlmostEqual(timeout['elapsed_seconds'],75.00580079999054)

    def test_unchanged_saved_boundary_passes(self):
        p.validate_boundary(self.binding,self.plan)

    def check_bad(self,edit):
        candidate=deepcopy(self.binding);edit(candidate)
        with self.assertRaises(ValueError):p.validate_boundary(candidate,self.plan)

    def test_cannot_drop_completed_game(self):
        self.check_bad(lambda b:b['games'].pop(0))

    def test_cannot_claim_an_extra_retry(self):
        self.check_bad(lambda b:b.update(disclosed_decision_retries=9))

    def test_cannot_change_deadline(self):
        self.check_bad(lambda b:b.update(decision_outer_seconds=90.))

    def test_cannot_relabel_prior_attempt(self):
        self.check_bad(lambda b:b.update(original_attempt_incomplete=False))

    def test_cannot_restart_completed_prefix(self):
        self.check_bad(lambda b:b['games'][8].update(prefix_count=0,prefix_rows=[]))

    def test_cannot_change_resume_snapshot_even_with_fresh_hash(self):
        def edit(b):
            x=b['games'][8];x['next_snapshot']['unauthorized_change']=True;x['next_snapshot_sha256']=p.c.digest(x['next_snapshot'])
        self.check_bad(edit)

    def test_cannot_change_original_slot(self):
        self.check_bad(lambda b:b['games'][8]['assignment'].update(slot=7))

    def test_completed_target_document_cannot_drop_a_preserved_row(self):
        item=self.binding['games'][0];rows=[p.data.read_artifact(x) for x in item['prefix_rows']]
        source=p.data.read_artifact;bad=deepcopy(source(item['document']));bad['rows'].pop(0)
        with patch.object(p.data,'read_artifact',side_effect=lambda ref:bad if ref==item['document'] else source(ref)):
            with self.assertRaises(ValueError):p._verify_document(item['spec'],rows,item['document'])

    def test_failed_operation_cannot_omit_successful_callbacks(self):
        item=self.binding['games'][14]
        receipts=p.c.read(p.CELL/'games'/item['game_id']/'service_calls'/f"{item['prefix_count']:06d}.json")
        self.assertTrue(receipts)
        raw=[{k:v for k,v in x.items() if k!='client_received_perf'} for x in receipts]
        journal=SimpleNamespace(decision=lambda *args,**kwargs:raw[:-1])
        execution=p.c.read(p.CONTAINER/'execution.json')
        with self.assertRaises(ValueError):p._attribute_failed(item,journal,execution)


class ContinuationEntry(unittest.TestCase):
    def fixture(self):
        game=dict(game_id='g2168',game_index=2168)
        before=dict(terminal=False,turn_index=27)
        prefix=[dict(action={'action_type':'BUY'},after=before) for _ in range(27)]
        refs=[dict(path=f'old/{i}',bytes=1,sha256=str(i)) for i in range(27)]
        item=dict(spec=game,game_id='g2168',prefix_rows=refs,next_snapshot=before,prefix_count=27,
            assignment=dict(client_id='worker_3',slot=3),failed_attempt={'path':'old/failure'})
        return game,before,prefix,refs,item

    def test_first_model_call_is_only_unfinished_move_and_uses_absolute_ply(self):
        game,before,prefix,refs,item=self.fixture();calls=[];writes=[]
        row=dict(action={'action_type':'BUY'},after=dict(terminal=True,turn_index=28))
        def decide(runtime,adapter,actual_before,actual_game,ply,folder):
            calls.append((actual_before,actual_game,ply));return row
        def finish(actual_game,rows,actual_refs,folder,assignment,**kw):
            self.assertEqual(rows[:27],prefix);self.assertEqual(rows[27:], [row])
            self.assertEqual(actual_refs[:27],refs);self.assertEqual(len(actual_refs),28)
            return {'status':'completed'}
        with patch.object(p.data,'read_artifact',side_effect=lambda x:prefix[int(x['path'].split('/')[-1])]), \
             patch.object(p.guard,'decode_training_snapshot',side_effect=lambda x:SimpleNamespace(terminal=x['terminal'])), \
             patch.object(p.root,'read',return_value={'saved_partial135':{'path':'frozen-boundary'}}), \
             patch.object(p.c,'save',side_effect=lambda path,value:writes.append((str(path),deepcopy(value)))), \
             patch.object(p.c,'artifact',side_effect=lambda x:{'path':str(x)}), \
             patch.object(p.collector,'decide_snapshot',side_effect=decide), \
             patch.object(p,'_finish_game',side_effect=finish):
            result=p._continue_game({},object(),object(),item,Path('new/g2168'),threading.Event(),lambda:None)
        self.assertEqual(result,{'status':'completed'})
        self.assertEqual(calls,[(before,game,27)])
        self.assertEqual(writes[0][1]['prefix_rows'],refs)
        self.assertEqual(writes[0][1]['disclosed_boundary_retry'],1)

    def test_a_failed_new_attempt_never_retries_itself(self):
        game,before,prefix,refs,item=self.fixture();writes=[];stopped=threading.Event()
        with patch.object(p.data,'read_artifact',side_effect=lambda x:prefix[int(x['path'].split('/')[-1])]), \
             patch.object(p.guard,'decode_training_snapshot',return_value=SimpleNamespace(terminal=False)), \
             patch.object(p.root,'read',return_value={'saved_partial135':{'path':'frozen-boundary'}}), \
             patch.object(p.c,'save',side_effect=lambda path,value:writes.append(value)), \
             patch.object(p.collector,'decide_snapshot',side_effect=TimeoutError('saved deadline failure')) as call:
            with self.assertRaises(TimeoutError):
                p._continue_game({},object(),object(),item,Path('new/g2168'),stopped,lambda:None)
        self.assertEqual(call.call_count,1);self.assertTrue(stopped.is_set())
        self.assertTrue(writes[-1]['no_further_retry']);self.assertEqual(writes[-1]['completed_moves'],27)


class LifecycleCounterexamples(unittest.TestCase):
    def fixture(self):
        cohort=dict(started_perf_counter=10.,started_unix=110.,gameplay_end_perf_counter=20.,gameplay_end_unix=120.,
            ended_perf_counter=25.,ended_unix=125.,wall_seconds=15.,gameplay_seconds=10.,target_loader_seconds=5.)
        started=dict(cell_id='c135',started_perf_counter=10.,started_unix=110.,amended_partial_cell=True,original_cell_clock_reset=False)
        execution=dict(monotonic_origin=0.,monotonic_end=30.,unix_origin=100.,unix_end=130.)
        games=[]
        for gid in p.UNFINISHED:
            finished=dict(game_id=gid,ended_perf_counter=19.,ended_unix=119.)
            outcome=dict(game_id=gid,status='completed',completed_moves=50,finished=finished,no_retry=True,
                disclosed_boundary_retry=1,completed_prefix_preserved=True)
            games.append(dict(game_id=gid,finished=finished,outcome=outcome,loaded={'plies':50}))
        outcomes=dict(outcomes={g['game_id']:g['outcome'] for g in games},errors=[],original_failed_games_preserved=True)
        return started,cohort,outcomes,games,execution,5.

    def test_valid_independent_lifecycle(self):
        p.validate_combined_lifecycle(*self.fixture())

    def test_saved_release_cannot_be_rebased(self):
        args=self.fixture();args[0]['started_perf_counter']=11.
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)

    def test_cannot_omit_game_outcome(self):
        args=self.fixture();del args[2]['outcomes']['g2168']
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)

    def test_cannot_hide_worker_error(self):
        args=self.fixture();args[2]['errors']=['failure']
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)

    def test_cannot_overrun_worker_clock(self):
        args=self.fixture();args[1]['ended_perf_counter']=35.;args[1]['wall_seconds']=25.;args[1]['target_loader_seconds']=15.
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)

    def test_cannot_rewrite_elapsed_time(self):
        args=self.fixture();args[1]['wall_seconds']=1.
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)

    def test_cannot_finish_game_after_target_loader(self):
        args=self.fixture();args[3][0]['finished']['ended_perf_counter']=23.
        with self.assertRaises(ValueError):p.validate_combined_lifecycle(*args)


class SavedTrajectoryIntegration(unittest.TestCase):
    def test_real_record_prefix_suffix_roundtrip_and_independent_targets(self):
        """Only the model call is replaced by exact immutable existing rows."""
        source=p.CELL/'games/g2160';game=p.c.read(source/'spec.json')
        original=p.c.read(source/'target_game.json');rows=[p.data.read_artifact(ref) for ref in original['rows']]
        split=20;assignment=p.c.read(p.CELL/'assignments/00.json')
        item=dict(game_id=game['game_id'],spec=game,assignment=assignment,completed=False,prefix_count=split,
            prefix_rows=original['rows'][:split],next_snapshot=rows[split]['before'],failed_attempt={'path':'excluded_saved_record_fixture'})
        output=p.root.OUT/'prelaunch/test_artifacts'/('partial135_saved_concat_'+uuid.uuid4().hex)
        cell=output/'cell';destination=cell/'games'/game['game_id'];calls=[]
        def saved_decision(runtime,adapter,before,spec,ply,folder):
            self.assertEqual(spec,game);self.assertEqual(before,rows[ply]['before']);self.assertEqual(folder,destination)
            calls.append(ply);p.c.save(folder/'moves'/f'{ply:06d}.json',rows[ply]);return rows[ply]
        read=p.root.read
        with patch.object(p.c,'OUT',output), \
             patch.object(p.root,'read',side_effect=lambda path:dict(saved_partial135={'path':'excluded_saved_record_fixture'}) if path==p.root.PLAN else read(path)), \
             patch.object(p.collector,'decide_snapshot',side_effect=saved_decision):
            ending=p._continue_game({},object(),object(),item,destination,threading.Event(),lambda:None)
            self.assertEqual(ending['status'],'completed');self.assertEqual(calls,list(range(split,len(rows))))
            ref=p.c.artifact(destination/'target_game.json');merged=p.c.read(destination/'target_game.json')
            self.assertEqual(merged['rows'][:split],original['rows'][:split])
            self.assertEqual([p.data.read_artifact(x) for x in merged['rows']],rows)
            self.assertEqual(merged['final'],original['final']);self.assertEqual(merged['winner'],original['winner'])
            manifest=p.data.build_manifest([ref],[game],route='cpu',cell_id='excluded_saved_record_fixture')
            def sink(spec,examples):
                p.c.save(cell/'targets'/(spec['game_id']+'.json'),dict(schema='gpu_collector_examples_v1',**p.data.IDENTITY,
                    spec=spec,examples=examples,official_terminal_only=True,teacher_simulations=512,training_updates=0))
            loaded=p.data.load_corpus(manifest,expected_specs=[game],sink=sink)
            proof=p._game_proof(item,ref,rows,cell,0)
            self.assertEqual(loaded['game_count'],1);self.assertEqual(loaded['games'],[proof['loaded']])
            self.assertEqual(proof['loaded']['plies'],56);self.assertEqual(proof['document'],ref)
            p.c.save(output/'fixture_receipt.json',dict(status='passed',source=p.c.artifact(source/'target_game.json'),
                prefix_preserved=split,suffix_supplied_from_existing_records=len(calls),model_calls=0,new_games=0,
                training_updates=0,excluded_from_production=True,complete_loader_and_independent_targets_equal=True))


if __name__=='__main__':unittest.main()
