"""Full saved-science qualification and incomplete-suffix negative tests; no models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import boundary as saved, partial


class _Callbacks:
    def __init__(self, rows):self.rows=rows
    def decision(self,client,started,*,owner,slot):return self.rows


class _Adapter:
    def __init__(self,client,slot):self.client_id,self.slot=client,slot
    def close(self):pass


class _Runtime:
    def __init__(self):self.adapters=[]
    def make_adapter(self,client,slot,quota):
        if quota != 512:raise AssertionError('Changed search quota')
        self.adapters.append((client,slot));return _Adapter(client,slot)


class PartialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding=saved.inspect_boundary()
        cls.item=cls.binding['selected_replays'][32]
        cls.operation=cls.binding['active_operation']
        cls.execution=partial.c.read(saved.CONTAINER/'execution.json')
        original=partial.c.read(saved.CELL/'games/g0666/service_calls/000040.json')
        # A deliberately short but exact retained prefix, used only as a negative-test fixture.
        cls.rows=[dict(x,complete_readback_perf=cls.operation['started_perf']+.1) for x in original[:2]]

    def test_saved_interrupted_prefix_exact(self):
        result=partial.attribute_interrupted(_Callbacks(self.rows),self.item,self.operation,self.execution)
        self.assertEqual(result['partial_callbacks'],2)
        self.assertFalse(result['completed_decision'])
        self.assertFalse(result['original_clock_reset'])

    def test_wrong_callback_output_is_rejected(self):
        rows=deepcopy(self.rows);rows[0]['output_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'exact prefix'):
            partial.attribute_interrupted(_Callbacks(rows),self.item,self.operation,self.execution)

    def test_callback_after_recorded_exit_is_rejected(self):
        rows=deepcopy(self.rows);rows[0]['complete_readback_perf']=self.execution['monotonic_end']+1.
        with self.assertRaisesRegex(ValueError,'exact prefix'):
            partial.attribute_interrupted(_Callbacks(rows),self.item,self.operation,self.execution)

    def test_changed_interrupted_clock_is_rejected(self):
        operation=deepcopy(self.operation);operation['started_perf']+=1.
        with self.assertRaisesRegex(ValueError,'interrupted verification request'):
            partial.attribute_interrupted(_Callbacks(self.rows),self.item,operation,self.execution)

    def test_supplement_routes_exactly_16_original_selections(self):
        destination=partial.root.ROOT/'outputs/splendorust_collection4096_resume_20261118_01/prelaunch/test_artifacts'
        destination.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='supplement_routing_',dir=destination))
        receipt=folder/'boundary.json';receipt.write_text(json.dumps(self.binding),encoding='utf-8')
        freeze=folder/'freeze.json';freeze.write_text(json.dumps({'saved_boundary':partial.root.artifact(receipt)}),encoding='utf-8')
        calls=[];runtime=_Runtime()
        def save(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as stream:json.dump(value,stream)
            return partial.c.artifact(path)
        def decide(runtime,adapter,before,game,ply,destination,*,kind):
            original=partial.c.read(saved.CELL/'games'/game['game_id']/'moves'/f'{ply:06d}.json')
            self.assertEqual(before,original['before']);self.assertEqual(kind,'selected_model_replay')
            calls.append((game['game_id'],ply,adapter.client_id,adapter.slot))
            save(Path(destination)/'moves'/f'{ply:06d}.json',original)
            return original
        with patch.object(partial.root,'PLAN',freeze), patch.object(partial.collector._legacy.q,'validate_runtime',lambda r:None), \
             patch.object(partial.collector,'warmup_runtime',lambda *args:None), \
             patch.object(partial.collector,'decide_snapshot',decide), patch.object(partial.c,'save',save):
            result=partial.run_supplement(partial.c.read(saved.ORIGINAL_PLAN),runtime,folder/'cell',lambda:None)
        expected=[(x['game_id'],x['ply'],'replay_'+x['game_id'],x['replay_slot']) for x in self.binding['selected_replays'][32:]]
        self.assertEqual(calls,expected);self.assertEqual(len(calls),16)
        self.assertEqual(result['repeated_interrupted_verification'],1)
        self.assertEqual(result['scientific_games_retried'],0)
        self.assertEqual(result['unstarted_verification_decisions'],15)

    def test_full_saved_878_move_audit(self):
        destination=partial.root.ROOT/'outputs/splendorust_collection4096_resume_20261118_01/prelaunch/test_artifacts'
        destination.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix='saved_partial_',dir=destination))
        receipt=folder/'boundary.json';receipt.write_text(json.dumps(self.binding),encoding='utf-8')
        freeze=folder/'freeze.json';freeze.write_text(json.dumps({'saved_boundary':partial.root.artifact(receipt)}),encoding='utf-8')
        with patch.object(partial.root,'PLAN',freeze):
            proof=partial.audit_partial(partial.c.read(saved.ORIGINAL_PLAN),lambda:None)
        self.assertEqual(proof['games'],16);self.assertEqual(proof['plies'],878)
        self.assertEqual(proof['eligible_roots'],878);self.assertEqual(len(proof['replay_prefix']),32)
        self.assertTrue(proof['journal']['all_callbacks_attributed_once'])
        self.assertFalse(proof['runtime_closure']['final_weights_verified'])
        self.assertEqual(proof['model_calls'],0)
        # Retain fixtures and outcomes even if later checks fail.
        target=folder/'saved_partial_qualification.json'
        target.write_text(json.dumps(proof,sort_keys=True,indent=2),encoding='utf-8')


if __name__=='__main__':unittest.main()
