"""Model-free tests against the immutable saved infrastructure failure."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.splendorust_search128_recovery_20261028 import recovery as r


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = r.ORIGINAL/'workers'/r.WORKER
        cls.row = load(r.ORIGINAL/r.GAME/'moves/0039.json')
        cls.receipt = load(cls.folder/(r.REQUEST_ID+'.json'))
        cls.parent = load(cls.folder/'parent_profiles'/(r.REQUEST_ID+'.json'))
        cls.child = load(cls.folder/(r.REQUEST_ID+'_child_profile.json'))
        cls.payload = load(cls.folder/'inputs'/(r.REQUEST_ID+'.json'))
        cls.stdout = gzip.decompress((cls.folder/'stdout.log.gz').read_bytes())
        cls.events = gzip.decompress((cls.folder/'events.jsonl.gz').read_bytes())
        cls.line = cls.stdout.splitlines(keepends=True)[-1]
        cls.initial = load(cls.folder/'initialization.json')
        cls.freeze = load(r.ORIGINAL/'freeze.json')

    def args(self):
        return [deepcopy(x) for x in (self.row,self.receipt,self.parent,self.child,self.payload)] + [self.line]

    def validate(self, args=None):
        return r.validate_emitted_response(*(self.args() if args is None else args),worker_pid=51608)

    def test_actual_packet_and_arrival_bound_without_clock_rewrite(self):
        args = self.args();before = deepcopy(args)
        proof = self.validate(args)
        self.assertEqual(args,before)
        self.assertEqual(proof['observed_complete_response_seconds'],1.0749384)
        self.assertEqual(proof['normalized_packet_sha256'],self.child['serialized_packet_utf8_sha256'])
        self.assertFalse(args[1]['accepted']);self.assertIsNone(args[1]['receipt_elapsed_seconds'])

    def test_changed_raw_packet_rejected(self):
        args=self.args();args[-1]=args[-1].replace(b'"result"',b'"resulx"',1)
        with self.assertRaisesRegex(ValueError,'serialized packet'):self.validate(args)

    def test_missing_newline_rejected(self):
        args=self.args();args[-1]=args[-1].rstrip(b'\r\n')
        with self.assertRaisesRegex(ValueError,'complete raw response'):self.validate(args)

    def test_multiple_raw_lines_rejected(self):
        args=self.args();args[-1]+=args[-1]
        with self.assertRaisesRegex(ValueError,'complete raw response'):self.validate(args)

    def test_duplicate_arrivals_rejected(self):
        args=self.args();args[2]['raw_arrivals_ns']*=2;args[0]['parent_profile']=args[2]
        with self.assertRaisesRegex(ValueError,'unambiguous'):self.validate(args)

    def test_dropped_event_rejected(self):
        args=self.args();args[2]['dropped_arrival_events']=1;args[0]['parent_profile']=args[2]
        with self.assertRaisesRegex(ValueError,'unambiguous'):self.validate(args)

    def test_expired_original_arrival_rejected(self):
        args=self.args();args[2]['raw_arrivals_ns']=[args[2]['request_start_ns']+10_000_000_001];args[0]['parent_profile']=args[2]
        with self.assertRaisesRegex(ValueError,'original deadline'):self.validate(args)

    def test_changed_original_origin_rejected(self):
        args=self.args();args[2]['request_start_ns']=args[2]['raw_arrivals_ns'][0]+1;args[0]['parent_profile']=args[2]
        with self.assertRaisesRegex(ValueError,'original deadline'):self.validate(args)

    def test_invented_accepted_receipt_rejected(self):
        args=self.args();args[1]['accepted']=True;args[0]['clock']=args[1]
        with self.assertRaisesRegex(ValueError,'infrastructure receipt'):self.validate(args)

    def test_other_resource_error_rejected(self):
        args=self.args();args[1]['detail']='RSS bound';args[0]['clock']=args[1]
        with self.assertRaisesRegex(ValueError,'pending-rename'):self.validate(args)

    def test_invented_consumed_arrival_rejected(self):
        args=self.args();args[2]['receipt_arrival_ns']=args[2]['raw_arrivals_ns'][0];args[0]['parent_profile']=args[2]
        with self.assertRaisesRegex(ValueError,'do not invent'):self.validate(args)

    def test_missing_child_flush_rejected(self):
        args=self.args();args[3]['outcome']='failed_before_successful_emission';args[0]['child_profile']=args[3]
        with self.assertRaisesRegex(ValueError,'successful original'):self.validate(args)

    def test_payload_mutation_rejected(self):
        args=self.args();args[4]['player']='reference'
        with self.assertRaisesRegex(ValueError,'lawful original'):self.validate(args)

    def test_result_hash_mutation_rejected(self):
        args=self.args();args[3]['result_sha256']='0'*64;args[0]['child_profile']=args[3]
        with self.assertRaisesRegex(ValueError,'scientific result'):self.validate(args)

    def test_original_inner_guard_unchanged(self):
        args=self.args();args[3]['request_started_seconds']-=9.;args[0]['child_profile']=args[3]
        with self.assertRaisesRegex(ValueError,'nine-second'):self.validate(args)

    def inventory(self):
        ids=[f'request-{i:08d}' for i in range(1,22)]
        return ({i:load(self.folder/(i+'.json')) for i in ids},
                {i:load(self.folder/'parent_profiles'/(i+'.json')) for i in ids})

    def test_complete_serial_inventory(self):
        receipts,parents=self.inventory()
        self.assertEqual(r.validate_wire_inventory(self.stdout,self.events,self.initial,receipts,parents),self.line)

    def test_duplicate_packet_inventory_rejected(self):
        receipts,parents=self.inventory()
        with self.assertRaisesRegex(ValueError,'exactly21'):
            r.validate_wire_inventory(self.stdout+self.line,self.events,self.initial,receipts,parents)

    def test_preceding_packet_unbound_rejected(self):
        receipts,parents=self.inventory();parents['request-00000020']['response_raw_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'preceding line'):
            r.validate_wire_inventory(self.stdout,self.events,self.initial,receipts,parents)

    def test_full_bridge_corrected_after_state_and_original_preserved(self):
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            r.ORIGINAL/r.GAME/'moves/0039.json',self.folder/(r.REQUEST_ID+'.json'),r.ORIGINAL/'completion.json')}
        value=r.build_bridge(self.freeze)
        self.assertEqual(value['bridge']['resume_ply'],40)
        self.assertEqual(value['before'],self.row['before'])
        self.assertNotEqual(value['before'],value['after'])
        self.assertEqual(value['source_row'],self.row)
        self.assertFalse(value['bridge']['original_clock_accepted'])
        self.assertFalse(value['bridge']['original_attempt_accepted'])
        self.assertEqual(value['bridge']['new_model_calls'],0)
        for path,pinned in before.items():self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pinned)
        reconstructed=r.verify_bridge(value['bridge'],self.freeze)
        self.assertEqual(value,reconstructed)

    def test_full_bridge_science_checker_is_mandatory(self):
        from tools.splendorust_search128_strength_20261028 import adapters
        with patch.object(adapters,'validate_result',side_effect=ValueError('science sentinel')):
            with self.assertRaisesRegex(ValueError,'science sentinel'):r.build_bridge(self.freeze)

    def test_full_bridge_guard_failure_does_not_publish(self):
        def expired():raise TimeoutError('shared deadline')
        with self.assertRaisesRegex(TimeoutError,'shared deadline'):r.build_bridge(self.freeze,expired)

    def test_changed_freeze_rejected(self):
        frozen=deepcopy(self.freeze);frozen['inputs']=[]
        with self.assertRaisesRegex(ValueError,'exact original freeze'):r.build_bridge(frozen)


if __name__=='__main__':unittest.main()
