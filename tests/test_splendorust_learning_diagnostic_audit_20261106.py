from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from tools.splendorust_learning_diagnostic_20261106 import common as c
from tools.splendorust_learning_diagnostic_20261106.audit import bind_prediction_rows, verify_stage, verify_qualification_sources


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.example = dict(game_index=192, ply=17, split='validation', root_input_sha256='a'*64,
            row_sha256='b'*64, mask=[True]+[False]*80, policy=[1.]+[0.]*80, value=[1.,-1.])
        self.row = dict(game_index=192, ply=17, split='validation', root_input_sha256='a'*64,
            row_sha256='b'*64, legal_mask81=[True]+[False]*80, policy_target81=[1.]+[0.]*80,
            value_target2=[1.,-1.], output83=[0.]*83)

    def test_correct_decoder_field_names_bind(self):
        bind_prediction_rows([self.row], [self.example])

    def test_changed_targets_masks_or_root_rejected(self):
        for field, value in [('game_index', 191), ('ply', 18), ('row_sha256', 'c'*64),
                             ('value_target2', [-1.,1.]), ('legal_mask81', [True]*81),
                             ('policy_target81', [0.,1.]+[0.]*79)]:
            changed = deepcopy(self.row)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_prediction_rows([changed], [self.example])

    def test_omitted_or_duplicated_rows_rejected(self):
        for rows in [[], [self.row,self.row]]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                bind_prediction_rows(rows, [self.example])

    def test_engineering_update_cannot_relabel_its_source(self):
        rechecked = {'qualification_examples': [self.example]*32, 'proof': {'actual': True}}
        saved = {'root_identities': [[192, 17, 'b'*64]]*32, 'data_qualification': {'proof': {'actual': True}}, 'scopes': []}
        verify_qualification_sources(saved, rechecked)
        for field, value in [('root_identities', [[191,17,'b'*64]]*32), ('data_qualification', {'proof': {'actual': False}})]:
            changed = deepcopy(saved); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): verify_qualification_sources(changed, rechecked)

    def test_saved_stage_cannot_expand_cap_or_change_original_claim(self):
        for cap, changed_claim in [(3600, False), (7200, False), (3600, True)]:
            with self.subTest(cap=cap, changed_claim=changed_claim), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                with patch.object(c, 'OUT', root), patch('tools.splendorust_learning_diagnostic_20261106.audit.require_absent'):
                    claim = c.save(root/'claim.json', {'monotonic_origin': 100., 'unix_origin': 10000.})
                    owner = dict(mode='fit', fit={'fit_id': 'test'}, identity={'pid': 1, 'created': 2.}, plan_sha256='a'*64,
                        windows_job_assigned=True, global_claim=claim, maximum_seconds=cap,
                        monotonic_origin=110., unix_origin=10010.)
                    if changed_claim: owner['global_claim'] = dict(claim, sha256='b'*64)
                    execution = dict(owner, status='passed', worker_exit_code=0, cleanup_errors=[], error=None, retries=0,
                        monotonic_end=120., unix_end=10020., observed_owned_identities=[])
                    result = {key: owner[key] for key in ('mode','fit','identity','plan_sha256')}
                    result['status'] = 'passed'
                    folder = root/'fit'
                    c.save(folder/'owner.json', owner); c.save(folder/'execution.json', execution); c.save(folder/'worker/result.json', result)
                    if cap == 3600 and not changed_claim:
                        verify_stage(folder, 'fit', {'fit_id':'test'}, 'a'*64, lambda: None)
                    else:
                        with self.assertRaises(ValueError): verify_stage(folder, 'fit', {'fit_id':'test'}, 'a'*64, lambda: None)


if __name__ == '__main__': unittest.main()
