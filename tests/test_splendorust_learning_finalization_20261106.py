import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from tools.splendorust_learning_finalization_20261106 import recovery as r


class RecoveryBoundaryTests(unittest.TestCase):
    def test_either_original_deadline_is_not_reset(self):
        plan=dict(original_deadline_monotonic=5000.,original_deadline_unix=15000.,
            original_audit_deadline_monotonic=1100.,original_audit_deadline_unix=11100.)
        claim=dict(monotonic_origin=1000.,unix_origin=11000.)
        r.check_deadlines(plan,claim,1099.,11099.)
        for mono,wall in [(1100.,11099.),(1099.,11100.),(999.,11001.)]:
            with self.subTest(mono=mono,wall=wall),self.assertRaises(ValueError):r.check_deadlines(plan,claim,mono,wall)

    def test_new_one_hour_cap_also_applies(self):
        plan=dict(original_deadline_monotonic=10000.,original_deadline_unix=20000.,
            original_audit_deadline_monotonic=10000.,original_audit_deadline_unix=20000.)
        claim=dict(monotonic_origin=1000.,unix_origin=11000.)
        with self.assertRaises(ValueError):r.check_deadlines(plan,claim,4600.,12000.)

    def test_original_outputs_cannot_be_written(self):
        with self.assertRaises(ValueError):r.save(r.SOURCE/'do_not_create.json',{})

    def test_exclusive_output_and_digest_check(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(r,'OUT',Path(folder)):
            item=r.save(Path(folder)/'new.json',{'first':True});r.verify([item])
            with self.assertRaises(FileExistsError):r.save(Path(folder)/'new.json',{'changed':True})
            Path(folder,'new.json').write_text('changed')
            with self.assertRaises(ValueError):r.verify([item])

    def test_openmp_bypass_rejected(self):
        with patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'TRUE'}):
            with self.assertRaises(ValueError):r.no_torch()

    def test_torch_import_rejected(self):
        with patch.dict('sys.modules',{'torch':object()}):
            with self.assertRaises(ValueError):r.no_torch()

    def test_private_facade_leaves_original_save_and_out_unchanged(self):
        from tools.splendorust_learning_diagnostic_20261106 import common as original
        source_out,source_save=original.OUT,original.save
        with tempfile.TemporaryDirectory() as folder,patch.object(r,'OUT',Path(folder)),patch.object(r,'read',return_value={'source':r.artifact(r.ANALYSIS_SOURCE)}):
            expected={'closed_saved_analysis':True}
            module=r.private_analysis(expected)
            self.assertIs(original.save,source_save);self.assertEqual(original.OUT,source_out)
            with self.assertRaises(ValueError):module.c.save(Path(folder)/'report/analysis.json',{'changed':True})
            module.c.save(Path(folder)/'report/analysis.json',expected)
            self.assertTrue(Path(folder,'report/analysis.json').exists())
            self.assertEqual(original.OUT,r.SOURCE)


if __name__=='__main__':unittest.main()
