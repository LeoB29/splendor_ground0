"""Independent actual-entrypoint source-guard regressions; no models or games."""
import os
import sys
from unittest.mock import patch

import pytest

os.environ.setdefault('ONEPASS_REPETITION', '0')
from tools import future_training_rules_20261021 as rules_guard
from tools.splendorust_onepass_strength_20261106 import segment
from tools.splendorust_onepass_strength_20261106.assessment import control


@pytest.mark.parametrize('mode', ['prepare', 'qualification', 'operational', 'main_block',
                                 'replay', 'saved', 'independent'])
def test_actual_segment_source_guard_rejects_before_any_work(mode):
    argv = ['segment', '--mode', mode]
    with patch.object(sys, 'argv', argv), \
         patch.object(rules_guard, 'verify_runtime_sources', side_effect=ValueError('SAVED_SOURCE_MISMATCH')) as guard, \
         patch.object(control, 'prepare') as prepare, patch.object(control, 'run_segment') as play, \
         patch.object(control, 'verify_saved') as saved:
        with pytest.raises(ValueError, match='SAVED_SOURCE_MISMATCH'):
            segment.main()
        guard.assert_called_once_with()
        prepare.assert_not_called(); play.assert_not_called(); saved.assert_not_called()


def test_prepare_uses_exact_single_startup_source_receipt(capsys):
    receipt = {'fixed_source': 'hash_fixture_no_model'}
    saved = []
    with patch.object(sys, 'argv', ['segment', '--mode', 'prepare']), \
         patch.object(rules_guard, 'verify_runtime_sources', return_value=receipt) as guard, \
         patch.object(control, 'prepare', return_value={'fixture': 'no_models'}) as prepare, \
         patch.object(segment.c, 'save', side_effect=lambda path, value: saved.append(value)), \
         patch.object(segment.c, 'artifact', return_value={'path': 'fixture', 'bytes': 1, 'sha256': '0'*64}):
        assert segment.main() == 0
        guard.assert_called_once_with(); prepare.assert_called_once_with()
    assert saved[0]['runtime_sources'] is receipt
    assert saved[0]['model_calls'] == saved[0]['games'] == 0
    assert 'prepared' in capsys.readouterr().out
