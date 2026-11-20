"""Independent actual-entrypoint source-guard regressions; no models or games."""
import os
import sys
from unittest.mock import patch

import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
from tools import future_training_rules_20261021 as rules_guard
from tools.splendorust_independent_data_20261107.strength import segment, report
from tools.splendorust_independent_data_20261107.strength import control


@pytest.mark.parametrize('mode', ['qualification', 'operational', 'main_block',
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


@pytest.mark.parametrize('mode', ['aggregate', 'reader'])
def test_actual_report_source_guard_rejects_before_saved_acceptance(mode):
    with patch.object(sys, 'argv', ['report', '--mode', mode, '--segment', 'unused', '--freeze-sha256', 'x']), \
         patch.object(rules_guard, 'verify_runtime_sources', side_effect=ValueError('SAVED_SOURCE_MISMATCH')) as guard:
        with pytest.raises(ValueError, match='SAVED_SOURCE_MISMATCH'): report.main()
        guard.assert_called_once_with()
