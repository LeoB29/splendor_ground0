"""Preservation boundaries and nonresettable watchdog cleanup."""
from pathlib import Path
import pytest
from tools.splendorust_inference64_20261025 import common as c, launch


def test_new_evidence_cannot_overwrite_previous_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'new')
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    for forbidden in (tmp_path/'old/result.json', c.OUT/'../old/result.json'):
        with pytest.raises(ValueError, match='outside'):
            c.save(forbidden, {'replacement': True})
        assert not forbidden.exists()
    path = c.OUT/'result.json'
    receipt = c.save(path, {'original': True})
    with pytest.raises(FileExistsError):
        c.save(path, {'replacement': True})
    c.verify([receipt])


@pytest.mark.parametrize('clock', ['monotonic', 'time'])
def test_cleanup_is_clamped_by_either_original_clock(monkeypatch, clock):
    for name, origin in [('monotonic', 100.), ('time', 1000.)]:
        now = origin+c.JOB_SECONDS-(.5 if name==clock else 100.)
        monkeypatch.setattr(launch.time, name, lambda value=now: value)
    receipt = launch.cleanup_owned(None, None, (), monotonic_origin=100., unix_origin=1000.)
    assert receipt['effective_allowance_seconds']==.5 and not receipt['errors']


def test_watchdog_dispatches_only_this_bounded_diagnostic():
    source = Path(launch.__file__).read_text(encoding='utf-8')
    assert "'tools.splendorust_inference64_20261025.runner'" in source
    assert "'tools.splendorust_buffered_reader_20261025.runner'" not in source
    assert 'CREATE_NO_WINDOW' in source
