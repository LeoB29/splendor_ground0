"""Resource forecast is based on both arms, full replay, and saved checkpoint size."""
import pytest
from . import resource_probe as p, common as c


def test_fixed_probe_covers_whole_prefix_without_outcomes():
    indices=p.batch_indices(7136)
    assert len(indices)==len(set(indices))==32
    assert indices[0]==0 and indices[-1]==7135 and indices==sorted(indices)


def test_forecast_prices_six_fits_and_replays_and_uses_slower_arm():
    x=p.projection({"raw":.1,"relations":.2},8_000_000,117,100*1024**3)
    expected=.2*28544*6*2*1.5/c.CONFIG["fit_workers"]+7200
    assert x["projected_learning_hours"]==expected/3600
    assert x["reasonable"] and x["includes_independent_full_learning_replay"]
    assert x["projected_total_hours"] == expected/3600+24


def test_storage_and_runtime_misses_do_not_silently_launch():
    assert not p.projection({"raw":2.,"relations":3.},8_000_000,117,100*1024**3)["reasonable"]
    assert not p.projection({"raw":.1,"relations":.2},8_000_000,117,6*1024**3)["reasonable"]
    with pytest.raises(ValueError):p.projection({"raw":float("nan"),"relations":.2},8_000_000,117,100)
