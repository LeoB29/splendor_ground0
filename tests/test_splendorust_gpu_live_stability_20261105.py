"""Packing itself is model-free and must preserve every client row exactly."""
import pytest
from tools.splendorust_gpu_live_20261105 import stability

@pytest.mark.parametrize('slot',range(8))
@pytest.mark.parametrize('variant',['zeros','forward','reverse'])
def test_fixed_anchor_is_independent_of_slot_and_companions(slot,variant):
    roots=[dict(tokens=[float(i)]*stability.WIDTH) for i in range(35)]
    values=stability.packing(roots,17,slot,variant)
    assert len(values)==8*stability.WIDTH
    assert values[slot*stability.WIDTH:(slot+1)*stability.WIDTH]==roots[17]['tokens']
    if variant=='zeros':
        assert all(values[i*stability.WIDTH:(i+1)*stability.WIDTH]==[0.]*stability.WIDTH for i in range(8) if i!=slot)
