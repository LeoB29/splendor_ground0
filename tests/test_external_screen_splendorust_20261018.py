"""Model-host boundary checks; no game outcomes or match search."""
import math
from pathlib import Path

import pytest
import torch

from tools.external_strength_screen_20261018_02 import splendorust_adapter as adapter


@pytest.fixture(scope='module')
def model_host():
    torch.set_num_threads(1)
    base = adapter.ROOT / 'outputs/external_opponent_search_20261018_02'
    host = object.__new__(adapter.Opponent)
    host.model, host.reference = adapter.load_model(
        base / 'splendorust/research/architecture_pivots/models.py', base / 'splendorust_model.pt')
    return host


def test_host_inference_matches_authenticated_upstream_forward(model_host):
    # Include nonzero public context, rules identity and signed legacy bit fields.
    x = torch.tensor([[math.sin(i * .23 + row) * 7 for i in range(525)]
                      for row in range(2)], dtype=torch.float32)
    with torch.inference_mode():
        tokens = model_host.reference['entities'](x)
        policy, value = model_host.model(x)
        actual = torch.tensor(model_host._infer(tokens.flatten().tolist(), 2))
    assert torch.equal(actual, torch.cat((policy, value), dim=1))


@pytest.mark.parametrize('batch,length', [(0, 0), (3, 4464), (1, 1487), (2, 1488)])
def test_host_rejects_malformed_token_contract(model_host, batch, length):
    with pytest.raises(ValueError, match='Malformed'):
        model_host._infer([0.] * length, batch)


def test_host_rejects_nonfinite_features(model_host):
    tokens = [0.] * 1488
    tokens[42] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite model input'):
        model_host._infer(tokens, 1)


def test_published_parameter_count_and_frozen_eval(model_host):
    assert sum(p.numel() for p in model_host.model.parameters()) == 4780883
    assert not model_host.model.training
    assert not model_host.model.has_history
