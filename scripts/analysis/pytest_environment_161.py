"""Test-only output isolation for a frozen160 guard test after160 completed.

The original test assumes its designated production directory is empty. Saved
game0000 now triggers the earlier FileExistsError guard instead. Run all original
assertions unchanged against an empty temporary directory designated as production
for this one test. Do not mask Path.exists, skip assertions or touch saved data.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated160_guard_output(request):
    if request.node.nodeid != "tests/test_world_budget_160.py::test_test_snapshot_and_test_output_cannot_enter_production":
        return
    target = request.getfixturevalue("tmp_path") / "designated_production160"
    patch = request.getfixturevalue("monkeypatch")
    module = request.node.module
    patch.setattr(module.h, "OUTPUT", target)
    patch.setattr(module.h, "_PRODUCTION_OUTPUT", target.resolve())
    patch.setattr(module.a, "PRODUCTION_OUTPUT", target.resolve())
