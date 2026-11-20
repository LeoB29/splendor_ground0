"""Model-free preservation of the qualified transport/placement and hard bounds."""
from pathlib import Path
import ast
from tools.splendorust_isolated_reliability_20261026 import common as c, adapters
from tools.splendorust_supervisor_qualification_20261026 import adapters as qualified


def test_qualified_adapter_and_placement_files_are_byte_identical():
    root=c.ROOT/'tools'
    for name in ('adapters.py','placement.py'):
        assert (root/'splendorust_isolated_reliability_20261026'/name).read_bytes()==(root/'splendorust_supervisor_qualification_20261026'/name).read_bytes()


def test_transport_timing_methods_are_identical_inherited_objects():
    for name in ('request','_receive','_stdout_reader','_stderr_reader'):
        assert getattr(adapters.PlacedProfiledStudyWorker,name) is getattr(qualified.PlacedProfiledStudyWorker,name)
    assert adapters.SIDECAR_WAIT_SECONDS==qualified.SIDECAR_WAIT_SECONDS==1.
    assert adapters.OBSERVER_MODULE==qualified.OBSERVER_MODULE


def test_global_stage_and_evidence_limits():
    assert c.JOB_SECONDS==21600
    assert c.COLLECTION_CUTOFF_SECONDS==19800
    assert c.STAGES==dict(qualification=600,operational=1800,main=16200,final=1800)
    assert c.MAX_EVIDENCE_BYTES==2*1024**3
    assert c.FINAL_RESERVE_BYTES==256*1024**2
    assert c.PREFINAL_EVIDENCE_BYTES+c.FINAL_RESERVE_BYTES==c.MAX_EVIDENCE_BYTES
    assert c.OUT.name=='splendorust_isolated_reliability_20261026_01'


def test_new_source_does_not_globally_assign_imported_module_attributes():
    # Local attributes and instance fields are expected; global rebinding of the
    # frozen package entrypoints would silently alter a historical player.
    for path in (c.ROOT/'tools/splendorust_isolated_reliability_20261026').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for statement in tree.body:
            if isinstance(statement,ast.Assign):
                assert not any(isinstance(target,ast.Attribute) for target in statement.targets),path.name
