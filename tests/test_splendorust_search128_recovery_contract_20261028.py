"""Model-free regression checks for the separately owned recovery lifecycle."""
import ast
from pathlib import Path
import pytest
from tools.splendorust_search128_recovery_20261028 import common as c
from tools.splendorust_search128_recovery_20261028 import launch_owner,guardian


def test_original_origin_and_global_deadline_cannot_restart():
    claim=c.read(c.OLD_OUT/'guardian_claim.json')
    assert (c.ORIGINAL_MONOTONIC,c.ORIGINAL_UNIX)==(claim['monotonic_origin'],claim['unix_origin'])
    assert c.JOB_SECONDS==86400 and c.COLLECTION_CUTOFF_SECONDS==72000
    tree=ast.parse(Path(guardian.__file__).read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='guardian_main')
    assignments=[n for n in ast.walk(main) if isinstance(n,ast.Assign)]
    assert any(ast.unparse(n)=='mono, wall = (c.ORIGINAL_MONOTONIC, c.ORIGINAL_UNIX)' for n in assignments)


def test_acceptance_is_clipped_to_original_global_and_reduced_final_window():
    claim=dict(monotonic_origin=c.ORIGINAL_MONOTONIC,unix_origin=c.ORIGINAL_UNIX,freeze_sha256='f')
    final_mono=c.ORIGINAL_MONOTONIC+70000+c.STAGES['final']-c.PRIOR_FINAL_SECONDS
    final_wall=c.ORIGINAL_UNIX+70000+c.STAGES['final']-c.PRIOR_FINAL_SECONDS
    completion=dict(freeze_sha256='f',independent_review_maximum_seconds=3600,
        independent_acceptance_deadline_monotonic=final_mono,
        independent_acceptance_deadline_unix=final_wall)
    ends=launch_owner.acceptance_deadlines(claim,completion,c.ORIGINAL_MONOTONIC+84000,c.ORIGINAL_UNIX+84000)
    assert ends==(final_mono,final_wall)
    assert ends[0]<=claim['monotonic_origin']+86400
    assert ends[1]<=claim['unix_origin']+86400


def test_recovery_caps_are_explicit_bounded_amendments():
    assert c.MAX_WORKERS==864==c.MAX_GRAPH_BUILDS
    assert c.PRIOR_FINAL_SECONDS==131 and c.PRIOR_RAM_WAIT_SECONDS==130
    assert c.RAM_WAIT_SECONDS==600 and c.STAGES['recovery_qualification']==1800
    assert c.MAX_EVIDENCE_BYTES==16*c.GiB and c.PREFINAL_EVIDENCE_BYTES==12*c.GiB
    assert c.SECONDS_PER_MOVE==10 and c.INTERNAL_SECONDS==9
    assert c.QUAL_OUT==c.OLD_OUT and c.OUT!=c.OLD_OUT


@pytest.mark.parametrize('module',['worker','adapters','audit','scientific','placement','analysis','schedule','corpus','qualification'])
def test_support_implementations_are_byte_identical(module):
    old=c.ROOT/'tools/splendorust_search128_strength_20261028'/f'{module}.py'
    new=c.ROOT/'tools/splendorust_search128_recovery_20261028'/f'{module}.py'
    assert old.read_bytes()==new.read_bytes()


def test_original_failure_is_not_a_timeout_or_accepted_move():
    from tools.splendorust_search128_strength_20261028 import scientific
    row=c.read(c.OLD_OUT/'games/main/p019_g38/moves/0039.json')
    assert row['clock']['status']=='resource_guard' and row['clock']['accepted'] is False
    assert row['clock']['receipt_elapsed_seconds'] is None and row['clock']['result'] is None
    assert row['completed'] is False and row['after'] is None
    with pytest.raises(ValueError,match='Only explicit nine-second'):
        scientific.classify_failure(row['clock'])


def test_frozen_original_sources_and_receipt_are_unchanged():
    assert c.file_hash(c.OLD_OUT/'freeze.json')==c.OLD_FREEZE_SHA
    freeze=c.read(c.OLD_OUT/'freeze.json')
    items=[x for x in freeze['inputs'] if x['path'].startswith('tools/splendorust_search128_strength_20261028/')]
    assert len(items)>=15
    c.verify(items)
    assert c.read(c.OLD_OUT/'completion.json')['status']=='incomplete'
    assert not(c.OLD_OUT/'postrun/independent_completion_review.json').exists()
