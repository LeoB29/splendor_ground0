"""Continuation preserves numerical identity and carries the original clocks."""
from pathlib import Path
import pytest
from tools.splendorust_search128_resume_20261029 import common as c
from tools.splendorust_search128_resume_20261029 import launch_owner


def test_actual_reader_command_and_exit_receipt_use_the_same_new_source():
    from tools.splendorust_search128_resume_20261029 import independent_review
    script, command = launch_owner.independent_reader_route('freeze-test')
    assert script.resolve() == Path(independent_review.__file__).resolve()
    assert command[command.index('-m') + 1] == independent_review.__name__
    assert command[-2:] == ['--freeze-sha256', 'freeze-test']
    assert c.file_hash(script) != c.file_hash(
        c.ROOT / 'tools/splendorust_search128_recovery_20261028/independent_review.py')


@pytest.mark.parametrize('name',[
    'worker.py','adapters.py','audit.py','scientific.py','placement.py',
    'analysis.py','schedule.py','corpus.py','qualification.py','accounting.py',
    'performance.py',
])
def test_qualified_implementation_bytes_unchanged(name):
    root=c.ROOT/'tools'
    assert (root/'splendorust_search128_resume_20261029'/name).read_bytes()==(
        root/'splendorust_search128_recovery_20261028'/name).read_bytes()


def test_original_clocks_resources_and_profile_source_are_retained():
    assert (c.ORIGINAL_MONOTONIC,c.ORIGINAL_UNIX)==(474768.187,1793218912.6661534)
    assert (c.PRIOR_FINAL_SECONDS,c.PRIOR_RAM_WAIT_SECONDS)==(190.,200.)
    assert (c.SECONDS_PER_MOVE,c.INTERNAL_SECONDS)==(10.,9.)
    assert c.QUAL_OUT==c.OLD_OUT
    assert c.MAX_WORKERS==c.MAX_GRAPH_BUILDS==864
    assert c.MAX_EVIDENCE_BYTES==16*c.GiB and c.FINAL_RESERVE_BYTES==4*c.GiB
    assert c.ROOT/'outputs/splendorust_search128_resume_20261029_01'==c.OUT
    assert c.FIRST_RECOVERY_OUT!=c.OUT and c.CLOSURE_OUT!=c.OUT


def test_new_acceptance_owner_cannot_extend_original_remaining_minute():
    claim={'freeze_sha256':'f','monotonic_origin':c.ORIGINAL_MONOTONIC,
           'unix_origin':c.ORIGINAL_UNIX}
    completion={'freeze_sha256':'f','independent_review_maximum_seconds':3600,
        'independent_acceptance_deadline_monotonic':c.ORIGINAL_MONOTONIC+c.JOB_SECONDS,
        'independent_acceptance_deadline_unix':c.ORIGINAL_UNIX+c.JOB_SECONDS}
    actual=launch_owner.acceptance_deadlines(claim,completion,
        c.ORIGINAL_MONOTONIC+c.JOB_SECONDS-60,c.ORIGINAL_UNIX+c.JOB_SECONDS-60)
    assert actual==(c.ORIGINAL_MONOTONIC+c.JOB_SECONDS,c.ORIGINAL_UNIX+c.JOB_SECONDS)


def test_final_stage_deadline_remains_stricter_than_fresh_review_budget():
    claim={'freeze_sha256':'f','monotonic_origin':c.ORIGINAL_MONOTONIC,
           'unix_origin':c.ORIGINAL_UNIX}
    completion={'freeze_sha256':'f','independent_review_maximum_seconds':3600,
        'independent_acceptance_deadline_monotonic':c.ORIGINAL_MONOTONIC+80000,
        'independent_acceptance_deadline_unix':c.ORIGINAL_UNIX+80000}
    actual=launch_owner.acceptance_deadlines(claim,completion,
        c.ORIGINAL_MONOTONIC+79970,c.ORIGINAL_UNIX+79970)
    assert actual==(c.ORIGINAL_MONOTONIC+80000,c.ORIGINAL_UNIX+80000)


def test_both_original_dispositions_and_saved_checker_stay_explicit():
    assert c.file_hash(c.OLD_OUT/'freeze.json')==c.OLD_FREEZE_SHA
    assert c.file_hash(c.FIRST_RECOVERY_OUT/'freeze.json')==c.FIRST_RECOVERY_SHA
    assert c.file_hash(c.CLOSURE_OUT/'closure_check.json')==c.CLOSURE_SHA
    for root in (c.OLD_OUT,c.FIRST_RECOVERY_OUT):
        assert c.read(root/'completion.json')['status']=='incomplete'
    receipt=c.read(c.CLOSURE_OUT/'closure_check.json')
    assert receipt['benchmark_accepted'] is False
    assert receipt['segmented_game']['proof']['transitions']==54
    assert receipt['new_model_calls']==receipt['new_games']==0
    row=c.read(c.OLD_OUT/'games/main/p019_g38/moves/0039.json')
    assert row['clock']['status']=='resource_guard'
    assert row['clock']['accepted'] is False
    assert row['clock']['receipt_elapsed_seconds'] is None
