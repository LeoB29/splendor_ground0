"""No-model classification and timing counterexamples for saved diagnosis."""
import copy
import pytest

from tools.splendorust_collection4096_resume_20261118.diagnostics.saved_timeout import classify_outcomes, summarize_receipts


def receipt():
    return dict(request_id='worker_1:1', submitted_perf=1., dispatch_perf=1.1,
                complete_readback_perf=1.2, client_received_perf=9., deadline_perf=60.,
                forward_roundtrip_seconds=.1)


def test_forward_is_not_postforward_wait():
    value = summarize_receipts([receipt()], 0.)
    assert value['maximum_forward_seconds'] == .1
    assert value['maximum_postforward_seconds'] == 7.8
    assert value['first_return_seconds'] == 9.


@pytest.mark.parametrize('change', [dict(client_received_perf=61.), dict(submitted_perf=-1.), dict(complete_readback_perf=10.)])
def test_impossible_successful_clock_rejected(change):
    row = dict(receipt(), **change)
    with pytest.raises(ValueError):
        summarize_receipts([row], 0.)


def test_duplicate_callback_rejected():
    with pytest.raises(ValueError):
        summarize_receipts([receipt(), copy.deepcopy(receipt())], 0.)


def test_only_timeout_classified_as_trigger_no_strength_claim():
    values = {'a': dict(status='completed'),
              'b': dict(status='integrity_failed', clock=dict(status='timeout', completed=False)),
              'c': dict(status='cancelled', clock=dict(status='error', completed=False))}
    result = classify_outcomes(values)
    assert result['triggering_recorded_timeout_games'] == ['b']
    assert result['cancelled_games'] == ['c']
    assert result['strength_inference'] is None
    assert result['underlying_cause'].startswith('unresolved')


def test_cancellation_not_silently_promoted_to_timeout():
    with pytest.raises(ValueError):
        classify_outcomes({'a': dict(status='cancelled', clock=dict(status='timeout', completed=False))})
