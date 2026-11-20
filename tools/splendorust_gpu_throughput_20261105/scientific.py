"""Explicit wrapper-only projection onto the unchanged scientific validator.

The original profile remains a source binding for model/search parameters. Its
old timing policy is not the effective execution policy of this new wrapper.
Raw responses are retained; projection never changes action, trace or weights.
"""
from copy import deepcopy

from tools.splendorust_search512_20261101 import adapters as original
from tools.splendorust_search512_20261101 import scientific as legacy
from .common import EXECUTION_POLICY_ID, require

RESULT_FIELDS = legacy.RESULT_FIELDS
WRAPPER_SUFFIX = "_throughput60s_outer75_v1"
audit_decision = legacy.audit_decision
validate_search_work = legacy.validate_search_work
check_search_phase = legacy.check_search_phase


def wrapper_id(quota):
    require(type(quota) is int and quota in (16, 512), "original16/512 scientific quota")
    return original.profile_id(quota, route="candidate") + WRAPPER_SUFFIX


def source_result(result, *, legacy=False):
    """Validate the explicit identity domain, then copy with one wrapper mapping.

`legacy=True` is reserved for already accepted historical evidence. New runtime
responses must use the default strict new-wrapper path.
    """
    require(type(result) is dict and set(result) == RESULT_FIELDS, "exact scientific envelope")
    quota = result["simulations_quota"]
    require(type(quota) is int and quota in (16, 512), "supported exact scientific quota")
    require(result["profile_id"] == original.profile_id(quota, route="candidate") and
            result["profile_sha256"] == original.profile_sha256(quota, route="candidate"),
            "unchanged authenticated scientific source profile")
    expected = original.wrapper_id(quota, route="candidate") if legacy else wrapper_id(quota)
    require(result["wrapper_id"] == expected, "explicit validated execution wrapper identity")
    projected = deepcopy(result)
    projected["wrapper_id"] = original.wrapper_id(quota, route="candidate")
    return projected


def normalize_result(result):
    return source_result(result, legacy=False)


def validate_result(result, observer, arm, expected_quota=None, *, quota=None):
    return legacy.validate_result(normalize_result(result), observer, arm, expected_quota, quota=quota)


def validate_legacy_result(result, observer, arm, expected_quota=None, *, quota=None):
    return legacy.validate_result(source_result(result, legacy=True), observer, arm, expected_quota, quota=quota)
