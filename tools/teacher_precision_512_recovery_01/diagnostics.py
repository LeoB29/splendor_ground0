"""Preserve frozen diagnostic arithmetic while repairing its budget dependency.

The outer report is teacher_precision_512.learning.training_report; its inner
report is independent_game_coverage_128_recovery_01.learning.training_report.
Their code objects, defaults, and other globals are retained exactly. Private
global dictionaries replace only the outer ``prior`` reference and the inner
``c`` reference. No ancestor module or configuration is modified.

The recovery runner must call install() in every spawned process. Budget lookup
remains dynamic so its current active-time/abort boundary also governs reports.
"""
from types import FunctionType, SimpleNamespace

from tools.teacher_precision_512 import common as c, learning


ORIGINAL_REPORT = learning.training_report
ORIGINAL_INNER_REPORT = learning.prior.training_report


def _current_budget_boundary(base):
    return c.budget_boundary(base)


def _clone(function, replacements):
    """Reuse exact frozen code with a private, narrowly replaced globals map."""
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result.__annotations__ = dict(function.__annotations__)
    result.__doc__ = function.__doc__
    result.__module__ = __name__
    result.__qualname__ = function.__qualname__
    return result


# The inner report accesses only c.require and c.budget_boundary. Keep the
# original validator and route the budget through the current experiment.
INNER_REPORT = _clone(ORIGINAL_INNER_REPORT, {
    "c": SimpleNamespace(require=ORIGINAL_INNER_REPORT.__globals__["c"].require,
                         budget_boundary=_current_budget_boundary),
})
REPORT = _clone(ORIGINAL_REPORT, {
    "prior": SimpleNamespace(training_report=INNER_REPORT),
})


def training_report(model, rows, base, config, initial=None, *, audit=False,
                    precision=False, data_scope="train",
                    diagnostic_scope="full_train", **kwargs):
    return REPORT(model, rows, base, config, initial, audit=audit,
                  precision=precision, data_scope=data_scope,
                  diagnostic_scope=diagnostic_scope, **kwargs)


def install():
    """Install this report locally, without changing frozen source files."""
    c.require(learning.training_report in (ORIGINAL_REPORT, training_report),
              "Unexpected diagnostic replacement; preserve and inspect")
    learning.training_report = training_report
    return training_report
