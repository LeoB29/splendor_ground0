"""Original service/search clocks with separately authenticated root-target capture."""
from types import SimpleNamespace

from tools.splendorust_gpu_throughput_20261105.adapters import ProxySearchOpponent
from tools.splendorust_selfplay_pilot_budget_20261102.capture_adapter import CaptureOpponent, capture_record


class CaptureProxySearchOpponent(ProxySearchOpponent):
    """The capture transport adds evidence; the original action trace is unaltered."""
    def __init__(self, profile_document, *, runtime, client_id, slot, native_artifact):
        self.runtime, self.client_id, self.slot = runtime, client_id, slot
        self.last_service_calls = []; self._closed = False
        runtime.service.register(client_id, slot)
        try:
            # The frozen constructor authenticates and starts only the capture
            # native bridge. Network evaluation still uses the inherited proxy.
            CaptureOpponent.__init__(self, profile_document, native_artifact=native_artifact,
                shared_from=SimpleNamespace(model=None, reference={}), capture_targets=True)
        except BaseException:
            if client_id in runtime.service.clients:
                runtime.service.unregister(client_id)
            raise

    def exchange(self, request):
        previous = self.active_deadline
        with self.runtime.operation(self.client_id, kind="native_exchange"):
            if previous is None:
                self._begin()
            try:
                return CaptureOpponent.exchange(self, request)
            finally:
                self.active_deadline = previous

    def choose(self, observer, *, diagnostic_capture=False):
        self.last_service_calls = []
        self.last_training_capture = None; self._main_capture = None; self._root_callback = None
        with self.runtime.operation(self.client_id, kind="complete_decision"):
            self._begin(); self.capture = bool(diagnostic_capture)
            try:
                result = self._choose_with_original_deadline(observer)
                _, trace = result
                main = {"kind": "Pass"} if trace["forced_pass"] else trace["phases"][0]["action"]
                self.last_training_capture = capture_record(observer, self.profile["iterations"],
                    self._main_capture, self._root_callback, main)
                # Target validation is part of this same admitted decision,
                # before its original60s/75s clocks finish. Never reset them.
                self.boundary()
                return result
            finally:
                self.capture = False; self._finish()


def source_equivalence():
    """Model-free identity of inherited math, actions, deadlines and capture code."""
    from . import common as c
    inherited = ("boundary", "_begin", "_infer", "_choose_with_original_deadline",
                 "_finish", "direct_choose", "reference_choose", "close")
    c.require(all(getattr(CaptureProxySearchOpponent, key) is getattr(ProxySearchOpponent, key)
                  for key in inherited), "original proxy inference/search/deadline methods inherited")
    return dict(status="capture_source_equivalence_verified", inherited_methods=list(inherited),
        original_proxy=c.artifact(c.ROOT/"tools/splendorust_gpu_throughput_20261105/adapters.py"),
        original_capture=c.artifact(c.ROOT/"tools/splendorust_selfplay_pilot_budget_20261102/capture_adapter.py"),
        new_adapter=c.artifact(__file__), original_inference_search_clocks=True,
        added_work="Authenticated capture transport and root-target validation inside original complete-decision clocks",
        target_evidence_separate_from_scientific_trace=True, models_run=0)
