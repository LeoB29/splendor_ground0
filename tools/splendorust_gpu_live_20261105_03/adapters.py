"""Original native PUCT and action flow with instance-local inference routing."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent
from .backend import require


class ProxySearchOpponent(SearchOpponent):
    """No local Torch model; one synchronous service call per original callback."""
    def __init__(self, profile_document, *, runtime, client_id, slot):
        self.runtime, self.client_id, self.slot = runtime, client_id, slot
        self.last_service_calls = []; self._closed = False
        runtime.service.register(client_id, slot)
        try:
            # The authenticated inherited shared_from constructor starts only
            # the byte-identical native transport; it performs no model load.
            super().__init__(profile_document, shared_from=SimpleNamespace(model=None, reference={}))
        except BaseException:
            runtime.service.unregister(client_id)
            raise

    def boundary(self):
        if self.active_deadline is not None and time.perf_counter() >= self.active_deadline:
            raise TimeoutError("GPU_LIVE_INTERNAL_DEADLINE: original complete9s budget")

    def _begin(self):
        if self.admission_guard is not None:
            self.admission_guard()
        self.active_deadline = self.request_deadline if self.request_deadline is not None else time.perf_counter() + 9.
        self.boundary()

    def _infer(self, tokens, batch):
        self.boundary()
        require(self.active_deadline is not None, "inference has an original operation deadline")
        outputs, receipt = self.runtime.service.infer(self.client_id, tokens, batch, deadline=self.active_deadline)
        self.boundary(); self.last_service_calls.append(receipt)
        input_sha, output_sha = digest(tokens), digest(outputs)
        item = dict(callback_sha256=digest([input_sha, output_sha, batch]), batch=batch)
        if self.capture:
            item.update(input_sha256=input_sha, output_sha256=output_sha)
            if not self.callbacks:
                item.update(tokens=copy.deepcopy(tokens), outputs=copy.deepcopy(outputs), output83=copy.deepcopy(outputs[0]))
        self.callbacks.append(item); self.boundary()
        return outputs

    def exchange(self, request):
        # Catalogue/inspect/parity qualification exchanges also have real outer
        # watchdog publication, even when not nested in a complete choose call.
        previous = self.active_deadline
        with self.runtime.operation(self.client_id, kind="native_exchange"):
            if previous is None:
                self._begin()
            try:
                return super().exchange(request)
            finally:
                self.active_deadline = previous

    def choose(self, observer, *, diagnostic_capture=False):
        self.last_service_calls = []
        with self.runtime.operation(self.client_id, kind="complete_decision"):
            self._begin(); self.capture = bool(diagnostic_capture)
            try:
                result = Opponent.choose(self, observer)
                self.boundary()
                return result
            finally:
                self.capture = False; self._finish()

    def direct_choose(self, observer, *, diagnostic_capture=True):
        return self.choose(observer, diagnostic_capture=diagnostic_capture)

    def reference_choose(self, observer, *, diagnostic_capture=True):
        require(self.profile["iterations"] == 16, "reference reproduction remains16")
        return self.choose(observer, diagnostic_capture=diagnostic_capture)

    def close(self):
        if self._closed:
            return
        try:
            Opponent.close(self)
        finally:
            # close is registered by the inherited constructor with atexit.
            # Only this owned native process and this client's slot are touched.
            if self.client_id not in self.runtime.service.pending:
                self.runtime.release_adapter(self.client_id)
            self._closed = True
