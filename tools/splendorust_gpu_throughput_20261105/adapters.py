"""Original native PUCT and action flow with instance-local inference routing."""
from __future__ import annotations

import copy
import time
from types import SimpleNamespace

from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent
from .backend import require
from .common import INTERNAL_SECONDS


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
            raise TimeoutError("GPU_LIVE_INTERNAL_DEADLINE: complete60s budget")

    def _begin(self):
        if self.admission_guard is not None:
            self.admission_guard()
        self.active_deadline = self.request_deadline if self.request_deadline is not None else time.perf_counter() + INTERNAL_SECONDS
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
                result = self._choose_with_original_deadline(observer)
                self.boundary()
                return result
            finally:
                self.capture = False; self._finish()

    def _choose_with_original_deadline(self, observer):
        from tools.external_strength_screen_20261018_02.bridge import encode_observation, main_actions, resolve_compound
        self.deadline = self.active_deadline
        require(self.deadline is not None, "preserved original complete-response deadline")
        self.inference_calls = self.inference_rows = 0
        seed = int(digest(observer)[:16], 16)
        phases = []

        def select(obs, choices):
            result = self.exchange({'op': 'select', 'observation': obs, 'legal': choices,
                                    'seed': seed, 'iterations': self.profile['iterations']})
            phases.append({k: v for k, v in result.items() if k != 'kind'})
            return result['action']

        try:
            observation = encode_observation(observer)
            legal = main_actions(observer)
            selected = legal[0] if legal == [{'kind': 'Pass'}] else select(observation, legal)
            action = resolve_compound(observer, selected, select)
            if time.perf_counter() >= self.deadline:
                raise TimeoutError('Internal fixed-work deadline exceeded')
            return action, {'opponent_id': 'splendorust', 'profile_id': self.profile['profile_id'],
                            'observer_seed': seed, 'phases': phases,
                            'inference_calls': self.inference_calls, 'inference_rows': self.inference_rows,
                            'forced_pass': selected['kind'] == 'Pass'}
        except (ValueError, KeyError, AssertionError, IndexError) as error:
            raise ValueError('ADAPTER_PREREQUISITE: ' + str(error)) from error
        finally:
            self.deadline = None


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
