"""Exact Entity policy with isolated per-choice public preparation reuse."""
from tools.move_evaluation_20261023.greedy_adapter import (
    GreedyOpponent as FrozenGreedyOpponent, PROFILE_ID, require, digest,
    serialize_action, canonical_index, first_max,
)
from . import prepared

IMPLEMENTATION_ID = prepared.IMPLEMENTATION_ID


class GreedyOpponent(FrozenGreedyOpponent):
    """Constructor, float32 arithmetic, model, native transport and seeds unchanged."""

    def choose(self, observer, *, diagnostic_capture=False):
        context = prepared.prepare(observer)
        observer_sha256 = context.observer_sha256
        seed = int(observer_sha256[:16], 16)
        phases = []

        def select(native_observation, legal):
            result = self.exchange({'op': 'greedy', 'observation': native_observation,
                                    'legal': legal, 'seed': seed}, capture=diagnostic_capture)
            required = {'kind', 'action', 'branch', 'native_legal_count', 'filtered', 'canonical_indices',
                        'selected_index', 'canonical_logits', 'simulations', 'leaf_calls'}
            require(set(result) == required, 'result schema')
            filtered = result['filtered']
            require(type(result['native_legal_count']) is int and result['native_legal_count'] == len(legal), 'legal count')
            require(isinstance(filtered, list) and filtered and all(a in legal for a in filtered), 'filtered support')
            indices = [legal.index(a) for a in filtered]
            require(indices == sorted(set(indices)), 'filter must preserve unique order')
            selected = result['selected_index']
            require(type(selected) is int and 0 <= selected < len(filtered) and result['action'] == filtered[selected], 'selection coupling')
            main = native_observation['phase']['kind'] == 'Main'
            expected = ('auxiliary_heuristic' if not main else 'single_choice' if len(filtered) == 1
                        else 'late_heuristic' if native_observation['turns'] >= 124 else 'neural_greedy')
            require(result['branch'] == expected, 'policy branch')
            neural = expected == 'neural_greedy'
            require(type(result['simulations']) is int and result['simulations'] == 0 and
                    type(result['leaf_calls']) is int and result['leaf_calls'] == int(neural) and
                    len(self.callbacks) == int(neural), 'greedy work count')
            require(result['canonical_indices'] == ([canonical_index(a) for a in filtered] if main else []), 'canonical index map')
            if neural:
                require(first_max(result['canonical_logits'], filtered) == selected, 'first maximum')
            else:
                require(result['canonical_logits'] is None, 'heuristic received logits')
            phase = {'phase': native_observation['phase']['kind'], 'observation_sha256': digest(native_observation),
                     'legal_sha256': digest(legal), 'branch': expected, 'native_legal_count': len(legal),
                     'retained_indices': indices, 'canonical_indices': result['canonical_indices'],
                     'selected_index': selected, 'action': result['action'],
                     'policy_logits_sha256': digest(result['canonical_logits']) if neural else None,
                     'inference': self.callbacks}
            if diagnostic_capture:
                phase['canonical_logits'] = result['canonical_logits']
            phases.append(phase)
            return result['action']

        native = prepared.encode_prepared(context)
        legal_main = prepared.main_prepared(context)
        forced = legal_main == [{'kind': 'Pass'}]
        selected = legal_main[0] if forced else select(native, legal_main)
        action = prepared.resolve_prepared(context, selected, select)
        require(action in context.legal, 'atomic action is illegal')
        trace = {'schema': 'entity_greedy_decision_v1', 'profile_id': PROFILE_ID,
                 'observer_sha256': observer_sha256, 'observer_seed': seed,
                 'action_sha256': digest(serialize_action(action)), 'forced_pass': forced,
                 'phases': phases, 'inference_calls': sum(len(p['inference']) for p in phases),
                 'simulations': 0}
        self.boundary()
        return action, trace
