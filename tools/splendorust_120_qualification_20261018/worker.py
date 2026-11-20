"""One fresh opponent and one decision; no fallback after internal exhaustion."""
import os
from pathlib import Path

from tools.local_player_qualification_128.clock import serve
from tools.splendorust_120_qualification_20261018 import common as c

INTERNAL_TIMEOUT_MESSAGE = 'Internal fixed-work deadline exceeded'


def initialize():
    # Authenticate every source before importing numerical/model code.
    profile_path = Path(os.environ['SPLENDOR_OPPONENT_PROFILE'])
    profile_sha = os.environ['SPLENDOR_OPPONENT_PROFILE_SHA256']
    profile = c.load_profile(profile_path, profile_sha)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
    opponent = Opponent(profile)
    return {'opponent': opponent, 'profile': profile, 'profile_sha256': profile_sha, 'used': False}


def handle(context, request):
    from tools.external_strength_screen_20261018 import rules
    if context['used']:
        raise ValueError('Exactly one decision is permitted per initialized worker')
    context['used'] = True
    if (not isinstance(request, dict) or set(request) != {'player', 'observer'} or
            request['player'] != 'opponent'):
        raise ValueError('Unexpected request fields or player')
    if rules.RULESET_ID != c.RULESET_ID or rules.CATALOGUE_SHA256 != c.CATALOGUE_SHA256:
        raise ValueError('Corrected rules identity changed')
    payload = request['observer']
    state = rules.state_from_observer_payload(payload)
    legal = rules.CorrectedEnv().legal_actions(state)
    if state.terminal or not legal:
        raise ValueError('ADAPTER_PREREQUISITE: terminal or empty legal set')
    identity = {'player': 'opponent', 'observer_sha256': c.digest(payload),
                'legal_actions_sha256': c.legal_digest(legal), 'legal_count': len(legal),
                'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
                'profile_sha256': context['profile_sha256']}
    try:
        action, diagnostics = context['opponent'].choose(payload)
    except TimeoutError as error:
        # Only the frozen adapter's exact internal fixed-work cutoff is a
        # qualification timeout. Transport/OS/adapter errors remain hard errors.
        if str(error) != INTERNAL_TIMEOUT_MESSAGE:
            raise
        return {**identity, 'status': 'internal_budget_exhausted',
                'internal_budget_seconds': c.INTERNAL_SECONDS,
                'opponent_id': c.OPPONENT_ID, 'profile_id': context['profile']['profile_id'],
                'reason': INTERNAL_TIMEOUT_MESSAGE}
    if (diagnostics['opponent_id'] != c.OPPONENT_ID or
            diagnostics['profile_id'] != context['profile']['profile_id']):
        raise ValueError('ADAPTER_PREREQUISITE: opponent identity changed')
    serialized = c.serialize_action(action)
    if serialized not in [c.serialize_action(a) for a in legal]:
        raise ValueError('Adapter returned an illegal action')
    return {**identity, 'status': 'completed', 'action': serialized, 'diagnostics': diagnostics}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle))
