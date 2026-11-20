"""One supervised Python worker and persistent observer-only Rust opponent."""
import os
from pathlib import Path
import time

from tools.local_player_qualification_128.clock import serve


def initialize():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
    from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
    from tools.external_strength_screen_20261018_02 import common as c
    profile_path = Path(os.environ['SPLENDOR_OPPONENT_PROFILE'])
    profile_sha = os.environ['SPLENDOR_OPPONENT_PROFILE_SHA256']
    profile = c.load_profile(profile_path, profile_sha)
    path = c.ROOT / 'outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt'
    if c.file_hash(path) != c.RAW035_SHA256:
        raise ValueError('Raw035 checkpoint identity mismatch')
    raw = LegalActionAttentionScorerBot(path, device='cpu')
    opponent = Opponent(profile)
    return {'raw035': raw, 'opponent': opponent, 'profile': profile, 'profile_sha256': profile_sha}


def handle(context, request):
    from tools.external_strength_screen_20261018 import rules
    from tools.external_strength_screen_20261018_02 import common as c
    if set(request) != {'player', 'observer'} or request['player'] not in ('raw035', 'opponent'):
        raise ValueError('Unexpected request fields or player')
    payload = request['observer']
    state = rules.state_from_observer_payload(payload)
    env = rules.CorrectedEnv()
    legal = env.legal_actions(state)
    if not legal:
        raise ValueError('ADAPTER_PREREQUISITE: terminal or empty legal set')
    if request['player'] == 'raw035':
        action = context['raw035'].choose_action(env, state, legal)
        result = {'action': c.serialize_action(action)}
    else:
        action, diagnostics = context['opponent'].choose(payload)
        if (diagnostics['opponent_id'] != c.OPPONENT_ID or
                diagnostics['profile_id'] != context['profile']['profile_id']):
            raise ValueError('ADAPTER_PREREQUISITE: opponent identity changed')
        result = {'action': c.serialize_action(action), 'diagnostics': diagnostics,
                  'profile_sha256': context['profile_sha256']}
    if result['action'] not in [c.serialize_action(a) for a in legal]:
        raise ValueError('Adapter returned an illegal action')
    return {**result, 'player': request['player'], 'observer_sha256': c.digest(payload),
            'legal_actions_sha256': c.legal_digest(legal), 'legal_count': len(legal),
            'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle))
