"""One timed process, two independent stateless policies; no true worlds enter."""
from tools.local_player_qualification_128.clock import serve


def initialize():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
    from tools.external_strength_screen_20261018 import ppo
    from tools.external_strength_screen_20261018.common import ROOT, file_hash, PPO_PROFILE
    path = ROOT / 'outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt'
    if file_hash(path) != '4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd':
        raise ValueError('Raw035 checkpoint identity mismatch')
    raw = LegalActionAttentionScorerBot(path, device='cpu')
    external = ppo.PPOAdapter(profile=PPO_PROFILE)
    return {'raw035': raw, 'ppo': external}


def handle(context, request):
    from tools.external_strength_screen_20261018 import rules
    from tools.external_strength_screen_20261018.common import serialize_action, legal_digest, digest
    if set(request) != {'player', 'observer'} or request['player'] not in context:
        raise ValueError('Unexpected request fields or player')
    payload = request['observer']
    state = rules.state_from_observer_payload(payload)
    env = rules.CorrectedEnv()
    legal = env.legal_actions(state)
    if not legal:
        raise ValueError('Terminal or empty legal set')
    if request['player'] == 'raw035':
        action = context['raw035'].choose_action(env, state, legal)
        result = {'action': serialize_action(action)}
    else:
        result = context['ppo'].choose(payload)
    if result['action'] not in [serialize_action(a) for a in legal]:
        raise ValueError('Adapter returned an illegal action')
    return {**result, 'player': request['player'], 'observer_sha256': digest(payload),
            'legal_actions_sha256': legal_digest(legal), 'legal_count': len(legal),
            'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle))
