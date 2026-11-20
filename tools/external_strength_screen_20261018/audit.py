"""Independent ledger transition audit; never call the environment's step method."""
from collections import Counter
from copy import deepcopy
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest

COLORS = ('white', 'blue', 'green', 'red', 'black')
TOKENS = COLORS + ('gold',)


def require(value, message):
    if not value:
        raise ValueError('Independent transition audit: ' + message)


def _affordable(player, card):
    return sum(max(0, card.cost.get(c, 0) - player['bonuses'][c] - player['tokens'][c])
               for c in COLORS) <= player['tokens']['gold']


def validate_ledger(snapshot):
    require(set(snapshot) == {'schema', 'ruleset_id', 'catalogue_sha256', 'state'} and
            snapshot['schema'] == rules.SNAPSHOT_SCHEMA and snapshot['ruleset_id'] == rules.RULESET_ID and
            snapshot['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'ruleset identity')
    state = snapshot['state']
    require(set(state) == {'current_player', 'turn_index', 'start_player', 'pending_round_end',
            'terminal', 'winner', 'bank_tokens', 'visible_tier_cards', 'hidden_tier_decks',
            'deck_counts', 'nobles', 'players'}, 'state fields')
    require(type(state['start_player']) is int and state['start_player'] in (0, 1) and
            type(state['current_player']) is int and type(state['turn_index']) is int and
            state['turn_index'] >= 0 and state['current_player'] ==
            (state['start_player'] + state['turn_index']) % 2, 'turn parity')
    require(type(state['terminal']) is bool and type(state['pending_round_end']) is bool and
            (state['winner'] is None or type(state['winner']) is int and state['winner'] in (0, 1)), 'terminal types')
    require(len(state['players']) == 2, 'player count')
    for key in ('visible_tier_cards', 'hidden_tier_decks', 'deck_counts'):
        require(set(state[key]) == {'1', '2', '3'}, 'tier fields')
    require(set(state['bank_tokens']) == set(TOKENS), 'bank fields')
    cards, nobles = rules.corrected_cards_by_id(), rules.corrected_nobles_by_id()
    allocated = []
    ns = list(state['nobles'])
    for tier in ('1', '2', '3'):
        allocated += state['visible_tier_cards'][tier] + state['hidden_tier_decks'][tier]
        require(type(state['deck_counts'][tier]) is int and state['deck_counts'][tier] >= 0 and
                state['deck_counts'][tier] == len(state['hidden_tier_decks'][tier]), 'deck count')
        require(len(state['visible_tier_cards'][tier]) <= 4, 'market size')
        require(all(cards[k].tier == int(tier) for k in state['visible_tier_cards'][tier] +
                    state['hidden_tier_decks'][tier]), 'tier placement')
    for seat, p in enumerate(state['players']):
        require(set(p) == {'player_id', 'tokens', 'bonuses', 'score', 'purchased_card_ids',
                'noble_ids', 'reserved_card_ids', 'reserved_card_sources'}, 'player fields')
        require(type(p['player_id']) is int and p['player_id'] == seat, 'seat identity')
        require(set(p['tokens']) == set(TOKENS) and set(p['bonuses']) == set(COLORS), 'player count fields')
        require(type(p['score']) is int and p['score'] >= 0 and all(type(v) is int and v >= 0
                for v in p['bonuses'].values()), 'score/bonus types')
        require(len(p['reserved_card_ids']) == len(p['reserved_card_sources']) <= 3, 'reserve slots')
        require(all(s in ('deck', 'visible') for s in p['reserved_card_sources']), 'reserve provenance')
        allocated += p['reserved_card_ids'] + p['purchased_card_ids']
        ns += p['noble_ids']
        bonus = Counter(cards[k].bonus_color for k in p['purchased_card_ids'])
        require(all(p['bonuses'][c] == bonus[c] for c in COLORS), 'bonus ledger')
        require(p['score'] == sum(cards[k].points for k in p['purchased_card_ids']) +
                sum(nobles[k].points for k in p['noble_ids']), 'score ledger')
        require(all(type(p['tokens'][c]) is int and p['tokens'][c] >= 0 for c in TOKENS)
                and sum(p['tokens'].values()) <= 10, 'player tokens')
    require(len(allocated) == len(set(allocated)) == 90 and set(allocated) == set(cards), 'card conservation')
    require(len(ns) == len(set(ns)) == 3, 'noble conservation')
    pending = any(p['score'] >= 15 for p in state['players'])
    require(state['pending_round_end'] == pending, 'pending finish threshold')
    require(state['terminal'] == (pending and state['current_player'] == state['start_player']), 'round completion')
    if not state['terminal']:
        require(state['winner'] is None, 'winner before terminal')
    else:
        scores = [p['score'] for p in state['players']]
        candidates = [i for i in (0, 1) if scores[i] == max(scores)]
        fewest = min(len(state['players'][i]['purchased_card_ids']) for i in candidates)
        winners = [i for i in candidates if len(state['players'][i]['purchased_card_ids']) == fewest]
        require(state['winner'] == (winners[0] if len(winners) == 1 else None), 'terminal winner/tiebreak')
    for c in TOKENS:
        require(type(state['bank_tokens'][c]) is int and state['bank_tokens'][c] >= 0 and state['bank_tokens'][c] +
                sum(p['tokens'][c] for p in state['players']) == (5 if c == 'gold' else 4), 'token conservation')


def transition(before, action):
    """Apply one action using a separate dictionary ledger implementation."""
    validate_ledger(before)
    out = deepcopy(before)
    s = out['state']
    require(not s['terminal'], 'action after terminal')
    actor = s['current_player']; p = s['players'][actor]
    cards, nobles = rules.corrected_cards_by_id(), rules.corrected_nobles_by_id()
    require(set(action) == {'action_type', 'tier', 'market_index', 'reserved_index', 'take_tokens',
            'return_tokens', 'spend_tokens', 'noble_id', 'metadata'}, 'action fields')
    require(all(isinstance(action[k], list) for k in ('take_tokens', 'return_tokens', 'spend_tokens', 'metadata')), 'action field types')
    for name in ('take_tokens', 'return_tokens', 'spend_tokens'):
        require(all(type(token) is str and token in TOKENS for token in action[name]), 'unknown token')
        require(action[name] == sorted(action[name], key=TOKENS.index), 'noncanonical token ordering')
    kind = action['action_type']; tier = str(action['tier'])
    gained, returned, spent = map(Counter, (action['take_tokens'], action['return_tokens'], action['spend_tokens']))
    require(not action['metadata'], 'unexpected action metadata')
    require(set(gained) | set(returned) | set(spent) <= set(TOKENS), 'unknown token')
    is_buy = kind in ('BUY_VISIBLE', 'BUY_RESERVED')
    is_reserve = kind in ('RESERVE_VISIBLE', 'RESERVE_DECK')
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE', 'RESERVE_DECK'):
        require(type(action['tier']) is int and action['tier'] in (1, 2, 3), 'action tier')
    else:
        require(action['tier'] is None, 'unexpected tier')
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE'):
        require(type(action['market_index']) is int and
                0 <= action['market_index'] < len(s['visible_tier_cards'][tier]), 'market index')
    else:
        require(action['market_index'] is None, 'unexpected market index')
    if kind == 'BUY_RESERVED':
        require(type(action['reserved_index']) is int and
                0 <= action['reserved_index'] < len(p['reserved_card_ids']), 'reserve index')
    else:
        require(action['reserved_index'] is None, 'unexpected reserve index')
    if kind == 'TAKE_TOKENS':
        require(not spent, 'take fields')
        available = sum(s['bank_tokens'][c] > 0 for c in COLORS)
        distinct = all(n == 1 and c != 'gold' for c, n in gained.items())
        double = len(gained) == 1 and sum(gained.values()) == 2 and 'gold' not in gained
        require((distinct and len(gained) == min(3, available) and available > 0) or
                (double and s['bank_tokens'][next(iter(gained))] >= 4), 'take legality')
    elif is_reserve:
        require(len(p['reserved_card_ids']) < 3 and not spent, 'reserve legality')
        require(gained == Counter({'gold': 1} if s['bank_tokens']['gold'] else {}), 'reserve gold')
        if kind == 'RESERVE_VISIBLE':
            card_id = s['visible_tier_cards'][tier].pop(action['market_index'])
            source = 'visible'
        else:
            card_id = s['hidden_tier_decks'][tier].pop(0)
            source = 'deck'
        p['reserved_card_ids'].append(card_id); p['reserved_card_sources'].append(source)
    elif is_buy:
        require(not gained and not returned, 'buy token fields')
        if kind == 'BUY_VISIBLE':
            card_id = s['visible_tier_cards'][tier].pop(action['market_index'])
        else:
            card_id = p['reserved_card_ids'].pop(action['reserved_index'])
            p['reserved_card_sources'].pop(action['reserved_index'])
        card = cards[card_id]
        required = {c: max(0, card.cost.get(c, 0) - p['bonuses'][c]) for c in COLORS}
        require(all(spent[c] <= required[c] and spent[c] <= p['tokens'][c] for c in COLORS), 'colored payment')
        require(spent['gold'] == sum(required[c] - spent[c] for c in COLORS) and
                spent['gold'] <= p['tokens']['gold'], 'gold payment')
        for c, n in spent.items(): p['tokens'][c] -= n; s['bank_tokens'][c] += n
        p['purchased_card_ids'].append(card_id); p['bonuses'][card.bonus_color] += 1; p['score'] += card.points
    elif kind == 'PASS':
        require(not gained and not returned and not spent, 'pass fields')
        require(not any(s['bank_tokens'][c] for c in COLORS), 'pass while tokens available')
        require(len(p['reserved_card_ids']) == 3 or not any(s['visible_tier_cards'][t] or
                s['hidden_tier_decks'][t] for t in ('1', '2', '3')), 'pass while reservation available')
        possible = p['reserved_card_ids'] + sum(s['visible_tier_cards'].values(), [])
        require(not any(_affordable(p, cards[k]) for k in possible), 'pass while purchase available')
    else:
        raise ValueError('Unknown action type')
    if kind == 'TAKE_TOKENS' or is_reserve:
        require(all(n <= s['bank_tokens'][c] for c, n in gained.items()), 'bank availability')
        for c, n in gained.items(): s['bank_tokens'][c] -= n; p['tokens'][c] += n
        require(sum(returned.values()) == max(0, sum(p['tokens'].values()) - 10), 'return count')
        require(all(n <= p['tokens'][c] for c, n in returned.items()), 'return ownership')
        for c, n in returned.items(): p['tokens'][c] -= n; s['bank_tokens'][c] += n
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE') and s['hidden_tier_decks'][tier]:
        s['visible_tier_cards'][tier].append(s['hidden_tier_decks'][tier].pop(0))
    eligible = [n for n in s['nobles'] if all(p['bonuses'][c] >= v for c, v in nobles[n].requirement.items())]
    require(action['noble_id'] in eligible if eligible else action['noble_id'] is None, 'mandatory end-of-turn noble choice')
    if eligible:
        n = action['noble_id']; s['nobles'].remove(n); p['noble_ids'].append(n); p['score'] += nobles[n].points
    s['deck_counts'] = {t: len(v) for t, v in s['hidden_tier_decks'].items()}
    s['turn_index'] += 1; s['current_player'] = 1 - actor
    s['pending_round_end'] = s['pending_round_end'] or p['score'] >= 15
    if s['pending_round_end'] and s['current_player'] == s['start_player']:
        s['terminal'] = True
        scores = [q['score'] for q in s['players']]
        contenders = [i for i, score in enumerate(scores) if score == max(scores)]
        fewest = min(len(s['players'][i]['purchased_card_ids']) for i in contenders)
        winners = [i for i in contenders if len(s['players'][i]['purchased_card_ids']) == fewest]
        s['winner'] = winners[0] if len(winners) == 1 else None
    validate_ledger(out)
    return out


def check_step(before, action, after):
    wanted = transition(before, action)
    require(wanted == after, 'independent transition differs from engine')
    return digest(after)


def check_observer(snapshot, payload):
    """Independently redact a referee ledger, without using the observer builder."""
    validate_ledger(snapshot)
    view = deepcopy(snapshot['state'])
    del view['hidden_tier_decks']
    cards = rules.corrected_cards_by_id()
    for seat, p in enumerate(view['players']):
        ids = p.pop('reserved_card_ids'); sources = p.pop('reserved_card_sources')
        p['reserves'] = [{'card_id': k if seat == view['current_player'] or source == 'visible' else None,
                          'tier': cards[k].tier, 'source': source} for k, source in zip(ids, sources)]
    wanted = {'schema': rules.OBSERVATION_SCHEMA, 'ruleset_id': rules.RULESET_ID,
              'catalogue_sha256': rules.CATALOGUE_SHA256, 'observation': view}
    require(payload == wanted, 'observer payload differs from independent redaction')
    return digest(payload)
