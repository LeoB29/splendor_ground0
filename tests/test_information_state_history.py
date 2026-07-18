from splendor_ai.cfr.reduced_splendor import ReducedSplendorCFRGame


def _blind_reserve_successors(game: ReducedSplendorCFRGame):
    roots = [game.apply_chance(game.root(), outcome) for outcome in (0, 1)]
    action_keys = [
        next(key for key in game.legal_action_keys(node) if key[0] == "RESERVE_DECK")
        for node in roots
    ]
    return roots, [
        game.apply_action(node, key) for node, key in zip(roots, action_keys, strict=True)
    ]


def test_hidden_order_is_masked_before_a_blind_reserve() -> None:
    game = ReducedSplendorCFRGame()
    roots, _children = _blind_reserve_successors(game)

    assert game.information_state_key(roots[0], 0) == game.information_state_key(roots[1], 0)


def test_blind_reserve_identity_is_private_but_owner_remembers_it() -> None:
    game = ReducedSplendorCFRGame()
    _roots, children = _blind_reserve_successors(game)

    assert game.information_state_key(children[0], 0) != game.information_state_key(children[1], 0)
    assert game.information_state_key(children[0], 1) == game.information_state_key(children[1], 1)
    assert children[0].public_history == children[1].public_history
    event_text = repr(children[0].public_history)
    assert "H_MIX" not in event_text
    assert "H_FREE" not in event_text

