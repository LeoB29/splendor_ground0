from __future__ import annotations

import subprocess, sys

from splendor_ai.bots.persistent_option_bot import (
    OPTION_DEPTH, OPTION_DETERMINIZATIONS, OPTION_INTERNAL_BRANCHING, OPTION_ROOT_BRANCHING,
    PersistentCardOptionSearch, select_option_target,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, RESERVE_SOURCE_VISIBLE

def test_frozen_search_contract() -> None:
    assert (OPTION_DEPTH, OPTION_ROOT_BRANCHING, OPTION_INTERNAL_BRANCHING, OPTION_DETERMINIZATIONS) == (3,8,4,4)

def test_reserved_qualifying_target_precedes_visible() -> None:
    state=SplendorEnv(seed=79).initial_state(); player=state.players[0]
    reserved=Card("reserved-option",3,"red",4,{"red":7})
    player.reserve_card(reserved,RESERVE_SOURCE_VISIBLE)
    assert select_option_target(state,0) == reserved

def test_option_tuple_orders_purchased_reserved_visible_lost() -> None:
    env=SplendorEnv(seed=3); state=env.initial_state(); target=state.visible_tier_cards[3][0]
    search=PersistentCardOptionSearch(target,3)
    visible=search._value(state,0)  # noqa: SLF001
    reserved_state=state.copy_shallow(); card=reserved_state.visible_tier_cards[3].pop(0); reserved_state.players[0].reserve_card(card,RESERVE_SOURCE_VISIBLE)
    reserved=search._value(reserved_state,0)  # noqa: SLF001
    purchased_state=reserved_state.copy_shallow(); card=purchased_state.players[0].pop_reserved_card(0); purchased_state.players[0].purchased_cards.append(card)
    purchased=search._value(purchased_state,0)  # noqa: SLF001
    lost_state=state.copy_shallow(); lost_state.visible_tier_cards[3].pop(0)
    lost=search._value(lost_state,0)  # noqa: SLF001
    assert purchased[1] == 3 and reserved[1] == 2 and visible[1] == 1 and lost[1] == 0
    assert purchased > reserved > visible > lost

def test_root_cli_help() -> None:
    result=subprocess.run([sys.executable,"run_persistent_card_option_planner_079.py","--help"],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
    assert "Persistent card-option planner" in result.stdout
