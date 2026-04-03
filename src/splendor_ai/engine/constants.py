"""Shared constants for the base Splendor environment."""

from __future__ import annotations

TOKEN_COLORS: tuple[str, ...] = ("white", "blue", "green", "red", "black")
ALL_TOKEN_TYPES: tuple[str, ...] = TOKEN_COLORS + ("gold",)

MAX_PLAYERS = 2
MAX_RESERVED_CARDS = 3
MAX_TOKENS_PER_PLAYER = 10
TARGET_PRESTIGE = 15

# Base-game setup constants for 2-player Splendor.
BANK_GEMS_PER_COLOR_2P = 4
GOLD_TOKENS = 5
VISIBLE_CARDS_PER_TIER = 4
NOBLES_IN_PLAY_2P = 3
