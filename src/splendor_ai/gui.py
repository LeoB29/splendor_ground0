"""Tkinter GUI for manual Splendor play against bots.

Board rendering mimics the physical game: a noble row on top, three tier rows
with the face-down deck on the left of four face-up cards, and the token bank
as chip stacks along the bottom. All drawing lives in ``gui_art``; this module
keeps game flow, action matching, and bot handling.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Import bots from their concrete modules (not the package) so the GUI stays
# launchable without torch; torch-backed bots are imported lazily on demand.
from splendor_ai.bots.heuristic_bot import GreedyHeuristicBot
from splendor_ai.bots.random_bot import RandomLegalBot
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble, SplendorState

from splendor_ai import gui_art as art


TOKEN_ORDER = ("white", "blue", "green", "red", "black", "gold")
TOKEN_SHORT = {
    "white": "W",
    "blue": "U",
    "green": "G",
    "red": "R",
    "black": "B",
    "gold": "Au",
}
TOKEN_COLOR = art.TOKEN_COLOR
BONUS_ORDER = ("white", "blue", "green", "red", "black")

PANEL_BG = "#efe8da"
PANEL_CARD_BG = "#f8f3e9"
TOP_BAR_BG = "#d9ceb8"


def _format_cost(cost: dict[str, int]) -> str:
    if not cost:
        return "free"
    return " ".join(f"{TOKEN_SHORT[color]}:{amount}" for color, amount in cost.items())


def describe_card(card: Card) -> str:
    return (
        f"{card.card_id}\n"
        f"T{card.tier}  pts:{card.points}  bonus:{TOKEN_SHORT[card.bonus_color]}\n"
        f"cost { _format_cost(card.cost) }"
    )


def describe_noble(noble: Noble) -> str:
    return f"{noble.noble_id}  pts:{noble.points}  req { _format_cost(noble.requirement) }"


def describe_action(action: Action, state: SplendorState) -> str:
    if action.action_type == ActionType.PASS:
        return "Pass"

    if action.action_type == ActionType.TAKE_TOKENS:
        text = f"Take {' '.join(TOKEN_SHORT[color] for color in action.take_tokens)}"
        if action.return_tokens:
            text += f" / return {' '.join(TOKEN_SHORT[color] for color in action.return_tokens)}"
        return text

    if action.action_type == ActionType.RESERVE_VISIBLE:
        card = state.visible_tier_cards[action.tier][action.market_index]
        text = f"Reserve board {card.card_id}"
        if action.take_tokens:
            text += " + gold"
        if action.return_tokens:
            text += f" / return {' '.join(TOKEN_SHORT[color] for color in action.return_tokens)}"
        return text

    if action.action_type == ActionType.RESERVE_DECK:
        text = f"Reserve deck T{action.tier}"
        if action.take_tokens:
            text += " + gold"
        if action.return_tokens:
            text += f" / return {' '.join(TOKEN_SHORT[color] for color in action.return_tokens)}"
        return text

    if action.action_type == ActionType.BUY_VISIBLE:
        card = state.visible_tier_cards[action.tier][action.market_index]
        text = f"Buy board {card.card_id}"
        if action.spend_tokens:
            text += f" / pay {' '.join(TOKEN_SHORT[color] for color in action.spend_tokens)}"
        if action.noble_id:
            text += f" / noble {action.noble_id}"
        return text

    if action.action_type == ActionType.BUY_RESERVED:
        card = state.players[state.current_player].reserved_cards[action.reserved_index]
        text = f"Buy reserved {card.card_id}"
        if action.spend_tokens:
            text += f" / pay {' '.join(TOKEN_SHORT[color] for color in action.spend_tokens)}"
        if action.noble_id:
            text += f" / noble {action.noble_id}"
        return text

    return repr(action)


HUMAN_GAMES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "human_games"


class _GuiGameRecorder:
    """Records GUI games in the replay-corpus JSONL row format.

    Rows match ``export_replay_games_jsonl`` (with state snapshots included),
    so the existing analysis tools can read human games directly. Serializer
    imports are lazy: without torch the recorder disables itself gracefully.
    """

    def __init__(self, log_fn) -> None:
        self._log = log_fn
        self._enabled: bool | None = None
        self._codec = None
        self._encode_observation = None
        self._serialize_action = None
        self._serialize_snapshot = None
        self._rows: list[dict] = []
        self._game_info: dict = {}

    def _ensure_ready(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        try:
            from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
            from splendor_ai.training.replay import serialize_action, serialize_state_snapshot

            self._codec = ActionCodec()
            self._encode_observation = encode_public_observation_tensor
            self._serialize_action = serialize_action
            self._serialize_snapshot = serialize_state_snapshot
            self._enabled = True
        except ImportError as exc:
            self._enabled = False
            self._log(f"Game recording disabled (missing dependency): {exc}")
        return self._enabled

    def start_game(self, seed: int, human_seat: int, opponent_kind: str, opponent_bot, model_path: str) -> None:
        self._rows = []
        if not self._ensure_ready():
            return
        opponent_meta: dict = {"bot_class": type(opponent_bot).__name__, "gui_opponent_kind": opponent_kind}
        if opponent_kind == "model" and model_path:
            opponent_meta["checkpoint_path"] = model_path
        human_meta = {"bot_class": "HumanGuiPlayer", "gui_opponent_kind": "human"}
        seat_names = ["HumanGuiPlayer", "HumanGuiPlayer"]
        seat_meta = [dict(human_meta), dict(human_meta)]
        seat_names[1 - human_seat] = type(opponent_bot).__name__
        seat_meta[1 - human_seat] = opponent_meta
        self._game_info = {
            "seed": seed,
            "human_seat": human_seat,
            "opponent_kind": opponent_kind,
            "model_path": model_path if opponent_kind == "model" else "",
            "bot_seats": seat_names,
            "bot_metadata": seat_meta,
        }

    def record_step(self, env, state, action) -> None:
        if not self._enabled or not self._game_info:
            return
        try:
            player_id = state.current_player
            legal_actions = env.legal_actions(state)
            self._rows.append(
                {
                    "seed": self._game_info["seed"],
                    "turn_index": state.turn_index,
                    "player_id": player_id,
                    "action_index": self._codec.encode(state, action),
                    "action_payload": self._serialize_action(action),
                    "legal_action_indices": list(self._codec.legal_action_indices(state, legal_actions)),
                    "observation_vector": list(self._encode_observation(state, player_id).vector),
                    "state_snapshot": self._serialize_snapshot(state),
                }
            )
        except Exception as exc:  # pragma: no cover - recording must never break play
            self._log(f"Recording error on step (game will not be saved): {exc}")
            self._rows = []
            self._game_info = {}

    def finalize(self, final_state) -> None:
        if not self._enabled or not self._game_info or not self._rows:
            return
        try:
            winner = final_state.winner
            info = self._game_info
            game_fields = {
                "game_seed": info["seed"],
                "game_turns": final_state.turn_index,
                "game_winner": winner,
                "game_final_scores": [final_state.players[0].score, final_state.players[1].score],
                "game_stalled": False,
                "game_timed_out": False,
                "game_termination_reason": "completed",
                "game_bot_seats": info["bot_seats"],
                "game_bot_metadata": info["bot_metadata"],
                "game_loop_fallback_triggers_by_seat": [0, 0],
                "game_model_loop_fallback_triggers": 0,
                "human_player_id": info["human_seat"],
                "recorded_by": "gui",
            }
            HUMAN_GAMES_DIR.mkdir(parents=True, exist_ok=True)
            replay_path = HUMAN_GAMES_DIR / "replays.jsonl"
            with replay_path.open("a", encoding="utf-8") as handle:
                for row in self._rows:
                    player_id = row["player_id"]
                    final_value = 0.0 if winner is None else (1.0 if winner == player_id else -1.0)
                    payload = {**game_fields, **row, "final_value": final_value, "winner": winner}
                    handle.write(json.dumps(payload))
                    handle.write("\n")

            human_won = winner is not None and winner == info["human_seat"]
            summary = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "seed": info["seed"],
                "opponent_kind": info["opponent_kind"],
                "model_path": info["model_path"],
                "human_seat": info["human_seat"],
                "final_scores": game_fields["game_final_scores"],
                "winner": winner,
                "human_won": human_won,
                "turns": final_state.turn_index,
                "moves_recorded": len(self._rows),
            }
            summary_path = HUMAN_GAMES_DIR / "games_summary.jsonl"
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary))
                handle.write("\n")
            self._log(f"Game recorded: {len(self._rows)} moves -> {replay_path}")
        except Exception as exc:  # pragma: no cover - recording must never break play
            self._log(f"Recording error on save: {exc}")
        finally:
            self._rows = []
            self._game_info = {}

    def discard(self) -> None:
        if self._rows:
            self._log(f"Discarded {len(self._rows)} unrecorded moves from unfinished game.")
        self._rows = []
        self._game_info = {}


class SplendorGUI:
    def __init__(
        self,
        root: tk.Tk,
        *,
        opponent: str = "greedy",
        model_path: str = "",
        seed: str = "0",
        human_seat: str = "0",
    ) -> None:
        self.root = root
        self.root.title("Splendor AI Sandbox")
        self.root.geometry("1600x1000")
        self.root.configure(bg=art.TABLE_FELT_EDGE)

        self.env: SplendorEnv | None = None
        self.state: SplendorState | None = None
        self.human_seat = 0
        self.bot = GreedyHeuristicBot(seed=0)
        self.pending_variant_actions: list[Action] = []
        self.pending_return_actions: list[Action] | None = None
        self.return_pick_counts: dict[str, int] = {}
        self.take_pick_counts: dict[str, int] = {}
        self.confirm_gold_purchases = True
        self.recorder = _GuiGameRecorder(self._log)

        self.seed_var = tk.StringVar(value=seed)
        self.bot_kind_var = tk.StringVar(value=opponent)
        self.human_seat_var = tk.StringVar(value=human_seat)
        self.model_path_var = tk.StringVar(value=model_path)
        self.status_var = tk.StringVar(value="Create a game to start.")
        self.detail_var = tk.StringVar(value="Click a card or move to inspect details.")

        self._build_layout()
        self._new_game()

    def _build_layout(self) -> None:
        top = tk.Frame(self.root, bg=TOP_BAR_BG, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Seed", bg=TOP_BAR_BG).pack(side="left")
        tk.Entry(top, textvariable=self.seed_var, width=8).pack(side="left", padx=(4, 12))

        tk.Label(top, text="Opponent", bg=TOP_BAR_BG).pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.bot_kind_var,
            values=("greedy", "random", "search", "model"),
            width=10,
            state="readonly",
        ).pack(side="left", padx=(4, 12))

        tk.Label(top, text="You are seat", bg=TOP_BAR_BG).pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.human_seat_var,
            values=("0", "1"),
            width=4,
            state="readonly",
        ).pack(side="left", padx=(4, 12))

        tk.Button(top, text="Model file...", command=self._choose_model_file).pack(side="left")
        tk.Entry(top, textvariable=self.model_path_var, width=42).pack(side="left", padx=8)
        tk.Button(top, text="New Game", command=self._new_game).pack(side="left", padx=12)

        tk.Label(
            top,
            textvariable=self.status_var,
            bg=TOP_BAR_BG,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", fill="x", expand=True, padx=12)

        body = tk.Frame(self.root, bg=art.TABLE_FELT_EDGE)
        body.pack(fill="both", expand=True)

        # Scrollable felt table on the left.
        table_holder = tk.Frame(body, bg=art.TABLE_FELT_EDGE)
        table_holder.pack(side="left", fill="both", expand=True)
        self.table_canvas = tk.Canvas(table_holder, bg=art.TABLE_FELT, highlightthickness=0)
        table_scroll = tk.Scrollbar(table_holder, orient="vertical", command=self.table_canvas.yview)
        self.table_canvas.configure(yscrollcommand=table_scroll.set)
        table_scroll.pack(side="right", fill="y")
        self.table_canvas.pack(side="left", fill="both", expand=True)
        self.left_panel = tk.Frame(self.table_canvas, bg=art.TABLE_FELT, padx=14, pady=10)
        self._table_window = self.table_canvas.create_window((0, 0), window=self.left_panel, anchor="nw")
        self.left_panel.bind(
            "<Configure>",
            lambda _e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all")),
        )
        self.table_canvas.bind(
            "<Configure>",
            lambda e: self.table_canvas.itemconfigure(self._table_window, width=e.width),
        )
        self.table_canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        self.right_panel = tk.Frame(body, bg=PANEL_BG, width=420, padx=10, pady=10)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        def felt_section(title: str) -> tk.Frame:
            holder = tk.Frame(self.left_panel, bg=art.TABLE_FELT)
            holder.pack(fill="x", pady=(0, 6))
            tk.Label(
                holder,
                text=title,
                bg=art.TABLE_FELT,
                fg="#bfe3cd",
                font=("Georgia", 10, "bold italic"),
                anchor="w",
            ).pack(fill="x")
            inner = tk.Frame(holder, bg=art.TABLE_FELT)
            inner.pack(anchor="w")
            return inner

        self.nobles_frame = felt_section("Nobles")

        self.board_frames: dict[int, tk.Frame] = {}
        for tier in (3, 2, 1):
            self.board_frames[tier] = felt_section(f"Tier {tier}")

        self.bank_frame = felt_section("Bank")

        players_row = tk.Frame(self.left_panel, bg=art.TABLE_FELT)
        players_row.pack(fill="both", expand=True, pady=(4, 0))
        self.player_area_frames: list[tk.Frame] = []
        self.player_area_titles: list[tk.Label] = []
        for seat in range(2):
            holder = tk.Frame(
                players_row,
                bg=art.TABLE_FELT_DARK,
                highlightbackground="#4e8a6b",
                highlightthickness=1,
                padx=8,
                pady=6,
            )
            holder.grid(row=0, column=seat, sticky="nsew", padx=(0 if seat == 0 else 8, 0))
            players_row.grid_columnconfigure(seat, weight=1)
            title = tk.Label(
                holder,
                text=f"Player {seat}",
                bg=art.TABLE_FELT_DARK,
                fg="#f2e5c4",
                font=("Georgia", 11, "bold"),
                anchor="w",
            )
            title.pack(fill="x")
            inner = tk.Frame(holder, bg=art.TABLE_FELT_DARK)
            inner.pack(fill="x", anchor="w")
            self.player_area_frames.append(inner)
            self.player_area_titles.append(title)

        tk.Label(self.right_panel, text="Inspector", bg=PANEL_BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            self.right_panel,
            textvariable=self.detail_var,
            bg=PANEL_CARD_BG,
            justify="left",
            anchor="w",
            wraplength=380,
            relief="ridge",
            bd=1,
            padx=8,
            pady=8,
        ).pack(fill="x", pady=(6, 12))

        tk.Label(self.right_panel, text="Quick Moves", bg=PANEL_BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.quick_moves_frame = tk.Frame(self.right_panel, bg=PANEL_BG)
        self.quick_moves_frame.pack(fill="x", pady=(6, 12))

        tk.Label(self.right_panel, text="Variants", bg=PANEL_BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        variants_holder = tk.Frame(self.right_panel, bg=PANEL_BG)
        variants_holder.pack(fill="x", pady=(6, 6))
        variants_scroll = tk.Scrollbar(variants_holder, orient="vertical")
        self.variants_list = tk.Listbox(variants_holder, height=8, yscrollcommand=variants_scroll.set)
        variants_scroll.configure(command=self.variants_list.yview)
        variants_scroll.pack(side="right", fill="y")
        self.variants_list.pack(side="left", fill="both", expand=True)
        self.variants_list.bind("<Double-Button-1>", lambda _event: self._apply_selected_variant())
        self.variants_list.bind("<<ListboxSelect>>", lambda _event: self._preview_selected_variant())
        tk.Button(self.right_panel, text="Apply Selected Move", command=self._apply_selected_variant).pack(fill="x")

        tk.Label(self.right_panel, text="Game Log", bg=PANEL_BG, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 0))
        self.log_text = tk.Text(self.right_panel, height=18, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))

    def _on_mouse_wheel(self, event) -> None:
        """Scroll the felt table, unless the pointer is over the right panel."""

        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self.right_panel:
                return
            widget = widget.master
        self.table_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _choose_model_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def _new_game(self) -> None:
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be an integer.")
            return

        try:
            self.bot = self._build_bot()
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("Bot load error", str(exc))
            return

        self.human_seat = int(self.human_seat_var.get())
        self.env = SplendorEnv(seed=seed)
        self.state = self.env.initial_state()
        self.pending_variant_actions = []
        self.pending_return_actions = None
        self.return_pick_counts = {}
        self.take_pick_counts = {}
        self.recorder.discard()
        self._clear_log()
        self._log_recent_games()
        self.recorder.start_game(
            seed,
            self.human_seat,
            self.bot_kind_var.get(),
            self.bot,
            self.model_path_var.get().strip(),
        )
        self._set_detail("New game created. Click a board card, noble, or move button to inspect details.")
        self._log(
            f"New game started. Human seat={self.human_seat}. "
            f"Opponent={self.bot_kind_var.get()} ({type(self.bot).__name__}). Seed={seed}."
        )
        self._refresh()
        self.root.after(100, self._maybe_run_bot_turn)

    def _log_recent_games(self) -> None:
        """Summarize recorded games so the next seed/seat is easy to choose."""

        try:
            summary_path = HUMAN_GAMES_DIR / "games_summary.jsonl"
            if not summary_path.exists():
                return
            games = [
                json.loads(line)
                for line in summary_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not games:
                return
            def record(subset: list[dict]) -> str:
                won = sum(1 for g in subset if g.get("human_won"))
                return f"{won}W-{len(subset) - won}L"

            seat0_games = [g for g in games if g.get("human_seat") == 0]
            seat1_games = [g for g in games if g.get("human_seat") != 0]
            self._log(
                f"Recorded so far: {len(games)} game(s) — your record {record(games)} "
                f"(seat 0: {record(seat0_games)} in {len(seat0_games)}, "
                f"seat 1: {record(seat1_games)} in {len(seat1_games)})."
            )
            last = games[-1]
            opponent = last.get("opponent_kind", "?")
            if opponent == "model" and last.get("model_path"):
                opponent = f"model {Path(last['model_path']).parent.name}"
            scores = last.get("final_scores", ["?", "?"])
            seat = last.get("human_seat", 0)
            you, them = scores[seat], scores[1 - seat]
            outcome = "won" if last.get("human_won") else ("drew" if last.get("winner") is None else "lost")
            self._log(
                f"Last game: seed {last.get('seed')}, seat {seat}, vs {opponent} — "
                f"{outcome} {you}-{them} in {last.get('turns')} turns."
            )
            seeds = sorted({g.get("seed") for g in games})
            self._log(f"Seeds already played: {', '.join(str(s) for s in seeds)}.")
        except Exception as exc:  # pragma: no cover - advisory only, never block startup
            self._log(f"(Could not read recorded-games history: {exc})")

    def _build_bot(self):
        bot_kind = self.bot_kind_var.get()
        if bot_kind == "greedy":
            return GreedyHeuristicBot(seed=1)
        if bot_kind == "random":
            return RandomLegalBot(seed=1)
        if bot_kind == "search":
            try:
                from splendor_ai.bots.search_bot import ShallowSearchBot
            except ImportError as exc:
                raise ValueError(f"Search opponent needs the full environment (torch): {exc}") from exc
            return ShallowSearchBot(seed=1)
        if bot_kind == "model":
            model_path = self.model_path_var.get().strip()
            if not model_path:
                raise ValueError("Select a model checkpoint before choosing the model opponent.")
            return self._load_model_bot(model_path)
        raise ValueError(f"Unsupported bot type: {bot_kind}")

    def _load_model_bot(self, model_path: str):
        """Load a checkpoint with the right bot class, detected from its config."""

        try:
            import torch

            from splendor_ai.bots.model_bot import (
                CheckpointPolicyBot,
                LegalActionAttentionScorerBot,
                LegalActionScorerBot,
                LegalActionSetScorerBot,
            )
        except ImportError as exc:
            raise ValueError(f"Model opponent needs the full environment (torch): {exc}") from exc

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("model_config", {})
        if "attention_layers" in config:
            kind, bot_class = "legal_action_attention_scorer", LegalActionAttentionScorerBot
        elif "set_context_hidden_size" in config:
            kind, bot_class = "legal_action_set_scorer", LegalActionSetScorerBot
        elif "action_hidden_size" in config:
            kind, bot_class = "legal_action_scorer", LegalActionScorerBot
        else:
            kind, bot_class = "flat_policy", CheckpointPolicyBot
        self._log(f"Model checkpoint detected as {kind}.")
        return bot_class(model_path, device="cpu")

    def _refresh(self) -> None:
        if self.state is None or self.env is None:
            return

        state = self.state
        self.status_var.set(self._status_text(state))
        self._render_bank(state)
        self._render_nobles(state)
        self._render_board(state)
        self._render_players(state)
        self._render_quick_moves(state)
        self._render_variants([])

    def _status_text(self, state: SplendorState) -> str:
        if state.terminal:
            if state.winner is None:
                return f"Game over: draw / unresolved tie. Scores {state.players[0].score}-{state.players[1].score}"
            return f"Game over: player {state.winner} wins. Scores {state.players[0].score}-{state.players[1].score}"
        current = "you" if state.current_player == self.human_seat else "bot"
        return (
            f"Turn {state.turn_index} | current player {state.current_player} ({current}) | "
            f"scores {state.players[0].score}-{state.players[1].score}"
        )

    def _render_bank(self, state: SplendorState) -> None:
        for widget in self.bank_frame.winfo_children():
            widget.destroy()
        chip = art.CHIP_SIZE
        pad = 12
        canvas = tk.Canvas(
            self.bank_frame,
            width=(chip + pad) * len(TOKEN_ORDER) + pad,
            height=chip + 22,
            bg=art.TABLE_FELT,
            highlightthickness=0,
        )
        canvas.pack(anchor="w")
        for index, color in enumerate(TOKEN_ORDER):
            count = state.bank_tokens[color]
            cx = pad + chip / 2 + index * (chip + pad)
            cy = 8 + chip / 2
            if count > 0:
                art.draw_token_chip(canvas, cx, cy, chip / 2, color, count=count)
            else:
                canvas.create_oval(
                    cx - chip / 2, cy - chip / 2, cx + chip / 2, cy + chip / 2,
                    outline="#4e8a6b", dash=(3, 3),
                )
                canvas.create_text(cx, cy, text="0", fill="#7fb598", font=("Segoe UI", 10, "bold"))

    def _render_nobles(self, state: SplendorState) -> None:
        for widget in self.nobles_frame.winfo_children():
            widget.destroy()
        if not state.nobles:
            tk.Label(self.nobles_frame, text="No nobles left", bg=art.TABLE_FELT, fg="#bfe3cd").pack(anchor="w")
            return
        for noble in state.nobles:
            canvas = tk.Canvas(
                self.nobles_frame,
                width=art.NOBLE_SIZE,
                height=art.NOBLE_SIZE,
                bg=art.TABLE_FELT,
                highlightthickness=0,
                cursor="hand2",
            )
            canvas.pack(side="left", padx=(0, 8))
            art.draw_noble_tile(canvas, art.NOBLE_SIZE, noble.points, noble.requirement, noble.noble_id)
            canvas.bind(
                "<Button-1>",
                lambda _event, current=noble: self._set_detail(
                    f"Noble {current.noble_id}\n\nPrestige: {current.points}\nRequirements: {_format_cost(current.requirement)}"
                ),
            )

    def _render_board(self, state: SplendorState) -> None:
        for tier, frame in self.board_frames.items():
            for widget in frame.winfo_children():
                widget.destroy()

            row = tk.Frame(frame, bg=art.TABLE_FELT)
            row.pack(anchor="w")

            # Face-down deck on the left, like the physical layout.
            reserve_deck_actions = self._matching_human_actions(
                action_type=ActionType.RESERVE_DECK,
                tier=tier,
            )
            deck_cell = tk.Frame(row, bg=art.TABLE_FELT)
            deck_cell.pack(side="left", padx=(0, 14))
            deck_canvas = tk.Canvas(
                deck_cell,
                width=art.CARD_WIDTH,
                height=art.CARD_HEIGHT,
                bg=art.TABLE_FELT,
                highlightthickness=0,
                cursor="hand2" if reserve_deck_actions else "arrow",
            )
            deck_canvas.pack()
            art.draw_card_back(
                deck_canvas,
                art.CARD_WIDTH,
                art.CARD_HEIGHT,
                tier,
                state.deck_counts[tier],
                highlight=bool(reserve_deck_actions),
            )
            if reserve_deck_actions:
                deck_canvas.bind(
                    "<Button-1>",
                    lambda _event, acts=reserve_deck_actions: self._handle_action_group(acts, "Reserve deck variants"),
                )
            else:
                deck_canvas.bind(
                    "<Button-1>",
                    lambda _event, t=tier, n=state.deck_counts[tier]: self._set_detail(
                        f"Tier {t} deck\n\n{n} hidden card(s).\nReserving from the deck is not available right now."
                    ),
                )
            tk.Label(
                deck_cell,
                text="click to reserve" if reserve_deck_actions else "deck",
                bg=art.TABLE_FELT,
                fg="#eebd4e" if reserve_deck_actions else "#7fb598",
                font=("Segoe UI", 8),
            ).pack()

            cards = state.visible_tier_cards[tier]
            if not cards:
                tk.Label(row, text="No visible cards", bg=art.TABLE_FELT, fg="#bfe3cd").pack(side="left")
                continue

            for market_index, card in enumerate(cards):
                buy_actions = self._matching_human_actions(
                    action_type=ActionType.BUY_VISIBLE,
                    tier=tier,
                    market_index=market_index,
                )
                reserve_actions = self._matching_human_actions(
                    action_type=ActionType.RESERVE_VISIBLE,
                    tier=tier,
                    market_index=market_index,
                )
                highlight = "buy" if buy_actions else "reserve" if reserve_actions else None

                cell = tk.Frame(row, bg=art.TABLE_FELT)
                cell.pack(side="left", padx=(0, 8))
                canvas = tk.Canvas(
                    cell,
                    width=art.CARD_WIDTH,
                    height=art.CARD_HEIGHT,
                    bg=art.TABLE_FELT,
                    highlightthickness=0,
                    cursor="hand2",
                )
                canvas.pack()
                art.draw_card_face(
                    canvas,
                    art.CARD_WIDTH,
                    art.CARD_HEIGHT,
                    card.tier,
                    card.points,
                    card.bonus_color,
                    card.cost,
                    card.card_id,
                    highlight=highlight,
                )
                canvas.bind(
                    "<Button-1>",
                    lambda _event, current=card, buys=buy_actions, reserves=reserve_actions: self._set_detail(
                        self._card_detail_text(current, buys, reserves)
                    ),
                )

                button_row = tk.Frame(cell, bg=art.TABLE_FELT)
                button_row.pack(pady=(3, 0))
                tk.Button(
                    button_row,
                    text="Buy",
                    font=("Segoe UI", 8),
                    padx=6,
                    state="normal" if buy_actions else "disabled",
                    command=lambda acts=buy_actions: self._handle_action_group(acts, "Buy variants"),
                ).pack(side="left", padx=(0, 4))
                tk.Button(
                    button_row,
                    text="Reserve",
                    font=("Segoe UI", 8),
                    padx=6,
                    state="normal" if reserve_actions else "disabled",
                    command=lambda acts=reserve_actions: self._handle_action_group(acts, "Reserve variants"),
                ).pack(side="left")

    def _render_players(self, state: SplendorState) -> None:
        for seat, frame in enumerate(self.player_area_frames):
            for widget in frame.winfo_children():
                widget.destroy()
            player = state.players[seat]
            owner = "You" if seat == self.human_seat else "Opponent"
            marker = "  ◀ to move" if (not state.terminal and state.current_player == seat) else ""
            token_total = sum(player.tokens.values())
            token_note = f"tokens {token_total}/10"
            if token_total >= 9:
                token_note += " ⚠"
            self.player_area_titles[seat].configure(
                text=(
                    f"Player {seat} — {owner}   |   {player.score} pts   |   "
                    f"{token_note}   |   nobles {len(player.nobles)}{marker}"
                ),
                fg="#ffe9a8" if state.current_player == seat and not state.terminal else "#f2e5c4",
            )

            # Tokens and permanent bonuses drawn as chips + mini card stacks.
            summary = tk.Canvas(
                frame,
                width=430,
                height=64,
                bg=art.TABLE_FELT_DARK,
                highlightthickness=0,
            )
            summary.pack(anchor="w", pady=(4, 2))
            chip_r = 15
            x = chip_r + 4
            for color in TOKEN_ORDER:
                count = player.tokens[color]
                if count > 0:
                    art.draw_token_chip(summary, x, 22, chip_r, color, count=count, stacked=False)
                else:
                    summary.create_oval(
                        x - chip_r, 22 - chip_r, x + chip_r, 22 + chip_r,
                        outline="#4e8a6b", dash=(2, 3),
                    )
                x += 2 * chip_r + 8
            x += 10
            summary.create_text(x, 22, text="bonuses:", fill="#bfe3cd", font=("Segoe UI", 8), anchor="w")
            x += 52
            for color in BONUS_ORDER:
                count = player.bonuses[color]
                if count > 0:
                    art.draw_bonus_stack(summary, x, 10, color, count)
                else:
                    summary.create_rectangle(x, 10, x + 17, 33, outline="#4e8a6b", dash=(2, 3))
                x += 26

            reserved_holder = tk.Frame(frame, bg=art.TABLE_FELT_DARK)
            reserved_holder.pack(anchor="w", pady=(2, 0))
            tk.Label(
                reserved_holder,
                text="Reserved",
                bg=art.TABLE_FELT_DARK,
                fg="#bfe3cd",
                font=("Segoe UI", 8, "bold"),
            ).pack(anchor="w")
            cards_row = tk.Frame(reserved_holder, bg=art.TABLE_FELT_DARK)
            cards_row.pack(anchor="w")

            if not player.reserved_cards:
                tk.Label(cards_row, text="none", bg=art.TABLE_FELT_DARK, fg="#7fb598", font=("Segoe UI", 8, "italic")).pack(anchor="w")
                continue

            for reserved_index, card in enumerate(player.reserved_cards):
                is_human = seat == self.human_seat
                publicly_visible = self._reserved_card_is_public(player, reserved_index)
                cell = tk.Frame(cards_row, bg=art.TABLE_FELT_DARK)
                cell.pack(side="left", padx=(0, 6))
                canvas = tk.Canvas(
                    cell,
                    width=art.MINI_CARD_WIDTH,
                    height=art.MINI_CARD_HEIGHT,
                    bg=art.TABLE_FELT_DARK,
                    highlightthickness=0,
                    cursor="hand2" if is_human else "arrow",
                )
                canvas.pack()

                if is_human or publicly_visible:
                    buy_actions = (
                        self._matching_human_actions(
                            action_type=ActionType.BUY_RESERVED,
                            reserved_index=reserved_index,
                        )
                        if is_human
                        else []
                    )
                    art.draw_card_face(
                        canvas,
                        art.MINI_CARD_WIDTH,
                        art.MINI_CARD_HEIGHT,
                        card.tier,
                        card.points,
                        card.bonus_color,
                        card.cost,
                        card.card_id,
                        highlight="buy" if buy_actions else None,
                    )
                    if is_human:
                        canvas.bind(
                            "<Button-1>",
                            lambda _event, current=card, buys=buy_actions: self._set_detail(
                                self._reserved_card_detail_text(current, buys)
                            ),
                        )
                        tk.Button(
                            cell,
                            text="Buy",
                            font=("Segoe UI", 8),
                            padx=6,
                            state="normal" if buy_actions else "disabled",
                            command=lambda acts=buy_actions: self._handle_action_group(acts, "Reserved buy variants"),
                        ).pack(pady=(2, 0))
                    else:
                        canvas.configure(cursor="hand2")
                        canvas.bind(
                            "<Button-1>",
                            lambda _event, current=card: self._set_detail(
                                "Opponent reserved card (public: reserved from the market)\n\n"
                                f"{describe_card(current)}"
                            ),
                        )
                else:
                    art.draw_card_back(canvas, art.MINI_CARD_WIDTH, art.MINI_CARD_HEIGHT, card.tier, 1)
                    canvas.create_text(
                        art.MINI_CARD_WIDTH / 2,
                        art.MINI_CARD_HEIGHT - 16,
                        text="hidden",
                        font=("Segoe UI", 8, "italic"),
                        fill="#5c5136",
                    )

    @staticmethod
    def _reserved_card_is_public(player, reserved_index: int) -> bool:
        """True when the opponent reserved this card face-up from the market.

        Uses reserve provenance when the engine provides it; falls back to
        hidden for legacy states without provenance.
        """

        try:
            source = player.reserved_card_source(reserved_index)
        except AttributeError:
            return False
        return source == "visible"

    def _render_quick_moves(self, state: SplendorState) -> None:
        for widget in self.quick_moves_frame.winfo_children():
            widget.destroy()

        if state.terminal or state.current_player != self.human_seat:
            self.pending_return_actions = None
            tk.Label(self.quick_moves_frame, text="Waiting for opponent or game over.", bg=PANEL_BG).pack(anchor="w")
            return

        if self.pending_return_actions:
            self._render_return_picker(state)
            return

        take_actions = self._matching_human_actions(action_type=ActionType.TAKE_TOKENS)
        pass_actions = self._matching_human_actions(action_type=ActionType.PASS)
        if not take_actions and not pass_actions:
            tk.Label(self.quick_moves_frame, text="No direct turn actions.", bg=PANEL_BG).pack(anchor="w")
            return

        take_groups: dict[tuple[str, ...], list[Action]] = {}
        for action in take_actions:
            key = tuple(sorted(action.take_tokens))
            take_groups.setdefault(key, []).append(action)

        if take_groups:
            self._render_take_picker(take_groups)

        # Shortcut row: one-click "take two of one colour" buttons.
        double_groups = [
            (key[0], group) for key, group in take_groups.items()
            if len(key) == 2 and key[0] == key[1]
        ]
        if double_groups:
            row = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=(6, 0))
            tk.Label(row, text="Take 2:", bg=PANEL_BG, font=("Segoe UI", 8)).pack(side="left")
            for color, group in double_groups:
                chip_r = 10
                canvas = tk.Canvas(
                    row, width=2 * chip_r + 6, height=2 * chip_r + 6,
                    bg=PANEL_BG, highlightthickness=0, cursor="hand2",
                )
                canvas.pack(side="left", padx=3)
                art.draw_token_chip(canvas, chip_r + 3, chip_r + 3, chip_r, color, count=None, stacked=False)
                canvas.bind(
                    "<Button-1>",
                    lambda _event, acts=group: self._handle_action_group(acts, "Take variants"),
                )

        # When the bank leaves only a handful of distinct takes, list them
        # directly as one-click buttons too.
        if 0 < len(take_groups) <= 4:
            grid = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
            grid.pack(fill="x", pady=(6, 0))
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)
            for index, group in enumerate(take_groups.values()):
                button = self._make_take_button(grid, group)
                button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)

        for action in pass_actions:
            tk.Button(
                self.quick_moves_frame,
                text=describe_action(action, state),
                command=lambda chosen=action: self._apply_action(chosen),
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=(6, 2))

    def _render_take_picker(self, take_groups: dict[tuple[str, ...], list[Action]]) -> None:
        """Build-your-take chip selector; ×2 badge marks double-takeable colours."""

        maxima: dict[str, int] = {}
        for key in take_groups:
            for color, count in Counter(key).items():
                maxima[color] = max(maxima.get(color, 0), count)
        for color in list(self.take_pick_counts):
            if color not in maxima:
                self.take_pick_counts.pop(color)

        selection = Counter({c: n for c, n in self.take_pick_counts.items() if n > 0})
        matched_group = None
        extendable = False
        if selection:
            for key, group in take_groups.items():
                key_counts = Counter(key)
                if key_counts == selection:
                    matched_group = group
                elif all(key_counts.get(c, 0) >= n for c, n in selection.items()):
                    extendable = True

        tk.Label(
            self.quick_moves_frame,
            text="Take tokens — click chips to build your take:",
            bg=PANEL_BG,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x")
        chips_row = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
        chips_row.pack(pady=2)
        self._render_chip_selector(
            chips_row,
            maxima,
            self.take_pick_counts,
            self._cycle_take_pick,
            ring_color="#2b8a3e",
            badge_double=True,
        )

        buttons = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
        buttons.pack(fill="x")
        needs_return = matched_group is not None and bool(matched_group[0].return_tokens)
        confirm_text = "Take (then choose returns)" if needs_return else "Take"
        tk.Button(
            buttons,
            text=confirm_text,
            state="normal" if matched_group is not None else "disabled",
            command=lambda: self._handle_action_group(matched_group or [], "Take variants"),
        ).pack(side="left", padx=(0, 6))
        if selection:
            tk.Button(buttons, text="Reset", command=self._reset_take_pick).pack(side="left")
        if selection and matched_group is None:
            if extendable:
                hint, color = "Keep picking…", "#666666"
            else:
                hint, color = "Not a legal take right now — adjust the selection.", "#a33"
            tk.Label(
                self.quick_moves_frame,
                text=hint,
                bg=PANEL_BG,
                font=("Segoe UI", 8, "italic"),
                fg=color,
                anchor="w",
            ).pack(fill="x")

    def _render_chip_selector(
        self,
        parent: tk.Frame,
        maxima: dict[str, int],
        counts: dict[str, int],
        on_click,
        ring_color: str,
        badge_double: bool = False,
    ) -> None:
        chip_r = 15
        for color in TOKEN_ORDER:
            if color not in maxima:
                continue
            counts.setdefault(color, 0)
            cell = tk.Frame(parent, bg=PANEL_BG)
            cell.pack(side="left", padx=4)
            canvas = tk.Canvas(
                cell, width=2 * chip_r + 6, height=2 * chip_r + 6,
                bg=PANEL_BG, highlightthickness=0, cursor="hand2",
            )
            canvas.pack()
            art.draw_token_chip(canvas, chip_r + 3, chip_r + 3, chip_r, color, count=None, stacked=False)
            picked = counts[color]
            if picked > 0:
                canvas.create_oval(2, 2, 2 * chip_r + 4, 2 * chip_r + 4, outline=ring_color, width=3)
            if badge_double and maxima[color] >= 2:
                canvas.create_oval(2 * chip_r - 9, 0, 2 * chip_r + 6, 14, fill="#f2e5c4", outline="#8a6d1f")
                canvas.create_text(
                    2 * chip_r - 1, 7, text="×2",
                    font=("Segoe UI", 7, "bold"), fill="#6d5313",
                )
            label = f"{picked}" if not badge_double else (f"{picked}" if picked else "")
            tk.Label(
                cell,
                text=label if label else " ",
                bg=PANEL_BG,
                font=("Segoe UI", 8, "bold"),
                fg=ring_color,
            ).pack()
            canvas.bind(
                "<Button-1>",
                lambda _event, c=color, m=maxima[color]: on_click(c, m),
            )

    def _cycle_take_pick(self, color: str, maximum: int) -> None:
        current = self.take_pick_counts.get(color, 0)
        self.take_pick_counts[color] = 0 if current >= maximum else current + 1
        if self.state is not None:
            self._render_quick_moves(self.state)

    def _reset_take_pick(self) -> None:
        self.take_pick_counts = {}
        if self.state is not None:
            self._render_quick_moves(self.state)

    def _make_take_button(self, parent: tk.Frame, group: list[Action]) -> tk.Frame:
        """One take-set button with gem chips; opens the return picker if needed."""

        holder = tk.Frame(parent, bg=PANEL_CARD_BG, relief="raised", bd=1, cursor="hand2")
        chip_r = 8
        take = list(group[0].take_tokens)
        width = 36 + len(take) * (2 * chip_r + 4) + 6
        canvas = tk.Canvas(holder, width=width, height=2 * chip_r + 6, bg=PANEL_CARD_BG, highlightthickness=0)
        canvas.pack(side="left", padx=(4, 0), pady=2)
        canvas.create_text(3, chip_r + 3, text="Take", anchor="w", font=("Segoe UI", 8))
        x = 36 + chip_r
        for color in take:
            art.draw_gem(canvas, x, chip_r + 3, chip_r, color)
            x += 2 * chip_r + 4
        hint = ""
        if group[0].return_tokens:
            hint = f"→ return {len(group[0].return_tokens)}"
        label = tk.Label(holder, text=hint, bg=PANEL_CARD_BG, font=("Segoe UI", 8), fg="#8a6d1f", anchor="w")
        label.pack(side="left", padx=2)
        for widget in (holder, canvas, label):
            widget.bind(
                "<Button-1>",
                lambda _event, acts=group: self._handle_action_group(acts, "Take variants"),
            )
        return holder

    @staticmethod
    def _differs_only_by_returns(actions: list[Action]) -> bool:
        signatures = {
            (
                action.action_type,
                action.tier,
                action.market_index,
                action.reserved_index,
                tuple(action.take_tokens),
                tuple(action.spend_tokens),
                action.noble_id,
            )
            for action in actions
        }
        return len(signatures) == 1

    def _base_move_text(self, action: Action) -> str:
        if action.action_type == ActionType.TAKE_TOKENS:
            return f"Take {' '.join(TOKEN_SHORT[c] for c in action.take_tokens)}"
        if action.action_type == ActionType.RESERVE_VISIBLE:
            card = self.state.visible_tier_cards[action.tier][action.market_index]
            gold = " (+gold)" if action.take_tokens else ""
            return f"Reserve {card.card_id}{gold}"
        if action.action_type == ActionType.RESERVE_DECK:
            gold = " (+gold)" if action.take_tokens else ""
            return f"Reserve deck T{action.tier}{gold}"
        return describe_action(action, self.state)

    def _open_return_picker(self, actions: list[Action]) -> None:
        self.pending_return_actions = list(actions)
        self.return_pick_counts = {}
        self._set_detail(
            "You are over the 10-token limit for this move.\n\n"
            "Step 2: click token chips to choose what to return, then Confirm."
        )
        if self.state is not None:
            self._render_quick_moves(self.state)

    def _render_return_picker(self, state: SplendorState) -> None:
        actions = self.pending_return_actions or []
        if not actions:
            return
        required = len(actions[0].return_tokens)
        maxima: dict[str, int] = {}
        for action in actions:
            for color, count in Counter(action.return_tokens).items():
                maxima[color] = max(maxima.get(color, 0), count)
        for color in maxima:
            self.return_pick_counts.setdefault(color, 0)

        selected_total = sum(self.return_pick_counts.values())
        player = state.players[self.human_seat]
        take_counts = Counter(actions[0].take_tokens)
        total_after = sum(player.tokens.values()) + len(actions[0].take_tokens) - required
        tk.Label(
            self.quick_moves_frame,
            text=f"{self._base_move_text(actions[0])}",
            bg=PANEL_BG,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            self.quick_moves_frame,
            text=(
                f"Step 2: return {required} token(s) — selected {selected_total}/{required} "
                f"(you will hold {total_after}/10)"
            ),
            bg=PANEL_BG,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x")

        chips_row = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
        chips_row.pack(pady=4)
        chip_r = 15
        for color in TOKEN_ORDER:
            if color not in maxima:
                continue
            cell = tk.Frame(chips_row, bg=PANEL_BG)
            cell.pack(side="left", padx=4)
            canvas = tk.Canvas(
                cell, width=2 * chip_r + 6, height=2 * chip_r + 6,
                bg=PANEL_BG, highlightthickness=0, cursor="hand2",
            )
            canvas.pack()
            art.draw_token_chip(canvas, chip_r + 3, chip_r + 3, chip_r, color, count=None, stacked=False)
            picked = self.return_pick_counts[color]
            if picked > 0:
                canvas.create_oval(2, 2, 2 * chip_r + 4, 2 * chip_r + 4, outline="#c92a2a", width=3)
            balance_after = player.tokens[color] + take_counts.get(color, 0) - picked
            tk.Label(
                cell,
                text=f"{picked}/{maxima[color]}\nkeep {balance_after}",
                bg=PANEL_BG,
                font=("Segoe UI", 8, "bold"),
                fg="#c92a2a" if picked else "#555555",
                justify="center",
            ).pack()
            canvas.bind(
                "<Button-1>",
                lambda _event, c=color, m=maxima[color]: self._cycle_return_pick(c, m),
            )

        target = Counter({c: n for c, n in self.return_pick_counts.items() if n > 0})
        matched = next(
            (a for a in actions if Counter(a.return_tokens) == target),
            None,
        )
        buttons = tk.Frame(self.quick_moves_frame, bg=PANEL_BG)
        buttons.pack(fill="x", pady=(4, 0))
        tk.Button(
            buttons,
            text="Confirm return",
            state="normal" if matched is not None else "disabled",
            command=lambda: self._apply_action(matched) if matched is not None else None,
        ).pack(side="left", padx=(0, 6))
        tk.Button(buttons, text="Cancel", command=self._cancel_return_picker).pack(side="left")
        if selected_total == required and matched is None:
            tk.Label(
                self.quick_moves_frame,
                text="That exact combination is not legal here — adjust the selection.",
                bg=PANEL_BG,
                font=("Segoe UI", 8, "italic"),
                fg="#a33",
                anchor="w",
            ).pack(fill="x")

    def _cycle_return_pick(self, color: str, maximum: int) -> None:
        current = self.return_pick_counts.get(color, 0)
        self.return_pick_counts[color] = 0 if current >= maximum else current + 1
        if self.state is not None:
            self._render_quick_moves(self.state)

    def _cancel_return_picker(self) -> None:
        self.pending_return_actions = None
        self.return_pick_counts = {}
        self._set_detail("Move cancelled. Pick another move.")
        if self.state is not None:
            self._render_quick_moves(self.state)

    def _render_variants(self, actions: list[Action], title: str | None = None) -> None:
        self.pending_variant_actions = list(actions)
        self.variants_list.delete(0, tk.END)
        if title:
            self._log(title)
        if self.state is None:
            return
        for action in actions:
            self.variants_list.insert(tk.END, describe_action(action, self.state))
        if actions:
            self.variants_list.selection_set(0)
            self._set_detail(
                "Multiple legal variants exist for this move.\n\n"
                f"Selected preview:\n{describe_action(actions[0], self.state)}"
            )

    def _matching_human_actions(self, **criteria) -> list[Action]:
        if self.state is None or self.env is None or self.state.current_player != self.human_seat:
            return []
        actions = self.env.legal_actions(self.state)
        matched: list[Action] = []
        for action in actions:
            if all(getattr(action, key) == value for key, value in criteria.items()):
                matched.append(action)
        return matched

    def _handle_action_group(self, actions: list[Action], title: str) -> None:
        if not actions:
            return
        if len(actions) == 1:
            self._apply_action(actions[0])
            return
        if self._differs_only_by_returns(actions):
            self._open_return_picker(actions)
            return
        self._render_variants(actions, title=title)

    def _apply_selected_variant(self) -> None:
        selection = self.variants_list.curselection()
        if not selection or not self.pending_variant_actions:
            return
        self._apply_action(self.pending_variant_actions[selection[0]])

    def _preview_selected_variant(self) -> None:
        selection = self.variants_list.curselection()
        if not selection or not self.pending_variant_actions or self.state is None:
            return
        action = self.pending_variant_actions[selection[0]]
        self._set_detail(f"Selected variant\n\n{describe_action(action, self.state)}")

    def _apply_action(self, action: Action) -> None:
        if self.state is None or self.env is None:
            return
        if (
            self.confirm_gold_purchases
            and action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
            and "gold" in action.spend_tokens
        ):
            gold_count = list(action.spend_tokens).count("gold")
            plural = "s" if gold_count > 1 else ""
            if not messagebox.askyesno(
                "Spend gold?",
                f"This purchase will use {gold_count} gold (joker) token{plural}. "
                "Are you sure you want to proceed?",
            ):
                self._set_detail("Purchase cancelled — gold not spent.")
                return
        old_player = self.state.current_player
        self.pending_return_actions = None
        self.return_pick_counts = {}
        self.take_pick_counts = {}
        self._set_detail(f"Applying move\n\n{describe_action(action, self.state)}")
        self._log(f"Player {old_player}: {describe_action(action, self.state)}")
        self.recorder.record_step(self.env, self.state, action)
        self.state = self.env.step(self.state, action)
        self._refresh()
        if self.state.terminal:
            self._log(self._status_text(self.state))
            self.recorder.finalize(self.state)
            return
        self.root.after(80, self._maybe_run_bot_turn)

    def _maybe_run_bot_turn(self) -> None:
        if self.state is None or self.env is None or self.state.terminal:
            return
        if self.state.current_player == self.human_seat:
            return

        legal_actions = self.env.legal_actions(self.state)
        if not legal_actions:
            self._log("No legal actions available for bot under current rule set.")
            self.status_var.set("Stalled state reached: no legal actions available.")
            return

        action = self.bot.choose_action(self.env, self.state, legal_actions)
        if action is None:
            self._log("Bot returned no action.")
            return
        self._set_detail(f"Bot chose\n\n{describe_action(action, self.state)}")
        self._log(f"Bot (player {self.state.current_player}): {describe_action(action, self.state)}")
        self.recorder.record_step(self.env, self.state, action)
        self.state = self.env.step(self.state, action)
        self._refresh()
        if self.state.terminal:
            self._log(self._status_text(self.state))
            self.recorder.finalize(self.state)

    def _token_line(self, tokens: dict[str, int]) -> str:
        return " ".join(f"{TOKEN_SHORT[color]}:{tokens[color]}" for color in TOKEN_ORDER)

    def _bonus_line(self, bonuses: dict[str, int]) -> str:
        return " ".join(f"{TOKEN_SHORT[color]}:{bonuses[color]}" for color in BONUS_ORDER)

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _set_detail(self, text: str) -> None:
        self.detail_var.set(text)

    def _card_detail_text(self, card: Card, buy_actions: list[Action], reserve_actions: list[Action]) -> str:
        lines = [
            f"Board card {card.card_id}",
            f"Tier: {card.tier}",
            f"Prestige: {card.points}",
            f"Bonus: {art.GEM_NAME[card.bonus_color]}",
            f"Cost: {_format_cost(card.cost)}",
            "",
            f"Buy variants available now: {len(buy_actions)}",
            f"Reserve variants available now: {len(reserve_actions)}",
        ]
        if buy_actions:
            lines.append(f"Example buy: {describe_action(buy_actions[0], self.state)}")
        if reserve_actions:
            lines.append(f"Example reserve: {describe_action(reserve_actions[0], self.state)}")
        return "\n".join(lines)

    def _reserved_card_detail_text(self, card: Card, buy_actions: list[Action]) -> str:
        lines = [
            f"Reserved card {card.card_id}",
            f"Tier: {card.tier}",
            f"Prestige: {card.points}",
            f"Bonus: {art.GEM_NAME[card.bonus_color]}",
            f"Cost: {_format_cost(card.cost)}",
            "",
            f"Buy variants available now: {len(buy_actions)}",
        ]
        if buy_actions:
            lines.append(f"Example: {describe_action(buy_actions[0], self.state)}")
        return "\n".join(lines)


def launch_gui(
    opponent: str = "greedy",
    model_path: str = "",
    seed: str = "0",
    human_seat: str = "0",
) -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    SplendorGUI(
        root,
        opponent=opponent,
        model_path=model_path,
        seed=seed,
        human_seat=human_seat,
    )
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    launch_gui()
