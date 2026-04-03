"""Tkinter GUI for manual Splendor play against bots."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from splendor_ai.bots import CheckpointPolicyBot, GreedyHeuristicBot, RandomLegalBot, ShallowSearchBot
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, Noble, SplendorState


TOKEN_ORDER = ("white", "blue", "green", "red", "black", "gold")
TOKEN_SHORT = {
    "white": "W",
    "blue": "U",
    "green": "G",
    "red": "R",
    "black": "B",
    "gold": "Au",
}
TOKEN_COLOR = {
    "white": "#f3f0e6",
    "blue": "#2d6cdf",
    "green": "#2b8a3e",
    "red": "#c92a2a",
    "black": "#212529",
    "gold": "#c99700",
}
BONUS_ORDER = ("white", "blue", "green", "red", "black")


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


class SplendorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Splendor AI Sandbox")
        self.root.geometry("1560x960")
        self.root.configure(bg="#e9e1d2")

        self.env: SplendorEnv | None = None
        self.state: SplendorState | None = None
        self.human_seat = 0
        self.bot = GreedyHeuristicBot(seed=0)
        self.pending_variant_actions: list[Action] = []

        self.seed_var = tk.StringVar(value="0")
        self.bot_kind_var = tk.StringVar(value="greedy")
        self.human_seat_var = tk.StringVar(value="0")
        self.model_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Create a game to start.")
        self.detail_var = tk.StringVar(value="Click a card or move to inspect details.")

        self._build_layout()
        self._new_game()

    def _build_layout(self) -> None:
        top = tk.Frame(self.root, bg="#d9ceb8", padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Seed", bg="#d9ceb8").pack(side="left")
        tk.Entry(top, textvariable=self.seed_var, width=8).pack(side="left", padx=(4, 12))

        tk.Label(top, text="Opponent", bg="#d9ceb8").pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.bot_kind_var,
            values=("greedy", "random", "search", "model"),
            width=10,
            state="readonly",
        ).pack(side="left", padx=(4, 12))

        tk.Label(top, text="You are seat", bg="#d9ceb8").pack(side="left")
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
            bg="#d9ceb8",
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=12)

        body = tk.Frame(self.root, bg="#e9e1d2")
        body.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(body, bg="#e9e1d2")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right_panel = tk.Frame(body, bg="#efe8da", width=420, padx=10, pady=10)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)

        self.bank_frame = tk.LabelFrame(self.left_panel, text="Bank", bg="#efe8da", padx=8, pady=8)
        self.bank_frame.pack(fill="x", pady=(0, 10))

        self.nobles_frame = tk.LabelFrame(self.left_panel, text="Nobles", bg="#efe8da", padx=8, pady=8)
        self.nobles_frame.pack(fill="x", pady=(0, 10))

        self.board_frames: dict[int, tk.LabelFrame] = {}
        for tier in (3, 2, 1):
            frame = tk.LabelFrame(
                self.left_panel,
                text=f"Tier {tier}",
                bg="#efe8da",
                padx=8,
                pady=8,
            )
            frame.pack(fill="x", pady=(0, 10))
            self.board_frames[tier] = frame

        self.players_frame = tk.Frame(self.left_panel, bg="#e9e1d2")
        self.players_frame.pack(fill="both", expand=True)

        self.player_area_frames: list[tk.LabelFrame] = []
        for seat in range(2):
            frame = tk.LabelFrame(
                self.players_frame,
                text=f"Player {seat}",
                bg="#efe8da",
                padx=8,
                pady=8,
            )
            frame.pack(fill="x", pady=(0, 10))
            self.player_area_frames.append(frame)

        tk.Label(self.right_panel, text="Quick Moves", bg="#efe8da", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            self.right_panel,
            textvariable=self.detail_var,
            bg="#f8f3e9",
            justify="left",
            anchor="w",
            wraplength=380,
            relief="ridge",
            bd=1,
            padx=8,
            pady=8,
        ).pack(fill="x", pady=(6, 12))

        tk.Label(self.right_panel, text="Quick Moves", bg="#efe8da", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.quick_moves_frame = tk.Frame(self.right_panel, bg="#efe8da")
        self.quick_moves_frame.pack(fill="x", pady=(6, 12))

        tk.Label(self.right_panel, text="Variants", bg="#efe8da", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.variants_list = tk.Listbox(self.right_panel, height=12)
        self.variants_list.pack(fill="x", pady=(6, 6))
        self.variants_list.bind("<Double-Button-1>", lambda _event: self._apply_selected_variant())
        self.variants_list.bind("<<ListboxSelect>>", lambda _event: self._preview_selected_variant())
        tk.Button(self.right_panel, text="Apply Selected Move", command=self._apply_selected_variant).pack(fill="x")

        tk.Label(self.right_panel, text="Game Log", bg="#efe8da", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 0))
        self.log_text = tk.Text(self.right_panel, height=22, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))

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
        self._clear_log()
        self._set_detail("New game created. Click a board card, noble, or move button to inspect details.")
        self._log(f"New game started. Human seat={self.human_seat}. Opponent={self.bot_kind_var.get()}. Seed={seed}.")
        self._refresh()
        self.root.after(100, self._maybe_run_bot_turn)

    def _build_bot(self):
        bot_kind = self.bot_kind_var.get()
        if bot_kind == "greedy":
            return GreedyHeuristicBot(seed=1)
        if bot_kind == "random":
            return RandomLegalBot(seed=1)
        if bot_kind == "search":
            return ShallowSearchBot(seed=1)
        if bot_kind == "model":
            model_path = self.model_path_var.get().strip()
            if not model_path:
                raise ValueError("Select a model checkpoint before choosing the model opponent.")
            return CheckpointPolicyBot(model_path, device="cpu")
        raise ValueError(f"Unsupported bot type: {bot_kind}")

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
        for color in TOKEN_ORDER:
            token = tk.Label(
                self.bank_frame,
                text=f"{TOKEN_SHORT[color]}\n{state.bank_tokens[color]}",
                bg=TOKEN_COLOR[color],
                fg="white" if color in {"blue", "green", "red", "black"} else "black",
                width=8,
                height=2,
                relief="ridge",
                bd=2,
            )
            token.pack(side="left", padx=4)

    def _render_nobles(self, state: SplendorState) -> None:
        for widget in self.nobles_frame.winfo_children():
            widget.destroy()
        if not state.nobles:
            tk.Label(self.nobles_frame, text="No nobles left", bg="#efe8da").pack(anchor="w")
            return
        for noble in state.nobles:
            label = tk.Label(
                self.nobles_frame,
                text=describe_noble(noble),
                bg="#efe8da",
                justify="left",
                anchor="w",
                cursor="hand2",
            )
            label.pack(fill="x", pady=2)
            label.bind(
                "<Button-1>",
                lambda _event, current=noble: self._set_detail(
                    f"Noble {current.noble_id}\n\nPrestige: {current.points}\nRequirements: {_format_cost(current.requirement)}"
                ),
            )

    def _render_board(self, state: SplendorState) -> None:
        for tier, frame in self.board_frames.items():
            for widget in frame.winfo_children():
                widget.destroy()
            cards = state.visible_tier_cards[tier]
            if not cards:
                tk.Label(frame, text="No visible cards", bg="#efe8da").pack(anchor="w")
                continue
            row = tk.Frame(frame, bg="#efe8da")
            row.pack(fill="x")
            for market_index, card in enumerate(cards):
                card_frame = tk.LabelFrame(
                    row,
                    text=f"Slot {market_index}",
                    bg="#f8f3e9",
                    padx=6,
                    pady=6,
                )
                card_frame.pack(side="left", padx=6, fill="y")
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

                badges: list[str] = []
                if buy_actions:
                    badges.append("BUYABLE")
                if reserve_actions:
                    badges.append("RESERVABLE")
                if not badges:
                    badges.append("VIEW")

                badge_label = tk.Label(
                    card_frame,
                    text=" | ".join(badges),
                    bg="#d7f5dd" if buy_actions else "#fff3bf" if reserve_actions else "#e9ecef",
                    anchor="w",
                    padx=4,
                )
                badge_label.pack(fill="x", pady=(0, 4))

                card_label = tk.Label(
                    card_frame,
                    text=describe_card(card),
                    bg="#f8f3e9",
                    justify="left",
                    anchor="w",
                    width=28,
                    cursor="hand2",
                )
                card_label.pack(anchor="w")
                card_label.bind(
                    "<Button-1>",
                    lambda _event, current=card, buys=buy_actions, reserves=reserve_actions: self._set_detail(
                        self._card_detail_text(current, buys, reserves)
                    ),
                )
                badge_label.bind(
                    "<Button-1>",
                    lambda _event, current=card, buys=buy_actions, reserves=reserve_actions: self._set_detail(
                        self._card_detail_text(current, buys, reserves)
                    ),
                )
                button_row = tk.Frame(card_frame, bg="#f8f3e9")
                button_row.pack(fill="x", pady=(6, 0))
                tk.Button(
                    button_row,
                    text="Buy",
                    state="normal" if buy_actions else "disabled",
                    command=lambda acts=buy_actions: self._handle_action_group(acts, "Buy variants"),
                ).pack(side="left", padx=(0, 4))
                tk.Button(
                    button_row,
                    text="Reserve",
                    state="normal" if reserve_actions else "disabled",
                    command=lambda acts=reserve_actions: self._handle_action_group(acts, "Reserve variants"),
                ).pack(side="left")

            deck_row = tk.Frame(frame, bg="#efe8da")
            deck_row.pack(fill="x", pady=(8, 0))
            reserve_deck_actions = self._matching_human_actions(
                action_type=ActionType.RESERVE_DECK,
                tier=tier,
            )
            tk.Button(
                deck_row,
                text=f"Reserve top of tier {tier} deck ({state.deck_counts[tier]} hidden)",
                state="normal" if reserve_deck_actions else "disabled",
                command=lambda acts=reserve_deck_actions: self._handle_action_group(acts, "Reserve deck variants"),
            ).pack(anchor="w")

    def _render_players(self, state: SplendorState) -> None:
        for seat, frame in enumerate(self.player_area_frames):
            for widget in frame.winfo_children():
                widget.destroy()
            player = state.players[seat]
            owner = "You" if seat == self.human_seat else "Opponent"
            frame.configure(text=f"Player {seat} ({owner})")
            header = tk.Label(
                frame,
                text=(
                    f"Score {player.score} | Tokens {self._token_line(player.tokens)} | "
                    f"Bonuses {self._bonus_line(player.bonuses)} | Nobles {len(player.nobles)}"
                ),
                bg="#efe8da",
                justify="left",
                anchor="w",
            )
            header.pack(fill="x")

            reserved_frame = tk.Frame(frame, bg="#efe8da")
            reserved_frame.pack(fill="x", pady=(6, 0))
            tk.Label(reserved_frame, text="Reserved", bg="#efe8da", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            if seat == self.human_seat:
                for reserved_index, card in enumerate(player.reserved_cards):
                    card_frame = tk.Frame(reserved_frame, bg="#f8f3e9", relief="ridge", bd=1, padx=6, pady=4)
                    card_frame.pack(fill="x", pady=2)
                    buy_actions = self._matching_human_actions(
                        action_type=ActionType.BUY_RESERVED,
                        reserved_index=reserved_index,
                    )
                    card_label = tk.Label(
                        card_frame,
                        text=describe_card(card),
                        bg="#f8f3e9",
                        justify="left",
                        anchor="w",
                        cursor="hand2",
                    )
                    card_label.pack(side="left")
                    card_label.bind(
                        "<Button-1>",
                        lambda _event, current=card, buys=buy_actions: self._set_detail(
                            self._reserved_card_detail_text(current, buys)
                        ),
                    )
                    tk.Button(
                        card_frame,
                        text="Buy reserved",
                        state="normal" if buy_actions else "disabled",
                        command=lambda acts=buy_actions: self._handle_action_group(acts, "Reserved buy variants"),
                    ).pack(side="right")
            else:
                if player.reserved_cards:
                    tk.Label(
                        reserved_frame,
                        text=f"{len(player.reserved_cards)} hidden reserved card(s)",
                        bg="#efe8da",
                    ).pack(anchor="w")
                else:
                    tk.Label(reserved_frame, text="None", bg="#efe8da").pack(anchor="w")

    def _render_quick_moves(self, state: SplendorState) -> None:
        for widget in self.quick_moves_frame.winfo_children():
            widget.destroy()

        if state.terminal or state.current_player != self.human_seat:
            tk.Label(self.quick_moves_frame, text="Waiting for opponent or game over.", bg="#efe8da").pack(anchor="w")
            return

        take_actions = self._matching_human_actions(action_type=ActionType.TAKE_TOKENS)
        pass_actions = self._matching_human_actions(action_type=ActionType.PASS)
        if not take_actions and not pass_actions:
            tk.Label(self.quick_moves_frame, text="No direct turn actions.", bg="#efe8da").pack(anchor="w")
            return

        for action in take_actions:
            tk.Button(
                self.quick_moves_frame,
                text=describe_action(action, state),
                command=lambda chosen=action: self._apply_action(chosen),
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=2)
        for action in pass_actions:
            tk.Button(
                self.quick_moves_frame,
                text=describe_action(action, state),
                command=lambda chosen=action: self._apply_action(chosen),
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=2)

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
        old_player = self.state.current_player
        self._set_detail(f"Applying move\n\n{describe_action(action, self.state)}")
        self._log(f"Player {old_player}: {describe_action(action, self.state)}")
        self.state = self.env.step(self.state, action)
        self._refresh()
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
        self.state = self.env.step(self.state, action)
        self._refresh()
        if self.state.terminal:
            self._log(self._status_text(self.state))

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
            f"Bonus: {TOKEN_SHORT[card.bonus_color]}",
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
            f"Bonus: {TOKEN_SHORT[card.bonus_color]}",
            f"Cost: {_format_cost(card.cost)}",
            "",
            f"Buy variants available now: {len(buy_actions)}",
        ]
        if buy_actions:
            lines.append(f"Example: {describe_action(buy_actions[0], self.state)}")
        return "\n".join(lines)


def launch_gui() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    SplendorGUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    launch_gui()
