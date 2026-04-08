"""
Stochastic Go implementation of the MCTS-FT GameInterface.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from game import (
    SGoState, SGoAction, END_TURN,
    P1, P2,
    legal_actions, apply_placement, apply_end_turn, advance_turn,
    placement_success,
)
from evaluate import win_prob as _win_prob
from mcts_ft import GameInterface


class SGoGameInterface(GameInterface):

    def legal_actions(self, state: SGoState, player: int) -> List[SGoAction]:
        return legal_actions(state)

    def is_stochastic(self, state: SGoState, action: SGoAction) -> bool:
        return action.type == "place"

    def apply_deterministic(self, state: SGoState, action: SGoAction) -> SGoState:
        assert action.type == "end_turn"
        return apply_end_turn(state)

    def dice_distribution(self, state: SGoState, action: SGoAction) -> List[Tuple[int, float]]:
        # 1D6: each face equally likely.
        return [(roll, 1 / 6) for roll in range(1, 7)]

    def apply_dice_outcome(self, state: SGoState, action: SGoAction, dice_value: int) -> SGoState:
        return apply_placement(state, action.coord, dice_value)

    def is_turn_end(self, state: SGoState, player: int) -> bool:
        return state.is_turn_end

    def is_terminal(self, state: SGoState) -> bool:
        return state.is_terminal()

    def current_player(self, state: SGoState) -> int:
        return state.current_player

    def hash_state(self, state: SGoState) -> int:
        return state.hash()

    def win_prob(self, state: SGoState) -> float:
        return _win_prob(state)

    def advance_turn(self, state: SGoState) -> SGoState:
        return advance_turn(state)

    def opponent(self, player: int) -> int:
        return P2 if player == P1 else P1
