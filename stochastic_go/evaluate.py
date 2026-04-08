"""
Value estimation for Stochastic Go.

win_prob(state) -> float in (0, 1): estimated probability that P1 wins from this state.
Uses a sigmoid over the stone count difference, with a steepness parameter k.
"""

import math
from game import SGoState, BOARD_SIZE, P1, P2, EMPTY

# Steepness of the sigmoid. Higher = more decisive; lower = flatter curve.
# With max possible difference of 64, k=0.1 gives a useful spread.
SIGMOID_K = 0.15


def win_prob(state: SGoState) -> float:
    """
    Estimate P1's probability of winning from state.
    Returns a value in (0, 1); 0.5 = equal position.
    """
    p1_stones = 0
    p2_stones = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if state.board[r][c] == P1:
                p1_stones += 1
            elif state.board[r][c] == P2:
                p2_stones += 1

    diff = p1_stones - p2_stones
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * diff))
