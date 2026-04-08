"""
Stochastic Go game logic.

Rules:
- 8x8 board, two players alternating turns.
- On your turn, place as many stones as desired (one at a time) or end turn voluntarily.
- Placement: roll 1D6. Success if roll > adjacent opponent stones. Roll 1 always fails, roll 6 always succeeds.
- Failure: stone not placed, turn ends immediately (turnover).
- Game ends after TOTAL_TURNS turns each, or when board is full.
- Score: P1_stones - P2_stones. P1 maximizes, P2 minimizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

BOARD_SIZE = 8
TOTAL_TURNS = 10  # turns per player
P1 = 1
P2 = 2
EMPTY = 0

# ─────────────────────────────────────────────
# Zobrist hash tables, initialized once.
# Every piece of mutable state gets its own table so hashes can be updated
# incrementally (XOR out old value, XOR in new value).
# ─────────────────────────────────────────────
import random as _rng
_rng.seed(0xDEADBEEF)

_ZOBRIST_BOARD: List[List[List[int]]] = [
    [[_rng.getrandbits(64) for _ in range(3)] for _ in range(BOARD_SIZE)]
    for _ in range(BOARD_SIZE)
]  # [row][col][piece=0/1/2]

_ZOBRIST_CURRENT_PLAYER: List[int] = [0, _rng.getrandbits(64), _rng.getrandbits(64)]
_ZOBRIST_TURN_END: List[int] = [_rng.getrandbits(64), _rng.getrandbits(64)]  # [False, True]
_ZOBRIST_P1_TURNS: List[int] = [_rng.getrandbits(64) for _ in range(TOTAL_TURNS + 1)]
_ZOBRIST_P2_TURNS: List[int] = [_rng.getrandbits(64) for _ in range(TOTAL_TURNS + 1)]

# Pre-built hash for the initial (empty) state so we don't recompute it.
_INITIAL_HASH: int = (
    _ZOBRIST_CURRENT_PLAYER[P1]
    ^ _ZOBRIST_TURN_END[0]
    ^ _ZOBRIST_P1_TURNS[TOTAL_TURNS]
    ^ _ZOBRIST_P2_TURNS[TOTAL_TURNS]
) & 0xFFFFFFFFFFFFFFFF

# ─────────────────────────────────────────────
# Pre-built action objects — reused to avoid per-call object allocation.
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class SGoAction:
    type: str  # "place" or "end_turn"
    coord: Optional[Tuple[int, int]] = None  # (row, col) for "place"

    def __str__(self) -> str:
        if self.type == "end_turn":
            return "end_turn"
        return f"place({self.coord[0]},{self.coord[1]})"

    def __repr__(self) -> str:
        return str(self)


END_TURN = SGoAction(type="end_turn")

# Singleton action object per coordinate — created once, reused everywhere.
_PLACE_ACTION: List[List[SGoAction]] = [
    [SGoAction(type="place", coord=(r, c)) for c in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE)
]


def place_action(r: int, c: int) -> SGoAction:
    return _PLACE_ACTION[r][c]


# ─────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────

@dataclass
class SGoState:
    board: List[List[int]]        # board[row][col] = EMPTY/P1/P2
    current_player: int           # P1 or P2
    p1_turns_remaining: int
    p2_turns_remaining: int
    is_turn_end: bool
    empty_cells: Set[Tuple[int, int]]  # set of (row, col) that are EMPTY; avoids O(64) scan
    _hash: Optional[int] = field(default=None, repr=False, compare=False)

    @classmethod
    def initial(cls) -> SGoState:
        board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        empty = {(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        return cls(
            board=board,
            current_player=P1,
            p1_turns_remaining=TOTAL_TURNS,
            p2_turns_remaining=TOTAL_TURNS,
            is_turn_end=False,
            empty_cells=empty,
            _hash=_INITIAL_HASH,
        )

    def compute_hash(self) -> int:
        """Full recompute — used only when incremental tracking fails (e.g. tests)."""
        h = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece != EMPTY:
                    h ^= _ZOBRIST_BOARD[r][c][piece]
        h ^= _ZOBRIST_CURRENT_PLAYER[self.current_player]
        h ^= _ZOBRIST_TURN_END[1 if self.is_turn_end else 0]
        h ^= _ZOBRIST_P1_TURNS[self.p1_turns_remaining]
        h ^= _ZOBRIST_P2_TURNS[self.p2_turns_remaining]
        return h & 0xFFFFFFFFFFFFFFFF

    def hash(self) -> int:
        if self._hash is None:
            self._hash = self.compute_hash()
        return self._hash

    def clone(self) -> SGoState:
        return SGoState(
            board=[row[:] for row in self.board],
            current_player=self.current_player,
            p1_turns_remaining=self.p1_turns_remaining,
            p2_turns_remaining=self.p2_turns_remaining,
            is_turn_end=self.is_turn_end,
            empty_cells=self.empty_cells.copy(),
            _hash=self._hash,  # carry forward; updated incrementally by callers
        )

    def is_terminal(self) -> bool:
        if self.p1_turns_remaining == 0 and self.p2_turns_remaining == 0:
            return True
        return len(self.empty_cells) == 0

    def score(self) -> int:
        """P1 stones minus P2 stones."""
        p1 = sum(self.board[r][c] == P1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        p2 = sum(self.board[r][c] == P2 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        return p1 - p2


# ─────────────────────────────────────────────
# Game mechanics
# ─────────────────────────────────────────────

def adjacent_opponent_count(board: List[List[int]], row: int, col: int, player: int) -> int:
    """Count orthogonal + diagonal cells occupied by the opponent."""
    opponent = P2 if player == P1 else P1
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r2, c2 = row + dr, col + dc
            if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE:
                if board[r2][c2] == opponent:
                    count += 1
    return count


def placement_success(roll: int, k: int) -> bool:
    """True if a 1D6 roll succeeds for a placement with k adjacent opponents."""
    if roll == 1:
        return False
    if roll == 6:
        return True
    return roll > k


def success_probability(k: int) -> float:
    """Probability of a successful placement with k adjacent opponent stones."""
    successes = sum(1 for r in range(1, 7) if placement_success(r, k))
    return successes / 6.0


def legal_actions(state: SGoState) -> List[SGoAction]:
    """All legal actions from state (for the current player)."""
    if state.is_turn_end or state.is_terminal():
        return []
    actions = [END_TURN]
    for (r, c) in state.empty_cells:
        actions.append(_PLACE_ACTION[r][c])
    return actions


def apply_placement(state: SGoState, coord: Tuple[int, int], roll: int) -> SGoState:
    """
    Apply a placement at coord with the given dice roll.
    Returns new state with incrementally updated hash.
    """
    r, c = coord
    k = adjacent_opponent_count(state.board, r, c, state.current_player)
    success = placement_success(roll, k)

    new_state = state.clone()  # carries _hash from parent
    base = new_state.hash()    # ensure we have a valid hash before incrementing

    if success:
        new_state.board[r][c] = state.current_player
        new_state.empty_cells.discard(coord)
        # Incremental hash: XOR in the newly placed piece (was EMPTY = no contribution).
        new_state._hash = (base ^ _ZOBRIST_BOARD[r][c][state.current_player]) & 0xFFFFFFFFFFFFFFFF
    else:
        new_state.is_turn_end = True
        # Incremental hash: flip is_turn_end flag.
        new_state._hash = (base ^ _ZOBRIST_TURN_END[0] ^ _ZOBRIST_TURN_END[1]) & 0xFFFFFFFFFFFFFFFF

    return new_state


def apply_end_turn(state: SGoState) -> SGoState:
    """Voluntarily end the current player's turn."""
    new_state = state.clone()
    base = new_state.hash()
    new_state.is_turn_end = True
    new_state._hash = (base ^ _ZOBRIST_TURN_END[0] ^ _ZOBRIST_TURN_END[1]) & 0xFFFFFFFFFFFFFFFF
    return new_state


def advance_turn(state: SGoState) -> SGoState:
    """
    Called when is_turn_end is True. Decrements the current player's turn counter,
    switches to the opponent, and resets is_turn_end.
    """
    assert state.is_turn_end
    new_state = state.clone()
    h = new_state.hash()

    # Flip is_turn_end: True → False
    new_state.is_turn_end = False
    h ^= _ZOBRIST_TURN_END[1] ^ _ZOBRIST_TURN_END[0]

    if state.current_player == P1:
        new_state.p1_turns_remaining -= 1
        new_state.current_player = P2
        h ^= _ZOBRIST_CURRENT_PLAYER[P1] ^ _ZOBRIST_CURRENT_PLAYER[P2]
        h ^= _ZOBRIST_P1_TURNS[state.p1_turns_remaining] ^ _ZOBRIST_P1_TURNS[state.p1_turns_remaining - 1]
    else:
        new_state.p2_turns_remaining -= 1
        new_state.current_player = P1
        h ^= _ZOBRIST_CURRENT_PLAYER[P2] ^ _ZOBRIST_CURRENT_PLAYER[P1]
        h ^= _ZOBRIST_P2_TURNS[state.p2_turns_remaining] ^ _ZOBRIST_P2_TURNS[state.p2_turns_remaining - 1]

    new_state._hash = h & 0xFFFFFFFFFFFFFFFF
    return new_state
