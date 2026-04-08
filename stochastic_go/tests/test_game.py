"""Unit tests for Stochastic Go game logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from game import (
    SGoState, SGoAction, END_TURN, P1, P2, EMPTY,
    BOARD_SIZE, TOTAL_TURNS,
    legal_actions, apply_placement, apply_end_turn, advance_turn,
    adjacent_opponent_count, placement_success, place_action,
)


# ─────────────────────────────────────────────
# legal_actions
# ─────────────────────────────────────────────

def test_legal_actions_empty_board():
    state = SGoState.initial()
    actions = legal_actions(state)
    place_actions = [a for a in actions if a.type == "place"]
    assert len(place_actions) == BOARD_SIZE * BOARD_SIZE
    assert END_TURN in actions
    assert len(actions) == BOARD_SIZE * BOARD_SIZE + 1


def test_legal_actions_no_actions_if_turn_ended():
    state = SGoState.initial()
    state = apply_end_turn(state)
    assert legal_actions(state) == []


def test_legal_actions_excludes_occupied_cells():
    state = SGoState.initial()
    # Use apply_placement to properly update board + empty_cells.
    state = apply_placement(state, (3, 3), roll=6)  # k=0, roll=6 → success
    actions = legal_actions(state)
    place_coords = [a.coord for a in actions if a.type == "place"]
    assert (3, 3) not in place_coords
    assert len(place_coords) == BOARD_SIZE * BOARD_SIZE - 1


# ─────────────────────────────────────────────
# placement_success rules
# ─────────────────────────────────────────────

def test_placement_always_fails_roll1():
    for k in range(9):
        assert placement_success(1, k) is False


def test_placement_always_succeeds_roll6():
    for k in range(9):
        assert placement_success(6, k) is True


def test_placement_success_k0():
    # k=0: rolls 2-6 succeed (5 out of 6)
    assert placement_success(1, 0) is False
    for roll in range(2, 7):
        assert placement_success(roll, 0) is True


def test_placement_success_k3():
    # k=3: only rolls 4,5,6 succeed → roll > 3
    assert placement_success(1, 3) is False
    assert placement_success(2, 3) is False
    assert placement_success(3, 3) is False
    assert placement_success(4, 3) is True
    assert placement_success(5, 3) is True
    assert placement_success(6, 3) is True


def test_placement_success_k5():
    # k=5: roll > 5 means only roll=6 succeeds
    for roll in range(1, 6):
        assert placement_success(roll, 5) is False
    assert placement_success(6, 5) is True


# ─────────────────────────────────────────────
# apply_placement outcomes
# ─────────────────────────────────────────────

def test_turnover_ends_turn_stone_not_placed():
    state = SGoState.initial()
    # k=0, roll=1 → failure
    new = apply_placement(state, (0, 0), roll=1)
    assert new.is_turn_end is True
    assert new.board[0][0] == EMPTY


def test_success_places_stone_turn_continues():
    state = SGoState.initial()
    # k=0, roll=6 → success
    new = apply_placement(state, (0, 0), roll=6)
    assert new.is_turn_end is False
    assert new.board[0][0] == P1


def test_placement_failure_with_adjacents():
    state = SGoState.initial()
    # Place 4 P2 stones diagonally adjacent to (3, 3): k=4.
    # placement_success(roll, 4): roll > 4 succeeds → rolls 5 and 6 succeed (2/6).
    state.board[2][2] = P2
    state.board[2][4] = P2
    state.board[4][2] = P2
    state.board[4][4] = P2
    state._hash = None

    # roll=4: 4 > 4 is False → fail
    new_fail = apply_placement(state, (3, 3), roll=4)
    assert new_fail.is_turn_end is True
    assert new_fail.board[3][3] == EMPTY

    # roll=5: 5 > 4 is True → success
    new_succ = apply_placement(state, (3, 3), roll=5)
    assert new_succ.is_turn_end is False
    assert new_succ.board[3][3] == P1

    # roll=6: always succeeds
    new_succ6 = apply_placement(state, (3, 3), roll=6)
    assert new_succ6.is_turn_end is False
    assert new_succ6.board[3][3] == P1


# ─────────────────────────────────────────────
# apply_end_turn
# ─────────────────────────────────────────────

def test_voluntary_end_turn():
    state = SGoState.initial()
    state = apply_placement(state, (2, 2), roll=6)  # success, board changes
    new = apply_end_turn(state)
    assert new.is_turn_end is True
    assert new.board[2][2] == P1  # stone remains


# ─────────────────────────────────────────────
# advance_turn
# ─────────────────────────────────────────────

def test_advance_turn_switches_player():
    state = SGoState.initial()
    state = apply_end_turn(state)
    assert state.current_player == P1
    new = advance_turn(state)
    assert new.current_player == P2
    assert new.is_turn_end is False
    assert new.p1_turns_remaining == TOTAL_TURNS - 1


def test_advance_turn_p2_to_p1():
    state = SGoState.initial()
    state = apply_end_turn(state)
    state = advance_turn(state)
    # Now P2's turn; end it.
    state = apply_end_turn(state)
    state = advance_turn(state)
    assert state.current_player == P1
    assert state.p2_turns_remaining == TOTAL_TURNS - 1


# ─────────────────────────────────────────────
# Terminal and score
# ─────────────────────────────────────────────

def test_game_ends_after_total_turns():
    state = SGoState.initial()
    for _ in range(TOTAL_TURNS):
        state = apply_end_turn(state)
        state = advance_turn(state)  # P1 → P2
        state = apply_end_turn(state)
        state = advance_turn(state)  # P2 → P1
    assert state.is_terminal()


def test_score_equals_stone_diff():
    state = SGoState.initial()
    state = apply_placement(state, (0, 0), roll=6)  # P1 places at (0,0)
    state = apply_placement(state, (0, 1), roll=6)  # P1 places at (0,1)
    # Switch to P2 and place one stone.
    state = apply_end_turn(state)
    state = advance_turn(state)
    state = apply_placement(state, (0, 2), roll=6)  # P2 places at (0,2)
    assert state.score() == 1  # 2 P1 - 1 P2


# ─────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────

def test_hash_state_deterministic():
    s1 = SGoState.initial()
    s2 = SGoState.initial()
    assert s1.hash() == s2.hash()


def test_hash_state_unique():
    import random
    seen = set()
    for _ in range(100):
        state = SGoState.initial()
        r, c = random.randint(0, 7), random.randint(0, 7)
        state.board[r][c] = P1
        state._hash = None
        h = state.hash()
        seen.add((r, c, h))
    # All (r, c, hash) triples should be unique — no two different boards same hash.
    hashes = [h for _, _, h in seen]
    assert len(hashes) == len(set(hashes))


def test_hash_changes_with_board():
    s1 = SGoState.initial()
    s2 = SGoState.initial()
    s2.board[3][3] = P1
    s2._hash = None
    assert s1.hash() != s2.hash()


def test_adjacent_opponent_count():
    state = SGoState.initial()
    state.board[2][2] = P2
    state.board[2][4] = P2
    state._hash = None
    # Placing P1 at (3, 3): diagonal neighbors at (2,2) and (2,4) are P2.
    k = adjacent_opponent_count(state.board, 3, 3, P1)
    assert k == 2
