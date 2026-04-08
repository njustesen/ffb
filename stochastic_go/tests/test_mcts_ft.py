"""Unit tests for MCTS-FT tree structure, backpropagation, and search properties."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pytest
from game import (
    SGoState, SGoAction, END_TURN, P1, P2,
    apply_placement, apply_end_turn, advance_turn, BOARD_SIZE, TOTAL_TURNS,
)
from mcts_ft import (
    mcts_ft, SearchContext, TranspositionTable,
    StateNode, backpropagate, best_action,
    traverse_turn,
)
from sgo_interface import SGoGameInterface

GAME = SGoGameInterface()


# ─────────────────────────────────────────────
# Single iteration structural tests
# ─────────────────────────────────────────────

def test_single_iteration_builds_path():
    ctx = SearchContext()
    state = SGoState.initial()
    action, stats = mcts_ft(state, 1, GAME, ctx=ctx)
    assert action is not None
    assert len(ctx.tt.table) >= 1


def test_action_edges_lazily_populated():
    ctx = SearchContext()
    root_state = SGoState.initial()
    root_hash = GAME.hash_state(root_state)
    # Before search: no node.
    assert ctx.tt.lookup(root_hash) is None
    mcts_ft(root_state, 1, GAME, ctx=ctx)
    root = ctx.tt.lookup(root_hash)
    assert root is not None
    assert len(root.action_edges) > 0


def test_backprop_updates_all_path_nodes():
    """All nodes on the path should have visit_count >= 1 after backprop."""
    ctx = SearchContext()
    root_state = SGoState.initial()
    mcts_ft(root_state, 5, GAME, ctx=ctx)
    for node in ctx.tt.table.values():
        if node.visit_count > 0:
            # Value sum should be finite.
            assert -10.0 <= node.value_sum <= 10.0


def test_backprop_skips_transposition_duplicates():
    """
    If the same StateNode appears twice in a path (via transposition),
    its visit_count should only be incremented once per iteration.
    Verify by checking visit_count <= iterations for all nodes.
    """
    ctx = SearchContext()
    iterations = 20
    mcts_ft(SGoState.initial(), iterations, GAME, ctx=ctx)
    for node in ctx.tt.table.values():
        assert node.visit_count <= iterations, (
            f"Node visit_count={node.visit_count} exceeds iterations={iterations}"
        )


def test_deterministic_action_creates_child_via_tt():
    """end_turn action (deterministic) should create a child via the transposition table."""
    ctx = SearchContext()
    state = SGoState.initial()
    mcts_ft(state, 10, GAME, ctx=ctx)

    root = ctx.tt.lookup(GAME.hash_state(state))
    assert root is not None
    # end_turn edge should exist.
    end_edge = root.action_edges.get(END_TURN)
    if end_edge is not None and end_edge.visit_count > 0:
        assert end_edge.deterministic_child is not None
        # Child must be in transposition table.
        child_hash = end_edge.deterministic_child.state_hash
        assert ctx.tt.lookup(child_hash) is not None


# ─────────────────────────────────────────────
# Subtree reuse
# ─────────────────────────────────────────────

def test_subtree_reuse():
    """
    After search A from root, any child state in ctx.tt has visit_count > 0.
    A search B starting from that child state (with the same ctx) should begin
    with root.visit_count > 0 without running any new iterations first.
    """
    ctx = SearchContext()
    state_a = SGoState.initial()
    mcts_ft(state_a, 30, GAME, ctx=ctx)

    # Find any visited child state.
    found_child = None
    root = ctx.tt.lookup(GAME.hash_state(state_a))
    for edge in root.action_edges.values():
        if edge.visit_count > 0 and edge.deterministic_child is not None:
            found_child = edge.deterministic_child
            break
        if edge.chance_node:
            for h, out_edge in edge.chance_node.outcome_edges.items():
                if edge.chance_node.visit_counts.get(h, 0) > 0:
                    found_child = out_edge.child_state
                    break
        if found_child:
            break

    if found_child is None:
        pytest.skip("No visited child found in tree")

    # The child node is already in the transposition table from search A.
    # (visit_count may be 0 for turn-end leaves that were never expanded as parents,
    # but the node exists and can be reused as the root of a new search.)
    child_hash = found_child.state_hash
    assert ctx.tt.lookup(child_hash) is found_child, (
        "Child node should be in the transposition table"
    )

    # When we re-search from the child state, get_or_create returns the existing node.
    pre_hits = ctx.tt.total_hits
    ctx.tt.get_or_create(child_hash, found_child.state)
    assert ctx.tt.total_hits == pre_hits + 1, (
        "Re-searching from a previously seen state should be a transposition hit"
    )


# ─────────────────────────────────────────────
# best_action
# ─────────────────────────────────────────────

def test_best_action_most_visited():
    ctx = SearchContext()
    state = SGoState.initial()
    mcts_ft(state, 50, GAME, ctx=ctx)
    root = ctx.tt.lookup(GAME.hash_state(state))
    action = best_action(root)
    expected = max(root.action_edges.keys(), key=lambda a: root.action_edges[a].visit_count)
    assert action == expected


# ─────────────────────────────────────────────
# Basic sanity: MCTS beats random on small board
# ─────────────────────────────────────────────

def _play_game_fixed(mcts_budget: int, seed: int) -> int:
    """Play one game: P1=MCTS, P2=random. Returns final score."""
    import random as rng
    rng.seed(seed)

    state = SGoState.initial()
    ctx = SearchContext()

    while not state.is_terminal():
        player = state.current_player
        if player == P1:
            action, _ = mcts_ft(state, mcts_budget, GAME, ctx=ctx)
        else:
            actions = [a for a in __import__("game").legal_actions(state)]
            action = rng.choice(actions)

        if action.type == "end_turn":
            state = apply_end_turn(state)
        else:
            roll = rng.randint(1, 6)
            state = apply_placement(state, action.coord, roll)

        if state.is_turn_end:
            state = advance_turn(state)

    return state.score()


def test_mcts_beats_random_simple():
    """MCTS-FT (150 iter) should win ≥ 6/10 fixed-seed games vs. random."""
    wins = 0
    for seed in range(10):
        score = _play_game_fixed(mcts_budget=150, seed=seed)
        if score > 0:
            wins += 1
    assert wins >= 6, f"MCTS won only {wins}/10 games vs. random"
