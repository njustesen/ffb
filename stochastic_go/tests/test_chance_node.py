"""Unit tests for chance node outcome aggregation and selection policy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from game import SGoState, SGoAction, P1, P2, EMPTY
from mcts_ft import ChanceNode, OutcomeEdge, TranspositionTable, SearchContext, select_outcome, expand_outcomes
from sgo_interface import SGoGameInterface

GAME = SGoGameInterface()


def make_placement_chance_node(k: int) -> tuple:
    """
    Build a chance node for placing P1's stone at (3, 3) with k adjacent P2 stones.
    Returns (ChanceNode, ctx).
    """
    state = SGoState.initial()
    # Place k opponent stones adjacent to (3, 3).
    adjacent = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
    for i in range(k):
        r, c = adjacent[i]
        state.board[r][c] = P2
    state._hash = None

    action = SGoAction(type="place", coord=(3, 3))
    cn = ChanceNode()
    ctx = SearchContext()
    expand_outcomes(cn, state, action, GAME, ctx)
    return cn, ctx


# ─────────────────────────────────────────────
# Outcome aggregation
# ─────────────────────────────────────────────

def test_expand_outcomes_k0_two_edges():
    cn, _ = make_placement_chance_node(0)
    assert len(cn.outcome_edges) == 2


def test_expand_outcomes_k0_probabilities():
    cn, _ = make_placement_chance_node(0)
    probs = sorted(e.probability for e in cn.outcome_edges.values())
    assert abs(probs[0] - 1/6) < 1e-9   # fail
    assert abs(probs[1] - 5/6) < 1e-9   # success


def test_expand_outcomes_k3_two_edges():
    cn, _ = make_placement_chance_node(3)
    assert len(cn.outcome_edges) == 2


def test_expand_outcomes_k3_probabilities():
    # k=3: placement_success(roll, 3): roll=1→F, roll=2→F, roll=3→F, roll=4→T, roll=5→T, roll=6→T
    # Success = rolls 4,5,6 = 3/6.  Fail = rolls 1,2,3 = 3/6.
    cn, _ = make_placement_chance_node(3)
    probs = sorted(e.probability for e in cn.outcome_edges.values())
    assert abs(probs[0] - 3/6) < 1e-9
    assert abs(probs[1] - 3/6) < 1e-9


def test_expand_outcomes_probabilities_sum_to_one():
    for k in range(6):
        cn, _ = make_placement_chance_node(k)
        total = sum(e.probability for e in cn.outcome_edges.values())
        assert abs(total - 1.0) < 1e-9


def test_transposition_reuse_in_expand():
    """Two expand_outcomes calls with the same state share StateNode instances."""
    state = SGoState.initial()
    action = SGoAction(type="place", coord=(0, 0))
    ctx = SearchContext()

    cn1 = ChanceNode()
    expand_outcomes(cn1, state, action, GAME, ctx)
    cn2 = ChanceNode()
    expand_outcomes(cn2, state, action, GAME, ctx)

    # Child nodes should be identical objects (same StateHash → same node in tt).
    hashes1 = set(cn1.outcome_edges.keys())
    hashes2 = set(cn2.outcome_edges.keys())
    assert hashes1 == hashes2
    for h in hashes1:
        assert cn1.outcome_edges[h].child_state is cn2.outcome_edges[h].child_state


# ─────────────────────────────────────────────
# Outcome selection policy
# ─────────────────────────────────────────────

def test_select_outcome_unexplored_high_prob_first():
    """First selection picks the highest-probability unexplored outcome."""
    cn, _ = make_placement_chance_node(0)
    # Success has p=5/6, fail has p=1/6. First pick should be success.
    out_hash, out_edge = select_outcome(cn)
    assert abs(out_edge.probability - 5/6) < 1e-9


def test_select_outcome_entropy_minimizing():
    """
    After visiting outcome A 9× and outcome B 1× (with p_A=p_B=0.5),
    the next selection returns B.
    """
    # Use k=2 for equal probabilities.
    cn, _ = make_placement_chance_node(2)
    assert len(cn.outcome_edges) == 2
    hashes = list(cn.outcome_edges.keys())
    h_a, h_b = hashes[0], hashes[1]

    # Simulate: explore A first (highest prob when equal → arbitrary, so force counts).
    cn.visit_counts[h_a] = 9
    cn.visit_counts[h_b] = 1
    cn.total_visits = 10

    # Next selection should pick B (most underrepresented: p/(n/N) = 0.5/(1/10) = 5.0 vs 0.5/(9/10) ≈ 0.56).
    out_hash, _ = select_outcome(cn)
    assert out_hash == h_b


def test_select_outcome_converges_to_distribution():
    """After many selections, visit ratio ≈ theoretical probability ratio."""
    cn, _ = make_placement_chance_node(0)  # p_success=5/6, p_fail=1/6
    hashes = list(cn.outcome_edges.keys())

    # Simulate repeated selections (no actual game application; just count).
    for i in range(1000):
        out_hash, _ = select_outcome(cn)
        cn.visit_counts[out_hash] = cn.visit_counts.get(out_hash, 0) + 1
        cn.total_visits += 1

    # Expected: ~833 success, ~167 fail.
    probs = {h: e.probability for h, e in cn.outcome_edges.items()}
    N = cn.total_visits
    for h, count in cn.visit_counts.items():
        ratio = count / N
        expected = probs[h]
        assert abs(ratio - expected) < 0.05, (
            f"Visit ratio {ratio:.3f} far from theoretical {expected:.3f}"
        )
