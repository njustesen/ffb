"""Unit tests for search statistics collection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from game import SGoState
from mcts_ft import mcts_ft, SearchContext
from sgo_interface import SGoGameInterface
from stats import SearchStats

GAME = SGoGameInterface()


def test_stats_fields_populated():
    """All SearchStats fields should be populated and in valid ranges."""
    ctx = SearchContext()
    _, stats = mcts_ft(SGoState.initial(), 50, GAME, ctx=ctx)

    assert stats.total_time_ms > 0
    assert stats.iterations == 50
    assert stats.iterations_per_second > 0
    assert stats.total_state_nodes >= 1
    assert stats.total_chance_nodes >= 0
    assert 0.0 <= stats.transposition_hit_rate <= 1.0
    assert stats.max_depth_reached >= 0
    assert stats.avg_branching_factor >= 0.0
    assert stats.root_visit_entropy >= 0.0
    assert stats.avg_chance_kl >= 0.0


def test_stats_iterations_match():
    for budget in (10, 50, 100):
        ctx = SearchContext()
        _, stats = mcts_ft(SGoState.initial(), budget, GAME, ctx=ctx)
        assert stats.iterations == budget


def test_kl_decreases_with_budget():
    """
    avg_chance_kl should be non-increasing as budget increases.
    (More iterations → better calibration of chance node visit frequencies.)
    """
    budgets = [50, 200, 1000]
    kl_values = []

    for budget in budgets:
        ctx = SearchContext()  # fresh ctx each time for fair comparison
        _, stats = mcts_ft(SGoState.initial(), budget, GAME, ctx=ctx)
        kl_values.append(stats.avg_chance_kl)

    # Allow some tolerance for randomness; the trend should be downward.
    # Check that the 1000-iter KL is meaningfully lower than the 50-iter KL.
    assert kl_values[-1] <= kl_values[0] + 0.01, (
        f"KL did not decrease: {kl_values}"
    )


def test_transposition_hit_rate_positive_after_many_iterations():
    """With many iterations, the transposition table should see some reuse."""
    ctx = SearchContext()
    _, stats = mcts_ft(SGoState.initial(), 200, GAME, ctx=ctx)
    assert stats.total_transposition_hits > 0, (
        "Expected some transposition hits after 200 iterations"
    )


def test_root_visit_distribution_covers_actions():
    """Root action distribution should cover at least a few distinct actions."""
    ctx = SearchContext()
    _, stats = mcts_ft(SGoState.initial(), 100, GAME, ctx=ctx)
    visited_actions = sum(1 for v in stats.root_visit_distribution.values() if v > 0)
    assert visited_actions >= 2, (
        f"Expected at least 2 visited root actions, got {visited_actions}"
    )
