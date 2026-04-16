package com.fumbbl.ffb.ai.mcts;

import java.util.HashMap;
import java.util.Map;

/**
 * A node in the Blood Bowl MCTS tree.
 *
 * <p>Each node corresponds to a game state at an {@code INIT_SELECTING} phase-1
 * decision point.  Children are keyed by the {@link BbAction} that leads from
 * this node's state to the child's state.
 *
 * <p>Priors (for MCTS-Script) are stored as a map from BbAction to a score.
 * When {@code priors} is {@code null} the node uses plain UCB.  When set, unvisited
 * children are explored in descending prior order before UCB takes over.
 */
public final class BbMctsNode {

    public int visitCount;
    public double valueSum;

    /** Children keyed by the activation that leads to them. */
    public final Map<BbAction, BbMctsNode> children = new HashMap<>();

    /**
     * Optional action scores for unvisited-arm ordering.
     * {@code null} = UCB mode (no ordering).  Populated once on first expansion
     * when an {@link IActionPrior} is set.
     */
    public Map<BbAction, Double> priors;

    /**
     * Cached candidate set for this node once it becomes internal.
     * Populated on the first internal visit to avoid repeated Dijkstra calls.
     * Holds both the candidate list and raw scores (for prior computation without
     * a second {@code selectPlayer()} call).
     * Set to {@code null} to invalidate (e.g., after a StackOverflowError indicates
     * the cached candidates caused an invalid action injection in a different
     * stochastic outcome).
     */
    public CandidateSet cachedSet;

    public double q() {
        return visitCount > 0 ? valueSum / visitCount : 0.0;
    }

    public BbMctsNode getOrCreateChild(BbAction action) {
        return children.computeIfAbsent(action, a -> new BbMctsNode());
    }

    /** Return the prior for {@code action}, or {@code 0.0} if not set. */
    public double getPrior(BbAction action) {
        if (priors == null) return 0.0;
        Double p = priors.get(action);
        return p != null ? p : 0.0;
    }
}
