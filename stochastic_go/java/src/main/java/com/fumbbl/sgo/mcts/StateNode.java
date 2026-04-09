package com.fumbbl.sgo.mcts;

import com.fumbbl.sgo.game.SGoAction;
import com.fumbbl.sgo.game.SGoState;

/**
 * A game state in the search tree.
 *
 * Action edges are stored in ActionEdge[TOTAL_CELLS+1] indexed by action.id
 * (0..TOTAL_CELLS-1 = place, TOTAL_CELLS = end_turn).
 * edgeIds holds the ids of actually populated edges for fast iteration.
 */
public final class StateNode {

    public final long stateHash;
    public SGoState state;
    public boolean isTurnEnd;

    // Direct-indexed by action id (0..TOTAL_CELLS). Null until first expanded.
    public ActionEdge[] edges = null; // lazily created in expandActions to avoid 520B per leaf node
    // Parallel list of populated edge ids for fast iteration
    public int[] edgeIds = null;
    public int edgeCount = 0;

    public int visitCount;
    public double valueSum;

    public StateNode(long stateHash, SGoState state, boolean isTurnEnd) {
        this.stateHash = stateHash;
        this.state = state;
        this.isTurnEnd = isTurnEnd;
    }

    public boolean isExpanded() {
        return edgeIds != null;
    }

    public double valueEstimate() {
        return visitCount > 0 ? valueSum / visitCount : 0.0;
    }
}
