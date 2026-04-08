package com.fumbbl.sgo.mcts;

/**
 * Edge for one action choice at a StateNode.
 * chanceNode is non-null for stochastic actions, deterministicChild for deterministic.
 */
public final class ActionEdge {
    public final int actionId;
    public ChanceNode chanceNode;           // null until edge is first selected (lazy)
    public StateNode deterministicChild;    // null for stochastic or not yet expanded
    public int visitCount;
    public double valueSum;

    public ActionEdge(int actionId) {
        this.actionId = actionId;
    }
}
