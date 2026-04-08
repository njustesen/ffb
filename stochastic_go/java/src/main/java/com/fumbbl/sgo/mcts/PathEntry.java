package com.fumbbl.sgo.mcts;

/**
 * Mutable path entry reused across iterations (pre-allocated in SearchContext).
 */
public final class PathEntry {
    public StateNode node;
    public int actionId;
    public ChanceNode chanceNode;  // null for deterministic actions
    public long outcomeHash;       // 0 for deterministic actions
}
