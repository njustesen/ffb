package com.fumbbl.sgo.mcts;

/**
 * Persists across multiple search() calls for subtree reuse.
 * Pre-allocates path buffer to eliminate per-iteration allocation.
 */
public final class SearchContext {

    public final TranspositionTable tt = new TranspositionTable();
    public int chanceNodeCount = 0;

    // Pre-allocated path buffer: reused each iteration
    public static final int MAX_PATH = 256;
    public final PathEntry[] path;
    public int pathSize = 0;

    public SearchContext() {
        path = new PathEntry[MAX_PATH];
        for (int i = 0; i < MAX_PATH; i++) {
            path[i] = new PathEntry();
        }
    }
}
