package com.fumbbl.sgo.mcts;

/**
 * Chance node grouping dice outcomes by resulting state hash.
 * Uses parallel arrays (max 6 outcomes) for zero-allocation linear scan.
 */
public final class ChanceNode {

    // Max 6 distinct outcomes (one per dice face, though usually 2: success/fail)
    public final long[] outcomeHashes = new long[6];
    public final OutcomeEdge[] outcomes = new OutcomeEdge[6];
    public final int[] visitCounts = new int[6];
    public int outcomeCount = 0;
    public int totalVisits = 0;

    public boolean isExpanded() {
        return outcomeCount > 0;
    }

    /** Find index of the given hash, or -1 if not present. */
    public int indexOf(long hash) {
        for (int i = 0; i < outcomeCount; i++) {
            if (outcomeHashes[i] == hash) return i;
        }
        return -1;
    }

    /** Add a new outcome. outcomeCount must be < 6. */
    public void addOutcome(long hash, double probability, StateNode child) {
        outcomeHashes[outcomeCount] = hash;
        outcomes[outcomeCount] = new OutcomeEdge(probability, child);
        outcomeCount++;
    }
}
