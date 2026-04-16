package com.fumbbl.ffb.ai.mcts;

import java.util.List;

/**
 * Per-game MCTS search statistics.
 *
 * <p>Accumulated across all MCTS decisions within a single game.  Use
 * {@link #merge(List)} to aggregate across multiple games.
 */
public final class BbMctsStats {

    /** Total tree iterations performed across all decisions this game. */
    public long totalIterations;

    /** Total search wall-clock time (nanoseconds). */
    public long totalSearchNs;

    /** Total MCTS decisions (= activations where MCTS ran). */
    public int decisions;

    /** Sum of candidate counts across all decisions (for avg branching factor). */
    public long branchSum;

    /** Maximum tree depth reached across all decisions. */
    public int maxDepth;

    /** Total unique tree nodes created (expanded) across all decisions. */
    public long totalNodes;

    /** Sum of max-depth values across all decisions (for average). */
    public long depthSum;

    // ── Derived stats (computed by computeDerived()) ──────────────────────────

    public double avgIterationMs;   // ms/iteration
    public double avgBranchFactor;  // mean candidates/decision
    public double itersPerSecond;   // throughput
    public double avgDepth;         // average max depth per decision

    public void recordDecision(int candidates, int iterations, long searchNs, int depth, int nodes) {
        decisions++;
        branchSum += candidates;
        totalIterations += iterations;
        totalSearchNs += searchNs;
        if (depth > maxDepth) maxDepth = depth;
        depthSum += depth;
        totalNodes += nodes;
    }

    public void computeDerived() {
        avgIterationMs  = totalIterations > 0 ? totalSearchNs / 1e6 / totalIterations : 0.0;
        avgBranchFactor = decisions > 0 ? (double) branchSum / decisions : 0.0;
        itersPerSecond  = totalSearchNs > 0 ? totalIterations / (totalSearchNs / 1e9) : 0.0;
        avgDepth        = decisions > 0 ? (double) depthSum / decisions : 0.0;
    }

    /** Merge a list of per-game stats into a single aggregate. */
    public static BbMctsStats merge(List<BbMctsStats> list) {
        BbMctsStats agg = new BbMctsStats();
        for (BbMctsStats s : list) {
            agg.totalIterations += s.totalIterations;
            agg.totalSearchNs   += s.totalSearchNs;
            agg.decisions       += s.decisions;
            agg.branchSum       += s.branchSum;
            agg.totalNodes      += s.totalNodes;
            agg.depthSum        += s.depthSum;
            if (s.maxDepth > agg.maxDepth) agg.maxDepth = s.maxDepth;
        }
        agg.computeDerived();
        return agg;
    }
}
