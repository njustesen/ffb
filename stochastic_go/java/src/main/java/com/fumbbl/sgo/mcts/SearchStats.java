package com.fumbbl.sgo.mcts;

import java.util.Map;
import java.util.LinkedHashMap;

public final class SearchStats {

    public long totalTimeMs;
    public int iterations;
    public double iterationsPerSecond;

    public int totalStateNodes;
    public int totalChanceNodes;
    public int totalTranspositionHits;
    public int totalTranspositionAttempts;
    public double transpositionHitRate;

    public int maxDepthReached;
    public double avgBranchingFactor;

    public Map<String, Integer> rootVisitDistribution = new LinkedHashMap<>();
    public double rootVisitEntropy;
    public double avgChanceKl;
    public int chanceNodeCountForKl;

    public void finalize(long elapsedNs, int budget) {
        this.totalTimeMs = elapsedNs / 1_000_000L;
        this.iterations = budget;
        double secs = elapsedNs / 1e9;
        this.iterationsPerSecond = secs > 0 ? budget / secs : 0;
        if (totalTranspositionAttempts > 0) {
            this.transpositionHitRate = (double) totalTranspositionHits / totalTranspositionAttempts;
        }
    }

    /** Shannon entropy of a visit distribution. */
    public static double entropy(Map<String, Integer> dist) {
        int total = 0;
        for (int v : dist.values()) total += v;
        if (total == 0) return 0.0;
        double h = 0.0;
        for (int v : dist.values()) {
            if (v > 0) {
                double p = (double) v / total;
                h -= p * Math.log(p) / Math.log(2);
            }
        }
        return h;
    }

    public String summary() {
        return String.format(
            "iterations=%d  time=%dms  iter/s=%.0f  nodes=%d  chance=%d  tt_hits=%d/%d (%.1f%%)  depth=%d  bf=%.1f  entropy=%.3f  kl=%.5f",
            iterations, totalTimeMs, iterationsPerSecond,
            totalStateNodes, totalChanceNodes,
            totalTranspositionHits, totalTranspositionAttempts,
            transpositionHitRate * 100,
            maxDepthReached, avgBranchingFactor,
            rootVisitEntropy, avgChanceKl
        );
    }
}
