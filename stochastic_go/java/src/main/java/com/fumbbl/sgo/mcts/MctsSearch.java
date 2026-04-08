package com.fumbbl.sgo.mcts;

import com.fumbbl.sgo.game.SGoAction;
import com.fumbbl.sgo.game.SGoRules;
import com.fumbbl.sgo.game.SGoState;

import java.util.Random;

/**
 * MCTS-FT: Full-Turn Monte Carlo Tree Search for Stochastic Go.
 *
 * Each iteration traverses a complete P1 turn then a complete P2 turn,
 * evaluates win probability delta, and backpropagates.
 */
public final class MctsSearch {

    // UCB exploration constant squared (C=sqrt(2), so C^2=2 — avoids Math.sqrt per edge).
    // We compare v + C*sqrt(logN/(n+1)) which is equivalent to comparing
    // v^2_term + C^2*logN/(n+1) only when v=0. For non-zero v we still need the
    // actual score, so we compute sqrt but keep the formula straightforward.
    private static final double C_ACTION = 1.41421356237; // sqrt(2)

    private final SearchContext ctx;
    private final Random rng;

    // Scratch array for backprop transposition dedup (path length is typically <50)
    private final long[] visitedHashes = new long[256];

    public MctsSearch(SearchContext ctx, Random rng) {
        this.ctx = ctx;
        this.rng = rng;
    }

    public MctsSearch() {
        this(new SearchContext(), new Random());
    }

    public SearchResult search(SGoState rootState, int budget) {
        return searchInternal(rootState, budget, Long.MAX_VALUE);
    }

    /** Run search for up to timeBudgetMs milliseconds. */
    public SearchResult searchForMs(SGoState rootState, long timeBudgetMs) {
        long deadlineNs = System.nanoTime() + timeBudgetMs * 1_000_000L;
        return searchInternal(rootState, Integer.MAX_VALUE, deadlineNs);
    }

    private SearchResult searchInternal(SGoState rootState, int budget, long deadlineNs) {
        TranspositionTable tt = ctx.tt;

        long rootHash = rootState.stateHash;
        boolean rootTurnEnd = rootState.isTurnEnd;
        StateNode root = tt.getOrCreate(rootHash, rootState, rootTurnEnd);
        double rootWp = SGoRules.winProb(rootState);

        int maxDepth = 0;
        long startNs = System.nanoTime();
        int iter = 0;

        // Check deadline every BATCH iterations to amortize System.nanoTime() cost.
        final int BATCH = 50;
        outer:
        while (iter < budget) {
            for (int b = 0; b < BATCH && iter < budget; b++, iter++) {
                int depth = 0;
                ctx.pathSize = 0;

                int firstPlayer = rootState.currentPlayer;
                int secondPlayer = (firstPlayer == SGoState.P1) ? SGoState.P2 : SGoState.P1;

                StateNode firstTerminal = traverseTurn(root, firstPlayer, depth);
                depth = getLastDepth();

                StateNode p2Terminal;
                if (!firstTerminal.state.isTerminal()) {
                    SGoState secondStartState = SGoRules.advanceTurn(firstTerminal.state);
                    long secondHash = secondStartState.stateHash;
                    StateNode secondStartNode = tt.getOrCreate(secondHash, secondStartState, secondStartState.isTurnEnd);
                    p2Terminal = traverseTurn(secondStartNode, secondPlayer, depth);
                    depth = getLastDepth();
                } else {
                    p2Terminal = firstTerminal;
                }

                double value = SGoRules.winProb(p2Terminal.state) - rootWp;
                backpropagate(ctx.pathSize, value);

                if (depth > maxDepth) maxDepth = depth;
            }
            if (System.nanoTime() >= deadlineNs) break outer;
        }

        long elapsedNs = System.nanoTime() - startNs;
        SGoAction bestAction = bestAction(root);
        SearchStats stats = collectStats(root, elapsedNs, iter, maxDepth);
        return new SearchResult(bestAction, stats);
    }

    // Temporary depth storage (avoids extra return value)
    private int lastDepth;

    private int getLastDepth() { return lastDepth; }

    /**
     * Walk one player's complete turn, appending to ctx.path[].
     * Returns the turn-end StateNode.
     */
    private StateNode traverseTurn(StateNode start, int player, int depthIn) {
        StateNode node = start;
        int depth = depthIn;

        while (!node.isTurnEnd) {
            SGoState state = node.state;

            if (state.isTerminal()) {
                node.isTurnEnd = true;
                break;
            }

            // Lazy action expansion
            if (!node.isExpanded()) {
                expandActions(node, state, player);
            }

            if (node.edgeCount == 0) {
                node.isTurnEnd = true;
                break;
            }

            int actionIdx = selectAction(node, player);
            ActionEdge edge = node.edges[node.edgeIds[actionIdx]];
            depth++;

            PathEntry pe = ctx.path[ctx.pathSize++];

            if (edge.actionId != SGoAction.END_TURN_ID) {
                // Stochastic action — lazily create ChanceNode on first selection
                if (edge.chanceNode == null) {
                    edge.chanceNode = new ChanceNode();
                    ctx.chanceNodeCount++;
                }
                ChanceNode cn = edge.chanceNode;
                if (!cn.isExpanded()) {
                    expandOutcomes(cn, state, edge.actionId);
                }

                int outIdx = selectOutcome(cn);
                long outHash = cn.outcomeHashes[outIdx];
                cn.visitCounts[outIdx]++;
                cn.totalVisits++;

                pe.node = node;
                pe.actionId = edge.actionId;
                pe.chanceNode = cn;
                pe.outcomeHash = outHash;

                node = cn.outcomes[outIdx].childState;
            } else {
                // Deterministic action
                if (edge.deterministicChild == null) {
                    SGoState nextState = SGoRules.applyEndTurn(state);
                    long h = nextState.stateHash;
                    edge.deterministicChild = ctx.tt.getOrCreate(h, nextState, nextState.isTurnEnd);
                }

                pe.node = node;
                pe.actionId = edge.actionId;
                pe.chanceNode = null;
                pe.outcomeHash = 0L;

                node = edge.deterministicChild;
            }
        }

        lastDepth = depth;
        return node;
    }

    /** Lazily populate action edges for a node. */
    private void expandActions(StateNode node, SGoState state, int player) {
        // Only place actions: turn ends naturally on failure (no voluntary END_TURN).
        // This eliminates the conservative-bias problem at low iteration counts where
        // MCTS would converge to END_TURN too early based on sparse sampling.
        int emptyCount = Long.bitCount(state.emptyCells);

        node.edgeIds = new int[emptyCount];
        int idx = 0;

        long bits = state.emptyCells;
        while (bits != 0L) {
            int cell = Long.numberOfTrailingZeros(bits);
            bits &= bits - 1L;
            node.edges[cell] = new ActionEdge(cell);
            node.edgeIds[idx++] = cell;
        }
        node.edgeCount = idx;
    }

    /**
     * UCB action selection. Returns index into node.edgeIds[].
     * P1 maximizes, P2 minimizes (flip sign for P2).
     */
    private int selectAction(StateNode node, int player) {
        int N = node.visitCount + 1;
        double logN = Math.log(N);
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestIdx = 0;
        int tieCount = 0;

        int[] edgeIds = node.edgeIds;
        ActionEdge[] edges = node.edges;
        int count = node.edgeCount;

        for (int i = 0; i < count; i++) {
            ActionEdge edge = edges[edgeIds[i]];
            int n = edge.visitCount;
            double v = n > 0 ? edge.valueSum / n : 0.0;
            if (player != SGoState.P1) v = -v;
            double score = v + C_ACTION * Math.sqrt(logN / (n + 1));

            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
                tieCount = 1;
            } else if (score == bestScore) {
                tieCount++;
                // Reservoir sampling for uniform tie-breaking
                if (rng.nextInt(tieCount) == 0) bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Enumerate all 6 dice outcomes for a placement, group by state hash.
     */
    private void expandOutcomes(ChanceNode cn, SGoState state, int actionId) {
        int cell = actionId; // actionId 0-63 = cell index
        double prob = 1.0 / 6.0;

        // We'll accumulate into a tiny scratch structure indexed by the 2 distinct hashes
        // There are at most 2 distinct outcomes (success hash, failure hash).
        // Use the cn arrays directly.
        for (int roll = 1; roll <= 6; roll++) {
            SGoState result = SGoRules.applyPlacement(state, cell, roll);
            long h = result.stateHash;

            int existingIdx = cn.indexOf(h);
            if (existingIdx >= 0) {
                cn.outcomes[existingIdx].probability += prob;
            } else {
                boolean turnEnd = result.isTurnEnd;
                StateNode child = ctx.tt.getOrCreate(h, result, turnEnd);
                cn.addOutcome(h, prob, child);
            }
        }
    }

    /**
     * Entropy-minimizing outcome selection.
     * Returns index into cn.outcomes[].
     */
    private int selectOutcome(ChanceNode cn) {
        int count = cn.outcomeCount;

        // Find any unexplored outcome; prefer highest probability among unexplored
        int bestUnexplored = -1;
        double bestUnexploredProb = -1.0;
        for (int i = 0; i < count; i++) {
            if (cn.visitCounts[i] == 0) {
                double p = cn.outcomes[i].probability;
                if (p > bestUnexploredProb) {
                    bestUnexploredProb = p;
                    bestUnexplored = i;
                }
            }
        }
        if (bestUnexplored >= 0) return bestUnexplored;

        // All explored: pick most underrepresented (max prob / empirical_fraction)
        int N = cn.totalVisits;
        double bestRatio = -1.0;
        int bestIdx = 0;
        for (int i = 0; i < count; i++) {
            double ratio = cn.outcomes[i].probability / ((double) cn.visitCounts[i] / N);
            if (ratio > bestRatio) {
                bestRatio = ratio;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Walk path in reverse, updating visit counts and value sums.
     * Transposition safety: update StateNode at most once per iteration.
     */
    private void backpropagate(int pathEnd, double value) {
        int visitedCount = 0;

        for (int i = pathEnd - 1; i >= 0; i--) {
            PathEntry pe = ctx.path[i];
            StateNode node = pe.node;
            ActionEdge edge = node.edges[pe.actionId];

            edge.visitCount++;
            edge.valueSum += value;

            // Check if we've already updated this node this iteration
            long hash = node.stateHash;
            boolean alreadyVisited = false;
            for (int j = 0; j < visitedCount; j++) {
                if (visitedHashes[j] == hash) {
                    alreadyVisited = true;
                    break;
                }
            }

            if (!alreadyVisited) {
                node.visitCount++;
                node.valueSum += value;
                visitedHashes[visitedCount++] = hash;
            }
        }
    }

    /**
     * Look up the best action for a state already in the TT without running new iterations.
     * Returns null if the state is unknown or unexpanded.
     */
    public SGoAction bestKnownAction(SGoState state) {
        StateNode node = ctx.tt.lookup(state.stateHash);
        if (node == null || !node.isExpanded()) return null;
        return bestAction(node);
    }

    /** Return the most-visited action at the root. */
    private SGoAction bestAction(StateNode root) {
        int bestVisits = -1;
        int bestId = -1;
        int[] edgeIds = root.edgeIds;
        if (edgeIds == null) return SGoAction.END_TURN;

        ActionEdge[] edges = root.edges;
        for (int i = 0; i < root.edgeCount; i++) {
            int id = edgeIds[i];
            ActionEdge edge = edges[id];
            if (edge.visitCount > bestVisits) {
                bestVisits = edge.visitCount;
                bestId = id;
            }
        }
        if (bestId < 0 || bestId == SGoAction.END_TURN_ID) return SGoAction.END_TURN;
        return SGoAction.PLACE_ACTIONS[bestId];
    }

    private SearchStats collectStats(StateNode root, long elapsedNs, int budget, int maxDepth) {
        SearchStats stats = new SearchStats();
        stats.maxDepthReached = maxDepth;

        TranspositionTable tt = ctx.tt;
        stats.totalStateNodes = tt.size();
        stats.totalChanceNodes = ctx.chanceNodeCount;
        stats.totalTranspositionHits = tt.totalHits;
        stats.totalTranspositionAttempts = tt.totalAttempts;

        // Branching factor: avg edges among expanded nodes
        StateNode[] allNodes = tt.allNodes();
        int expandedCount = 0;
        long totalEdges = 0;
        for (StateNode n : allNodes) {
            if (n.isExpanded()) {
                totalEdges += n.edgeCount;
                expandedCount++;
            }
        }
        if (expandedCount > 0) {
            stats.avgBranchingFactor = (double) totalEdges / expandedCount;
        }

        // Root visit distribution
        if (root.isExpanded()) {
            int[] edgeIds = root.edgeIds;
            ActionEdge[] edges = root.edges;
            for (int i = 0; i < root.edgeCount; i++) {
                int id = edgeIds[i];
                ActionEdge edge = edges[id];
                String key = id == SGoAction.END_TURN_ID ? "end_turn"
                        : "place(" + (id / SGoState.BOARD_SIZE) + "," + (id % SGoState.BOARD_SIZE) + ")";
                stats.rootVisitDistribution.put(key, edge.visitCount);
            }
            stats.rootVisitEntropy = SearchStats.entropy(stats.rootVisitDistribution);
        }

        // KL divergence across chance nodes
        double klSum = 0.0;
        int klCount = 0;
        for (StateNode n : allNodes) {
            if (!n.isExpanded()) continue;
            ActionEdge[] edges = n.edges;
            int[] edgeIds = n.edgeIds;
            for (int i = 0; i < n.edgeCount; i++) {
                ActionEdge edge = edges[edgeIds[i]];
                ChanceNode cn = edge.chanceNode;
                if (cn != null && cn.totalVisits > 0) {
                    klSum += klDivergence(cn);
                    klCount++;
                }
            }
        }
        stats.avgChanceKl = klCount > 0 ? klSum / klCount : 0.0;
        stats.chanceNodeCountForKl = klCount;

        stats.finalize(elapsedNs, budget);
        return stats;
    }

    /** KL(empirical || theoretical) for a chance node. */
    private double klDivergence(ChanceNode cn) {
        double kl = 0.0;
        int N = cn.totalVisits;
        for (int i = 0; i < cn.outcomeCount; i++) {
            double empirical = (double) cn.visitCounts[i] / N;
            double theoretical = cn.outcomes[i].probability;
            if (empirical > 0 && theoretical > 0) {
                kl += empirical * Math.log(empirical / theoretical);
            }
        }
        return kl;
    }

    public static final class SearchResult {
        public final SGoAction action;
        public final SearchStats stats;

        public SearchResult(SGoAction action, SearchStats stats) {
            this.action = action;
            this.stats = stats;
        }
    }
}
