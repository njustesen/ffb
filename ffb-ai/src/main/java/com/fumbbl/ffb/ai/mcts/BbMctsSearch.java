package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.net.commands.ClientCommandActingPlayer;
import com.fumbbl.ffb.net.commands.ClientCommandEndTurn;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.ai.simulation.GameSnapshot;
import com.fumbbl.ffb.ai.simulation.HeadlessFantasyFootballServer;
import com.fumbbl.ffb.ai.simulation.MatchRunner;
import com.fumbbl.ffb.ai.simulation.RolloutSetup;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Blood Bowl MCTS activation selector with static leaf evaluation.
 *
 * <p>At each {@code INIT_SELECTING} phase-1 decision point this class runs
 * {@code budget} MCTS iterations and returns the most-visited candidate.
 *
 * <h3>Algorithm</h3>
 * Each iteration:
 * <ol>
 *   <li>Restores the cloned game state from a {@link GameSnapshot} (taken once per
 *       {@link #selectActivation} call, ~0.2 ms) instead of re-cloning via JSON.</li>
 *   <li><b>Selection</b>: traverse from root using UCB (or prior ordering for
 *       unvisited arms), advancing one activation per level via
 *       {@link MatchRunner#advanceToNextActivationOrTurnEnd}.  Candidate lists are
 *       cached on each {@link BbMctsNode} after their first internal visit — the
 *       game state at any node is deterministic, so the cache is always valid.</li>
 *   <li><b>Expansion</b>: advance the chosen unvisited action.</li>
 *   <li><b>Evaluation</b>: evaluate the new leaf state with {@link #staticEval}
 *       (~0.05 ms) instead of a scripted rollout (~5 ms).  staticEval uses
 *       score differential + standing-player count.</li>
 *   <li><b>Backpropagation</b>: update {@code visitCount}/{@code valueSum}.</li>
 * </ol>
 * The tree grows by exactly one node per iteration.  Per-iteration cost is
 * dominated by ~1 activation of game-step execution (~0.3 ms) plus one
 * {@code buildCandidates} call per new node (~1 ms, cached thereafter).
 * Target: ~1,000–2,000 iter/s single-threaded.
 *
 * <h3>MCTS-UCB vs MCTS-Prior</h3>
 * With an {@link IActionPrior} set: unvisited arms are explored in descending
 * prior order.  Once all arms are visited, plain UCB governs.
 *
 * <h3>Evaluation</h3>
 * {@code staticEval} combines score delta, ball possession + advancement,
 * on-field player count, and turn urgency.  {@code winProb} is the score-only
 * version used for turn-end and game-over leaves.
 */
public final class BbMctsSearch {

    /** UCB exploration constant. */
    private static final double C_UCB = 1.41421356237;

    /** Sigmoid scale: one TD lead → win_prob ≈ 0.95. */
    private static final double TD_SCALE = 3.0;

    public static final int DEFAULT_BUDGET = 10;

    /**
     * Maximum activation depth per tree iteration.
     * Capped at 4: enough to see 2 activations of ours + opponent response,
     * while preventing PUCT from following high-value paths into pathologically
     * deep trees that make each iteration extremely expensive.
     * (Old value 15 caused MCTS-Prior with ball-possession eval to build
     * depth-15 trees, costing ~75ms/iter × 200 budget = 15s/decision.)
     */
    private static final int MAX_TRAVERSE_DEPTH = 4;

    /**
     * Weight for the on-field-player-count signal in {@link #staticEval}.
     * A difference of 11 standing players contributes this much to the raw logit
     * (relative to TD_SCALE=3 for a one-TD lead).
     */
    private static final double STANDING_SCALE = 0.15;

    /**
     * Weight for the ball-possession + advancement signal in {@link #staticEval}.
     * At 1.0: ball carrier near endzone contributes ~0.8 to raw logit → sigmoid ≈ 0.69.
     * Kept modest to avoid PUCT over-exploiting high-Q "ball near endzone" states
     * when the eval is not sufficiently calibrated for the search budget.
     */
    private static final double BALL_SCALE = 1.0;

    /**
     * Weight for the turn-urgency modifier in {@link #staticEval}.
     * Amplifies the score-delta signal proportionally to game progress so that
     * a deficit late in the second half is penalised more than the same deficit
     * in the first half.
     */
    private static final double URGENCY_SCALE = 0.5;

    /** Shared thread pool for root parallelism (daemon threads so JVM exits cleanly). */
    private static final ExecutorService POOL =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors(), r -> {
            Thread t = Executors.defaultThreadFactory().newThread(r);
            t.setDaemon(true);
            return t;
        });

    // ── Configuration ─────────────────────────────────────────────────────────

    private final HeadlessFantasyFootballServer server;

    /**
     * Scripted runner used to advance sub-steps within an activation
     * (action commands, dialog responses, etc.) and to complete turns via
     * {@link MatchRunner#advanceTurnEndScripted}.
     */
    private final MatchRunner scriptedRunner;

    private final int budget;

    /**
     * Wall-clock time budget per decision in milliseconds.
     * When {@code > 0}, {@link #selectActivation(Game, boolean)} uses a time-based
     * stopping criterion instead of the fixed iteration count; each thread runs
     * iterations until the deadline, maximising search depth for the given time.
     * When {@code 0} (default), the fixed {@link #budget} iteration count is used.
     */
    private long timeBudgetMs = 0;

    /**
     * Maximum threads to use for parallel search.
     * Defaults to all available processors.  Set to {@code 1} to disable
     * parallelism entirely (useful for ablation studies).
     */
    private int maxThreads = Runtime.getRuntime().availableProcessors();

    /** Optional prior for MCTS-Prior mode.  Null = plain UCB. */
    private IActionPrior actionPrior;

    /** Leaf-node evaluator.  Defaults to the fast heuristic static eval. */
    private ILeafEval leafEval = new StaticLeafEval();

    /**
     * When true, MCTS traversal crosses turn boundaries: after a home-team turn
     * ends, the opponent's full scripted turn is simulated and the search continues
     * into the home team's NEXT activation.  This gives genuine multi-turn
     * look-ahead (vs the default which evaluates at turn-end with score-only
     * {@code winProb}).
     *
     * <p>Cost: each turn-crossing iteration calls {@code advanceTurnEndScripted}
     * (~5–15 ms vs ~1.7 ms baseline), so effective iterations per 100 ms budget
     * drop from ~60 to ~10–20.  Use time-based search (≥ 200 ms) when enabled.
     */
    private boolean crossTurnSearch = false;

    public void setCrossTurnSearch(boolean enabled) { this.crossTurnSearch = enabled; }

    private final BbMctsStats stats = new BbMctsStats();

    /**
     * Persistent rollout {@link GameState} for single-threaded search.
     * Created once, synced per decision — eliminates 29 ms JSON clone each time.
     */
    private GameState singleThreadedCtx;

    /**
     * Per-thread-index persistent rollout state for parallel search.
     * Index {@code ti} is only accessed by the task submitted as thread {@code ti},
     * so no synchronization needed.
     */
    private static final int MAX_THREADS = Runtime.getRuntime().availableProcessors();
    private final GameState[]                        parallelCtxs    = new GameState[MAX_THREADS];
    private final HeadlessFantasyFootballServer[]    parallelServers = new HeadlessFantasyFootballServer[MAX_THREADS];
    private final MatchRunner[]                      parallelRunners = new MatchRunner[MAX_THREADS];

    // ── Construction ──────────────────────────────────────────────────────────

    public BbMctsSearch(HeadlessFantasyFootballServer server,
                        MatchRunner scriptedRunner,
                        int budget) {
        this.server = server;
        this.scriptedRunner = scriptedRunner;
        this.budget = budget;
    }

    public BbMctsSearch(HeadlessFantasyFootballServer server, MatchRunner scriptedRunner) {
        this(server, scriptedRunner, DEFAULT_BUDGET);
    }

    public void setActionPrior(IActionPrior prior) {
        this.actionPrior = prior;
    }

    public void setLeafEval(ILeafEval eval) {
        this.leafEval = eval;
    }

    /**
     * Switch this agent to time-based search.
     * Each call to {@link #selectActivation} will run all threads for {@code ms}
     * milliseconds of wall-clock time, then return the most-visited candidate.
     * Overrides the fixed iteration {@link #budget}.
     */
    public void setTimeBudgetMs(long ms) {
        this.timeBudgetMs = ms;
    }

    /**
     * Cap the number of parallel threads.  Pass {@code 1} to run single-threaded
     * (no root parallelism); pass {@link Runtime#availableProcessors()} to restore
     * the default.  Affects both count-based and time-based search.
     */
    public void setMaxThreads(int maxThreads) {
        this.maxThreads = Math.max(1, maxThreads);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public BbAction selectActivation(Game liveGame, boolean isHome) {
        if (timeBudgetMs > 0) {
            return selectActivationTimed(liveGame, isHome, timeBudgetMs);
        }
        return selectActivation(liveGame, isHome, budget);
    }

    public BbAction selectActivation(Game liveGame, boolean isHome, int searchBudget) {
        CandidateSet rootSet = buildCandidates(liveGame, isHome, ThreadLocalRandom.current());
        if (rootSet.actions.isEmpty()) return BbAction.END_TURN;

        double winProbRoot = winProb(liveGame, isHome);

        // Minimum iterations per thread so createFromMidGame overhead is amortized.
        // With budget < MIN_ITERS_PER_THREAD, use single-threaded to avoid 8× clone cost.
        final int MIN_ITERS_PER_THREAD = 10;
        int nThreads = Math.min(maxThreads,
            Math.max(1, searchBudget / MIN_ITERS_PER_THREAD));
        if (nThreads <= 1) {
            return selectSingleThreaded(liveGame, isHome, searchBudget, rootSet, winProbRoot);
        }
        return selectParallel(liveGame, isHome, searchBudget, rootSet, winProbRoot, nThreads);
    }

    /**
     * Time-based search: each thread runs iterations until the wall-clock deadline,
     * then all results are merged.  The number of iterations varies per decision
     * (complex states get fewer iterations because each one takes longer).
     */
    public BbAction selectActivationTimed(Game liveGame, boolean isHome, long budgetMs) {
        CandidateSet rootSet = buildCandidates(liveGame, isHome, ThreadLocalRandom.current());
        if (rootSet.actions.isEmpty()) return BbAction.END_TURN;
        double winProbRoot = winProb(liveGame, isHome);
        long deadlineNs = System.nanoTime() + budgetMs * 1_000_000L;
        return selectParallelTimed(liveGame, isHome, rootSet, winProbRoot, maxThreads, deadlineNs);
    }

    /** Single-threaded search (used when budget is very small or for leaf decisions). */
    private BbAction selectSingleThreaded(Game liveGame, boolean isHome,
                                          int searchBudget,
                                          CandidateSet rootSet,
                                          double winProbRoot) {
        List<BbAction> rootCandidates = rootSet.actions;
        BbMctsNode root = new BbMctsNode();
        if (actionPrior != null) setPriors(root, rootSet, liveGame);

        // First call: create the rollout GameState (29 ms JSON clone + init).
        // Subsequent calls: sync the existing state from the live game (~0.1 ms).
        GameSnapshot snapshot;
        if (singleThreadedCtx == null) {
            try {
                singleThreadedCtx = RolloutSetup.createFromMidGame(liveGame, server);
            } catch (Exception e) {
                return rootCandidates.get(0);
            }
            snapshot = GameSnapshot.take(singleThreadedCtx.getGame());
        } else {
            try {
                snapshot = RolloutSetup.syncFromLiveGame(singleThreadedCtx, liveGame);
            } catch (Exception e) {
                // Sync failed (e.g. game structure diverged) — fall back to fresh clone.
                singleThreadedCtx = null;
                try {
                    singleThreadedCtx = RolloutSetup.createFromMidGame(liveGame, server);
                } catch (Exception e2) {
                    return rootCandidates.get(0);
                }
                snapshot = GameSnapshot.take(singleThreadedCtx.getGame());
            }
        }
        GameState ctx = singleThreadedCtx;

        long searchNsTotal = 0;
        int maxDepth = 0;
        int totalNodes = 0;

        GameState.setStepDepthLimitEnabled(true);
        try {
            for (int iter = 0; iter < searchBudget; iter++) {
                RolloutSetup.resetForIteration(ctx, snapshot);
                long t0 = System.nanoTime();
                int[] nodeCount = {0};
                int depth = traverse(root, ctx, isHome, winProbRoot, rootCandidates, 0,
                    ThreadLocalRandom.current(), nodeCount, scriptedRunner);
                searchNsTotal += System.nanoTime() - t0;
                if (depth > maxDepth) maxDepth = depth;
                totalNodes += nodeCount[0];
            }
        } finally {
            GameState.setStepDepthLimitEnabled(false);
        }

        stats.recordDecision(rootCandidates.size(), searchBudget,
            searchNsTotal, maxDepth, totalNodes);

        return rootCandidates.get(argmaxVisits(root, rootCandidates));
    }

    /** Parallel search: N independent trees merged by visit counts. */
    private BbAction selectParallel(Game liveGame, boolean isHome,
                                    int searchBudget,
                                    CandidateSet rootSet,
                                    double winProbRoot,
                                    int nThreads) {
        List<BbAction> rootCandidates = rootSet.actions;
        int budgetPerThread = Math.max(1, searchBudget / nThreads);
        @SuppressWarnings("unchecked")
        BbMctsNode[] threadRoots = new BbMctsNode[nThreads];
        long[] threadSearchNs = new long[nThreads];
        int[] threadMaxDepth = new int[nThreads];
        int[] threadNodes = new int[nThreads];

        List<Future<?>> futures = new ArrayList<>();
        for (int t = 0; t < nThreads; t++) {
            final int ti = t;
            threadRoots[ti] = new BbMctsNode();
            if (actionPrior != null) setPriors(threadRoots[ti], rootSet, liveGame);

            futures.add(POOL.submit(() -> {
                // First call for this thread index: create rollout state (29 ms JSON clone).
                // Subsequent calls: sync from live game (~0.1 ms).
                GameState tCtx = parallelCtxs[ti];
                GameSnapshot tSnap;
                if (tCtx == null) {
                    parallelServers[ti] = new HeadlessFantasyFootballServer();
                    try {
                        tCtx = RolloutSetup.createFromMidGame(liveGame, parallelServers[ti]);
                    } catch (Exception e) {
                        return;
                    }
                    parallelCtxs[ti] = tCtx;
                    parallelRunners[ti] = new MatchRunner(null, null,
                        MatchRunner.AgentMode.SCRIPTED_ARGMAX, MatchRunner.AgentMode.SCRIPTED_ARGMAX);
                    tSnap = GameSnapshot.take(tCtx.getGame());
                } else {
                    try {
                        tSnap = RolloutSetup.syncFromLiveGame(tCtx, liveGame);
                    } catch (Exception e) {
                        // Sync failed — recreate this thread's context.
                        parallelServers[ti] = new HeadlessFantasyFootballServer();
                        try {
                            tCtx = RolloutSetup.createFromMidGame(liveGame, parallelServers[ti]);
                        } catch (Exception e2) {
                            return;
                        }
                        parallelCtxs[ti] = tCtx;
                        tSnap = GameSnapshot.take(tCtx.getGame());
                    }
                }
                // Each thread uses its own MatchRunner (no shared-state races on rng/comm).
                MatchRunner tRunner = parallelRunners[ti];
                Random rng = ThreadLocalRandom.current();

                long nsTotal = 0;
                int maxD = 0;
                int nodes = 0;

                GameState.setStepDepthLimitEnabled(true);
                try {
                    for (int iter = 0; iter < budgetPerThread; iter++) {
                        RolloutSetup.resetForIteration(tCtx, tSnap);
                        long t0 = System.nanoTime();
                        int[] nodeCount = {0};
                        int depth = traverse(threadRoots[ti], tCtx, isHome,
                            winProbRoot, rootCandidates, 0, rng, nodeCount, tRunner);
                        nsTotal += System.nanoTime() - t0;
                        if (depth > maxD) maxD = depth;
                        nodes += nodeCount[0];
                    }
                } finally {
                    GameState.setStepDepthLimitEnabled(false);
                }
                threadSearchNs[ti]  = nsTotal;
                threadMaxDepth[ti]  = maxD;
                threadNodes[ti]     = nodes;
            }));
        }

        // Wait for all threads.
        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception e) { /* ignore */ }
        }

        // Merge roots: sum visitCount/valueSum for each candidate's child.
        BbMctsNode merged = mergeRoots(threadRoots, rootCandidates);

        long totalNs = 0; int maxDepth = 0; int totalNodes = 0;
        for (int t = 0; t < nThreads; t++) {
            totalNs    += threadSearchNs[t];
            if (threadMaxDepth[t] > maxDepth) maxDepth = threadMaxDepth[t];
            totalNodes += threadNodes[t];
        }
        stats.recordDecision(rootCandidates.size(), budgetPerThread * nThreads,
            totalNs, maxDepth, totalNodes);

        return rootCandidates.get(argmaxVisits(merged, rootCandidates));
    }

    /** Time-based parallel search: each thread runs until {@code deadlineNs}. */
    private BbAction selectParallelTimed(Game liveGame, boolean isHome,
                                         CandidateSet rootSet,
                                         double winProbRoot,
                                         int nThreads,
                                         long deadlineNs) {
        List<BbAction> rootCandidates = rootSet.actions;
        BbMctsNode[] threadRoots = new BbMctsNode[nThreads];
        long[] threadSearchNs = new long[nThreads];
        int[] threadMaxDepth = new int[nThreads];
        int[] threadNodes = new int[nThreads];
        int[] threadIters = new int[nThreads];

        List<Future<?>> futures = new ArrayList<>();
        for (int t = 0; t < nThreads; t++) {
            final int ti = t;
            threadRoots[ti] = new BbMctsNode();
            if (actionPrior != null) setPriors(threadRoots[ti], rootSet, liveGame);

            futures.add(POOL.submit(() -> {
                GameState tCtx = parallelCtxs[ti];
                GameSnapshot tSnap;
                if (tCtx == null) {
                    parallelServers[ti] = new HeadlessFantasyFootballServer();
                    try {
                        tCtx = RolloutSetup.createFromMidGame(liveGame, parallelServers[ti]);
                    } catch (Exception e) { return; }
                    parallelCtxs[ti] = tCtx;
                    parallelRunners[ti] = new MatchRunner(null, null,
                        MatchRunner.AgentMode.SCRIPTED_ARGMAX, MatchRunner.AgentMode.SCRIPTED_ARGMAX);
                    tSnap = GameSnapshot.take(tCtx.getGame());
                } else {
                    try {
                        tSnap = RolloutSetup.syncFromLiveGame(tCtx, liveGame);
                    } catch (Exception e) {
                        parallelServers[ti] = new HeadlessFantasyFootballServer();
                        try {
                            tCtx = RolloutSetup.createFromMidGame(liveGame, parallelServers[ti]);
                        } catch (Exception e2) { return; }
                        parallelCtxs[ti] = tCtx;
                        tSnap = GameSnapshot.take(tCtx.getGame());
                    }
                }
                MatchRunner tRunner = parallelRunners[ti];
                Random rng = ThreadLocalRandom.current();

                long nsTotal = 0;
                int maxD = 0, nodes = 0, iters = 0;

                GameState.setStepDepthLimitEnabled(true);
                try {
                    while (System.nanoTime() < deadlineNs) {
                        RolloutSetup.resetForIteration(tCtx, tSnap);
                        long t0 = System.nanoTime();
                        int[] nodeCount = {0};
                        int depth = traverse(threadRoots[ti], tCtx, isHome,
                            winProbRoot, rootCandidates, 0, rng, nodeCount, tRunner);
                        nsTotal += System.nanoTime() - t0;
                        if (depth > maxD) maxD = depth;
                        nodes += nodeCount[0];
                        iters++;
                    }
                } finally {
                    GameState.setStepDepthLimitEnabled(false);
                }
                threadSearchNs[ti] = nsTotal;
                threadMaxDepth[ti] = maxD;
                threadNodes[ti]    = nodes;
                threadIters[ti]    = iters;
            }));
        }

        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception e) { /* ignore */ }
        }

        BbMctsNode merged = mergeRoots(threadRoots, rootCandidates);

        long totalNs = 0; int maxDepth = 0, totalNodes = 0, totalIters = 0;
        for (int t = 0; t < nThreads; t++) {
            totalNs    += threadSearchNs[t];
            if (threadMaxDepth[t] > maxDepth) maxDepth = threadMaxDepth[t];
            totalNodes += threadNodes[t];
            totalIters += threadIters[t];
        }
        stats.recordDecision(rootCandidates.size(), totalIters, totalNs, maxDepth, totalNodes);

        return rootCandidates.get(argmaxVisits(merged, rootCandidates));
    }

    private BbMctsNode mergeRoots(BbMctsNode[] roots, List<BbAction> candidates) {
        BbMctsNode merged = new BbMctsNode();
        for (BbMctsNode root : roots) {
            merged.visitCount += root.visitCount;
            merged.valueSum   += root.valueSum;
            for (BbAction action : candidates) {
                BbMctsNode child = root.children.get(action);
                if (child != null) {
                    BbMctsNode mc = merged.getOrCreateChild(action);
                    mc.visitCount += child.visitCount;
                    mc.valueSum   += child.valueSum;
                }
            }
        }
        return merged;
    }

    public BbMctsStats getStats() {
        return stats;
    }

    // ── Tree traversal ────────────────────────────────────────────────────────

    /**
     * Standard MCTS-PR traversal: selection + expansion + simulation (rollout) +
     * backpropagation, all in one recursive call.
     *
     * <p>The key invariant: when {@code child.visitCount == 0} (new node), we run
     * a scripted-argmax rollout to turn end (<em>do not recurse</em>).  This is
     * standard MCTS — one new node expanded per iteration, then rollout.
     *
     * @param nodeCount accumulator incremented on every new node creation
     * @return the depth reached
     */
    private int traverse(BbMctsNode node, GameState gameState,
                         boolean isHome, double winProbRoot,
                         List<BbAction> candidates, int depth,
                         Random rng, int[] nodeCount, MatchRunner runner) {
        return traverse(node, gameState, isHome, winProbRoot, candidates, depth, rng, nodeCount, runner, 0);
    }

    /**
     * @param turnCrossings number of times this iteration has already crossed a turn
     *   boundary via cross-turn search; limits recursive turn-crossings to 1 per
     *   iteration to prevent each iteration from simulating 3+ full opponent turns.
     */
    private int traverse(BbMctsNode node, GameState gameState,
                         boolean isHome, double winProbRoot,
                         List<BbAction> candidates, int depth,
                         Random rng, int[] nodeCount, MatchRunner runner,
                         int turnCrossings) {
        Game game = gameState.getGame();

        if (candidates.isEmpty() || depth >= MAX_TRAVERSE_DEPTH) {
            // No legal moves (or safety cap): end turn and evaluate.
            try {
                MatchRunner.inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                runner.advanceToNextActivationOrTurnEnd(gameState, isHome, MatchRunner.MAX_ACTIVATION_ITERATIONS);
            } catch (StackOverflowError | GameState.StepChainDepthLimitException e) {
                // ignore — fall through to evaluate current state
            }
            double value = winProb(game, isHome) - winProbRoot;
            node.visitCount++;
            node.valueSum += value;
            return depth;
        }

        BbAction action = selectAction(node, candidates, rng);
        // Pre-fetch the child node so we can invalidate its cache on StackOverflowError.
        BbMctsNode child = node.getOrCreateChild(action);

        // Inject the chosen activation.
        if (action.isEndTurn()) {
            MatchRunner.inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
        } else {
            try {
                MatchRunner.inject(gameState,
                    new ClientCommandActingPlayer(action.player.getId(), action.action, false));
            } catch (StackOverflowError | GameState.StepChainDepthLimitException e) {
                // Infinite step-chain recursion in the server (caught early via
                // StepChainDepthLimitException at depth 200, or via JVM StackOverflowError
                // as a last-resort fallback).  The cached candidates at this child may
                // be stale; invalidate so the next visit recomputes them.
                child.cachedSet = null;
                node.visitCount++;
                node.valueSum += 0.0;
                return depth;
            }
        }

        // Advance through sub-steps until next phase-1 INIT_SELECTING or turn end.
        boolean sameTurn;
        try {
            sameTurn = runner.advanceToNextActivationOrTurnEnd(gameState, isHome, MatchRunner.MAX_ACTIVATION_ITERATIONS);
        } catch (StackOverflowError | GameState.StepChainDepthLimitException e) {
            child.cachedSet = null;
            node.visitCount++;
            node.valueSum += 0.0;
            return depth;
        }

        double value = 0.0;
        int reachedDepth = depth + 1;

        if (!sameTurn || game.getFinished() != null) {
            // Turn ended or game over.
            if (game.getFinished() != null) {
                // Terminal: score is final.
                value = winProb(game, isHome) - winProbRoot;
            } else if (crossTurnSearch && depth < MAX_TRAVERSE_DEPTH - 1 && turnCrossings == 0) {
                // ── CROSS-TURN SEARCH ───────────────────────────────────────
                // Simulate the opponent's full scripted turn, then continue
                // the MCTS tree into the home team's next activation.
                // Limited to 1 turn crossing per iteration (turnCrossings==0 guard)
                // to prevent exponential iteration cost from multiple Dijkstra calls.
                boolean continued = false;
                try {
                    runner.advanceTurnEndScripted(gameState, !isHome);
                    if (game.getFinished() == null) {
                        CandidateSet nextSet = buildCandidates(game, isHome, rng);
                        if (!nextSet.actions.isEmpty()) {
                            if (actionPrior != null && child.priors == null)
                                setPriors(child, nextSet, game);
                            reachedDepth = traverse(child, gameState, isHome, winProbRoot,
                                nextSet.actions, depth + 1, rng, nodeCount, runner, 1);
                            value = child.q();
                            continued = true;
                        }
                    }
                } catch (StackOverflowError | GameState.StepChainDepthLimitException e) {
                    // fall through
                }
                if (!continued) {
                    value = winProb(game, isHome) - winProbRoot;
                    if (child.visitCount == 0) nodeCount[0]++;
                    child.visitCount++;
                    child.valueSum += value;
                }
                node.visitCount++;
                node.valueSum += value;
                return reachedDepth;
            } else {
                // Score-based eval at turn-end (rich eval over-exploits ball signal).
                value = winProb(game, isHome) - winProbRoot;
            }
            if (child.visitCount == 0) nodeCount[0]++;
            child.visitCount++;
            child.valueSum += value;

        } else if (child.visitCount == 0) {
            // ── EXPANSION with Argmax-quality leaf eval ─────────────────────
            // Call buildCandidates now (same Dijkstra call that would happen on the
            // second visit for selection) and cache the result.  This way:
            // 1. We get rawScores (Argmax action scores) for a better eval signal.
            // 2. The second visit (selection) reuses the cache — no extra Dijkstra.
            //
            // Leaf eval = blend of rawScore-based signal + staticEval.
            // rawScore normalization: scores in roughly [-20, 50]; /8.0 maps that to
            // sigmoid range [-2.5, 6.25] → probability [0.08, 1.00].
            nodeCount[0]++;
            CandidateSet leafSet = buildCandidates(game, isHome, rng);
            child.cachedSet = leafSet;  // pre-cache so selection visits don't re-run Dijkstra
            if (actionPrior != null) setPriors(child, leafSet, game);
            double argmaxSignal;
            if (leafSet.rawScores != null && leafSet.rawScores.length > 0) {
                double maxScore = leafSet.rawScores[0];
                for (double s : leafSet.rawScores) if (s > maxScore) maxScore = s;
                // rawScores are softmaxScore() = 1.0 + p*v*c ∈ [0, 2], baseline 1.0 = neutral.
                // sigmoid(2*(maxScore-1)) maps: 0.75→0.38, 1.0→0.50, 1.5→0.73, 2.0→0.88.
                // Comparable scale to winProb turn-end signal (0 for draw, ±0.45 for ±1 TD).
                argmaxSignal = 1.0 / (1.0 + Math.exp(-2.0 * (maxScore - 1.0)));
            } else {
                argmaxSignal = leafEval.evaluate(game, isHome);
            }
            value = argmaxSignal - winProbRoot;
            child.visitCount++;
            child.valueSum += value;

        } else {
            // ── SELECTION: internal node (visited) ──────────────────────────
            // Use cached candidates to avoid repeated Dijkstra calls.  Blood Bowl
            // is stochastic (dice), so the game state at this node varies across
            // iterations.  Validate the cache before use: if any non-EndTurn
            // candidate player is no longer active/standing-or-prone, the cache is
            // stale and must be recomputed (prevents invalid action injection which
            // causes server infinite recursion).
            CandidateSet nextSet = child.cachedSet;
            if (nextSet == null || !areCandidatesValid(nextSet.actions, game)) {
                nextSet = buildCandidates(game, isHome, rng);
                child.cachedSet = nextSet;
            }
            // Priors already set at expansion time (via child.cachedSet pre-cache).
            // Recompute only if the cache was rebuilt (stale candidates) and priors are missing.
            if (actionPrior != null && child.priors == null && !nextSet.actions.isEmpty()) {
                setPriors(child, nextSet, game);
            }
            reachedDepth = traverse(child, gameState, isHome, winProbRoot,
                nextSet.actions, depth + 1, rng, nodeCount, runner, turnCrossings);
            value = child.q();
        }

        // Backpropagate.
        node.visitCount++;
        node.valueSum += value;
        return reachedDepth;
    }

    // ── Action selection ──────────────────────────────────────────────────────

    private BbAction selectAction(BbMctsNode node, List<BbAction> candidates, Random rng) {
        boolean hasPriors = (node.priors != null);

        // Phase 1: find the best unvisited candidate.
        BbAction bestUnvisited = null;
        double bestUnvisitedScore = Double.NEGATIVE_INFINITY;

        for (BbAction action : candidates) {
            BbMctsNode child = node.children.get(action);
            if (child == null || child.visitCount == 0) {
                double score = hasPriors ? node.getPrior(action) : -candidates.indexOf(action);
                if (score > bestUnvisitedScore) {
                    bestUnvisitedScore = score;
                    bestUnvisited = action;
                }
            }
        }
        if (bestUnvisited != null) return bestUnvisited;

        // Phase 2: all visited — UCB.
        double logN = Math.log(node.visitCount + 1);
        double bestUcb = Double.NEGATIVE_INFINITY;
        BbAction bestAction = candidates.get(0);
        int tieCount = 0;

        for (BbAction action : candidates) {
            BbMctsNode child = node.children.get(action);
            int n = (child != null) ? child.visitCount : 0;
            double q = (child != null) ? child.q() : 0.0;
            double ucb = q + C_UCB * Math.sqrt(logN / (n + 1));
            if (ucb > bestUcb) {
                bestUcb = ucb;
                bestAction = action;
                tieCount = 1;
            } else if (ucb == bestUcb) {
                tieCount++;
                if (rng.nextInt(tieCount) == 0) bestAction = action;
            }
        }
        return bestAction;
    }

    private int argmaxVisits(BbMctsNode root, List<BbAction> candidates) {
        int best = 0;
        int bestVisits = -1;
        for (int i = 0; i < candidates.size(); i++) {
            BbMctsNode child = root.children.get(candidates.get(i));
            int v = (child != null) ? child.visitCount : 0;
            if (v > bestVisits) {
                bestVisits = v;
                best = i;
            }
        }
        return best;
    }

    // ── Prior computation ─────────────────────────────────────────────────────

    private void setPriors(BbMctsNode node, CandidateSet cs, Game game) {
        double[] priorValues = actionPrior.computePriorFromScores(cs.actions, cs.rawScores, game);
        if (priorValues == null) return;
        node.priors = new HashMap<>();
        for (int i = 0; i < cs.actions.size(); i++) {
            node.priors.put(cs.actions.get(i), priorValues[i]);
        }
    }

    // ── Evaluation ───────────────────────────────────────────────────────────

    /**
     * Heuristic win probability for the acting team.
     * {@code sigmoid(TD_SCALE × scoreDelta)} — one TD ahead → ~0.95.
     */
    private static double winProb(Game game, boolean isHome) {
        GameResult r = game.getGameResult();
        int delta = isHome
            ? (r.getScoreHome() - r.getScoreAway())
            : (r.getScoreAway() - r.getScoreHome());
        return 1.0 / (1.0 + Math.exp(-TD_SCALE * delta));
    }

    /**
     * Fast static evaluation at a leaf node — replaces the scripted rollout.
     *
     * <p>Calling cost: O(22 players) ≈ 0.05–0.1 ms — versus ~5 ms scripted rollout.
     *
     * <p>Signals:
     * <ul>
     *   <li>{@code TD_SCALE × scoreDelta} — score advantage (primary)
     *   <li>{@code BALL_SCALE × ballSignal} — ball possession and advancement
     *       toward opponent endzone; [−0.8, +0.8] range
     *   <li>{@code STANDING_SCALE × onFieldAdv / 11} — on-field player count
     *       advantage (counts standing + prone + stunned; excludes KO/injured)
     *   <li>{@code URGENCY_SCALE × gameProgress × scoreDelta} — amplifies score
     *       deficit signal proportionally to how late in the game it is
     * </ul>
     *
     * <p>Coordinate convention: home team attacks toward {@code x = 25}
     * (away endzone), confirmed from {@code MoveDecisionEngine.advanceScore = x/25.0}.
     */
    public static double staticEval(Game game, boolean isHome) {
        GameResult r = game.getGameResult();
        int scoreDelta = isHome
            ? (r.getScoreHome() - r.getScoreAway())
            : (r.getScoreAway() - r.getScoreHome());

        // ── On-field player count advantage ───────────────────────────────────
        // Count players physically deployed on the field (standing, prone, stunned).
        // Excludes KO'd and injured players who can no longer contribute.
        FieldModel fm = game.getFieldModel();
        int myOnField = 0;
        int oppOnField = 0;
        for (Player<?> p : game.getTeamHome().getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (ps != null) {
                int base = ps.getBase();
                boolean onField = (base == PlayerState.STANDING || base == PlayerState.MOVING
                    || base == PlayerState.PRONE || base == PlayerState.STUNNED);
                if (onField) { if (isHome) myOnField++; else oppOnField++; }
            }
        }
        for (Player<?> p : game.getTeamAway().getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (ps != null) {
                int base = ps.getBase();
                boolean onField = (base == PlayerState.STANDING || base == PlayerState.MOVING
                    || base == PlayerState.PRONE || base == PlayerState.STUNNED);
                if (onField) { if (isHome) oppOnField++; else myOnField++; }
            }
        }

        // ── Ball possession + field advancement ────────────────────────────────
        // Home attacks toward x=25; away attacks toward x=0.
        // ballSignal ∈ [−0.8, +0.8]:
        //   +0.3..+0.8  we carry the ball (more = closer to their endzone)
        //   −0.3..−0.8  they carry the ball (more negative = closer to our endzone)
        //   ≈ 0         loose ball or ball not in play
        double ballSignal = 0.0;
        FieldCoordinate ballCoord = fm.getBallCoordinate();
        if (ballCoord != null && fm.isBallInPlay()) {
            double normalizedX = isHome
                ? ballCoord.getX() / 25.0
                : (25.0 - ballCoord.getX()) / 25.0;
            Player<?> carrier = fm.getPlayer(ballCoord);
            if (carrier != null) {
                boolean carrierIsHome = false;
                for (Player<?> p : game.getTeamHome().getPlayers()) {
                    if (p == carrier) { carrierIsHome = true; break; }
                }
                boolean weHave = (carrierIsHome == isHome);
                if (weHave) {
                    ballSignal = 0.3 + 0.5 * normalizedX;          // [0.30, 0.80]
                } else {
                    ballSignal = -0.3 - 0.5 * (1.0 - normalizedX); // [−0.80, −0.30]
                }
            } else {
                // Loose ball: small bias toward opponent's half
                ballSignal = 0.05 * (normalizedX - 0.5);
            }
        }

        // ── Turn urgency ───────────────────────────────────────────────────────
        // A score deficit is more costly late in the game than early on.
        int myTurns  = (isHome ? game.getTurnDataHome() : game.getTurnDataAway()).getTurnNr();
        int oppTurns = (isHome ? game.getTurnDataAway() : game.getTurnDataHome()).getTurnNr();
        double gameProgress = ((game.getHalf() - 1) * 8 + Math.max(myTurns, oppTurns)) / 16.0;

        double raw = TD_SCALE      * scoreDelta
            + BALL_SCALE           * ballSignal
            + STANDING_SCALE       * (myOnField - oppOnField) / 11.0
            + URGENCY_SCALE        * gameProgress * scoreDelta;
        return 1.0 / (1.0 + Math.exp(-raw));
    }

    // ── Candidate enumeration ─────────────────────────────────────────────────

    /**
     * Returns {@code true} if all player-action candidates in the list are still
     * activatable in the current game state.
     *
     * <p>A cached candidate is stale when the stochastic game progression (dice)
     * reached a different state than when the cache was built — e.g., a player was
     * knocked down or injured.  This O(n) check (n ≈ 5–10 candidates) is far
     * cheaper than a full {@link #buildCandidates} (which runs Dijkstra per player).
     */
    private static boolean areCandidatesValid(List<BbAction> candidates, Game game) {
        FieldModel fm = game.getFieldModel();
        for (BbAction a : candidates) {
            if (a.isEndTurn()) continue;
            PlayerState ps = fm.getPlayerState(a.player);
            if (ps == null || !ps.isActive()) return false;
            int base = ps.getBase();
            if (base != PlayerState.STANDING && base != PlayerState.PRONE) return false;
        }
        return true;
    }

    private CandidateSet buildCandidates(Game game, boolean isHome, Random rng) {
        Team myTeam  = isHome ? game.getTeamHome() : game.getTeamAway();
        Team oppTeam = isHome ? game.getTeamAway() : game.getTeamHome();
        MoveDecisionEngine.PlayerSelection sel =
            MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, isHome, isHome, rng, false);
        List<BbAction> list = new ArrayList<>();
        List<Player<?>> players = sel.candidatePlayers;
        List<com.fumbbl.ffb.PlayerAction> actions = sel.candidateActions;
        for (int i = 0; i < players.size(); i++) {
            Player<?> p = players.get(i);
            if (p != null) list.add(new BbAction(p, actions.get(i)));
        }
        // Extract raw scores parallel with the filtered candidate list.
        double[] rawScores = null;
        if (sel.rawScores != null && !list.isEmpty()) {
            rawScores = new double[list.size()];
            int li = 0;
            for (int i = 0; i < players.size(); i++) {
                if (players.get(i) != null) rawScores[li++] = sel.rawScores[i];
            }
        }
        return new CandidateSet(list, rawScores);
    }
}
