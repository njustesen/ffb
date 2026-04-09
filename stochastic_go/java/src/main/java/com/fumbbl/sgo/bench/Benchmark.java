package com.fumbbl.sgo.bench;

import com.fumbbl.sgo.game.SGoAction;
import com.fumbbl.sgo.game.SGoRules;
import com.fumbbl.sgo.game.SGoState;
import com.fumbbl.sgo.mcts.MctsSearch;
import com.fumbbl.sgo.mcts.SearchContext;
import com.fumbbl.sgo.mcts.SearchStats;

import java.util.Random;

/**
 * Benchmark and tournament runner for Java MCTS-FT.
 *
 * Usage:
 *   java -jar stochastic-go-jar-with-dependencies.jar [--bench|--tournament|--single]
 */
public final class Benchmark {

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "--bench";

        switch (mode) {
            case "--bench":
                runBench();
                break;
            case "--tournament":
                runTournament();
                break;
            case "--selfplay":
                runSelfPlay();
                break;
            case "--selfplay-score":
                runSelfPlayScore();
                break;
            case "--pilot":
                runPilot(args.length > 1 ? Integer.parseInt(args[1]) : 30);
                break;
            case "--single":
                runSingle(args.length > 1 ? Integer.parseInt(args[1]) : 500);
                break;
            default:
                System.err.println("Unknown mode: " + mode);
                System.exit(1);
        }
    }

    /** Speed benchmark: warm up then measure iterations/second. */
    private static void runBench() {
        int budget = 10000;
        System.out.println("=== Java MCTS-FT Benchmark ===");
        System.out.println("Budget: " + budget + " iterations");

        // Warm up (3 rounds, discarded)
        System.out.print("Warming up...");
        for (int w = 0; w < 3; w++) {
            MctsSearch search = new MctsSearch(new SearchContext(), new Random(42));
            search.search(SGoState.initial(), budget);
            System.out.print(" " + (w + 1));
        }
        System.out.println(" done.");

        // Measure (3 rounds)
        double totalIps = 0;
        for (int r = 0; r < 3; r++) {
            MctsSearch search = new MctsSearch(new SearchContext(), new Random(42));
            MctsSearch.SearchResult result = search.search(SGoState.initial(), budget);
            SearchStats stats = result.stats;
            System.out.printf("Run %d: %s%n", r + 1, stats.summary());
            totalIps += stats.iterationsPerSecond;
        }
        System.out.printf("%nAverage: %.0f iter/s%n", totalIps / 3);
    }

    /** Tournament across time budgets (ms per MCTS call) to show improvement with more search. */
    private static void runTournament() {
        // {timeBudgetMs, games}: each P1 turn costs budgetMs, ~10 turns/game
        // game wall time ≈ budget × 10ms; games scaled so each row takes ~60–120s
        long[][] schedule = {{10, 300}, {50, 150}, {250, 50}, {1250, 12}};
        long seed = 1234;

        System.out.println("=== MCTS vs Random: win rate by time budget per decision ===");
        System.out.printf("%-10s  %5s  %6s  %6s  %6s  %7s%n",
                "budget(ms)", "games", "wins", "losses", "draws", "win%");
        System.out.println("-".repeat(50));

        for (long[] row : schedule) {
            long timeBudgetMs = row[0];
            int games = (int) row[1];
            int wins = 0, losses = 0, draws = 0;
            Random rng = new Random(seed);
            for (int g = 0; g < games; g++) {
                int result = playGameTimed(timeBudgetMs, rng);
                if (result > 0) wins++;
                else if (result < 0) losses++;
                else draws++;
            }
            System.out.printf("%-10d  %5d  %6d  %6d  %6d  %6.1f%%%n",
                    timeBudgetMs, games, wins, losses, draws, 100.0 * wins / games);
            System.out.flush();
        }
    }

    /**
     * Play one game: P1=MCTS, P2=random.
     * Returns positive if P1 wins, negative if P2 wins, 0 if draw.
     */
    /**
     * Play one game using a time budget per P1 turn.
     * MCTS searches once at the START of each P1 turn; subsequent placements within
     * the same turn are looked up from the existing tree at no extra cost.
     */
    private static int playGameTimed(long timeBudgetMs, Random rng) {
        SGoState state = SGoState.initial();
        Random mctsRng = new Random(rng.nextLong());
        SearchContext ctx = null;
        MctsSearch search = null;
        boolean p1TurnActive = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                if (!p1TurnActive) {
                    // Start of P1's turn: spend the full time budget building the tree.
                    ctx = new SearchContext();
                    search = new MctsSearch(ctx, mctsRng);
                    p1TurnActive = true;
                    search.searchForMs(state, timeBudgetMs);
                }
                // Look up best action from the existing tree (no new iterations).
                SGoAction action = search.bestKnownAction(state);
                if (action == null) {
                    // State not yet in tree (can happen after a surprise dice outcome).
                    action = search.searchForMs(state, timeBudgetMs).action;
                }
                state = applyAction(state, action, rng);
            } else {
                p1TurnActive = false;
                state = applyRandomMove(state, rng);
            }

            if (state.isTurnEnd && !state.isTerminal()) {
                state = SGoRules.advanceTurn(state);
            }
        }

        return Integer.compare(state.score(), 0);
    }

    private static int playGame(int budget, Random rng) {
        SGoState state = SGoState.initial();
        Random mctsRng = new Random(rng.nextLong());
        // Reset SearchContext each P1 turn to keep TT bounded at high budgets.
        SearchContext ctx = null;
        MctsSearch search = null;
        boolean p1TurnActive = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                // Start of a new P1 turn: fresh context for memory safety
                if (!p1TurnActive) {
                    ctx = new SearchContext();
                    search = new MctsSearch(ctx, mctsRng);
                    p1TurnActive = true;
                }
                MctsSearch.SearchResult result = search.search(state, budget);
                state = applyAction(state, result.action, rng);
            } else {
                p1TurnActive = false;
                state = applyRandomMove(state, rng);
            }

            if (state.isTurnEnd && !state.isTerminal()) {
                state = SGoRules.advanceTurn(state);
            }
        }

        return Integer.compare(state.score(), 0);
    }

    private static SGoState applyAction(SGoState state, SGoAction action, Random rng) {
        if (!action.isPlace()) {
            return SGoRules.applyEndTurn(state);
        }
        int roll = rng.nextInt(6) + 1;
        return SGoRules.applyPlacement(state, action.id, roll);
    }

    private static SGoState applyRandomMove(SGoState state, Random rng) {
        // Collect legal actions
        long emptyCells = state.emptyCells;
        int emptyCount = Long.bitCount(emptyCells);
        int total = 1 + emptyCount;

        int choice = rng.nextInt(total);
        if (choice == 0) {
            return SGoRules.applyEndTurn(state);
        }

        // Pick the (choice-1)-th empty cell
        long bits = emptyCells;
        int idx = 0;
        int target = choice - 1;
        while (bits != 0L) {
            int cell = Long.numberOfTrailingZeros(bits);
            if (idx == target) {
                int roll = rng.nextInt(6) + 1;
                return SGoRules.applyPlacement(state, cell, roll);
            }
            bits &= bits - 1L;
            idx++;
        }
        return SGoRules.applyEndTurn(state); // fallback
    }

    /**
     * Self-play tournament using deterministic iteration-based search.
     *
     * searchForMs() is non-deterministic (result depends on machine load), so the
     * same seed can give different outcomes across runs. Instead we:
     *   1. Measure iter/s with a short warmup.
     *   2. Convert the target ms budgets to fixed iteration counts.
     *   3. Use search(state, iters), which is fully deterministic given the same seed.
     *
     * Comparison: each budget vs the 10ms baseline (one-tailed z-test).
     * Game counts are sized so the ~8–10 pp cumulative improvement is detectable
     * at 95% confidence: n_base=600 (cheap), scaling down for expensive rows.
     *
     * Typical total wall time: ~55–65 min.
     */
    private static void runSelfPlay() {
        // --- calibrate iter/s ---
        System.out.print("Calibrating iter/s...");
        System.out.flush();
        MctsSearch warmup = new MctsSearch(new SearchContext(), new Random(1));
        for (int w = 0; w < 2; w++) warmup.search(SGoState.initial(), 5000);
        long calStart = System.nanoTime();
        MctsSearch cal = new MctsSearch(new SearchContext(), new Random(1));
        MctsSearch.SearchResult calResult = cal.search(SGoState.initial(), 10000);
        double itersPerMs = 10000.0 / ((System.nanoTime() - calStart) / 1_000_000.0);
        System.out.printf(" %.0f iter/ms%n", itersPerMs);

        // Convert target ms budgets to iteration counts (rounded to nearest 10).
        // Scale starts at 1ms so the baseline MCTS is truly weak (≤350 iters on a
        // 25-cell board: ~14 visits/branch — too few to do multi-turn planning).
        // The 10ms step discovers the 2-turn cluster plan (~10pp jump),
        // 100ms adds 3-4 turn lookahead, 1000ms near-optimal planning.
        int[] msTargets   = {1, 10, 100, 1000};
        int[] p1ItersArr  = new int[msTargets.length];
        for (int i = 0; i < msTargets.length; i++) {
            p1ItersArr[i] = Math.max(1, (int) Math.round(itersPerMs * msTargets[i]));
        }

        System.out.printf("P1 budgets:  %d, %d, %d, %d iters (≈1, 10, 100, 1000ms)%n",
                p1ItersArr[0], p1ItersArr[1], p1ItersArr[2], p1ItersArr[3]);
        System.out.println("P2 baseline: random placement (no MCTS)");
        System.out.println();

        // Game counts: 1ms row is very cheap so use 1000 games for tight CI.
        // Higher-budget rows scale down; n=400 at 10ms gives z>3 for 10pp jumps.
        int[] gameCounts = {1000, 600, 400, 200};

        long seed = 5678;
        System.out.println("=== MCTS Self-Play: deterministic, P1(MCTS budget) vs P2(random) ===");
        System.out.println("Significance: one-tailed z-test vs 1ms row (H1: win% higher)");
        System.out.printf("%-12s  %8s  %5s  %6s  %6s  %7s  %7s  %s%n",
                "budget", "iters", "games", "wins", "losses", "win%", "95% CI", "vs 1ms");
        System.out.println("-".repeat(82));

        int baseWins = -1, baseGames = -1;

        for (int i = 0; i < msTargets.length; i++) {
            int p1Iters = p1ItersArr[i];
            int games   = gameCounts[i];
            int wins = 0, losses = 0;
            Random rng = new Random(seed);
            for (int g = 0; g < games; g++) {
                int result = playGameMctsVsRandom(p1Iters, rng);
                if (result > 0) wins++;
                else if (result < 0) losses++;
            }

            double p = (double) wins / games;
            double ci = 1.96 * Math.sqrt(p * (1 - p) / games);

            String comparison;
            if (baseWins < 0) {
                comparison = "— (baseline)";
                baseWins = wins;
                baseGames = games;
            } else {
                double p0 = (double) baseWins / baseGames;
                double pHat = (double)(baseWins + wins) / (baseGames + games);
                double se = Math.sqrt(pHat * (1 - pHat) * (1.0 / baseGames + 1.0 / games));
                double zStat = (p - p0) / se;
                double pVal = 1.0 - normalCdf(zStat);
                comparison = String.format("z=%.2f p=%.3f %s",
                        zStat, pVal, pVal < 0.05 ? "✓ p<0.05" : "✗");
            }

            System.out.printf("%-12d  %8d  %5d  %6d  %6d  %6.1f%%  ±%5.1f%%  %s%n",
                    msTargets[i], p1Iters, games, wins, losses,
                    100.0 * p, 100.0 * ci, comparison);
            System.out.flush();
        }
    }

    /** Standard normal CDF via Horner's method (Abramowitz & Stegun 26.2.17). */
    private static double normalCdf(double z) {
        if (z < -8) return 0.0;
        if (z >  8) return 1.0;
        double t = 1.0 / (1.0 + 0.2316419 * Math.abs(z));
        double poly = t * (0.319381530 + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        double pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        double upper = pdf * poly;
        return z >= 0 ? 1.0 - upper : upper;
    }

    /**
     * P1=MCTS(p1Iters), P2=purely random placement (no MCTS).
     * Random P2 picks a uniformly random empty cell each step.
     */
    private static int playGameMctsVsRandom(int p1Iters, Random rng) {
        SGoState state = SGoState.initial();
        Random p1Rng = new Random(rng.nextLong());
        rng.nextLong(); // consume P2 seed slot for reproducibility
        SearchContext p1Ctx = null; MctsSearch p1Search = null; boolean p1Active = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                if (!p1Active) {
                    p1Ctx = new SearchContext(); p1Search = new MctsSearch(p1Ctx, p1Rng);
                    p1Active = true; p1Search.search(state, p1Iters);
                }
                SGoAction action = p1Search.bestKnownAction(state);
                if (action == null) {
                    // State not yet expanded in main tree (deep dice outcome). Use a fresh
                    // lightweight search rather than bloating the main TT with another full budget.
                    action = new MctsSearch(new SearchContext(), p1Rng).search(state, 1000).action;
                }
                state = applyAction(state, action, rng);
            } else {
                if (p1Active) {
                    // Explicitly release search tree to allow GC before allocating the next turn's tree.
                    p1Ctx = null; p1Search = null; p1Active = false;
                }
                state = applyRandomPlacement(state, rng);
            }
            if (state.isTurnEnd && !state.isTerminal()) state = SGoRules.advanceTurn(state);
        }
        return Integer.compare(state.score(), 0);
    }

    /** Random placement: pick uniformly random empty cell. No END_TURN option. */
    private static SGoState applyRandomPlacement(SGoState state, Random rng) {
        long emptyCells = state.emptyCells;
        int emptyCount = Long.bitCount(emptyCells);
        if (emptyCount == 0) return SGoRules.applyEndTurn(state);
        int target = rng.nextInt(emptyCount);
        long bits = emptyCells;
        int idx = 0;
        while (bits != 0L) {
            int cell = Long.numberOfTrailingZeros(bits);
            if (idx == target) {
                int roll = rng.nextInt(6) + 1;
                return SGoRules.applyPlacement(state, cell, roll);
            }
            bits &= bits - 1L;
            idx++;
        }
        return SGoRules.applyEndTurn(state); // unreachable
    }

    /**
     * Play one game: P1=MCTS(p1BudgetMs), P2=MCTS(p2BudgetMs).
     * Each player searches for its time budget at the start of its turn.
     */
    private static int playGameSelfPlay(long p1BudgetMs, long p2BudgetMs, Random rng) {
        SGoState state = SGoState.initial();
        Random p1Rng = new Random(rng.nextLong());
        Random p2Rng = new Random(rng.nextLong());

        SearchContext p1Ctx = null;
        MctsSearch p1Search = null;
        boolean p1TurnActive = false;

        SearchContext p2Ctx = null;
        MctsSearch p2Search = null;
        boolean p2TurnActive = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                p2TurnActive = false;
                if (!p1TurnActive) {
                    p1Ctx = new SearchContext();
                    p1Search = new MctsSearch(p1Ctx, p1Rng);
                    p1TurnActive = true;
                    p1Search.searchForMs(state, p1BudgetMs);
                }
                SGoAction action = p1Search.bestKnownAction(state);
                if (action == null) action = p1Search.searchForMs(state, p1BudgetMs).action;
                state = applyAction(state, action, rng);
            } else {
                p1TurnActive = false;
                if (!p2TurnActive) {
                    p2Ctx = new SearchContext();
                    p2Search = new MctsSearch(p2Ctx, p2Rng);
                    p2TurnActive = true;
                    p2Search.searchForMs(state, p2BudgetMs);
                }
                SGoAction action = p2Search.bestKnownAction(state);
                if (action == null) action = p2Search.searchForMs(state, p2BudgetMs).action;
                state = applyAction(state, action, rng);
            }

            if (state.isTurnEnd && !state.isTerminal()) {
                state = SGoRules.advanceTurn(state);
            }
        }

        return Integer.compare(state.score(), 0);
    }

    /**
     * Deterministic variant: uses search(state, iters) so outcomes depend only on
     * the RNG seed, not on wall-clock time or machine load.
     */
    private static int playGameSelfPlayIter(int p1Iters, int p2Iters, Random rng) {
        SGoState state = SGoState.initial();
        Random p1Rng = new Random(rng.nextLong());
        Random p2Rng = new Random(rng.nextLong());
        SearchContext p1Ctx = null; MctsSearch p1Search = null; boolean p1Active = false;
        SearchContext p2Ctx = null; MctsSearch p2Search = null; boolean p2Active = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                p2Active = false;
                if (!p1Active) {
                    p1Ctx = new SearchContext(); p1Search = new MctsSearch(p1Ctx, p1Rng);
                    p1Active = true; p1Search.search(state, p1Iters);
                }
                SGoAction action = p1Search.bestKnownAction(state);
                if (action == null) action = p1Search.search(state, p1Iters).action;
                state = applyAction(state, action, rng);
            } else {
                p1Active = false;
                if (!p2Active) {
                    p2Ctx = new SearchContext(); p2Search = new MctsSearch(p2Ctx, p2Rng);
                    p2Active = true; p2Search.search(state, p2Iters);
                }
                SGoAction action = p2Search.bestKnownAction(state);
                if (action == null) action = p2Search.search(state, p2Iters).action;
                state = applyAction(state, action, rng);
            }
            if (state.isTurnEnd && !state.isTerminal()) state = SGoRules.advanceTurn(state);
        }
        return Integer.compare(state.score(), 0);
    }

    /**
     * Score-based self-play: uses final score (P1 pieces - P2 pieces) instead of
     * binary win/loss. The continuous metric has ~10× lower variance, so Welch's
     * t-test detects step improvements with ~90 games/row rather than ~480.
     * All four budgets vs P2=1ms. Each row compares to the PREVIOUS row (adjacent).
     *
     * Game wall time ≈ (p1Budget + 1) × TOTAL_TURNS ms.
     * With 100 games/row: total ≈ 100 × (11+1010+5010+10010) ms ≈ 27 min.
     */
    private static void runSelfPlayScore() {
        long baselineMs = 1;
        long[] budgets = {10, 100, 500, 1000};
        int gamesPerRow = 100;
        long seed = 5678;

        System.out.println("=== MCTS Self-Play (score metric): P1(budget) vs P2(" + baselineMs + "ms) ===");
        System.out.println("Significance: one-tailed Welch t-test vs previous row (H1: score higher)");
        System.out.printf("%-12s  %5s  %8s  %8s  %s%n",
                "P1 budget", "games", "avg score", "std dev", "vs prev");
        System.out.println("-".repeat(65));

        double[] prevScores = null;

        for (long budget : budgets) {
            double[] scores = new double[gamesPerRow];
            Random rng = new Random(seed);
            for (int g = 0; g < gamesPerRow; g++) {
                scores[g] = playGameScore(budget, baselineMs, rng);
            }

            double mean = 0;
            for (double s : scores) mean += s;
            mean /= gamesPerRow;

            double variance = 0;
            for (double s : scores) variance += (s - mean) * (s - mean);
            variance /= (gamesPerRow - 1);
            double stddev = Math.sqrt(variance);
            double ci = 1.96 * stddev / Math.sqrt(gamesPerRow);

            String comparison = "—";
            if (prevScores != null) {
                // Welch's t-test (one-tailed)
                double prevMean = 0, prevVar = 0;
                for (double s : prevScores) prevMean += s;
                prevMean /= prevScores.length;
                for (double s : prevScores) prevVar += (s - prevMean) * (s - prevMean);
                prevVar /= (prevScores.length - 1);

                double welchSe = Math.sqrt(prevVar / prevScores.length + variance / gamesPerRow);
                double tStat = (mean - prevMean) / welchSe;
                // Degrees of freedom (Welch–Satterthwaite)
                double df = Math.pow(prevVar / prevScores.length + variance / gamesPerRow, 2)
                        / (Math.pow(prevVar / prevScores.length, 2) / (prevScores.length - 1)
                           + Math.pow(variance / gamesPerRow, 2) / (gamesPerRow - 1));
                // One-tailed p-value via normal approximation (accurate for df > 30)
                double pVal = 1.0 - normalCdf(tStat);
                comparison = String.format("t=%.2f (df=%.0f) p=%.3f %s",
                        tStat, df, pVal, pVal < 0.05 ? "✓ p<0.05" : "✗");
            }

            System.out.printf("%-12d  %5d  %+7.2f±%.2f  %7.2f  %s%n",
                    budget, gamesPerRow, mean, ci, stddev, comparison);
            System.out.flush();

            prevScores = scores;
        }
    }

    /**
     * Play one game and return the raw score (P1 pieces - P2 pieces).
     * P1=MCTS(p1BudgetMs), P2=MCTS(p2BudgetMs).
     */
    private static int playGameScore(long p1BudgetMs, long p2BudgetMs, Random rng) {
        SGoState state = SGoState.initial();
        Random p1Rng = new Random(rng.nextLong());
        Random p2Rng = new Random(rng.nextLong());
        SearchContext p1Ctx = null; MctsSearch p1Search = null; boolean p1Active = false;
        SearchContext p2Ctx = null; MctsSearch p2Search = null; boolean p2Active = false;

        while (!state.isTerminal()) {
            if (state.currentPlayer == SGoState.P1) {
                p2Active = false;
                if (!p1Active) {
                    p1Ctx = new SearchContext(); p1Search = new MctsSearch(p1Ctx, p1Rng);
                    p1Active = true; p1Search.searchForMs(state, p1BudgetMs);
                }
                SGoAction action = p1Search.bestKnownAction(state);
                if (action == null) action = p1Search.searchForMs(state, p1BudgetMs).action;
                state = applyAction(state, action, rng);
            } else {
                p1Active = false;
                if (!p2Active) {
                    p2Ctx = new SearchContext(); p2Search = new MctsSearch(p2Ctx, p2Rng);
                    p2Active = true; p2Search.searchForMs(state, p2BudgetMs);
                }
                SGoAction action = p2Search.bestKnownAction(state);
                if (action == null) action = p2Search.searchForMs(state, p2BudgetMs).action;
                state = applyAction(state, action, rng);
            }
            if (state.isTurnEnd && !state.isTerminal()) state = SGoRules.advanceTurn(state);
        }
        return state.score();
    }

    /**
     * Quick pilot: run `gamesPerRow` games at each of the four target budgets
     * vs P2=1ms, to estimate per-step effect sizes before the full run.
     */
    private static void runPilot(int gamesPerRow) {
        System.out.println("=== Pilot: estimating effect sizes (P2=1ms, " + gamesPerRow + " games/row) ===");
        long[] budgets = {10, 100, 500, 1000};
        long baselineMs = 1;
        long seed = 5678;

        System.out.printf("%-12s  %5s  %7s  %7s%n", "P1 budget", "games", "win%", "95% CI");
        System.out.println("-".repeat(40));
        for (long budget : budgets) {
            int wins = 0, games = gamesPerRow;
            Random rng = new Random(seed);
            for (int g = 0; g < games; g++) {
                if (playGameSelfPlay(budget, baselineMs, rng) > 0) wins++;
            }
            double p = (double) wins / games;
            double ci = 1.96 * Math.sqrt(p * (1 - p) / games);
            System.out.printf("%-12d  %5d  %6.1f%%  ±%5.1f%%%n", budget, games, 100*p, 100*ci);
            System.out.flush();
        }
    }

    /** Single run with stats printout. */
    private static void runSingle(int budget) {
        System.out.println("=== Single MCTS-FT Run, budget=" + budget + " ===");
        MctsSearch search = new MctsSearch(new SearchContext(), new Random(42));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), budget);
        System.out.println("Best action: " + result.action);
        System.out.println(result.stats.summary());
        System.out.println("Root visits by action:");
        for (java.util.Map.Entry<String, Integer> e : result.stats.rootVisitDistribution.entrySet()) {
            System.out.printf("  %-20s %d%n", e.getKey(), e.getValue());
        }
    }
}
