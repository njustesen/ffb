package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.server.GameState;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Benchmarks headless game simulation speed.
 *
 * <p>Runs {@code N} full games between Lizardman (Kalimar) and Human (BattleLore)
 * using two random agents and reports per-game timing statistics.
 *
 * <h3>Usage</h3>
 * <pre>
 *   java -cp ... com.fumbbl.ffb.ai.simulation.SimulationBenchmark [projectRoot]
 * </pre>
 * If {@code projectRoot} is omitted the parent of the current working directory
 * is used (works when run from {@code ffb-ai/}).
 */
public class SimulationBenchmark {

    private static final String HOME_TEAM_ID = "teamLizardmanKalimar";
    private static final String AWAY_TEAM_ID = "teamHumanBattleLore";
    private static final String HOME_SETUP_FILE = "setups/setup_lizardman_Kalimar.xml";
    private static final String AWAY_SETUP_FILE = "setups/setup_human_BattleLore.xml";

    public static void main(String[] args) throws Exception {
        File projectRoot = args.length > 0
            ? new File(args[0])
            : new File(System.getProperty("user.dir")).getParentFile();
        File serverDir = new File(projectRoot, "ffb-server");

        System.out.println("=== FFB Headless Simulation Benchmark ===");
        System.out.println("Server dir : " + serverDir.getAbsolutePath());
        System.out.println("Home team  : " + HOME_TEAM_ID);
        System.out.println("Away team  : " + AWAY_TEAM_ID);
        System.out.println();

        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();

        // One-time setup: load teams + setups using a throwaway game for XML parsing
        System.out.print("Loading teams and setups... ");
        GameState setupState = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
        Game setupGame = setupState.getGame();
        TeamSetup homeSetup = HeadlessGameSetup.loadTeamSetup(setupGame,
            new File(serverDir, HOME_SETUP_FILE));
        TeamSetup awaySetup = HeadlessGameSetup.loadTeamSetup(setupGame,
            new File(serverDir, AWAY_SETUP_FILE));
        System.out.println("done.");

        SimulationLoop loop = new SimulationLoop(homeSetup, awaySetup);

        // JIT warmup run (not measured)
        System.out.print("Warming up (JIT)... ");
        loop.runGame(HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir));
        System.out.println("done.");
        System.out.println();

        int N = 20;
        long[] setupNs = new long[N];
        long[] kickoffNs = new long[N];
        long[] driveNs = new long[N];
        long[] turns = new long[N];
        int completed = 0;

        // Accumulated step timings and per-granularity samples across all games
        Map<String, Long> allStepTimeNs = new HashMap<>();
        Map<String, Integer> allStepCounts = new HashMap<>();
        java.util.List<Long> allGameNs = new java.util.ArrayList<>();
        java.util.List<Long> allDriveNs = new java.util.ArrayList<>();
        java.util.List<Long> allTurnNs = new java.util.ArrayList<>();
        java.util.List<Long> allPlayerTurnNs = new java.util.ArrayList<>();

        System.out.printf("Running %d games...%n%n", N);

        for (int i = 0; i < N; i++) {
            long t0 = System.nanoTime();
            GameState gs = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
            setupNs[i] = System.nanoTime() - t0;

            SimulationLoop.GameResult result = loop.runGameSplit(gs);
            kickoffNs[i] = result.kickoffNs;
            driveNs[i] = result.driveNs;
            turns[i] = result.turns;
            result.stepTimeNs.forEach((k, v) -> allStepTimeNs.merge(k, v, Long::sum));
            result.stepCounts.forEach((k, v) -> allStepCounts.merge(k, v, Integer::sum));
            allGameNs.add(result.kickoffNs + result.driveNs);
            allDriveNs.addAll(result.perDriveNs);
            allTurnNs.addAll(result.perTurnNs);
            allPlayerTurnNs.addAll(result.perPlayerTurnNs);

            boolean done = gs.getGame().getFinished() != null;
            if (done) {
                completed++;
            }
            long usPerTurn = turns[i] > 0 ? (driveNs[i] / 1_000) / turns[i] : 0;
            System.out.printf("  Game %2d: setup=%4d ms  kickoff=%4d ms  drive=%4d ms  turns=%3d  us/turn=%4d  %s%n",
                i + 1,
                setupNs[i] / 1_000_000,
                kickoffNs[i] / 1_000_000,
                driveNs[i] / 1_000_000,
                turns[i],
                usPerTurn,
                done ? "FINISHED" : "INCOMPLETE");
        }

        System.out.println();
        System.out.printf("Games completed: %d / %d%n", completed, N);
        System.out.println();

        System.out.println("--- Setup time (GameState creation) ---");
        printStats("setup  ", setupNs, N);
        System.out.println();

        System.out.println("--- Kickoff time (setup + kickoff phases) ---");
        printStats("kickoff", kickoffNs, N);
        System.out.println();

        System.out.println("--- Drive time (regular play only) ---");
        printStats("drive  ", driveNs, N);
        System.out.println();

        long totalTurns = 0;
        long totalDriveNs = 0;
        for (int i = 0; i < N; i++) {
            totalTurns += turns[i];
            totalDriveNs += driveNs[i];
        }
        System.out.printf("--- Drive time per turn (avg across all %d games) ---%n", N);
        System.out.printf("  turns total: %d  avg per game: %d  us/turn: %d%n",
            totalTurns, totalTurns / N, totalTurns > 0 ? (totalDriveNs / 1_000) / totalTurns : 0);
        System.out.println();

        System.out.println("--- Total time (setup + kickoff + drive) ---");
        long[] totalNs = new long[N];
        for (int i = 0; i < N; i++) {
            totalNs[i] = setupNs[i] + kickoffNs[i] + driveNs[i];
        }
        printStats("total  ", totalNs, N);
        System.out.println();

        System.out.printf("--- Timing by granularity (drive time only, %d games) ---%n", N);
        System.out.printf("  %-14s  %7s  %10s  %10s  %10s%n",
            "granularity", "samples", "min", "avg", "max");
        printGranularity("game (drive)", allGameNs);
        printGranularity("drive", allDriveNs);
        printGranularity("turn", allTurnNs);
        printGranularity("player turn", allPlayerTurnNs);
        System.out.println("  (player turn = INIT_SELECTING dispatch; simulation ends turn immediately on first selection)");
        System.out.println();

        System.out.printf("--- Hottest steps (all %d games, top 20 by total time) ---%n", N);
        long grandTotalNs = allStepTimeNs.values().stream().mapToLong(Long::longValue).sum();
        List<Map.Entry<String, Long>> sorted = allStepTimeNs.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .limit(20)
            .collect(Collectors.toList());
        System.out.printf("  %-45s  %6s  %8s  %7s  %5s%n",
            "step (+dialog)", "hits", "total ms", "avg µs", "share");
        for (Map.Entry<String, Long> e : sorted) {
            String key = e.getKey();
            long tNs = e.getValue();
            int hits = allStepCounts.getOrDefault(key, 0);
            long avgUs = hits > 0 ? (tNs / 1_000) / hits : 0;
            double pct = grandTotalNs > 0 ? 100.0 * tNs / grandTotalNs : 0;
            System.out.printf("  %-45s  %6d  %8d  %7d  %4.1f%%%n",
                key, hits, tNs / 1_000_000, avgUs, pct);
        }
    }

    private static void printGranularity(String label, java.util.List<Long> samples) {
        if (samples.isEmpty()) {
            System.out.printf("  %-14s  %7d  %10s  %10s  %10s%n", label, 0, "-", "-", "-");
            return;
        }
        long min = Long.MAX_VALUE, max = 0, sum = 0;
        for (long v : samples) { min = Math.min(min, v); max = Math.max(max, v); sum += v; }
        long avg = sum / samples.size();
        // Choose unit based on magnitude of average
        if (avg >= 1_000_000L) {
            System.out.printf("  %-14s  %7d  %9d ms  %9d ms  %9d ms%n",
                label, samples.size(), min / 1_000_000, avg / 1_000_000, max / 1_000_000);
        } else {
            System.out.printf("  %-14s  %7d  %9d µs  %9d µs  %9d µs%n",
                label, samples.size(), min / 1_000, avg / 1_000, max / 1_000);
        }
    }

    private static void printStats(String label, long[] ns, int n) {
        long min = Long.MAX_VALUE, max = 0, sum = 0;
        for (long t : ns) {
            min = Math.min(min, t);
            max = Math.max(max, t);
            sum += t;
        }
        System.out.printf("  %s: min=%4d ms  avg=%4d ms  max=%4d ms%n",
            label, min / 1_000_000, (sum / n) / 1_000_000, max / 1_000_000);
    }
}
