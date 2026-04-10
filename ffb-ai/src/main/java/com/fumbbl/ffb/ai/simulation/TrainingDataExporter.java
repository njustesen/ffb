package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.ai.strategy.ScriptedStrategy;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Generates per-decision training data (JSONL) from headless simulation games.
 *
 * <p>Each game runs entirely in-process using {@link HeadlessFantasyFootballServer}.
 * All three decision types are captured:
 * <ul>
 *   <li>{@code "dialog"}        — ScriptedStrategy dialog response</li>
 *   <li>{@code "player_select"} — MoveDecisionEngine player selection (INIT_SELECTING)</li>
 *   <li>{@code "move_target"}   — MoveDecisionEngine move target</li>
 * </ul>
 *
 * <p>One {@code .jsonl} file is written per worker thread to avoid locking.
 * Concatenate shard files for full dataset.
 *
 * <pre>{@code
 * Usage: TrainingDataExporter [options]
 *   --output DIR        output directory (default: ffb-ml/data)
 *   --games N           total games to generate (default: 1000)
 *   --temperature T     ScriptedStrategy temperature 0.0–1.0 (default: 0.5)
 *   --races r1,r2,...   comma-separated race filter (default: all)
 *   --threads K         parallel workers (default: min(4, cpus/2))
 * }</pre>
 */
public class TrainingDataExporter {

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        // Defaults
        File outputDir   = new File("ffb-ml/data");
        int  totalGames  = 1000;
        double temperature = 0.5;
        List<String> raceFilter = Collections.emptyList();
        int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));

        // Parse CLI
        for (int i = 0; i < args.length - 1; i++) {
            switch (args[i]) {
                case "--output":      outputDir   = new File(args[++i]);               break;
                case "--games":       totalGames  = Integer.parseInt(args[++i]);       break;
                case "--temperature": temperature = Double.parseDouble(args[++i]);     break;
                case "--races":       raceFilter  = Arrays.asList(args[++i].split(",")); break;
                case "--threads":     threads     = Integer.parseInt(args[++i]);       break;
            }
        }

        final List<ReplayGenerator.TeamEntry> pool = buildPool(raceFilter);
        if (pool.size() < 2) {
            System.err.println("ERROR: race pool has fewer than 2 entries.");
            System.exit(1);
        }

        File projectRoot = new File(System.getProperty("user.dir")).getParentFile();
        File serverDir   = new File(projectRoot, "ffb-server");
        if (!serverDir.isDirectory()) serverDir = new File(System.getProperty("user.dir"), "ffb-server");
        if (!serverDir.isDirectory()) {
            System.err.println("ERROR: cannot find ffb-server directory. Tried: " + serverDir);
            System.exit(1);
        }

        outputDir.mkdirs();

        String raceList = pool.stream().map(e -> e.raceName).distinct()
            .sorted().collect(Collectors.joining(", "));
        System.out.println("=== TrainingDataExporter ===");
        System.out.printf("  Games       : %d%n", totalGames);
        System.out.printf("  Temperature : %.2f%n", temperature);
        System.out.printf("  Races       : %s (%d pool entries)%n", raceList, pool.size());
        System.out.printf("  Threads     : %d%n", threads);
        System.out.printf("  Output dir  : %s%n", outputDir.getAbsolutePath());
        System.out.println();

        final double finalTemperature = temperature;
        final File   finalServerDir   = serverDir;
        final File   finalOutputDir   = outputDir;
        final AtomicInteger saved   = new AtomicInteger(0);
        final AtomicInteger retries = new AtomicInteger(0);
        final int reportEvery = Math.max(1, totalGames / 10);
        final long startTime = System.currentTimeMillis();

        // One JSONL file per thread — avoids locking during concurrent writes.
        // We open them all upfront so they exist even if the pool errors early.
        final BufferedWriter[] writers = new BufferedWriter[threads];
        for (int i = 0; i < threads; i++) {
            File shardFile = new File(finalOutputDir, String.format("shard_%03d.jsonl", i));
            writers[i] = Files.newBufferedWriter(shardFile.toPath());
        }

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        ExecutorCompletionService<Boolean> ecs = new ExecutorCompletionService<>(executor);

        // Track which writer belongs to which thread using a thread-local index.
        final int finalThreads = threads;
        final java.util.concurrent.atomic.AtomicInteger nextWriterIdx = new java.util.concurrent.atomic.AtomicInteger(0);
        final ThreadLocal<Integer> writerIdx = ThreadLocal.withInitial(nextWriterIdx::getAndIncrement);

        Callable<Boolean> gameTask = () -> runOneGame(
            finalServerDir, writers[writerIdx.get() % finalThreads], pool, finalTemperature);

        for (int i = 0; i < totalGames; i++) ecs.submit(gameTask);
        int inFlight = totalGames;

        while (saved.get() < totalGames) {
            boolean ok;
            try {
                ok = ecs.take().get();
            } catch (Exception e) {
                ok = false;
            }
            inFlight--;

            if (ok) {
                int n = saved.incrementAndGet();
                if (n % reportEvery == 0 || n == totalGames) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double rate = n / (elapsed / 1000.0);
                    System.out.printf("  [%d/%d] %.1f games/sec, ~%.0f/hr%n",
                        n, totalGames, rate, rate * 3600);
                }
            } else {
                retries.incrementAndGet();
                ecs.submit(gameTask);
                inFlight++;
            }
        }
        executor.shutdownNow();

        // Flush and close all shard writers
        for (BufferedWriter w : writers) {
            try { w.flush(); w.close(); } catch (IOException ignored) {}
        }

        long totalMs = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.printf("=== Done: %d games, %d retries, %.1f s total (%.0f/hr) ===%n",
            saved.get(), retries.get(), totalMs / 1000.0,
            saved.get() / (totalMs / 3_600_000.0));
        System.out.printf("  Output: %s/shard_000.jsonl .. shard_%03d.jsonl%n",
            finalOutputDir.getAbsolutePath(), threads - 1);
    }

    // ── Single-game runner ────────────────────────────────────────────────────

    private static boolean runOneGame(File serverDir, BufferedWriter writer,
                                      List<ReplayGenerator.TeamEntry> pool,
                                      double temperature) throws IOException {
        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();

        Random rng = new Random();
        ReplayGenerator.TeamEntry home, away;
        home = pool.get(rng.nextInt(pool.size()));
        do {
            away = pool.get(rng.nextInt(pool.size()));
        } while (away == home);

        com.fumbbl.ffb.server.GameState setupGs =
            HeadlessGameSetup.create(server, home.teamId, away.teamId, serverDir);
        TeamSetup homeSetup = HeadlessGameSetup.loadTeamSetup(
            setupGs.getGame(), new File(serverDir, home.setupPath));
        TeamSetup awaySetup = HeadlessGameSetup.loadTeamSetup(
            setupGs.getGame(), new File(serverDir, away.setupPath));

        ScriptedStrategy.setTemperature(temperature);

        com.fumbbl.ffb.server.GameState gs =
            HeadlessGameSetup.create(server, home.teamId, away.teamId, serverDir);

        MatchRunner runner = new MatchRunner(homeSetup, awaySetup,
            MatchRunner.AgentMode.SCRIPTED_SAMPLE, MatchRunner.AgentMode.SCRIPTED_SAMPLE);

        // Attach the JSONL collector to this game's runner
        JsonlTrainingDataCollector collector = new JsonlTrainingDataCollector(writer);
        runner.setCollector(collector);

        com.fumbbl.ffb.model.GameResult result = runner.runGame(gs);

        // null = timed out
        return result != null;
    }

    // ── Pool builder (delegates to ReplayGenerator's pool entries) ────────────

    private static List<ReplayGenerator.TeamEntry> buildPool(List<String> raceFilter) {
        if (raceFilter.isEmpty()) {
            return new ArrayList<>(ReplayGenerator.ALL_ENTRIES);
        }
        List<String> normalized = raceFilter.stream()
            .map(String::trim).map(String::toLowerCase)
            .collect(Collectors.toList());
        return ReplayGenerator.ALL_ENTRIES.stream()
            .filter(e -> normalized.contains(e.raceName))
            .collect(Collectors.toList());
    }
}
