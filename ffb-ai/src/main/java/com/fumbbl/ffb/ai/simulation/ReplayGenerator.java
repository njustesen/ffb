package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.ai.strategy.ScriptedStrategy;
import com.fumbbl.ffb.json.UtilJson;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.server.GameState;

import java.io.BufferedOutputStream;
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
 * Generates replay files from headless simulation games.
 *
 * <p>Each game is run entirely in-process (no DB, no WebSocket, no Swing).
 * Replays are written as gzip-compressed JSON ({@code .ffbr}) to an output directory.
 *
 * <pre>{@code
 * Usage: ReplayGenerator [options]
 *   --output DIR        output directory (default: ./replays)
 *   --games N           total games to generate (default: 100)
 *   --temperature T     ScriptedStrategy temperature 0.0–1.0 (default: 0.5)
 *   --races r1,r2,...   comma-separated race filter (default: all)
 *   --threads K         parallel workers (default: min(4, cpus/2))
 * }</pre>
 */
public class ReplayGenerator {

    // ── Race pool ─────────────────────────────────────────────────────────────

    /**
     * Each entry describes one playable team: the team XML id, the path to its
     * setup XML relative to the server directory, and a short race name used for
     * filtering and replay file naming.
     *
     * <p>"human" maps to <em>two</em> entries (Kalimar + BattleLore) so that
     * Human-vs-Human matchups are possible.  All other races have one entry.
     */
    static final class TeamEntry {
        final String teamId;
        final String setupPath;
        final String raceName;

        TeamEntry(String teamId, String setupPath, String raceName) {
            this.teamId    = teamId;
            this.setupPath = setupPath;
            this.raceName  = raceName;
        }

        @Override public String toString() { return raceName + "(" + teamId + ")"; }
    }

    static final List<TeamEntry> ALL_ENTRIES = Collections.unmodifiableList(Arrays.asList(
        new TeamEntry("teamAmazonKalimar",       "setups/setup_amazon_Kalimar.xml",        "amazon"),
        new TeamEntry("teamChaosBattleLore",      "setups/setup_chaos_BattleLore.xml",      "chaos"),
        new TeamEntry("teamChaosDwarfKalimar",    "setups/setup_chaos_dwarf_Kalimar.xml",   "chaos_dwarf"),
        new TeamEntry("teamDwarfBattleLore",      "setups/setup_dwarf_BattleLore.xml",      "dwarf"),
        new TeamEntry("teamElfKalimar",           "setups/setup_elf_Kalimar.xml",           "elf"),
        new TeamEntry("teamGoblinKalimar",        "setups/setup_goblin_Kalimar.xml",        "goblin"),
        new TeamEntry("teamHighElfBattleLore",    "setups/setup_high_elf_BattleLore.xml",   "high_elf"),
        new TeamEntry("teamHumanKalimar",         "setups/setup_human_Kalimar.xml",         "human"),
        new TeamEntry("teamHumanBattleLore",      "setups/setup_human_BattleLore.xml",      "human"),
        new TeamEntry("teamLizardmanKalimar",     "setups/setup_lizardman_Kalimar.xml",     "lizardman"),
        new TeamEntry("teamNecromanticBattleLore","setups/setup_necromantic_BattleLore.xml","necromantic"),
        new TeamEntry("teamNorseKalimar",         "setups/setup_norse_Kalimar.xml",         "norse"),
        new TeamEntry("teamOrcBattleLore",        "setups/setup_orc_BattleLore.xml",        "orc"),
        new TeamEntry("teamSkavenBattleLore",     "setups/setup_skaven_BattleLore.xml",     "skaven"),
        new TeamEntry("teamUndeadKalimar",        "setups/setup_undead_Kalimar.xml",        "undead"),
        new TeamEntry("teamUnderworldKalimar",    "setups/setup_underworld_Kalimar.xml",    "underworld"),
        new TeamEntry("teamVampireBattleLore",    "setups/setup_vampire_BattleLore.xml",    "vampire"),
        new TeamEntry("teamWoodElfKalimar",       "setups/setup_wood_elf_Kalimar.xml",      "wood_elf")
    ));

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        // Defaults
        File outputDir   = new File("replays");
        int  totalGames  = 100;
        double temperature = 0.5;
        List<String> raceFilter = Collections.emptyList();
        int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));

        // Parse CLI
        for (int i = 0; i < args.length - 1; i++) {
            switch (args[i]) {
                case "--output":      outputDir   = new File(args[++i]);                             break;
                case "--games":       totalGames  = Integer.parseInt(args[++i]);                     break;
                case "--temperature": temperature = Double.parseDouble(args[++i]);                   break;
                case "--races":       raceFilter  = Arrays.asList(args[++i].split(","));             break;
                case "--threads":     threads     = Integer.parseInt(args[++i]);                     break;
            }
        }

        // Build filtered pool
        final List<TeamEntry> pool = buildPool(raceFilter);
        if (pool.size() < 2) {
            System.err.println("ERROR: race pool has fewer than 2 entries.");
            System.err.println("  The game engine requires two teams with distinct IDs — a team cannot play itself.");
            System.err.println("  Only 'human' has two entries (Kalimar + BattleLore) and supports a single-race pool.");
            System.err.println("  For all other races, specify at least 2 different races with --races.");
            System.err.println("Available races: " + ALL_ENTRIES.stream()
                .map(e -> e.raceName).distinct().collect(Collectors.joining(", ")));
            System.exit(1);
        }

        // Locate server dir (parent of working dir or explicit sibling)
        File projectRoot = new File(System.getProperty("user.dir")).getParentFile();
        File serverDir   = new File(projectRoot, "ffb-server");
        if (!serverDir.isDirectory()) {
            // Fallback: working dir might already be project root
            serverDir = new File(System.getProperty("user.dir"), "ffb-server");
        }
        if (!serverDir.isDirectory()) {
            System.err.println("ERROR: cannot find ffb-server directory. Tried: " + serverDir);
            System.exit(1);
        }

        outputDir.mkdirs();

        // Summary
        String raceList = pool.stream().map(e -> e.raceName).distinct()
            .sorted().collect(Collectors.joining(", "));
        double estMinutes = (totalGames * 0.5) / threads / 60.0;
        System.out.println("=== ReplayGenerator ===");
        System.out.printf("  Games       : %d%n", totalGames);
        System.out.printf("  Temperature : %.2f%n", temperature);
        System.out.printf("  Races       : %s (%d pool entries)%n", raceList, pool.size());
        System.out.printf("  Threads     : %d%n", threads);
        System.out.printf("  Output dir  : %s%n", outputDir.getAbsolutePath());
        System.out.printf("  Server dir  : %s%n", serverDir.getAbsolutePath());
        System.out.printf("  Est. time   : ~%.0f min (rough estimate)%n", estMinutes);
        System.out.println();

        final double finalTemperature = temperature;
        final File   finalServerDir   = serverDir;
        final File   finalOutputDir   = outputDir;
        final int    finalTotalGames  = totalGames;
        final AtomicInteger saved    = new AtomicInteger(0);
        final AtomicInteger retries  = new AtomicInteger(0);
        final int reportEvery = Math.max(1, totalGames / 10);
        final long startTime = System.currentTimeMillis();

        // Use a CompletionService so we can resubmit timed-out games and guarantee
        // exactly totalGames saved replays regardless of how many time out.
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        ExecutorCompletionService<Boolean> ecs = new ExecutorCompletionService<>(executor);

        Callable<Boolean> gameTask = () -> runOneGame(finalServerDir, finalOutputDir, pool, finalTemperature);

        // Seed the pool with the initial batch
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
                if (n % reportEvery == 0 || n == finalTotalGames) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double rate = n / (elapsed / 1000.0);
                    System.out.printf("  [%d/%d] %.1f games/sec, ~%.0f/hr%n",
                        n, finalTotalGames, rate, rate * 3600);
                }
            } else {
                // Game timed out or errored — resubmit to hit the target count
                retries.incrementAndGet();
                ecs.submit(gameTask);
                inFlight++;
            }
        }
        executor.shutdownNow();

        long totalMs = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.printf("=== Done: %d replays saved, %d retries, %.1f s total (%.0f/hr) ===%n",
            saved.get(), retries.get(), totalMs / 1000.0,
            saved.get() / (totalMs / 3_600_000.0));
    }

    // ── Single-game runner — returns true if a replay was saved, false on timeout ─

    private static boolean runOneGame(File serverDir, File outputDir,
                                      List<TeamEntry> pool, double temperature)
            throws IOException {

        // Each task creates its own server — no shared mutable state between threads.
        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();

        // Pick two distinct entries at random
        Random rng = new Random();
        TeamEntry home, away;
        home = pool.get(rng.nextInt(pool.size()));
        do {
            away = pool.get(rng.nextInt(pool.size()));
        } while (away == home);  // identity check — ensures two different TeamEntry objects

        // Load setup files (requires a fresh GameState to parse XML in game context)
        GameState setupGs = HeadlessGameSetup.create(server, home.teamId, away.teamId, serverDir);
        TeamSetup homeSetup = HeadlessGameSetup.loadTeamSetup(
            setupGs.getGame(), new File(serverDir, home.setupPath));
        TeamSetup awaySetup = HeadlessGameSetup.loadTeamSetup(
            setupGs.getGame(), new File(serverDir, away.setupPath));

        // Apply temperature before running
        ScriptedStrategy.setTemperature(temperature);

        // Create a fresh game state and run
        GameState gs = HeadlessGameSetup.create(server, home.teamId, away.teamId, serverDir);
        MatchRunner runner = new MatchRunner(homeSetup, awaySetup,
            MatchRunner.AgentMode.SCRIPTED_SAMPLE, MatchRunner.AgentMode.SCRIPTED_SAMPLE);
        GameResult result = runner.runGame(gs);

        // null means the game timed out — discard and let the caller retry
        if (result == null) {
            return false;
        }

        // Serialize and write — use nanoTime for uniqueness since headless game IDs are all 0
        String filename = String.format("replay_%d_%s_vs_%s.ffbr",
            System.nanoTime(), home.raceName, away.raceName);
        File outFile = new File(outputDir, filename);
        byte[] gzipped = UtilJson.gzip(gs.toJsonValue());
        try (BufferedOutputStream out = new BufferedOutputStream(Files.newOutputStream(outFile.toPath()))) {
            out.write(gzipped);
        }
        return true;
    }

    // ── Pool builder ──────────────────────────────────────────────────────────

    private static List<TeamEntry> buildPool(List<String> raceFilter) {
        if (raceFilter.isEmpty()) {
            return new ArrayList<>(ALL_ENTRIES);
        }
        List<String> normalized = raceFilter.stream()
            .map(String::trim).map(String::toLowerCase)
            .collect(Collectors.toList());
        return ALL_ENTRIES.stream()
            .filter(e -> normalized.contains(e.raceName))
            .collect(Collectors.toList());
    }
}
