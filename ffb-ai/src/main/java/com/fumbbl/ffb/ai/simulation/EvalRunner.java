package com.fumbbl.ffb.ai.simulation;

import ai.onnxruntime.OrtException;
import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.TeamSetup;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Evaluates the trained BC model against Random and ScriptedStrategy opponents.
 *
 * <pre>
 * mvn -pl ffb-ai exec:java \
 *   -Dexec.mainClass=com.fumbbl.ffb.ai.simulation.EvalRunner \
 *   -Dexec.args="--model /tmp/ffb-ckpt2/bc_model --vocab /tmp/ffb-features/vocab.json --games 100"
 * </pre>
 *
 * <p>Conditions run:
 * <ol>
 *   <li>BC model (home) vs Random (away)</li>
 *   <li>BC model (home) vs ScriptedArgmax (away)</li>
 *   <li>ScriptedArgmax (home) vs Random (away)  — reference baseline</li>
 *   <li>Random (home) vs Random (away)           — null baseline</li>
 * </ol>
 */
public class EvalRunner {

    // Same team/setup IDs as MatchRunner
    private static final String HOME_TEAM_ID = MatchRunner.HOME_TEAM_ID;
    private static final String AWAY_TEAM_ID = MatchRunner.AWAY_TEAM_ID;
    private static final String HOME_SETUP   = MatchRunner.HOME_SETUP;
    private static final String AWAY_SETUP   = MatchRunner.AWAY_SETUP;

    // ── Main ──────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        // ── Parse args ────────────────────────────────────────────────────────
        String modelPrefix = "/tmp/ffb-ckpt2/bc_model";
        String vocabPath   = "/tmp/ffb-features/vocab.json";
        String rootPath    = null;
        int nGames  = 100;
        int threads = 1;

        for (int i = 0; i < args.length; i++) {
            if ("--model".equals(args[i]) && i + 1 < args.length)  modelPrefix = args[++i];
            if ("--vocab".equals(args[i]) && i + 1 < args.length)  vocabPath   = args[++i];
            if ("--root".equals(args[i])  && i + 1 < args.length)  rootPath    = args[++i];
            if ("--games".equals(args[i]) && i + 1 < args.length)  nGames      = Integer.parseInt(args[++i]);
            if ("--threads".equals(args[i]) && i + 1 < args.length) threads   = Integer.parseInt(args[++i]);
        }

        File projectRoot = rootPath != null
            ? new File(rootPath)
            : new File(System.getProperty("user.dir")).getParentFile();
        File serverDir = new File(projectRoot, "ffb-server");

        System.out.println("=== EvalRunner: BC Model Behavioral Evaluation ===");
        System.out.println("Model prefix : " + modelPrefix);
        System.out.println("Vocab        : " + vocabPath);
        System.out.println("Server dir   : " + serverDir.getAbsolutePath());
        System.out.println("Games/cond   : " + nGames);
        System.out.println("Threads      : " + threads);
        System.out.println();

        // ── Load vocab ────────────────────────────────────────────────────────
        Map<String, Integer> skillVocab = new HashMap<>();
        int nDialogTypes;
        try (FileReader fr = new FileReader(vocabPath)) {
            JsonObject root = Json.parse(fr).asObject();
            JsonObject skills = root.get("skills").asObject();
            for (JsonObject.Member m : skills) {
                skillVocab.put(m.getName(), m.getValue().asInt());
            }
            JsonObject dialogs = root.get("dialog_types").asObject();
            nDialogTypes = dialogs.size();
            System.out.printf("Vocab loaded: %d skills, %d dialog types%n",
                skillVocab.size(), nDialogTypes);
        }

        FeatureExtractor extractor = new FeatureExtractor(skillVocab, nDialogTypes);

        // ── Check model files ─────────────────────────────────────────────────
        for (String suffix : new String[]{"_dialog.onnx", "_player_select.onnx", "_move_target.onnx"}) {
            File f = new File(modelPrefix + suffix);
            if (!f.exists()) {
                System.err.println("ERROR: model file not found: " + f.getAbsolutePath());
                System.exit(1);
            }
        }

        // ── Setup ─────────────────────────────────────────────────────────────
        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();
        GameState setupState = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
        com.fumbbl.ffb.model.Game setupGame = setupState.getGame();
        TeamSetup homeSetup = HeadlessGameSetup.loadTeamSetup(setupGame, new File(serverDir, HOME_SETUP));
        TeamSetup awaySetup = HeadlessGameSetup.loadTeamSetup(setupGame, new File(serverDir, AWAY_SETUP));

        // JIT warm-up
        System.out.print("Warming up (JIT)... ");
        new MatchRunner(homeSetup, awaySetup, MatchRunner.AgentMode.SCRIPTED_ARGMAX, MatchRunner.AgentMode.RANDOM)
            .runGame(HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir));
        System.out.println("done.");
        System.out.println();

        // ── Measure model inference speed ──────────────────────────────────────
        System.out.println("Measuring ONNX inference speed...");
        measureInferenceSpeed(modelPrefix, extractor, nDialogTypes, setupGame);

        // ── Run conditions ────────────────────────────────────────────────────
        EvalResult condA = runCondition("A: BC      vs Random ",
            MatchRunner.AgentMode.MODEL, MatchRunner.AgentMode.RANDOM,
            modelPrefix, extractor, nDialogTypes,
            homeSetup, awaySetup, server, serverDir, nGames, threads);

        EvalResult condB = runCondition("B: BC      vs Argmax ",
            MatchRunner.AgentMode.MODEL, MatchRunner.AgentMode.SCRIPTED_ARGMAX,
            modelPrefix, extractor, nDialogTypes,
            homeSetup, awaySetup, server, serverDir, nGames, threads);

        EvalResult condC = runCondition("C: Argmax  vs Random ",
            MatchRunner.AgentMode.SCRIPTED_ARGMAX, MatchRunner.AgentMode.RANDOM,
            null, extractor, nDialogTypes,   // no model needed for pure Argmax
            homeSetup, awaySetup, server, serverDir, nGames, threads);

        EvalResult condD = runCondition("D: Random  vs Random ",
            MatchRunner.AgentMode.RANDOM, MatchRunner.AgentMode.RANDOM,
            null, extractor, nDialogTypes,
            homeSetup, awaySetup, server, serverDir, nGames, threads);

        // ── Print report ──────────────────────────────────────────────────────
        printReport(nGames, condA, condB, condC, condD);
    }

    // ── Condition runner ──────────────────────────────────────────────────────

    private static EvalResult runCondition(String label,
            MatchRunner.AgentMode homeMode, MatchRunner.AgentMode awayMode,
            String modelPrefix, FeatureExtractor extractor, int nDialogTypes,
            TeamSetup homeSetup, TeamSetup awaySetup,
            HeadlessFantasyFootballServer server,
            File serverDir, int n, int threads) throws Exception {

        System.out.printf("Running %s (%d games, %d thread(s))...%n", label.trim(), n, threads);
        EvalResult result = new EvalResult(label, homeMode, awayMode);

        if (threads <= 1) {
            // Single-threaded: share one model agent pair
            OnnxModelAgent homeAgent = null, awayAgent = null;
            try {
                if (homeMode == MatchRunner.AgentMode.MODEL && modelPrefix != null) {
                    homeAgent = OnnxModelAgent.load(modelPrefix, extractor, nDialogTypes);
                }
                if (awayMode == MatchRunner.AgentMode.MODEL && modelPrefix != null) {
                    awayAgent = OnnxModelAgent.load(modelPrefix, extractor, nDialogTypes);
                }

                MatchRunner runner = new MatchRunner(homeSetup, awaySetup, homeMode, awayMode);
                if (homeAgent != null || awayAgent != null) {
                    runner.setModelAgents(homeAgent, awayAgent);
                }

                for (int i = 1; i <= n; i++) {
                    GameState gs = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
                    try {
                        GameResult gr = runner.runGame(gs);
                        recordResult(result, gr);
                    } catch (Exception e) {
                        result.errors++;
                    }
                }
            } finally {
                if (homeAgent != null) homeAgent.close();
                if (awayAgent != null) awayAgent.close();
            }
        } else {
            // Multi-threaded: each thread gets its own model agent
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<Future<int[]>> futures = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                final String prefix = modelPrefix;
                final int nDT = nDialogTypes;
                futures.add(pool.submit(() -> {
                    OnnxModelAgent hAgent = null, aAgent = null;
                    try {
                        if (homeMode == MatchRunner.AgentMode.MODEL && prefix != null) {
                            hAgent = OnnxModelAgent.load(prefix, extractor, nDT);
                        }
                        if (awayMode == MatchRunner.AgentMode.MODEL && prefix != null) {
                            aAgent = OnnxModelAgent.load(prefix, extractor, nDT);
                        }
                        MatchRunner runner = new MatchRunner(homeSetup, awaySetup, homeMode, awayMode);
                        if (hAgent != null || aAgent != null) {
                            runner.setModelAgents(hAgent, aAgent);
                        }
                        GameState gs = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
                        GameResult gr = runner.runGame(gs);
                        int hs = gr.getScoreHome(), as = gr.getScoreAway();
                        return new int[]{hs > as ? 1 : 0, as > hs ? 1 : 0, hs == as ? 1 : 0, 0};
                    } catch (Exception e) {
                        return new int[]{0, 0, 0, 1};
                    } finally {
                        if (hAgent != null) try { hAgent.close(); } catch (Exception ignored) {}
                        if (aAgent != null) try { aAgent.close(); } catch (Exception ignored) {}
                    }
                }));
            }
            pool.shutdown();
            for (Future<int[]> f : futures) {
                int[] r = f.get();
                result.homeWins += r[0];
                result.awayWins += r[1];
                result.draws    += r[2];
                result.errors   += r[3];
            }
        }

        System.out.printf("  Done: %dW / %dD / %dL (%d errors)%n",
            result.homeWins, result.draws, result.awayWins, result.errors);
        return result;
    }

    private static void recordResult(EvalResult result, GameResult gr) {
        if (gr == null) { result.errors++; return; }
        int hs = gr.getScoreHome(), as = gr.getScoreAway();
        if      (hs > as) result.homeWins++;
        else if (as > hs) result.awayWins++;
        else              result.draws++;
    }

    // ── Inference speed measurement ───────────────────────────────────────────

    private static void measureInferenceSpeed(String modelPrefix,
                                               FeatureExtractor extractor,
                                               int nDialogTypes,
                                               com.fumbbl.ffb.model.Game game) throws OrtException {
        ai.onnxruntime.OrtEnvironment env = ai.onnxruntime.OrtEnvironment.getEnvironment();
        ai.onnxruntime.OrtSession.SessionOptions opts = new ai.onnxruntime.OrtSession.SessionOptions();
        opts.setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT);

        float[] spatial  = extractor.buildSpatialBoard(game);
        float[] ns       = extractor.buildNonSpatial(game);
        int[]   skillIds = new int[FeatureExtractor.MAX_CANDS * FeatureExtractor.MAX_SKILLS];
        float[] stats    = new float[FeatureExtractor.MAX_CANDS * 5];
        float[] psMask   = new float[FeatureExtractor.MAX_CANDS + 1];
        psMask[0] = 1.0f; psMask[1] = 1.0f;
        float[] mtMask   = new float[FeatureExtractor.BOARD_W * FeatureExtractor.BOARD_H + 1];
        for (int i = 0; i < 10; i++) mtMask[i] = 1.0f;

        try (ai.onnxruntime.OrtSession psSession =
                 env.createSession(modelPrefix + "_player_select.onnx", opts);
             ai.onnxruntime.OrtSession mtSession =
                 env.createSession(modelPrefix + "_move_target.onnx", opts)) {

            int N = 1000;
            // Warm-up
            for (int w = 0; w < 20; w++) {
                runPsInference(env, psSession, spatial, ns, skillIds, stats, psMask);
                runMtInference(env, mtSession, spatial, ns, mtMask);
            }

            long t0 = System.nanoTime();
            for (int i = 0; i < N; i++) runPsInference(env, psSession, spatial, ns, skillIds, stats, psMask);
            double psMs = (System.nanoTime() - t0) / 1e6 / N;

            t0 = System.nanoTime();
            for (int i = 0; i < N; i++) runMtInference(env, mtSession, spatial, ns, mtMask);
            double mtMs = (System.nanoTime() - t0) / 1e6 / N;

            System.out.printf("  player_select  inference: %.3fms / call  (%,.0f calls/sec)%n",
                psMs, 1000.0 / psMs);
            System.out.printf("  move_target    inference: %.3fms / call  (%,.0f calls/sec)%n",
                mtMs, 1000.0 / mtMs);
            System.out.println();
        }
    }

    private static void runPsInference(ai.onnxruntime.OrtEnvironment env,
                                        ai.onnxruntime.OrtSession session,
                                        float[] spatial, float[] ns,
                                        int[] skillIds, float[] stats, float[] mask)
            throws OrtException {
        long[] tolong = new long[skillIds.length];
        for (int i = 0; i < skillIds.length; i++) tolong[i] = skillIds[i];
        float[] candPos = new float[(FeatureExtractor.MAX_CANDS + 1) * FeatureExtractor.CAND_POS_DIM];
        Map<String, ai.onnxruntime.OnnxTensor> inputs = new HashMap<>();
        inputs.put("spatial",        ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(spatial),
            new long[]{1, FeatureExtractor.N_BOARD_CHANNELS, FeatureExtractor.BOARD_W, FeatureExtractor.BOARD_H}));
        inputs.put("non_spatial",    ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(ns), new long[]{1, FeatureExtractor.NS_DIM}));
        inputs.put("cand_skill_ids", ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.LongBuffer.wrap(tolong),
            new long[]{1, FeatureExtractor.MAX_CANDS, FeatureExtractor.MAX_SKILLS}));
        inputs.put("cand_stats",     ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(stats), new long[]{1, FeatureExtractor.MAX_CANDS, 5}));
        inputs.put("cand_mask",      ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(mask), new long[]{1, FeatureExtractor.MAX_CANDS + 1}));
        inputs.put("cand_pos",       ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(candPos),
            new long[]{1, FeatureExtractor.MAX_CANDS + 1, FeatureExtractor.CAND_POS_DIM}));
        try (ai.onnxruntime.OrtSession.Result r = session.run(inputs)) { /* consume */ }
    }

    private static void runMtInference(ai.onnxruntime.OrtEnvironment env,
                                        ai.onnxruntime.OrtSession session,
                                        float[] spatial, float[] ns, float[] mask)
            throws OrtException {
        Map<String, ai.onnxruntime.OnnxTensor> inputs = new HashMap<>();
        inputs.put("spatial",        ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(spatial),
            new long[]{1, FeatureExtractor.N_BOARD_CHANNELS, FeatureExtractor.BOARD_W, FeatureExtractor.BOARD_H}));
        inputs.put("non_spatial",    ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(ns), new long[]{1, FeatureExtractor.NS_DIM}));
        inputs.put("candidate_mask", ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.FloatBuffer.wrap(mask),
            new long[]{1, FeatureExtractor.BOARD_W * FeatureExtractor.BOARD_H + 1}));
        try (ai.onnxruntime.OrtSession.Result r = session.run(inputs)) { /* consume */ }
    }

    // ── Report ────────────────────────────────────────────────────────────────

    private static void printReport(int n, EvalResult... conds) {
        System.out.println();
        System.out.printf("=== BC Model Evaluation Results (N=%d per condition, 95%% CI) ===%n", n);
        System.out.println();

        System.out.println("Win Rates (home-team perspective):");
        System.out.printf("  %-24s  %-8s  %-6s  %-8s  %s%n",
            "Condition", "W", "D", "L", "WinRate [95% CI]");
        System.out.println("  " + "-".repeat(75));

        for (EvalResult c : conds) {
            int p = c.homeWins + c.awayWins + c.draws;
            if (p == 0) {
                System.out.printf("  %-24s  (no completed games, %d errors)%n", c.label, c.errors);
                continue;
            }
            double[] ci = wilsonCI(c.homeWins, p);
            System.out.printf("  %-24s  %-8d  %-6d  %-8d  %5.1f%%  [%5.1f%%–%5.1f%%]%n",
                c.label, c.homeWins, c.draws, c.awayWins,
                100.0 * c.homeWins / p, 100.0 * ci[0], 100.0 * ci[1]);
            if (c.errors > 0) {
                System.out.printf("  %-24s  (%d errors skipped)%n", "", c.errors);
            }
        }

        System.out.println();
        System.out.println("Key:");
        System.out.println("  BC      = ONNX model (player-select + move-target; dialog = ScriptedArgmax)");
        System.out.println("  Argmax  = ScriptedStrategy deterministic (argmax temperature=0)");
        System.out.println("  Random  = always end turn");
        System.out.println();

        // Interpretation
        EvalResult bcVsRandom  = conds[0];
        EvalResult bcVsArgmax  = conds[1];
        EvalResult refVsRandom = conds[2];
        EvalResult baseline    = conds[3];

        int pBcR  = bcVsRandom.homeWins + bcVsRandom.awayWins + bcVsRandom.draws;
        int pBcA  = bcVsArgmax.homeWins  + bcVsArgmax.awayWins  + bcVsArgmax.draws;
        int pRef  = refVsRandom.homeWins + refVsRandom.awayWins + refVsRandom.draws;

        if (pBcR > 0 && pRef > 0) {
            double bcWr  = 100.0 * bcVsRandom.homeWins  / pBcR;
            double refWr = 100.0 * refVsRandom.homeWins / pRef;
            System.out.printf("BC vs Random win rate:       %.1f%% (Argmax reference: %.1f%%)%n", bcWr, refWr);
            System.out.printf("BC captures %.0f%% of Argmax's advantage over Random%n",
                pRef > 0 ? 100.0 * (bcWr - 50.0) / Math.max(1.0, refWr - 50.0) : 0.0);
        }
        if (pBcA > 0) {
            double bcVsArgmaxWr = 100.0 * bcVsArgmax.homeWins / pBcA;
            System.out.printf("BC vs Argmax win rate:       %.1f%% (50%% = perfect imitation)%n", bcVsArgmaxWr);
        }
    }

    // ── Wilson CI ─────────────────────────────────────────────────────────────

    static double[] wilsonCI(int k, int n) {
        if (n == 0) return new double[]{0, 0};
        double z = 1.96, p = (double) k / n;
        double denom = 1.0 + z * z / n;
        double center = (p + z * z / (2 * n)) / denom;
        double margin = z * Math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom;
        return new double[]{Math.max(0, center - margin), Math.min(1, center + margin)};
    }

    // ── Result container ──────────────────────────────────────────────────────

    static final class EvalResult {
        final String label;
        final MatchRunner.AgentMode homeMode, awayMode;
        int homeWins, awayWins, draws, errors;

        EvalResult(String label, MatchRunner.AgentMode homeMode, MatchRunner.AgentMode awayMode) {
            this.label    = label;
            this.homeMode = homeMode;
            this.awayMode = awayMode;
        }
    }
}
