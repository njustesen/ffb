package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.Direction;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.FieldCoordinateBounds;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.Pushback;
import com.fumbbl.ffb.PushbackSquare;
import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.PathProbabilityFinder;
import com.fumbbl.ffb.ai.strategy.RandomStrategy;
import com.fumbbl.ffb.ai.strategy.ScriptedStrategy;
import com.fumbbl.ffb.dialog.DialogArgueTheCallParameter;
import com.fumbbl.ffb.dialog.DialogBriberyAndCorruptionParameter;
import com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.net.commands.ClientCommand;
import com.fumbbl.ffb.net.commands.ClientCommandActingPlayer;
import com.fumbbl.ffb.net.commands.ClientCommandBlock;
import com.fumbbl.ffb.net.commands.ClientCommandEndTurn;
import com.fumbbl.ffb.net.commands.ClientCommandFoul;
import com.fumbbl.ffb.net.commands.ClientCommandKickoff;
import com.fumbbl.ffb.net.commands.ClientCommandMove;
import com.fumbbl.ffb.net.commands.ClientCommandPushback;
import com.fumbbl.ffb.net.commands.ClientCommandStartGame;
import com.fumbbl.ffb.net.commands.ClientCommandTouchback;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.net.ReceivedCommand;
import com.fumbbl.ffb.server.step.IStep;
import com.fumbbl.ffb.server.step.StepId;
import com.fumbbl.ffb.util.UtilPlayer;

import org.eclipse.jetty.websocket.api.Session;

import ai.onnxruntime.OrtException;
import com.eclipsesource.json.Json;
import com.eclipsesource.json.JsonObject;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runs N headless games comparing three agent modes:
 * <ul>
 *   <li><b>SCRIPTED_SAMPLE</b> — probabilistic (softmax) dialog decisions</li>
 *   <li><b>SCRIPTED_ARGMAX</b> — deterministic (best-score) dialog decisions</li>
 *   <li><b>RANDOM</b> — always end turn</li>
 * </ul>
 *
 * <p>Three conditions run back-to-back, N games each:
 * <ol>
 *   <li>ScriptedSample (home) vs Random (away)</li>
 *   <li>ScriptedArgmax (home) vs Random (away)</li>
 *   <li>ScriptedSample attacker (home) vs ScriptedArgmax defender (away) — move logic
 *       runs only for home; away uses argmax for dialog decisions (block dice, re-rolls)
 *       but ends its own turn without making moves.</li>
 * </ol>
 *
 * <p>Outputs win rates with 95% Wilson confidence intervals and per-level
 * timing statistics (mean ± σ) for decision, activation, turn, drive, half,
 * and full game.
 *
 * <p><b>All games run purely in-process via {@link HeadlessFantasyFootballServer}.</b>
 * No WebSocket server, no Swing client, no AI clients are launched.
 */
public class MatchRunner {

    // ── Agent mode ────────────────────────────────────────────────────────────

    public enum AgentMode { RANDOM, SCRIPTED_SAMPLE, SCRIPTED_ARGMAX, MODEL,
                            MCTS_UCB, MCTS_PRIOR }

    // ── Teams (Human vs Human) ────────────────────────────────────────────────

    static final String HOME_TEAM_ID = "teamHumanKalimar";
    static final String AWAY_TEAM_ID  = "teamHumanBattleLore";
    static final String HOME_SETUP    = "setups/setup_human_Kalimar.xml";
    static final String AWAY_SETUP    = "setups/setup_human_BattleLore.xml";

    private static final int MAX_ITERATIONS = 100_000;

    /**
     * Steps that belong to the kickoff phase (post-TD or half-time drive setup).
     * If a rollout reaches any of these, the current turn has ended naturally —
     * the rollout should stop rather than trying to handle kickoff dialogs.
     */
    private static final Set<StepId> KICKOFF_STEPS = EnumSet.of(
        StepId.SETUP,
        StepId.INIT_KICKOFF,
        StepId.KICKOFF,
        StepId.KICKOFF_ANIMATION,
        StepId.KICKOFF_RESULT_ROLL,
        StepId.KICKOFF_RETURN,
        StepId.KICKOFF_SCATTER_ROLL,
        StepId.KICKOFF_SCATTER_ROLL_ASK_AFTER,
        StepId.APPLY_KICKOFF_RESULT,
        StepId.END_KICKOFF,
        StepId.BLITZ_TURN,
        StepId.MASTER_CHEF,
        StepId.PRAYER,
        StepId.PRAYERS,
        StepId.SWARMING,
        StepId.TOUCHBACK
    );

    // ── Timing container ──────────────────────────────────────────────────────

    static final class GameTimings {
        long gameNs;
        long decisions;   long decisionNs;
        // Per-team decision counts and cumulative nanoseconds.
        // Used to derive per-agent-type timing across conditions.
        long homeDecisions; long homeDecisionNs;
        long awayDecisions; long awayDecisionNs;
        long activations; long activationNs;
        long turns;       long turnNs;
        long drives;      long driveNs;
        long halves;      long halfNs;
    }

    // ── Condition result container ────────────────────────────────────────────

    static final class ConditionResult {
        final String label;
        final AgentMode homeMode, awayMode;
        int homeWins, awayWins, draws, errors;
        final List<GameTimings> timings = new ArrayList<>();
        /** Aggregated MCTS search stats (null for non-MCTS conditions). */
        com.fumbbl.ffb.ai.mcts.BbMctsStats mctsStats;

        ConditionResult(String label, AgentMode homeMode, AgentMode awayMode) {
            this.label = label;
            this.homeMode = homeMode;
            this.awayMode = awayMode;
        }
        int played() { return homeWins + awayWins + draws; }
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        // Suppress verbose Jetty / server logging
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        File projectRoot = args.length > 0
            ? new File(args[0])
            : new File(System.getProperty("user.dir")).getParentFile();
        int n = args.length > 1 ? Integer.parseInt(args[1]) : 200;
        int mctsBudget = 0;
        long mctsTimeMs = 0;
        int mctsThreads    = Runtime.getRuntime().availableProcessors(); // count-based (E/F)
        int mctsTimeThreads = Runtime.getRuntime().availableProcessors(); // time-based (G/H)
        boolean mctsVsArgmax = false;   // also run E2/F2 / G2/H2: MCTS vs Argmax
        boolean mctsCrossTurn = false;  // enable cross-turn tree search (G3/H3 conditions)
        String  leafEvalMode = "static"; // "static" or "onnx"
        String  onnxValuePath = null;    // path to value.onnx
        String  onnxVocabPath = null;    // path to vocab.json for FeatureExtractor
        for (int i = 0; i < args.length - 1; i++) {
            if ("--mcts-budget".equals(args[i])) {
                mctsBudget = Integer.parseInt(args[i + 1]);
            } else if ("--mcts-time-ms".equals(args[i])) {
                mctsTimeMs = Long.parseLong(args[i + 1]);
            } else if ("--mcts-threads".equals(args[i])) {
                mctsThreads = Integer.parseInt(args[i + 1]);
            } else if ("--time-threads".equals(args[i])) {
                mctsTimeThreads = Integer.parseInt(args[i + 1]);
            } else if ("--leaf-eval".equals(args[i])) {
                leafEvalMode = args[i + 1];
            } else if ("--onnx-value".equals(args[i])) {
                onnxValuePath = args[i + 1];
            } else if ("--onnx-vocab".equals(args[i])) {
                onnxVocabPath = args[i + 1];
            }
        }
        for (String arg : args) {
            if ("--mcts-vs-argmax".equals(arg)) mctsVsArgmax = true;
            if ("--mcts-cross-turn".equals(arg)) mctsCrossTurn = true;
        }

        // Load ONNX value leaf eval if requested
        com.fumbbl.ffb.ai.mcts.ILeafEval onnxLeafEval = null;
        if ("onnx".equals(leafEvalMode) && onnxValuePath != null && onnxVocabPath != null) {
            try (FileReader fr = new FileReader(onnxVocabPath)) {
                JsonObject vocabRoot = Json.parse(fr).asObject();
                Map<String, Integer> skillVocab = new HashMap<>();
                for (JsonObject.Member m : vocabRoot.get("skills").asObject()) {
                    skillVocab.put(m.getName(), m.getValue().asInt());
                }
                int nDialogTypes = vocabRoot.get("dialog_types").asObject().size();
                FeatureExtractor extractor = new FeatureExtractor(skillVocab, nDialogTypes);
                onnxLeafEval = new OnnxLeafEval(onnxValuePath, extractor);
                System.out.println("ONNX leaf eval loaded: " + onnxValuePath);
            } catch (Exception e) {
                System.err.println("WARNING: failed to load ONNX leaf eval: " + e.getMessage());
            }
        }

        // Force line-buffered output when stdout is a pipe (Maven exec:java buffers by default).
        System.setOut(new java.io.PrintStream(System.out, true));

        File serverDir = new File(projectRoot, "ffb-server");
        System.out.println("=== MatchRunner: Human vs Human comparative experiment ===");
        System.out.println("Server dir : " + serverDir.getAbsolutePath());
        System.out.println("Games/cond : " + n);
        if (mctsBudget > 0) System.out.println("MCTS budget: " + mctsBudget);
        if (mctsTimeMs > 0) System.out.println("MCTS time  : " + mctsTimeMs + "ms/decision" + (mctsCrossTurn ? " + cross-turn" : ""));
        System.out.println("MCTS threads (count): " + mctsThreads + "  (time): " + mctsTimeThreads + "  (available: " + Runtime.getRuntime().availableProcessors() + ")");
        System.out.println();

        HeadlessFantasyFootballServer server = new HeadlessFantasyFootballServer();
        GameState setupState = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
        Game setupGame = setupState.getGame();
        TeamSetup homeSetup = HeadlessGameSetup.loadTeamSetup(setupGame, new File(serverDir, HOME_SETUP));
        TeamSetup awaySetup = HeadlessGameSetup.loadTeamSetup(setupGame, new File(serverDir, AWAY_SETUP));

        // JIT warm-up
        System.out.print("Warming up (JIT)... ");
        new MatchRunner(homeSetup, awaySetup, AgentMode.SCRIPTED_SAMPLE, AgentMode.RANDOM)
            .runGame(HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir));
        System.out.println("done.");
        System.out.println();

        // ── Four scripted conditions ──────────────────────────────────────────

        ConditionResult condA = runCondition("A: Sample  vs Random ",
            AgentMode.SCRIPTED_SAMPLE, AgentMode.RANDOM,
            homeSetup, awaySetup, server, serverDir, n);

        ConditionResult condB = runCondition("B: Argmax  vs Random ",
            AgentMode.SCRIPTED_ARGMAX, AgentMode.RANDOM,
            homeSetup, awaySetup, server, serverDir, n);

        ConditionResult condC = runCondition("C: Sample  vs Argmax ",
            AgentMode.SCRIPTED_SAMPLE, AgentMode.SCRIPTED_ARGMAX,
            homeSetup, awaySetup, server, serverDir, n);

        ConditionResult condD = runCondition("D: Random  vs Random ",
            AgentMode.RANDOM, AgentMode.RANDOM,
            homeSetup, awaySetup, server, serverDir, n);

        // ── Optional MCTS conditions ──────────────────────────────────────────

        List<ConditionResult> allConds = new ArrayList<>();
        allConds.add(condA); allConds.add(condB); allConds.add(condC); allConds.add(condD);

        if (mctsBudget > 0) {
            // Hint GC to reclaim A-D game state before allocating 12 parallel MCTS contexts.
            System.gc();
            try { Thread.sleep(500); } catch (InterruptedException ignored) {}
            MatchRunner scriptedRunner = new MatchRunner(homeSetup, awaySetup,
                AgentMode.SCRIPTED_ARGMAX, AgentMode.SCRIPTED_ARGMAX);

            // Condition E: MCTS with plain UCB (no prior)
            com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbAgent =
                new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
            mctsUcbAgent.setMaxThreads(mctsThreads);

            ConditionResult condE = runMctsCondition("E: MCTS-UCB    vs Random ",
                AgentMode.MCTS_UCB, AgentMode.RANDOM,
                mctsUcbAgent, null,
                homeSetup, awaySetup, server, serverDir, n);
            allConds.add(condE);

            // Condition F: MCTS with scripted action ordering (Script prior)
            com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsScriptedAgent =
                new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
            mctsScriptedAgent.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
            mctsScriptedAgent.setMaxThreads(mctsThreads);

            ConditionResult condF = runMctsCondition("F: MCTS-Prior  vs Random ",
                AgentMode.MCTS_PRIOR, AgentMode.RANDOM,
                mctsScriptedAgent, null,
                homeSetup, awaySetup, server, serverDir, n);
            allConds.add(condF);

            // Conditions E2/F2: MCTS vs Argmax (head-to-head quality check)
            if (mctsVsArgmax) {
                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbVsArgmax =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
                mctsUcbVsArgmax.setMaxThreads(mctsThreads);

                ConditionResult condE2 = runMctsCondition("E2: MCTS-UCB    vs Argmax ",
                    AgentMode.MCTS_UCB, AgentMode.SCRIPTED_ARGMAX,
                    mctsUcbVsArgmax, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condE2);

                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsPriorVsArgmax =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
                mctsPriorVsArgmax.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
                mctsPriorVsArgmax.setMaxThreads(mctsThreads);

                ConditionResult condF2 = runMctsCondition("F2: MCTS-Prior  vs Argmax ",
                    AgentMode.MCTS_PRIOR, AgentMode.SCRIPTED_ARGMAX,
                    mctsPriorVsArgmax, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condF2);

                // Conditions I/J: MCTS with ONNX value leaf eval vs Argmax
                if (onnxLeafEval != null) {
                    com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbOnnx =
                        new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
                    mctsUcbOnnx.setMaxThreads(mctsThreads);
                    mctsUcbOnnx.setLeafEval(onnxLeafEval);

                    ConditionResult condI = runMctsCondition("I:  MCTS-UCB+ONNX vs Argmax ",
                        AgentMode.MCTS_UCB, AgentMode.SCRIPTED_ARGMAX,
                        mctsUcbOnnx, null,
                        homeSetup, awaySetup, server, serverDir, n);
                    allConds.add(condI);

                    com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsPriorOnnx =
                        new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner, mctsBudget);
                    mctsPriorOnnx.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
                    mctsPriorOnnx.setMaxThreads(mctsThreads);
                    mctsPriorOnnx.setLeafEval(onnxLeafEval);

                    ConditionResult condJ = runMctsCondition("J:  MCTS-Prior+ONNX vs Argmax ",
                        AgentMode.MCTS_PRIOR, AgentMode.SCRIPTED_ARGMAX,
                        mctsPriorOnnx, null,
                        homeSetup, awaySetup, server, serverDir, n);
                    allConds.add(condJ);
                }
            }
        }

        if (mctsTimeMs > 0) {
            MatchRunner scriptedRunner = new MatchRunner(homeSetup, awaySetup,
                AgentMode.SCRIPTED_ARGMAX, AgentMode.SCRIPTED_ARGMAX);

            // Condition G: MCTS-UCB with time budget vs Random
            com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbTimed =
                new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
            mctsUcbTimed.setTimeBudgetMs(mctsTimeMs);
            mctsUcbTimed.setMaxThreads(mctsTimeThreads);

            ConditionResult condG = runMctsCondition("G: MCTS-UCB    vs Random (time)",
                AgentMode.MCTS_UCB, AgentMode.RANDOM,
                mctsUcbTimed, null,
                homeSetup, awaySetup, server, serverDir, n);
            allConds.add(condG);

            // Condition H: MCTS-Prior with time budget vs Random
            com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsScriptedTimed =
                new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
            mctsScriptedTimed.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
            mctsScriptedTimed.setTimeBudgetMs(mctsTimeMs);
            mctsScriptedTimed.setMaxThreads(mctsTimeThreads);

            ConditionResult condH = runMctsCondition("H: MCTS-Prior  vs Random (time)",
                AgentMode.MCTS_PRIOR, AgentMode.RANDOM,
                mctsScriptedTimed, null,
                homeSetup, awaySetup, server, serverDir, n);
            allConds.add(condH);

            // Conditions G2/H2: time-based MCTS vs Argmax (head-to-head quality check)
            if (mctsVsArgmax) {
                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbTimedVsArgmax =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
                mctsUcbTimedVsArgmax.setTimeBudgetMs(mctsTimeMs);
                mctsUcbTimedVsArgmax.setMaxThreads(mctsTimeThreads);

                ConditionResult condG2 = runMctsCondition("G2: MCTS-UCB    vs Argmax (time)",
                    AgentMode.MCTS_UCB, AgentMode.SCRIPTED_ARGMAX,
                    mctsUcbTimedVsArgmax, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condG2);

                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsPriorTimedVsArgmax =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
                mctsPriorTimedVsArgmax.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
                mctsPriorTimedVsArgmax.setTimeBudgetMs(mctsTimeMs);
                mctsPriorTimedVsArgmax.setMaxThreads(mctsTimeThreads);

                ConditionResult condH2 = runMctsCondition("H2: MCTS-Prior  vs Argmax (time)",
                    AgentMode.MCTS_PRIOR, AgentMode.SCRIPTED_ARGMAX,
                    mctsPriorTimedVsArgmax, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condH2);
            }

            // Conditions G3/H3: time-based MCTS with cross-turn search vs Argmax
            if (mctsCrossTurn && mctsVsArgmax) {
                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsUcbCrossTurn =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
                mctsUcbCrossTurn.setTimeBudgetMs(mctsTimeMs);
                mctsUcbCrossTurn.setMaxThreads(mctsTimeThreads);
                mctsUcbCrossTurn.setCrossTurnSearch(true);

                ConditionResult condG3 = runMctsCondition("G3: MCTS-UCB+XT  vs Argmax (time)",
                    AgentMode.MCTS_UCB, AgentMode.SCRIPTED_ARGMAX,
                    mctsUcbCrossTurn, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condG3);

                com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsPriorCrossTurn =
                    new com.fumbbl.ffb.ai.mcts.BbMctsSearch(server, scriptedRunner);
                mctsPriorCrossTurn.setActionPrior(new com.fumbbl.ffb.ai.mcts.ScriptedActionPrior());
                mctsPriorCrossTurn.setTimeBudgetMs(mctsTimeMs);
                mctsPriorCrossTurn.setMaxThreads(mctsTimeThreads);
                mctsPriorCrossTurn.setCrossTurnSearch(true);

                ConditionResult condH3 = runMctsCondition("H3: MCTS-Prior+XT vs Argmax (time)",
                    AgentMode.MCTS_PRIOR, AgentMode.SCRIPTED_ARGMAX,
                    mctsPriorCrossTurn, null,
                    homeSetup, awaySetup, server, serverDir, n);
                allConds.add(condH3);
            }
        }

        // ── Print results ─────────────────────────────────────────────────────
        printReport(n, allConds.toArray(new ConditionResult[0]));
    }

    private static ConditionResult runMctsCondition(String label,
            AgentMode homeMode, AgentMode awayMode,
            com.fumbbl.ffb.ai.mcts.BbMctsSearch homeMcts,
            com.fumbbl.ffb.ai.mcts.BbMctsSearch awayMcts,
            TeamSetup homeSetup, TeamSetup awaySetup,
            HeadlessFantasyFootballServer server,
            File serverDir, int n) throws Exception {

        System.out.printf("Running condition %s (%d games)...%n", label.trim(), n);
        ConditionResult result = new ConditionResult(label, homeMode, awayMode);
        MatchRunner runner = new MatchRunner(homeSetup, awaySetup, homeMode, awayMode);
        runner.setMctsAgents(homeMcts, awayMcts);

        long condStart = System.nanoTime();
        for (int i = 1; i <= n; i++) {
            long gameStart = System.nanoTime();
            GameState gs = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
            GameTimings[] timingOut = new GameTimings[1];
            GameResult gr;
            try {
                gr = runner.runGame(gs, timingOut);
            } catch (RuntimeException e) {
                result.errors++;
                continue;
            }
            if (gr == null) { result.errors++; continue; }
            int hs = gr.getScoreHome(), as = gr.getScoreAway();
            if      (hs > as) result.homeWins++;
            else if (as > hs) result.awayWins++;
            else              result.draws++;
            result.timings.add(timingOut[0]);
            {
                long elapsed = System.nanoTime() - condStart;
                long gameMs = (System.nanoTime() - gameStart) / 1_000_000;
                System.out.printf("  [%d/%d] %s %dms  W/D/L: %d/%d/%d  elapsed: %.0fs%n",
                    i, n, label.trim(), gameMs,
                    result.homeWins, result.draws, result.awayWins,
                    elapsed / 1e9);
            }
        }
        System.out.printf("  Done: %dW / %dD / %dL (%d errors)%n",
            result.homeWins, result.draws, result.awayWins, result.errors);

        // Collect and aggregate MCTS stats from the agent(s).
        com.fumbbl.ffb.ai.mcts.BbMctsSearch activeAgent = (homeMcts != null) ? homeMcts : awayMcts;
        if (activeAgent != null) {
            activeAgent.getStats().computeDerived();
            result.mctsStats = activeAgent.getStats();
        }
        return result;
    }

    private static ConditionResult runCondition(String label,
            AgentMode homeMode, AgentMode awayMode,
            TeamSetup homeSetup, TeamSetup awaySetup,
            HeadlessFantasyFootballServer server,
            File serverDir, int n) throws Exception {

        System.out.printf("Running condition %s (%d games)...%n", label.trim(), n);
        ConditionResult result = new ConditionResult(label, homeMode, awayMode);
        MatchRunner runner = new MatchRunner(homeSetup, awaySetup, homeMode, awayMode);

        for (int i = 1; i <= n; i++) {
            GameState gs = HeadlessGameSetup.create(server, HOME_TEAM_ID, AWAY_TEAM_ID, serverDir);
            GameTimings[] timingOut = new GameTimings[1];
            GameResult gr;
            try {
                gr = runner.runGame(gs, timingOut);
            } catch (RuntimeException e) {
                result.errors++;
                continue;
            }

            if (gr == null) {
                result.errors++;
                continue;
            }
            int hs = gr.getScoreHome(), as = gr.getScoreAway();
            if      (hs > as) result.homeWins++;
            else if (as > hs) result.awayWins++;
            else              result.draws++;
            result.timings.add(timingOut[0]);
        }
        System.out.printf("  Done: %dW / %dD / %dL (%d errors)%n",
            result.homeWins, result.draws, result.awayWins, result.errors);
        return result;
    }

    // ── Report ────────────────────────────────────────────────────────────────

    private static void printReport(int n, ConditionResult... conds) {
        System.out.println();
        System.out.printf("=== Experiment Results (N=%d per condition, Human vs Human, 95%% CI) ===%n", n);
        System.out.println();

        // ── Win rates ─────────────────────────────────────────────────────────
        System.out.println("Win Rates (home perspective):");
        System.out.printf("  %-22s  %5s  %5s  %5s  %s%n",
            "Condition", "Wins", "Draws", "Loss", "WinRate ± 95% CI");
        System.out.println("  " + "-".repeat(72));
        for (ConditionResult c : conds) {
            int p = c.played();
            if (p == 0) {
                System.out.printf("  %-22s  (no completed games)%n", c.label);
                continue;
            }
            double[] ci = wilsonCI(c.homeWins, p);
            System.out.printf("  %-22s  %5d  %5d  %5d  %.1f%% [%.1f%%–%.1f%%]%n",
                c.label, c.homeWins, c.draws, c.awayWins,
                100.0 * c.homeWins / p, 100.0 * ci[0], 100.0 * ci[1]);
        }

        // ── Per-level timing by condition ─────────────────────────────────────
        System.out.println();
        System.out.println("Timing by level (mean ± σ, all times in ms except µs/activation):");
        System.out.printf("  %-22s  %14s  %14s  %14s  %14s  %14s%n",
            "Condition", "ms/match", "ms/half", "ms/drive", "ms/turn", "µs/activation");
        System.out.println("  " + "-".repeat(100));
        for (ConditionResult c : conds) {
            if (c.timings.isEmpty()) { System.out.printf("  %-22s  (no data)%n", c.label); continue; }
            TimingStat match = stat(c.timings, t -> t.gameNs / 1e6);
            TimingStat half  = stat(c.timings, t -> t.halves      > 0 ? t.halfNs      / 1e6 / t.halves      : 0);
            TimingStat drive = stat(c.timings, t -> t.drives      > 0 ? t.driveNs     / 1e6 / t.drives      : 0);
            TimingStat turn  = stat(c.timings, t -> t.turns       > 0 ? t.turnNs      / 1e6 / t.turns       : 0);
            TimingStat act   = stat(c.timings, t -> t.activations > 0 ? t.activationNs / 1e3 / t.activations : 0);
            System.out.printf("  %-22s  %14s  %14s  %14s  %14s  %14s%n",
                c.label, fmt(match), fmt(half), fmt(drive), fmt(turn), fmt(act));
        }

        // ── Per-agent decision timing ─────────────────────────────────────────
        // Collect home-decision and away-decision timings grouped by agent type.
        System.out.println();
        System.out.println("Per-agent decision timing (µs/decision, mean ± σ):");
        System.out.printf("  %-16s  %16s  %16s%n", "Agent", "µs/decision", "from conditions");
        System.out.println("  " + "-".repeat(60));
        for (AgentMode mode : AgentMode.values()) {
            List<Double> samples = new ArrayList<>();
            StringBuilder from = new StringBuilder();
            for (ConditionResult c : conds) {
                boolean homeIs = c.homeMode == mode;
                boolean awayIs = c.awayMode == mode;
                if (!homeIs && !awayIs) continue;
                for (GameTimings t : c.timings) {
                    if (homeIs && t.homeDecisions > 0)
                        samples.add(t.homeDecisionNs / 1e3 / t.homeDecisions);
                    if (awayIs && t.awayDecisions > 0)
                        samples.add(t.awayDecisionNs / 1e3 / t.awayDecisions);
                }
                if (from.length() > 0) from.append(", ");
                from.append(homeIs ? "home" : "").append(homeIs && awayIs ? "/" : "").append(awayIs ? "away" : "")
                    .append("(").append(c.label.trim()).append(")");
            }
            if (samples.isEmpty()) continue;
            double sum = 0, sum2 = 0;
            for (double v : samples) { sum += v; sum2 += v * v; }
            int cnt = samples.size();
            double mean = sum / cnt;
            double sd = cnt > 1 ? Math.sqrt((sum2 - sum * sum / cnt) / (cnt - 1)) : 0;
            System.out.printf("  %-16s  %16s  %s%n",
                modeName(mode), String.format("%.2f ± %.2f", mean, sd), from);
        }
        // ── MCTS search statistics ─────────────────────────────────────────────
        boolean hasMcts = false;
        for (ConditionResult c : conds) {
            if (c.mctsStats != null) { hasMcts = true; break; }
        }
        if (hasMcts) {
            System.out.println();
            System.out.println("MCTS Search Statistics:");
            System.out.printf("  %-22s  %12s  %12s  %12s  %12s  %12s  %12s%n",
                "Condition", "iter/s", "ms/iter", "ms/dec", "avg depth", "max depth", "nodes/dec");
            System.out.println("  " + "-".repeat(104));
            for (ConditionResult c : conds) {
                if (c.mctsStats == null) continue;
                com.fumbbl.ffb.ai.mcts.BbMctsStats s = c.mctsStats;
                double msPerIter = s.avgIterationMs;
                double msPerDec  = s.decisions > 0 ? s.totalSearchNs / 1e6 / s.decisions : 0;
                double nodesPerDec = s.decisions > 0 ? (double) s.totalNodes / s.decisions : 0;
                System.out.printf("  %-22s  %12.0f  %12.3f  %12.1f  %12.1f  %12d  %12.1f%n",
                    c.label, s.itersPerSecond, msPerIter, msPerDec,
                    s.avgDepth, s.maxDepth, nodesPerDec);
            }
        }
        System.out.println();
    }

    private static String modeName(AgentMode mode) {
        switch (mode) {
            case SCRIPTED_SAMPLE: return "Sample";
            case SCRIPTED_ARGMAX: return "Argmax";
            case RANDOM:          return "Random";
            case MCTS_UCB:        return "MCTS-UCB";
            case MCTS_PRIOR:      return "MCTS-Prior";
            default:              return mode.name();
        }
    }

    // ── Wilson score interval (95% CI) ────────────────────────────────────────

    /** Returns [low, high] Wilson score interval for k successes out of n trials at z=1.96. */
    static double[] wilsonCI(int k, int n) {
        if (n == 0) return new double[]{0, 0};
        double z = 1.96;
        double p = (double) k / n;
        double z2 = z * z;
        double centre = (p + z2 / (2 * n)) / (1 + z2 / n);
        double margin = (z / (1 + z2 / n)) * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
        return new double[]{Math.max(0, centre - margin), Math.min(1, centre + margin)};
    }

    // ── Timing helpers ────────────────────────────────────────────────────────

    @FunctionalInterface
    interface TimingExtractor { double extract(GameTimings t); }

    static final class TimingStat { double mean, sd; }

    static TimingStat stat(List<GameTimings> list, TimingExtractor f) {
        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (GameTimings t : list) {
            double v = f.extract(t);
            sum += v; sum2 += v * v; cnt++;
        }
        TimingStat s = new TimingStat();
        if (cnt > 0) {
            s.mean = sum / cnt;
            s.sd = cnt > 1 ? Math.sqrt((sum2 - sum * sum / cnt) / (cnt - 1)) : 0;
        }
        return s;
    }

    static String fmt(TimingStat s) {
        return String.format("%.2f ± %.2f", s.mean, s.sd);
    }

    // ── Instance ──────────────────────────────────────────────────────────────

    private final TeamSetup homeSetup;
    private final TeamSetup awaySetup;
    private final AgentMode homeMode;
    private final AgentMode awayMode;
    private final CapturingClientCommunication comm = new CapturingClientCommunication();
    private final Random rng = new Random();

    /** Optional collector for training data; null = disabled (normal operation). */
    private ITrainingDataCollector collector;

    /** Optional ONNX model agents (one per side; null = not using MODEL mode for that side). */
    private OnnxModelAgent homeModelAgent;
    private OnnxModelAgent awayModelAgent;

    /** Optional MCTS search agents (one per side; null = not using MCTS mode for that side). */
    private com.fumbbl.ffb.ai.mcts.BbMctsSearch homeMctsAgent;
    private com.fumbbl.ffb.ai.mcts.BbMctsSearch awayMctsAgent;

    public MatchRunner(TeamSetup homeSetup, TeamSetup awaySetup,
                       AgentMode homeMode, AgentMode awayMode) {
        this.homeSetup = homeSetup;
        this.awaySetup = awaySetup;
        this.homeMode  = homeMode;
        this.awayMode  = awayMode;
    }

    /** Attach a training-data collector.  Must be called before {@link #runGame}. */
    public void setCollector(ITrainingDataCollector collector) {
        this.collector = collector;
    }

    /** Attach ONNX model agents for MODEL mode.  Must be called before {@link #runGame}. */
    public void setModelAgents(OnnxModelAgent homeAgent, OnnxModelAgent awayAgent) {
        this.homeModelAgent = homeAgent;
        this.awayModelAgent = awayAgent;
    }

    /** Attach MCTS agents for {@code MCTS_UCB} / {@code MCTS_PRIOR} mode.  Must be called before {@link #runGame}. */
    public void setMctsAgents(com.fumbbl.ffb.ai.mcts.BbMctsSearch homeAgent,
                              com.fumbbl.ffb.ai.mcts.BbMctsSearch awayAgent) {
        this.homeMctsAgent = homeAgent;
        this.awayMctsAgent = awayAgent;
    }

    /** Run a single game. If timingOut is non-null, its [0] element is set to the collected timings. */
    public GameResult runGame(GameState gameState) {
        return runGame(gameState, null);
    }

    public GameResult runGame(GameState gameState, GameTimings[] timingOut) {
        Game game = gameState.getGame();

        injectForTeam(gameState, new ClientCommandStartGame(), true);
        injectForTeam(gameState, new ClientCommandStartGame(), false);

        GameTimings t = new GameTimings();
        long gameStart = System.nanoTime();

        // Mutable period-start timestamps: [driveStart, halfStart, turnStart]
        long[] ps = {gameStart, gameStart, gameStart};
        int prevHalf = -1;
        // Track which team last owned an INIT_SELECTING step to detect turn boundaries.
        // -1 = no INIT_SELECTING seen yet; 0 = home; 1 = away.
        int lastInitSelectingTeam = -1;

        // Per-turn phase-2 visit counter: tracks how many times each player enters
        // INIT_SELECTING phase 2 within the current team's turn. Reset when the active
        // team changes. If any player exceeds MAX_PHASE2_VISITS, force end-turn.
        java.util.Map<String, Integer> phase2Visits = new java.util.HashMap<>();
        boolean prevHomePlaying = game.isHomePlaying();
        final int MAX_PHASE2_VISITS = 20;

        // SETUP-phase stuck detector: if the same (stepId, homePlaying) pair repeats
        // without the game advancing we're in a SETUP_ERROR feedback loop. Abort fast
        // instead of spinning for 100 000 iterations.
        StepId setupStuckStep = null;
        boolean setupStuckHome = false;
        int setupStuckCount = 0;
        final int MAX_SETUP_REPEATS = 12;

        // General stuck-step detector: if a non-SETUP, non-INIT_SELECTING step repeats
        // many times in a row without any dialog, it's cycling (e.g. vampire blood-lust
        // foul-move loop). Inject EndTurn to break out; fast-fail if EndTurn is rejected.
        StepId generalStuckStep = null;
        int generalStuckCount = 0;
        int generalStuckEndTurns = 0;
        final int MAX_GENERAL_REPEATS = 50;
        final int MAX_GENERAL_END_TURNS = 3;

        int iter = 0;
        while (game.getFinished() == null && ++iter < MAX_ITERATIONS) {
            IStep step = gameState.getCurrentStep();
            if (step == null) break;

            IDialogParameter dialog = game.getDialogParameter();
            StepId stepId = step.getId();

            // Track half transitions
            int currentHalf = game.getHalf();
            if (currentHalf != prevHalf && prevHalf >= 0) {
                t.halfNs += System.nanoTime() - ps[1];
                t.halves++;
                ps[1] = System.nanoTime();
            }
            prevHalf = currentHalf;

            // Track turn boundaries: a new turn starts when INIT_SELECTING switches teams.
            if (stepId == StepId.INIT_SELECTING && (dialog == null || dialog.getId() == null)) {
                int thisTeam = game.isHomePlaying() ? 0 : 1;
                if (lastInitSelectingTeam >= 0 && thisTeam != lastInitSelectingTeam) {
                    long now2 = System.nanoTime();
                    t.turnNs += now2 - ps[2];
                    t.turns++;
                    ps[2] = now2;
                }
                lastInitSelectingTeam = thisTeam;
            }

            // Per-turn stuck detector: reset the map when the active team changes.
            boolean curHomePlaying = game.isHomePlaying();
            if (curHomePlaying != prevHomePlaying) {
                phase2Visits.clear();
                prevHomePlaying = curHomePlaying;
            }
            if (stepId == StepId.INIT_SELECTING && dialog == null) {
                ActingPlayer apCheck = game.getActingPlayer();
                if (apCheck != null && apCheck.getPlayerId() != null) {
                    int visits = phase2Visits.merge(apCheck.getPlayerId(), 1, Integer::sum);
                    if (visits >= MAX_PHASE2_VISITS) {
                        inject(gameState, new com.fumbbl.ffb.net.commands.ClientCommandEndTurn(game.getTurnMode(), null));
                        phase2Visits.clear();
                        iter++; t.decisions++;
                        continue;
                    }
                }
            }

            // SETUP-phase stuck detector: count how many consecutive non-dialog iterations
            // share the same (stepId, homePlaying) pair.  Dialogs don't reset the count
            // because a stuck loop can consist of alternating step + dialog iterations.
            // Only reset when we leave the SETUP phase entirely.
            if (game.getTurnMode() == TurnMode.SETUP) {
                if (dialog == null) {
                    boolean curHome = game.isHomePlaying();
                    if (stepId == setupStuckStep && curHome == setupStuckHome) {
                        if (++setupStuckCount >= MAX_SETUP_REPEATS) {
                            return null;
                        }
                    } else {
                        setupStuckStep = stepId;
                        setupStuckHome = curHome;
                        setupStuckCount = 0;
                    }
                }
                // dialogs: leave count unchanged — they interleave but don't mean progress
            } else {
                setupStuckStep = null;
                setupStuckCount = 0;
            }

            // General stuck-step detector (non-SETUP, non-INIT_SELECTING, no dialog)
            if (game.getTurnMode() != TurnMode.SETUP
                    && stepId != StepId.INIT_SELECTING && dialog == null) {
                if (stepId == generalStuckStep) {
                    if (++generalStuckCount >= MAX_GENERAL_REPEATS) {
                        if (++generalStuckEndTurns > MAX_GENERAL_END_TURNS) {
                            return null;
                        }
                        inject(gameState, new com.fumbbl.ffb.net.commands.ClientCommandEndTurn(game.getTurnMode(), null));
                        generalStuckCount = 0;
                        continue;
                    }
                } else {
                    generalStuckStep = stepId;
                    generalStuckCount = 1;
                    generalStuckEndTurns = 0;
                }
            } else if (dialog != null || stepId == StepId.INIT_SELECTING) {
                generalStuckStep = null;
                generalStuckCount = 0;
                generalStuckEndTurns = 0;
            }

            boolean homeStep = game.isHomePlaying();
            long stepStart = System.nanoTime();

            if (dialog != null && stepId != StepId.INIT_SELECTING) {
                handleDialog(dialog, game, gameState, t);
            } else {
                handleStep(stepId, game, gameState, t, ps);
            }

            long stepNs = System.nanoTime() - stepStart;
            t.decisions++;
            t.decisionNs += stepNs;
            if (homeStep) { t.homeDecisions++; t.homeDecisionNs += stepNs; }
            else          { t.awayDecisions++;  t.awayDecisionNs  += stepNs; }
        }

        // Close final open periods
        long now = System.nanoTime();
        if (prevHalf >= 0) { t.halfNs += now - ps[1]; t.halves++; }
        t.driveNs += now - ps[0]; t.drives++;
        if (lastInitSelectingTeam >= 0) { t.turnNs += now - ps[2]; t.turns++; }

        t.gameNs = now - gameStart;
        if (timingOut != null) timingOut[0] = t;

        if (iter >= MAX_ITERATIONS && game.getFinished() == null) {
            IStep stuck = gameState.getCurrentStep();
            ActingPlayer apStuck = game.getActingPlayer();
            System.err.println("[MatchRunner] Timeout iter=" + iter
                + " ms=" + (t.gameNs / 1_000_000)
                + " step=" + (stuck != null ? stuck.getId() : "null")
                + " player=" + (apStuck != null ? apStuck.getPlayerId() : "null")
                + " action=" + (apStuck != null ? apStuck.getPlayerAction() : "null")
                + " move=" + (apStuck != null ? apStuck.getCurrentMove() : "null")
                + " home=" + game.isHomePlaying()
                + " turn=" + game.getTurnMode()
                + " half=" + game.getHalf()
                + " scoreH=" + game.getGameResult().getScoreHome()
                + " scoreA=" + game.getGameResult().getScoreAway());
            return null;
        }
        return game.getGameResult();
    }

    // ── Step handling ─────────────────────────────────────────────────────────

    /**
     * ps[0] = driveStart, ps[1] = halfStart (managed in runGame), ps[2] = turnStart.
     * handleStep updates ps[0] and ps[2] at drive/turn boundaries.
     */
    private void handleStep(StepId stepId, Game game, GameState gameState,
                            GameTimings t, long[] ps) {
        boolean home = game.isHomePlaying();
        AgentMode mode = home ? homeMode : awayMode;

        switch (stepId) {

            case SETUP: {
                resetCurrentTeam(game);
                placeReserves(game, gameState);
                inject(gameState, new ClientCommandEndTurn(TurnMode.SETUP, null));
                break;
            }

            case KICKOFF: {
                // Close the previous drive and start a new one
                long now = System.nanoTime();
                if (t.drives > 0) {              // skip the very first kickoff (drive 0 already open)
                    t.driveNs += now - ps[0];
                    t.drives++;
                }
                ps[0] = now;
                // Kick into the opponent's half (coordinates in the kicker's perspective).
                // Home kicks to absolute X=13..24; away kicks to absolute X=1..12.
                // The server transforms away-team coordinates, so away must send the
                // pre-transformed version of their intended target.
                int kx, ky;
                ky = 1 + rng.nextInt(13); // 1..13 (absolute)
                if (home) {
                    kx = 13 + rng.nextInt(12); // 13..24 = away half (absolute)
                    inject(gameState, new ClientCommandKickoff(new FieldCoordinate(kx, ky)));
                } else {
                    // Away team wants to kick to absolute X=1..12.
                    // Pre-transform: server will call .transform() → we send (25-kx, 14-ky) so server gets (kx, ky).
                    kx = 1 + rng.nextInt(12); // target X=1..12 absolute
                    FieldCoordinate target = new FieldCoordinate(kx, ky);
                    inject(gameState, new ClientCommandKickoff(target.transform()));
                }
                break;
            }

            case APPLY_KICKOFF_RESULT:
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;

            case INIT_SELECTING: {
                if (mode == AgentMode.RANDOM) {
                    inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                    break;
                }
                ActingPlayer ap = game.getActingPlayer();
                if (ap == null || ap.getPlayerId() == null) {
                    // Phase 1: choose which player to activate
                    long actStart = System.nanoTime();
                    Team myTeam  = home ? game.getTeamHome() : game.getTeamAway();
                    Team oppTeam = home ? game.getTeamAway() : game.getTeamHome();

                    if (mode == AgentMode.MCTS_UCB || mode == AgentMode.MCTS_PRIOR) {
                        com.fumbbl.ffb.ai.mcts.BbMctsSearch mctsAgent = home ? homeMctsAgent : awayMctsAgent;
                        if (mctsAgent != null) {
                            com.fumbbl.ffb.ai.mcts.BbAction best = mctsAgent.selectActivation(game, home);
                            t.activations++;
                            t.activationNs += System.nanoTime() - actStart;
                            if (best.isEndTurn()) {
                                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                            } else {
                                inject(gameState, new ClientCommandActingPlayer(
                                    best.player.getId(), best.action, false));
                            }
                            break;
                        }
                        // Fall through to scripted argmax if no agent attached.
                    }

                    MoveDecisionEngine.PlayerSelection sel;
                    if (mode == AgentMode.MODEL) {
                        OnnxModelAgent agent = home ? homeModelAgent : awayModelAgent;
                        sel = (agent != null)
                            ? agent.selectPlayer(game, myTeam, oppTeam, home, home)
                            : MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, home, home, rng, true);
                    } else {
                        boolean argmax = (mode == AgentMode.SCRIPTED_ARGMAX);
                        sel = MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, home, home, rng, argmax);
                    }
                    t.activations++;
                    t.activationNs += System.nanoTime() - actStart;
                    if (collector != null) collector.onPlayerSelect(game, sel, mode.name());
                    if (sel.player == null) {
                        inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                    } else {
                        inject(gameState, new ClientCommandActingPlayer(
                            sel.player.getId(), sel.action, false));
                    }
                } else {
                    // Phase 2: player is selected — send their action command
                    handleActionForSelectedPlayer(game, gameState, ap, home);
                }
                break;
            }

            case INIT_MOVING: {
                // Normally the player has already moved via the full path sent in INIT_SELECTING.
                // For blood-lust FOUL_MOVE, StepEndFouling pushes a fresh Move sequence with no
                // prior move — INIT_MOVING is the first step and the vampire still needs to foul.
                // StepInitMoving handles CLIENT_FOUL for FOUL_MOVE by dispatching the Foul sequence.
                ActingPlayer apMove = game.getActingPlayer();
                if (apMove != null && apMove.getPlayerAction() == PlayerAction.FOUL_MOVE
                        && !apMove.hasFouled()) {
                    Team oppTeamMove = home ? game.getTeamAway() : game.getTeamHome();
                    FieldCoordinate posMove = game.getFieldModel().getPlayerCoordinate(apMove.getPlayer());
                    Player<?>[] foulTargets = posMove != null
                        ? UtilPlayer.findAdjacentPronePlayers(game, oppTeamMove, posMove)
                        : new Player<?>[0];
                    if (foulTargets != null && foulTargets.length > 0) {
                        inject(gameState, new ClientCommandFoul(apMove.getPlayerId(), foulTargets[0].getId(), false));
                        break;
                    }
                }
                inject(gameState, new ClientCommandActingPlayer(null, null, false));
                break;
            }

            case TOUCHBACK:
                handleTouchback(game, gameState, home);
                break;

            case PUSHBACK:
                handlePushback(game, gameState, home);
                break;

            case KICKOFF_RETURN:
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;

            default:
                inject(gameState, new ClientCommandActingPlayer(null, null, false));
                break;
        }
    }

    // ── Phase 2: player selected — send action command ────────────────────────

    private void handleActionForSelectedPlayer(Game game, GameState gameState,
                                               ActingPlayer ap, boolean home) {
        AgentMode mode = home ? homeMode : awayMode;
        Team myTeam  = home ? game.getTeamHome() : game.getTeamAway();
        Team oppTeam = home ? game.getTeamAway() : game.getTeamHome();
        PlayerAction action = ap.getPlayerAction();

        if (action == PlayerAction.MOVE || action == PlayerAction.FOUL_MOVE) {
            MoveDecisionEngine.MoveResult mr;
            if (mode == AgentMode.MODEL) {
                OnnxModelAgent agent = home ? homeModelAgent : awayModelAgent;
                mr = (agent != null)
                    ? agent.selectMoveTarget(game, ap, myTeam, oppTeam, home)
                    : MoveDecisionEngine.selectMoveTarget(game, ap, myTeam, oppTeam, home, rng, true);
            } else {
                boolean argmax = (mode == AgentMode.SCRIPTED_ARGMAX);
                mr = MoveDecisionEngine.selectMoveTarget(game, ap, myTeam, oppTeam, home, rng, argmax);
            }
            if (collector != null) collector.onMoveTarget(game, ap, mr, mode.name());
            PathProbabilityFinder.PathEntry entry = mr.chosen;
            if (entry == null || entry.path == null || entry.path.length == 0) {
                // If no candidates at all (hasEndOption=false), the server may reject deselect
                // while the player is active — end the turn to break the cycle.
                if (!mr.hasEndOption) {
                    inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                } else {
                    inject(gameState, new ClientCommandActingPlayer(null, null, false));
                }
                return;
            }
            FieldCoordinate from = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
            // The server applies .transform() to away-team coordinates, so pre-transform.
            FieldCoordinate fromSend = home ? from : from.transform();
            FieldCoordinate[] pathSend = home ? entry.path : transformPath(entry.path);
            inject(gameState, new ClientCommandMove(ap.getPlayerId(), fromSend, pathSend, null));

        } else if (action == PlayerAction.BLITZ) {
            boolean argmax = (mode == AgentMode.SCRIPTED_ARGMAX || mode == AgentMode.MODEL);
            if (ap.getCurrentMove() > 0) {
                // Player has already moved this BLITZ activation — try to block now.
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
                Player<?>[] targets = pos != null
                    ? UtilPlayer.findAdjacentBlockablePlayers(game, oppTeam, pos)
                    : new Player<?>[0];
                if (targets != null && targets.length > 0) {
                    inject(gameState, new ClientCommandBlock(ap.getPlayerId(), targets[0].getId(),
                        false, false, false, false, false));
                } else {
                    inject(gameState, new ClientCommandActingPlayer(null, null, false));
                }
                return;
            }
            // Haven't moved yet — navigate toward a target.
            MoveDecisionEngine.MoveResult blitzMr;
            if (mode == AgentMode.MODEL) {
                OnnxModelAgent agent = home ? homeModelAgent : awayModelAgent;
                blitzMr = (agent != null)
                    ? agent.selectMoveTarget(game, ap, myTeam, oppTeam, home)
                    : MoveDecisionEngine.selectMoveTarget(game, ap, myTeam, oppTeam, home, rng, true);
            } else {
                blitzMr = MoveDecisionEngine.selectMoveTarget(game, ap, myTeam, oppTeam, home, rng, argmax);
            }
            if (collector != null) collector.onMoveTarget(game, ap, blitzMr, mode.name());
            PathProbabilityFinder.PathEntry entry = blitzMr.chosen;
            if (entry == null || entry.path == null || entry.path.length == 0) {
                // Can't or shouldn't move — try an immediate block if adjacent.
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
                Player<?>[] targets = pos != null
                    ? UtilPlayer.findAdjacentBlockablePlayers(game, oppTeam, pos)
                    : new Player<?>[0];
                if (targets != null && targets.length > 0) {
                    inject(gameState, new ClientCommandBlock(ap.getPlayerId(), targets[0].getId(),
                        false, false, false, false, false));
                } else {
                    inject(gameState, new ClientCommandActingPlayer(null, null, false));
                }
                return;
            }
            FieldCoordinate from = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
            FieldCoordinate fromSend = home ? from : from.transform();
            FieldCoordinate[] pathSend = home ? entry.path : transformPath(entry.path);
            inject(gameState, new ClientCommandMove(ap.getPlayerId(), fromSend, pathSend, null));

        } else if (action == PlayerAction.BLOCK) {
            FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
            Player<?>[] targets = pos != null
                ? UtilPlayer.findAdjacentBlockablePlayers(game, oppTeam, pos)
                : new Player<?>[0];
            if (targets == null || targets.length == 0) {
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                return;
            }
            Player<?> target = targets[0];
            FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();
            for (Player<?> tt : targets) {
                FieldCoordinate tc = game.getFieldModel().getPlayerCoordinate(tt);
                if (tc != null && tc.equals(ballCoord)) { target = tt; break; }
            }
            inject(gameState, new ClientCommandBlock(ap.getPlayerId(), target.getId(),
                false, false, false, false, false));

        } else if (action == PlayerAction.FOUL) {
            // Don't re-foul if already used this turn (server rejects it → cycle).
            if (game.getTurnData().isFoulUsed()) {
                inject(gameState, new ClientCommandActingPlayer(null, null, false));
                return;
            }
            FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(ap.getPlayer());
            Player<?>[] targets = pos != null
                ? UtilPlayer.findAdjacentPronePlayers(game, oppTeam, pos)
                : new Player<?>[0];
            if (targets == null || targets.length == 0) {
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                return;
            }
            inject(gameState, new ClientCommandFoul(ap.getPlayerId(), targets[0].getId(), false));

        } else {
            inject(gameState, new ClientCommandActingPlayer(null, null, false));
        }
    }

    // ── Path transform helper ─────────────────────────────────────────────────

    /** Pre-transform a path for the away team: server will apply .transform() itself. */
    private static FieldCoordinate[] transformPath(FieldCoordinate[] path) {
        FieldCoordinate[] result = new FieldCoordinate[path.length];
        for (int i = 0; i < path.length; i++) result[i] = path[i].transform();
        return result;
    }

    // ── TOUCHBACK ─────────────────────────────────────────────────────────────

    private void handleTouchback(Game game, GameState gameState, boolean home) {
        Team myTeam = home ? game.getTeamHome() : game.getTeamAway();
        FieldModel fm = game.getFieldModel();

        // Best ball carrier on our team
        Player<?> best = null;
        int bestScore = -1;
        for (Player<?> p : myTeam.getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            FieldCoordinate coord = fm.getPlayerCoordinate(p);
            if (ps == null || coord == null || !ps.isActive() || ps.getBase() != PlayerState.STANDING) continue;
            int score = ScriptedStrategy.ballCarrierScore(p);
            if (score > bestScore) { bestScore = score; best = p; }
        }

        if (best != null) {
            FieldCoordinate coord = fm.getPlayerCoordinate(best);
            if (coord != null) {
                inject(gameState, new ClientCommandTouchback(home ? coord : coord.transform()));
                return;
            }
        }
        // Fallback: first standing player on our team
        for (Player<?> p : myTeam.getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            FieldCoordinate coord = fm.getPlayerCoordinate(p);
            if (ps != null && ps.getBase() == PlayerState.STANDING && coord != null) {
                inject(gameState, new ClientCommandTouchback(home ? coord : coord.transform()));
                return;
            }
        }
        inject(gameState, new ClientCommandTouchback(null));
    }

    // ── PUSHBACK ──────────────────────────────────────────────────────────────

    private void handlePushback(Game game, GameState gameState, boolean home) {
        PushbackSquare[] squares = game.getFieldModel().getPushbackSquares();

        // Find a non-locked pushback square owned by the current side; pick the one
        // that moves the defender furthest into their endzone (off-pitch is best).
        PushbackSquare best = null;
        int bestScore = Integer.MIN_VALUE;
        if (squares != null) {
            for (PushbackSquare sq : squares) {
                if (sq.isLocked()) continue;
                FieldCoordinate coord = sq.getCoordinate();
                int score = FieldCoordinateBounds.FIELD.isInBounds(coord)
                    ? (home ? coord.getX() : 25 - coord.getX())
                    : 1000;
                if (score > bestScore) { bestScore = score; best = sq; }
            }
        }

        Pushback pushback = null;
        if (best != null) {
            // Derive pushed player from the square's direction (same as live client).
            FieldCoordinate toCoord = best.getCoordinate();
            FieldCoordinate fromCoord = computeFromCoord(toCoord, best.getDirection());
            Player<?> pushedPlayer = fromCoord != null ? game.getFieldModel().getPlayer(fromCoord) : null;
            if (pushedPlayer == null) {
                // Fallback: use defender from game state
                pushedPlayer = game.getDefender();
            }
            if (pushedPlayer != null) {
                FieldCoordinate sendCoord = home ? toCoord : toCoord.transform();
                pushback = new Pushback(pushedPlayer.getId(), sendCoord);
            }
        }

        if (pushback == null) {
            // No valid pushback available — build a fallback off-pitch push
            Player<?> defender = game.getDefender();
            if (defender == null && game.getDefenderId() != null) {
                defender = game.getPlayerById(game.getDefenderId());
            }
            String pid = defender != null ? defender.getId() : null;
            FieldCoordinate fallback = home ? new FieldCoordinate(26, 1) : new FieldCoordinate(1, 1);
            pushback = new Pushback(pid, fallback);
        }

        inject(gameState, new ClientCommandPushback(pushback));
    }

    /** Reverse a pushback direction to find the 'from' coordinate. */
    private static FieldCoordinate computeFromCoord(FieldCoordinate toCoord, Direction direction) {
        if (toCoord == null || direction == null) return null;
        switch (direction) {
            case NORTH:     return toCoord.add(0, 1);
            case NORTHEAST: return toCoord.add(-1, 1);
            case EAST:      return toCoord.add(-1, 0);
            case SOUTHEAST: return toCoord.add(-1, -1);
            case SOUTH:     return toCoord.add(0, -1);
            case SOUTHWEST: return toCoord.add(1, -1);
            case WEST:      return toCoord.add(1, 0);
            case NORTHWEST: return toCoord.add(1, 1);
            default:        return null;
        }
    }

    // ── Dialog handling ───────────────────────────────────────────────────────

    private void handleDialog(IDialogParameter dialog, Game game, GameState gameState,
                              GameTimings t) {
        switch (dialog.getId()) {
            case KICKOFF_RETURN:
            case SETUP_ERROR:
            case SWARMING_ERROR:
            case INVALID_SOLID_DEFENCE:
                game.setDialogParameter(null);
                break;
            default:
                comm.clearCaptured();
                boolean homeDialog = isHomeDialog(dialog, game);
                AgentMode mode = homeDialog ? homeMode : awayMode;

                if (mode == AgentMode.RANDOM) {
                    RandomStrategy.respondToDialog(dialog, game, comm);
                } else {
                    // MODEL mode falls through to ScriptedStrategy argmax for dialog
                    // (dialog execution is coupled to ScriptedStrategy's command dispatch)
                    double temp = (mode == AgentMode.SCRIPTED_ARGMAX || mode == AgentMode.MODEL) ? 0.0 : 0.5;
                    ScriptedStrategy.setTemperature(temp);
                    if (collector != null) ScriptedStrategy.startLogging();
                    ScriptedStrategy.respondToDialog(dialog, game, comm);
                    if (collector != null) {
                        com.fumbbl.ffb.ai.strategy.DecisionLog dlog = ScriptedStrategy.getAndClearLog();
                        collector.onDialog(dialog, game, dlog, mode.name());
                    }
                }

                ClientCommand captured = comm.getCapturedCommand();
                if (captured != null) {
                    String dialogTeamId = getDialogTeamId(dialog);
                    try {
                        if (dialogTeamId != null) {
                            injectForTeam(gameState, captured, dialogTeamId.equals(game.getTeamHome().getId()));
                        } else {
                            inject(gameState, captured);
                        }
                    } catch (RuntimeException e) {
                        // Command was rejected — clear the dialog to avoid a hang
                        game.setDialogParameter(null);
                    }
                } else {
                    game.setDialogParameter(null);
                }
                break;
        }
    }

    private boolean isHomeDialog(IDialogParameter dialog, Game game) {
        String teamId = getDialogTeamId(dialog);
        if (teamId != null) return teamId.equals(game.getTeamHome().getId());
        try {
            java.lang.reflect.Method m = dialog.getClass().getMethod("getPlayerId");
            String playerId = (String) m.invoke(dialog);
            if (playerId != null) {
                Player<?> p = game.getPlayerById(playerId);
                if (p != null) return game.getTeamHome().hasPlayer(p);
            }
        } catch (Exception ignored) {}
        return game.isHomePlaying();
    }

    private static String getDialogTeamId(IDialogParameter dialog) {
        if (dialog instanceof DialogArgueTheCallParameter) {
            return ((DialogArgueTheCallParameter) dialog).getTeamId();
        }
        if (dialog instanceof DialogBriberyAndCorruptionParameter) {
            return ((DialogBriberyAndCorruptionParameter) dialog).getTeamId();
        }
        if (dialog instanceof DialogPlayerChoiceParameter) {
            return ((DialogPlayerChoiceParameter) dialog).getTeamId();
        }
        if (dialog instanceof com.fumbbl.ffb.dialog.DialogBribesParameter) {
            return ((com.fumbbl.ffb.dialog.DialogBribesParameter) dialog).getTeamId();
        }
        return null;
    }

    // ── Setup helpers ─────────────────────────────────────────────────────────

    private static void resetCurrentTeam(Game game) {
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();
        FieldModel fm = game.getFieldModel();
        for (Player<?> p : team.getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (ps.canBeSetUpNextDrive()) {
                fm.setPlayerState(p, ps.changeBase(PlayerState.RESERVE));
                com.fumbbl.ffb.util.UtilBox.putPlayerIntoBox(game, p);
            }
        }
    }

    private void placeReserves(Game game, GameState gameState) {
        if (homeSetup == null || awaySetup == null) {
            // No setup files available (e.g., MCTS rollout in GUI mode) — skip placement.
            return;
        }
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();
        FieldModel fm = game.getFieldModel();

        int available = 0, onField = 0, onLos = 0;
        for (Player<?> p : team.getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (!ps.canBeSetUpNextDrive()) continue;
            available++;
            FieldCoordinate coord = fm.getPlayerCoordinate(p);
            boolean inHalf = homePlaying
                ? FieldCoordinateBounds.HALF_HOME.isInBounds(coord)
                : FieldCoordinateBounds.HALF_AWAY.isInBounds(coord);
            if (inHalf) {
                onField++;
                if (homePlaying ? FieldCoordinateBounds.LOS_HOME.isInBounds(coord)
                               : FieldCoordinateBounds.LOS_AWAY.isInBounds(coord)) onLos++;
            }
        }
        int losNeeded   = (available >= 3) ? Math.max(0, 3 - onLos)     : Math.max(0, available - onLos);
        int fieldNeeded = Math.max(0, Math.min(available, 11) - onField);
        if (losNeeded == 0 && fieldNeeded == 0) return;

        int[][] losSquares = {{12,7},{12,6},{12,8},{12,5},{12,9},{12,4},{12,10}};
        int[][] overflowSq = {{5,5},{5,7},{5,9},{6,6},{6,8},{4,6},{4,8},{3,6},{3,8},{2,5},{2,9},{1,7}};
        int li = 0, oi = 0;

        for (Player<?> p : team.getPlayers()) {
            if (losNeeded <= 0 && fieldNeeded <= 0) break;
            PlayerState ps = fm.getPlayerState(p);
            if (!ps.canBeSetUpNextDrive() || ps.getBase() != PlayerState.RESERVE) continue;

            if (losNeeded > 0) {
                while (li < losSquares.length) {
                    int ox = losSquares[li][0], oy = losSquares[li++][1];
                    FieldCoordinate gc = homePlaying ? new FieldCoordinate(ox,oy) : new FieldCoordinate(ox,oy).transform();
                    if (fm.getPlayer(gc) == null) {
                        com.fumbbl.ffb.server.util.UtilServerSetup.setupPlayer(gameState, p.getId(), new FieldCoordinate(ox,oy));
                        losNeeded--; fieldNeeded--; break;
                    }
                }
            } else {
                while (oi < overflowSq.length) {
                    int ox = overflowSq[oi][0], oy = overflowSq[oi++][1];
                    FieldCoordinate gc = homePlaying ? new FieldCoordinate(ox,oy) : new FieldCoordinate(ox,oy).transform();
                    if (fm.getPlayer(gc) == null) {
                        com.fumbbl.ffb.server.util.UtilServerSetup.setupPlayer(gameState, p.getId(), new FieldCoordinate(ox,oy));
                        fieldNeeded--; break;
                    }
                }
            }
        }

    }

    // ── Injection helpers ─────────────────────────────────────────────────────

    public static void inject(GameState gameState, ClientCommand cmd) {
        boolean home = gameState.getGame().isHomePlaying();
        Session session = home ? HeadlessFantasyFootballServer.HOME_SESSION
                               : HeadlessFantasyFootballServer.AWAY_SESSION;
        gameState.handleCommand(new ReceivedCommand(cmd, session));
    }

    public static void injectForTeam(GameState gameState, ClientCommand cmd, boolean homeTeam) {
        Session session = homeTeam ? HeadlessFantasyFootballServer.HOME_SESSION
                                   : HeadlessFantasyFootballServer.AWAY_SESSION;
        gameState.handleCommand(new ReceivedCommand(cmd, session));
    }

    /**
     * Run simulation for at most {@code maxActivations} player activations
     * (INIT_SELECTING phase-1 visits), then stop.  Intended for MCTS rollouts.
     *
     * <p>The caller is responsible for having already injected the first
     * {@code ClientCommandActingPlayer} (phase-1 injection of the candidate).
     * This method drives from that point onward until either the activation
     * budget is exhausted or the game ends.
     *
     * @param gameState     state to drive (must have INIT_SELECTING as current step)
     * @param maxActivations number of additional phase-1 INIT_SELECTING visits before stopping
     */
    public void runForActivations(GameState gameState, int maxActivations) {
        Game game = gameState.getGame();
        int activationsRemaining = maxActivations;
        int iter = 0;

        // Dummy timing containers (rollouts don't track timing).
        GameTimings dummyTimings = new GameTimings();
        long[] dummyPs = new long[]{0L, 0L, 0L};

        // Minimal stuck-detectors to avoid infinite loops in rollouts.
        StepId generalStuckStep = null;
        int generalStuckCount = 0;
        int generalEndTurnCount = 0;
        final int MAX_GENERAL_REPEATS = 30;
        final int MAX_END_TURN_RETRIES = 3;

        while (game.getFinished() == null && ++iter < MAX_ITERATIONS) {
            IStep step = gameState.getCurrentStep();
            if (step == null) break;

            IDialogParameter dialog = game.getDialogParameter();
            StepId stepId = step.getId();

            // A kickoff-phase step means a TD was scored / half ended — end rollout.
            if (KICKOFF_STEPS.contains(stepId)) {
                break;
            }

            // Check activation budget at phase-1 entry.
            if (stepId == StepId.INIT_SELECTING && dialog == null) {
                ActingPlayer ap = game.getActingPlayer();
                if (ap == null || ap.getPlayerId() == null) {
                    if (activationsRemaining <= 0) break;
                    activationsRemaining--;
                }
            }

            // Stuck-step guard.
            if (stepId != StepId.INIT_SELECTING && dialog == null) {
                if (stepId == generalStuckStep) {
                    if (++generalStuckCount >= MAX_GENERAL_REPEATS) {
                        if (++generalEndTurnCount >= MAX_END_TURN_RETRIES) {
                            break;
                        }
                        inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                        generalStuckCount = 0;
                        continue;
                    }
                } else {
                    generalStuckStep = stepId;
                    generalStuckCount = 1;
                    generalEndTurnCount = 0;
                }
            } else {
                generalStuckStep = null;
                generalStuckCount = 0;
                generalEndTurnCount = 0;
            }

            if (dialog != null && stepId != StepId.INIT_SELECTING) {
                handleDialog(dialog, game, gameState, dummyTimings);
            } else {
                handleStep(stepId, game, gameState, dummyTimings, dummyPs);
            }
        }
    }

    /**
     * Maximum step-loop iterations for a single activation advance in MCTS rollouts.
     * Legitimate activations (move + block/foul cascade) complete in &lt;200 steps;
     * capping at 500 prevents stuck activations from looping indefinitely while
     * leaving a safe margin for unusual sequences.
     */
    public static final int MAX_ACTIVATION_ITERATIONS = 500;

    /**
     * Drive all sub-steps of the current activation until the next phase-1
     * {@code INIT_SELECTING} or until the turn ends.
     *
     * <p>Stops <em>before</em> handling the next phase-1 {@code INIT_SELECTING}
     * so the caller retains control over which player to activate next.
     *
     * @param gameState   the game state to advance
     * @param isHome      which team is the acting team for this turn
     * @param maxIter     maximum step-loop iterations before returning {@code false};
     *                    use {@link #MAX_ITERATIONS} for full-game simulation or
     *                    {@link #MAX_ACTIVATION_ITERATIONS} for MCTS rollouts
     * @return {@code true} if stopped at a phase-1 INIT_SELECTING for {@code isHome}'s
     *         team (turn continues); {@code false} if the turn ended, switched teams,
     *         game finished, or iteration cap reached
     */
    public boolean advanceToNextActivationOrTurnEnd(GameState gameState, boolean isHome,
                                                    int maxIter) {
        Game game = gameState.getGame();
        int iter = 0;

        GameTimings dummyTimings = new GameTimings();
        long[] dummyPs = new long[]{0L, 0L, 0L};

        StepId generalStuckStep = null;
        int generalStuckCount = 0;
        int generalEndTurnCount = 0;
        final int MAX_GENERAL_REPEATS = 30;
        // After this many EndTurn injections on the same stuck step without progress,
        // give up and treat as turn-end.  Limits worst-case to MAX_GENERAL_REPEATS ×
        // MAX_END_TURN_RETRIES = 90 iterations per stuck step, not 100,000.
        final int MAX_END_TURN_RETRIES = 3;

        while (game.getFinished() == null && ++iter < maxIter) {
            IStep step = gameState.getCurrentStep();
            if (step == null) return false;

            IDialogParameter dialog = game.getDialogParameter();
            StepId stepId = step.getId();

            // A kickoff-phase step means a TD was scored / half ended — treat as turn end.
            if (KICKOFF_STEPS.contains(stepId)) {
                return false;
            }

            // Stop at phase-1 INIT_SELECTING — let the caller decide the next action.
            if (stepId == StepId.INIT_SELECTING && dialog == null) {
                ActingPlayer ap = game.getActingPlayer();
                if (ap == null || ap.getPlayerId() == null) {
                    // Phase 1: return whether the same team is still acting.
                    return game.isHomePlaying() == isHome;
                }
            }

            // Stuck-step guard.
            if (stepId != StepId.INIT_SELECTING && dialog == null) {
                if (stepId == generalStuckStep) {
                    if (++generalStuckCount >= MAX_GENERAL_REPEATS) {
                        if (++generalEndTurnCount >= MAX_END_TURN_RETRIES) {
                            // EndTurn hasn't helped — treat as turn-end boundary.
                            return false;
                        }
                        inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                        generalStuckCount = 0;
                        continue;
                    }
                } else {
                    generalStuckStep = stepId;
                    generalStuckCount = 1;
                    generalEndTurnCount = 0;
                }
            } else {
                generalStuckStep = null;
                generalStuckCount = 0;
                generalEndTurnCount = 0;
            }

            if (dialog != null && stepId != StepId.INIT_SELECTING) {
                handleDialog(dialog, game, gameState, dummyTimings);
            } else {
                handleStep(stepId, game, gameState, dummyTimings, dummyPs);
            }
        }
        return false;
    }

    /** Convenience overload using the full {@link #MAX_ITERATIONS} cap (normal simulation). */
    public boolean advanceToNextActivationOrTurnEnd(GameState gameState, boolean isHome) {
        return advanceToNextActivationOrTurnEnd(gameState, isHome, MAX_ITERATIONS);
    }

    /**
     * Drive the remaining activations of a turn to completion using scripted-argmax
     * player selection, starting from the current {@code INIT_SELECTING} phase-1
     * decision point.
     *
     * <p>Unlike {@link #advanceToNextActivationOrTurnEnd}, which stops <em>before</em>
     * the next phase-1 {@code INIT_SELECTING}, this method handles {@code INIT_SELECTING}
     * internally and loops until the turn ends or the game is over.  It is used by the
     * MCTS rollout simulation step to complete a turn after expanding a new tree node.
     *
     * @param gameState the rollout game state; must be at a phase-1 {@code INIT_SELECTING}
     *                  for {@code isHome}'s team when called
     * @param isHome    which team's turn is being driven
     */
    public void advanceTurnEndScripted(GameState gameState, boolean isHome) {
        Game game = gameState.getGame();
        int safetyIter = 0;
        final int MAX_ACTIVATIONS = 20; // Blood Bowl: ≤11 players + some safety margin

        while (game.getFinished() == null && safetyIter++ < MAX_ACTIVATIONS) {
            // We are at INIT_SELECTING phase 1 for isHome's team.
            // Apply scripted-argmax activation selection.
            Team myTeam  = isHome ? game.getTeamHome() : game.getTeamAway();
            Team oppTeam = isHome ? game.getTeamAway() : game.getTeamHome();
            MoveDecisionEngine.PlayerSelection sel =
                MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, isHome, isHome, rng, true);
            if (sel.player == null) {
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
            } else {
                inject(gameState, new ClientCommandActingPlayer(
                    sel.player.getId(), sel.action, false));
            }
            // Advance through sub-steps until the next INIT_SELECTING phase 1 or turn end.
            boolean sameTurn = advanceToNextActivationOrTurnEnd(gameState, isHome);
            if (!sameTurn) return;
        }
    }

}

