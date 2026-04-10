package com.fumbbl.ffb.ai.simulation;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.PathProbabilityFinder;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;

import java.io.Closeable;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * ONNX-based agent that uses three trained BC models for player-select and
 * move-target decisions. Dialog decisions fall back to ScriptedStrategy.
 *
 * <p>Load with {@link #load(String, FeatureExtractor)} after model ONNX files
 * have been exported. The base path should point to the directory or prefix
 * used when exporting (e.g., "/path/to/bc_model"), and the three files
 * {@code bc_model_dialog.onnx}, {@code bc_model_player_select.onnx}, and
 * {@code bc_model_move_target.onnx} must exist next to it.
 */
public final class OnnxModelAgent implements Closeable {

    private static final int BOARD_W = FeatureExtractor.BOARD_W;
    private static final int BOARD_H = FeatureExtractor.BOARD_H;
    private static final int N_CH    = FeatureExtractor.N_BOARD_CHANNELS;
    private static final int NS_DIM  = FeatureExtractor.NS_DIM;
    private static final int MAX_CANDS = FeatureExtractor.MAX_CANDS;
    private static final int MAX_SKILLS = FeatureExtractor.MAX_SKILLS;

    private final OrtEnvironment env;
    private final OrtSession sessionDialog;
    private final OrtSession sessionPlayerSelect;
    private final OrtSession sessionMoveTarget;
    private final FeatureExtractor extractor;
    private final int nDialogTypes;
    private final Random rng = new Random(42);

    private OnnxModelAgent(OrtEnvironment env,
                            OrtSession sessionDialog,
                            OrtSession sessionPlayerSelect,
                            OrtSession sessionMoveTarget,
                            FeatureExtractor extractor,
                            int nDialogTypes) {
        this.env                  = env;
        this.sessionDialog        = sessionDialog;
        this.sessionPlayerSelect  = sessionPlayerSelect;
        this.sessionMoveTarget    = sessionMoveTarget;
        this.extractor            = extractor;
        this.nDialogTypes         = nDialogTypes;
    }

    /**
     * Load three ONNX model files from the given base path prefix.
     *
     * @param modelPrefix  e.g. "/tmp/ffb-ckpt2/bc_model"
     * @param extractor    pre-configured FeatureExtractor with the skill vocab
     * @param nDialogTypes number of dialog type classes (from vocab.json)
     */
    public static OnnxModelAgent load(String modelPrefix, FeatureExtractor extractor,
                                       int nDialogTypes) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        OrtSession dialog       = env.createSession(modelPrefix + "_dialog.onnx",       opts);
        OrtSession playerSelect = env.createSession(modelPrefix + "_player_select.onnx", opts);
        OrtSession moveTarget   = env.createSession(modelPrefix + "_move_target.onnx",   opts);

        return new OnnxModelAgent(env, dialog, playerSelect, moveTarget, extractor, nDialogTypes);
    }

    // ── Player selection ──────────────────────────────────────────────────────

    /**
     * Selects which player to activate (and with what action) using the
     * player-select ONNX head.
     *
     * <p>Candidate generation delegates to {@link MoveDecisionEngine#selectPlayer}
     * with argmax=true (to obtain the candidates list); the model then picks
     * the best index from those candidates.
     *
     * @return a {@link MoveDecisionEngine.PlayerSelection} whose {@code player}
     *         and {@code action} reflect the model's choice (null player = end turn)
     */
    public MoveDecisionEngine.PlayerSelection selectPlayer(Game game, Team myTeam, Team oppTeam,
                                                            boolean isHome, boolean allowBlock) {
        // Get candidates from MoveDecisionEngine (argmax=true just to populate the list)
        MoveDecisionEngine.PlayerSelection ref =
            MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, isHome, allowBlock, rng, true);

        List<Player<?>> cands   = ref.candidatePlayers;
        List<PlayerAction> acts = ref.candidateActions;

        // Count real (non-null) candidates
        int nReal = 0;
        List<Player<?>> realCands = new ArrayList<>();
        List<PlayerAction> realActs = new ArrayList<>();
        for (int i = 0; i < cands.size(); i++) {
            if (cands.get(i) != null) {
                realCands.add(cands.get(i));
                realActs.add(acts.get(i));
                nReal++;
            }
        }

        if (nReal == 0) {
            // Only end-turn available
            return ref;
        }

        try {
            float[] spatial    = extractor.buildSpatialBoard(game);
            float[] ns         = extractor.buildNonSpatial(game);
            int[]   skillIds   = extractor.buildCandidateSkillIds(realCands);
            float[] stats      = extractor.buildCandidateStats(realCands);
            float[] candMask   = extractor.buildCandidateMaskPs(nReal);

            // Build candidate positions (std coords) for player-select
            FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();
            List<FieldCoordinate> candCoords = new ArrayList<>();
            for (Player<?> p : realCands) {
                candCoords.add(p != null ? game.getFieldModel().getPlayerCoordinate(p) : null);
            }
            float[] candPos = extractor.buildCandidatePos(candCoords, ballCoord, isHome);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("spatial",        OnnxTensor.createTensor(env,
                FloatBuffer.wrap(spatial), new long[]{1, N_CH, BOARD_W, BOARD_H}));
            inputs.put("non_spatial",    OnnxTensor.createTensor(env,
                FloatBuffer.wrap(ns), new long[]{1, NS_DIM}));
            inputs.put("cand_skill_ids", OnnxTensor.createTensor(env,
                LongBuffer.wrap(tolong(skillIds)), new long[]{1, MAX_CANDS, MAX_SKILLS}));
            inputs.put("cand_stats",     OnnxTensor.createTensor(env,
                FloatBuffer.wrap(stats), new long[]{1, MAX_CANDS, 5}));
            inputs.put("cand_mask",      OnnxTensor.createTensor(env,
                FloatBuffer.wrap(candMask), new long[]{1, MAX_CANDS + 1}));
            inputs.put("cand_pos",       OnnxTensor.createTensor(env,
                FloatBuffer.wrap(candPos), new long[]{1, MAX_CANDS + 1, FeatureExtractor.CAND_POS_DIM}));

            try (OrtSession.Result result = sessionPlayerSelect.run(inputs)) {
                float[][] logProbs = (float[][]) result.get(0).getValue();
                int best = argmax(logProbs[0], MAX_CANDS + 1);
                // best >= nReal means end-turn
                if (best >= nReal) {
                    return new MoveDecisionEngine.PlayerSelection(null, null,
                        ref.rawScores, cands, acts);
                } else {
                    return new MoveDecisionEngine.PlayerSelection(realCands.get(best), realActs.get(best),
                        ref.rawScores, cands, acts);
                }
            }
        } catch (OrtException e) {
            // Fall back to ScriptedStrategy selection on error
            return ref;
        }
    }

    // ── Move target selection ─────────────────────────────────────────────────

    /**
     * Selects a move target (or end-activation) using the move-target ONNX head.
     *
     * <p>Candidate generation delegates to {@link MoveDecisionEngine#selectMoveTarget}
     * with argmax=true; the model re-ranks the same candidates.
     */
    public MoveDecisionEngine.MoveResult selectMoveTarget(Game game, ActingPlayer ap,
                                                           Team myTeam, Team oppTeam, boolean isHome) {
        MoveDecisionEngine.MoveResult ref =
            MoveDecisionEngine.selectMoveTarget(game, ap, myTeam, oppTeam, isHome, rng, true);

        List<FieldCoordinate> candidates = ref.candidates;
        if (candidates.isEmpty() && !ref.hasEndOption) {
            return ref;
        }

        try {
            float[] spatial  = extractor.buildSpatialBoard(game);
            float[] ns       = extractor.buildNonSpatial(game);
            float[] candMask = extractor.buildMoveCandidateMask(candidates, ref.hasEndOption, isHome);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("spatial",        OnnxTensor.createTensor(env,
                FloatBuffer.wrap(spatial), new long[]{1, N_CH, BOARD_W, BOARD_H}));
            inputs.put("non_spatial",    OnnxTensor.createTensor(env,
                FloatBuffer.wrap(ns), new long[]{1, NS_DIM}));
            inputs.put("candidate_mask", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(candMask), new long[]{1, BOARD_W * BOARD_H + 1}));

            try (OrtSession.Result result = sessionMoveTarget.run(inputs)) {
                float[][] logProbs = (float[][]) result.get(0).getValue();
                float[] scores = logProbs[0];

                // Find the masked argmax
                int best = -1;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < BOARD_W * BOARD_H; i++) {
                    if (candMask[i] > 0 && scores[i] > bestVal) {
                        bestVal = scores[i];
                        best = i;
                    }
                }
                // Check end option
                int endIdx = BOARD_W * BOARD_H;
                if (ref.hasEndOption && scores[endIdx] > bestVal) {
                    best = endIdx;
                    bestVal = scores[endIdx];
                }

                if (best < 0 || best == endIdx) {
                    // End activation
                    return new MoveDecisionEngine.MoveResult(null, candidates, ref.rawScores,
                        ref.hasEndOption, ref.isBallCarrier, ref.isBallRetriever,
                        ref.isReceiver, ref.playerCoord);
                }

                // Map flat index back to FieldCoordinate → PathEntry
                // The index is in standardised x coords; inverse-flip to get actual board x
                int bxs = best / BOARD_H;
                int by  = best % BOARD_H;
                int bx  = FeatureExtractor.stdX(bxs, isHome);  // stdX is its own inverse
                FieldCoordinate target = new FieldCoordinate(bx, by);

                // Find the PathEntry for this coordinate
                PathProbabilityFinder.PathEntry entry = findEntry(game, ap, target);
                if (entry == null) {
                    // Coordinate not actually reachable (shouldn't happen but fallback)
                    return ref;
                }
                return new MoveDecisionEngine.MoveResult(entry, candidates, ref.rawScores,
                    ref.hasEndOption, ref.isBallCarrier, ref.isBallRetriever,
                    ref.isReceiver, ref.playerCoord);
            }
        } catch (OrtException e) {
            return ref;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private PathProbabilityFinder.PathEntry findEntry(Game game, ActingPlayer ap,
                                                       FieldCoordinate target) {
        Map<FieldCoordinate, PathProbabilityFinder.PathEntry> paths =
            PathProbabilityFinder.findAllPaths(game, ap);
        return paths.get(target);
    }

    private static int argmax(float[] arr, int n) {
        int best = 0;
        for (int i = 1; i < Math.min(arr.length, n); i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    private static long[] tolong(int[] ints) {
        long[] longs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) longs[i] = ints[i];
        return longs;
    }

    @Override
    public void close() {
        try { sessionDialog.close(); }       catch (OrtException ignored) {}
        try { sessionPlayerSelect.close(); } catch (OrtException ignored) {}
        try { sessionMoveTarget.close(); }   catch (OrtException ignored) {}
        try { env.close(); }                 catch (Exception ignored) {}
    }
}
