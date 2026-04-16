package com.fumbbl.ffb.ai.simulation;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fumbbl.ffb.ai.mcts.BbMctsSearch;
import com.fumbbl.ffb.ai.mcts.ILeafEval;
import com.fumbbl.ffb.model.Game;

import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * {@link ILeafEval} implementation backed by a trained ONNX value head.
 *
 * <p>Loads {@code value.onnx} — exported via {@code export_model.py --value}.
 * Features are extracted from the acting team's perspective via {@link FeatureExtractor}
 * (matches training-time extraction). The resulting win probability is flipped
 * when the currently-acting team differs from {@code isHome}.
 *
 * <p>The underlying {@link OrtSession} is thread-safe for concurrent reads.
 * Falls back to {@link BbMctsSearch#staticEval} on {@link OrtException}.
 */
public final class OnnxLeafEval implements ILeafEval, Closeable {

    private static final int N_CH    = FeatureExtractor.N_BOARD_CHANNELS;
    private static final int BOARD_W = FeatureExtractor.BOARD_W;
    private static final int BOARD_H = FeatureExtractor.BOARD_H;
    private static final int NS_DIM  = FeatureExtractor.NS_DIM;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final FeatureExtractor extractor;

    public OnnxLeafEval(String valuePath, FeatureExtractor extractor) throws OrtException {
        this.env       = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session   = env.createSession(valuePath, opts);
        this.extractor = extractor;
    }

    @Override
    public double evaluate(Game game, boolean isHome) {
        try {
            float[] spatialFlat = extractor.buildSpatialBoard(game);
            float[] ns          = extractor.buildNonSpatial(game);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("spatial",     OnnxTensor.createTensor(env,
                FloatBuffer.wrap(spatialFlat), new long[]{1, N_CH, BOARD_W, BOARD_H}));
            inputs.put("non_spatial", OnnxTensor.createTensor(env,
                FloatBuffer.wrap(ns), new long[]{1, NS_DIM}));

            try (OrtSession.Result result = session.run(inputs)) {
                float winProb = ((float[]) result.get(0).getValue())[0];
                // Model outputs acting-team win probability; flip if acting team != isHome
                boolean actingIsHome = game.isHomePlaying();
                return (actingIsHome == isHome) ? winProb : 1.0 - winProb;
            }
        } catch (OrtException e) {
            return BbMctsSearch.staticEval(game, isHome);
        }
    }

    @Override
    public void close() throws IOException {
        try {
            session.close();
        } catch (OrtException e) {
            throw new IOException("Failed to close OrtSession", e);
        }
    }
}
