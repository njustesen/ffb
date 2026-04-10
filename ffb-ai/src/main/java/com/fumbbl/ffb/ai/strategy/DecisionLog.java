package com.fumbbl.ffb.ai.strategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Per-dialog decision log captured by {@link ScriptedStrategy} when logging is enabled.
 *
 * <p>Each call to {@code pick()} or {@code pickBool()} appends one entry:
 * the raw softmax-input scores and the index that was chosen.
 *
 * <p>For {@code pickBool()}, the scores array has length 2: {@code [scoreTrue, scoreFalse]}.
 * The chosen index is 0 for {@code true}, 1 for {@code false}.
 *
 * <p>A single {@code respondToDialog()} call typically produces exactly one entry, but
 * complex dialog handlers may call {@code pick()} multiple times (e.g. selecting
 * among N players in a PLAYER_CHOICE dialog produces N rounds of scoring).
 */
public final class DecisionLog {

    private final List<double[]> scores  = new ArrayList<>();
    private final List<Integer>  chosen  = new ArrayList<>();

    /** Called by {@link ScriptedStrategy#pick} after choosing. */
    void add(double[] scoreArray, int chosenIndex) {
        scores.add(scoreArray.clone());
        chosen.add(chosenIndex);
    }

    /** Called by {@link ScriptedStrategy#pickBool} after choosing. */
    void addBool(double scoreTrue, double scoreFalse, boolean choice) {
        scores.add(new double[]{scoreTrue, scoreFalse});
        chosen.add(choice ? 0 : 1);
    }

    /** Number of pick/pickBool calls recorded. */
    public int size() { return chosen.size(); }

    /** Raw scores for the i-th pick (caller must not modify the returned array). */
    public double[] getScores(int i) { return scores.get(i); }

    /** Chosen index for the i-th pick. */
    public int getChosen(int i) { return chosen.get(i); }

    /** Scores for the first (and usually only) pick — convenience accessor. */
    public double[] firstScores() { return scores.isEmpty() ? new double[0] : scores.get(0); }

    /** Chosen index for the first pick. */
    public int firstChosen() { return chosen.isEmpty() ? 0 : chosen.get(0); }
}
