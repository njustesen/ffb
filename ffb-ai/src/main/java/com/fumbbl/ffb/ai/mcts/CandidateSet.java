package com.fumbbl.ffb.ai.mcts;

import java.util.List;

/**
 * Candidate player activations at an {@code INIT_SELECTING} node, together with
 * the raw scores produced by {@code MoveDecisionEngine.selectPlayer()}.
 *
 * <p>Caching both together means the prior distribution can be derived from the
 * pre-computed scores without a second {@code selectPlayer()} call.
 */
final class CandidateSet {

    /** Candidate player activations. Never {@code null}; may be empty (end-turn only). */
    final List<BbAction> actions;

    /**
     * Raw scores from {@code selectPlayer()}, parallel with {@code actions}.
     * {@code null} if {@code selectPlayer()} returned no scores.
     */
    final double[] rawScores;

    CandidateSet(List<BbAction> actions, double[] rawScores) {
        this.actions = actions;
        this.rawScores = rawScores;
    }
}
