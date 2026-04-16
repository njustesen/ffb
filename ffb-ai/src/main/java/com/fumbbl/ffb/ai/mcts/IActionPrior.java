package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.model.Game;

import java.util.List;

/**
 * Action prior distribution for PUCT-based Blood Bowl MCTS.
 *
 * <p>Implement this interface to provide a prior probability distribution
 * over a set of candidate player activations.  When injected into
 * {@link BbMctsSearch}, the plain UCB formula is replaced with PUCT:
 *
 * <pre>
 *   U(a) = Q(a) + C_PUCT × P(a) × sqrt(N) / (1 + n(a))
 * </pre>
 */
public interface IActionPrior {

    /**
     * Compute prior probabilities for the given candidates.
     *
     * @param candidates the candidate player activations
     * @param game       current game state (read-only)
     * @return array of length {@code candidates.size()} with probabilities summing to 1.0,
     *         or {@code null} to fall back to plain UCB
     */
    double[] computePrior(List<BbAction> candidates, Game game);

    /**
     * Compute prior probabilities using pre-computed raw scores from
     * {@code MoveDecisionEngine.selectPlayer()}.  Avoids a second Dijkstra pass.
     *
     * <p>Default implementation ignores {@code rawScores} and delegates to
     * {@link #computePrior}.  Override to use the scores directly.
     *
     * @param candidates the candidate player activations (parallel with rawScores)
     * @param rawScores  raw scores from selectPlayer(), or {@code null} if unavailable
     * @param game       current game state (read-only)
     * @return array of probabilities, or {@code null} to fall back to plain UCB
     */
    default double[] computePriorFromScores(List<BbAction> candidates, double[] rawScores, Game game) {
        return computePrior(candidates, game);
    }
}
