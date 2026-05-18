package com.fumbbl.sgo.mcts;

/**
 * Optional PUCT prior for {@link MctsSearch}.
 *
 * <p>Implement this interface to provide a learned or scripted action prior
 * distribution over the available edge IDs at a given state.  When set on
 * a {@link MctsSearch}, the UCB formula is replaced with PUCT:
 *
 * <pre>
 *   U(a) = Q(a) + C_PUCT × P(a) × sqrt(N) / (1 + n(a))
 * </pre>
 *
 * <p>Return {@code null} to fall back to plain UCB for a specific state.
 */
public interface IActionPrior {

    /**
     * Compute a prior probability distribution over the given edge IDs.
     *
     * @param stateHash Zobrist hash of the current state
     * @param edgeIds   array of action IDs (cell indices 0–63 or END_TURN)
     * @param count     number of valid entries in {@code edgeIds}
     * @return array of length {@code count} with probabilities summing to 1.0,
     *         or {@code null} to use UCB for this state
     */
    double[] computePrior(long stateHash, int[] edgeIds, int count);
}
