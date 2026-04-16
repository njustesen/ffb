package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.model.Game;

/**
 * Pluggable leaf-node evaluator for MCTS.
 *
 * <p>Returns a win probability in [0, 1] from the specified team's perspective.
 * Implementations include {@link StaticLeafEval} (fast heuristic) and
 * {@code OnnxLeafEval} (trained value head).
 */
public interface ILeafEval {

    /**
     * Evaluates the given game state and returns a win probability in [0, 1]
     * from {@code isHome}'s perspective.
     *
     * @param game   current game state at the leaf node
     * @param isHome {@code true} to evaluate from the home team's perspective
     * @return win probability in [0, 1]
     */
    double evaluate(Game game, boolean isHome);
}
