package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.model.Game;

/**
 * {@link ILeafEval} wrapper around {@link BbMctsSearch#staticEval}.
 */
final class StaticLeafEval implements ILeafEval {

    @Override
    public double evaluate(Game game, boolean isHome) {
        return BbMctsSearch.staticEval(game, isHome);
    }
}
