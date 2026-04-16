package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.model.Game;

import java.util.List;

/**
 * Uniform action prior — equivalent to UCB when used with PUCT.
 *
 * <p>Returns {@code 1/n} for each of the {@code n} candidates, which
 * makes PUCT reduce to UCB since all priors are equal.  Used as the
 * {@code MCTS_UNIFORM} baseline.
 */
public final class UniformActionPrior implements IActionPrior {

    public static final UniformActionPrior INSTANCE = new UniformActionPrior();

    @Override
    public double[] computePrior(List<BbAction> candidates, Game game) {
        int n = candidates.size();
        if (n == 0) return new double[0];
        double uniform = 1.0 / n;
        double[] probs = new double[n];
        for (int i = 0; i < n; i++) probs[i] = uniform;
        return probs;
    }
}
