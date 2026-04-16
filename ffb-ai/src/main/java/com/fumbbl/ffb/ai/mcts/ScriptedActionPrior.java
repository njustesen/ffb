package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.PolicySampler;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;

import java.util.List;
import java.util.Random;

/**
 * Action prior derived from the scripted agent's player-selection scores.
 *
 * <p>Uses {@link MoveDecisionEngine#selectPlayer} to score each candidate,
 * then converts the raw scores to probabilities via softmax (T=0.5).
 * This provides the same distribution the scripted agent uses for its own
 * decisions, making it a natural prior for PUCT.
 */
public final class ScriptedActionPrior implements IActionPrior {

    /** Softmax temperature — same as scripted policy player-selection temperature. */
    private static final double T = 0.5;

    private static final Random DUMMY_RNG = new Random(0);

    @Override
    public double[] computePrior(List<BbAction> candidates, Game game) {
        if (candidates.isEmpty()) return new double[0];

        boolean home = game.isHomePlaying();
        Team myTeam  = home ? game.getTeamHome() : game.getTeamAway();
        Team oppTeam = home ? game.getTeamAway() : game.getTeamHome();

        // Get the raw scores from the scripted agent (same call MCTS makes for candidate
        // enumeration — we just re-use the rawScores array from it).
        MoveDecisionEngine.PlayerSelection sel =
            MoveDecisionEngine.selectPlayer(game, myTeam, oppTeam, home, home, DUMMY_RNG, false);

        if (sel.rawScores == null || sel.rawScores.length == 0) {
            return null; // fall back to UCB
        }

        // Map each BbAction candidate to its score in the PlayerSelection.
        // candidatePlayers/candidateActions are parallel with rawScores.
        double[] scores = new double[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            BbAction action = candidates.get(i);
            scores[i] = findScore(sel, action.player, action.action, sel.rawScores);
        }

        return PolicySampler.softmax(scores, T);
    }

    /**
     * Fast path: softmax directly on pre-computed scores — no second Dijkstra pass.
     */
    @Override
    public double[] computePriorFromScores(List<BbAction> candidates, double[] rawScores, Game game) {
        if (rawScores == null || rawScores.length != candidates.size()) {
            return computePrior(candidates, game);
        }
        return PolicySampler.softmax(rawScores, T);
    }

    private double findScore(MoveDecisionEngine.PlayerSelection sel,
                             Player<?> player,
                             com.fumbbl.ffb.PlayerAction playerAction,
                             double[] rawScores) {
        List<Player<?>> players = sel.candidatePlayers;
        List<com.fumbbl.ffb.PlayerAction> actions = sel.candidateActions;
        for (int i = 0; i < players.size(); i++) {
            if (players.get(i) == player && actions.get(i) == playerAction) {
                return rawScores[i];
            }
        }
        // Candidate not found — use a small baseline score.
        return 0.5;
    }
}
