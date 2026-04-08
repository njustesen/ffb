package com.fumbbl.ffb.ai;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.property.NamedProperties;
import com.fumbbl.ffb.model.skill.Skill;
import com.fumbbl.ffb.util.UtilPlayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Shared move-decision logic used by both the live AI client and the headless simulation.
 *
 * <p>All scoring and selection is done here via {@link PathProbabilityFinder} and
 * {@link PolicySampler}, so behavior is identical regardless of where the agent runs.
 *
 * <p>All methods are stateless. Callers are responsible for translating the returned
 * decisions into the appropriate network commands and, in the headless simulation,
 * applying any coordinate transforms for the away team.
 */
public final class MoveDecisionEngine {

    /** Softmax temperatures — must match AiDecisionEngine constants. */
    public static final double T_PLAYER  = 0.50;
    public static final double T_MOVE    = 0.60;
    public static final double T_KICKOFF = 1.20;

    private MoveDecisionEngine() {}

    // ── Result types ──────────────────────────────────────────────────────────

    /** Result of {@link #selectPlayer}. player==null means end turn. */
    public static final class PlayerSelection {
        public final Player<?> player;
        public final PlayerAction action;
        /** Raw softmax-input scores (parallel with candidatePlayers/candidateActions). */
        public final double[] rawScores;
        public final List<Player<?>> candidatePlayers;
        public final List<PlayerAction> candidateActions;

        PlayerSelection(Player<?> player, PlayerAction action,
                double[] rawScores, List<Player<?>> candidatePlayers,
                List<PlayerAction> candidateActions) {
            this.player = player;
            this.action = action;
            this.rawScores = rawScores;
            this.candidatePlayers = candidatePlayers;
            this.candidateActions = candidateActions;
        }
    }

    /** Result of {@link #selectMoveTarget}. chosen==null means end activation. */
    public static final class MoveResult {
        /** The chosen PathEntry, or null to end activation. */
        public final PathProbabilityFinder.PathEntry chosen;
        /** All reachable coordinates (for visualization). */
        public final List<FieldCoordinate> candidates;
        /** Raw softmax-input scores, parallel with candidates; last entry is "end" if hasEndOption. */
        public final double[] rawScores;
        public final boolean hasEndOption;
        public final boolean isBallCarrier;
        public final boolean isBallRetriever;
        public final boolean isReceiver;
        public final FieldCoordinate playerCoord;

        MoveResult(PathProbabilityFinder.PathEntry chosen,
                List<FieldCoordinate> candidates, double[] rawScores,
                boolean hasEndOption, boolean isBallCarrier, boolean isBallRetriever,
                boolean isReceiver, FieldCoordinate playerCoord) {
            this.chosen = chosen;
            this.candidates = candidates;
            this.rawScores = rawScores;
            this.hasEndOption = hasEndOption;
            this.isBallCarrier = isBallCarrier;
            this.isBallRetriever = isBallRetriever;
            this.isReceiver = isReceiver;
            this.playerCoord = playerCoord;
        }

        public boolean isEndAction() { return chosen == null; }

        /** Role string for visualization labels. */
        public String role() {
            return isBallCarrier ? "ball carrier"
                : isBallRetriever ? "retriever"
                : isReceiver ? "receiver"
                : "support";
        }
    }

    // ── Player selection ──────────────────────────────────────────────────────

    /**
     * Scores and selects which player to activate and with which action.
     *
     * @param myTeam       the team whose turn it is
     * @param opponentTeam the opposing team
     * @param isHome       true if myTeam attacks toward x=25
     * @param allowBlock   true to include BLOCK candidates; pass false for the away team
     *                     in the headless simulation to avoid server-side deselect loops
     * @param argmax       true to always pick the highest-scored option; false to sample
     */
    public static PlayerSelection selectPlayer(Game game, Team myTeam, Team opponentTeam,
            boolean isHome, boolean allowBlock, Random rng, boolean argmax) {

        FieldModel fieldModel = game.getFieldModel();
        FieldCoordinate ballCoord = fieldModel.getBallCoordinate();
        Player<?> ballCarrierPlayer = (ballCoord != null) ? fieldModel.getPlayer(ballCoord) : null;
        boolean opponentHasBall = (ballCarrierPlayer != null) && opponentTeam.hasPlayer(ballCarrierPlayer);
        boolean ballIsLoose = (ballCoord != null) && (ballCarrierPlayer == null);

        int remainingActivations = 0;
        for (Player<?> p : myTeam.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(p);
            if (ps != null && ps.isActive() && ps.getBase() == PlayerState.STANDING) remainingActivations++;
        }

        List<Player<?>> candPlayers = new ArrayList<>();
        List<PlayerAction> candActions = new ArrayList<>();
        List<Double> candScores = new ArrayList<>();

        for (Player<?> p : myTeam.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(p);
            FieldCoordinate pCoord = fieldModel.getPlayerCoordinate(p);
            if (ps == null || pCoord == null || !ps.isActive()) continue;
            boolean standing = ps.getBase() == PlayerState.STANDING;
            boolean prone = ps.getBase() == PlayerState.PRONE;
            if (!standing && !prone) continue;

            // Prone players: activate to stand up (costs 3 MA)
            if (prone) {
                boolean proneHasBall = pCoord.equals(ballCoord);
                boolean proneIsRetriever = !proneHasBall && ballIsLoose && ballCoord != null
                    && chebyshev(pCoord, ballCoord) <= p.getMovementWithModifiers();
                double rawScore = bestActivationSoftmaxScore(
                    p, PlayerAction.MOVE, game, ballCoord, pCoord,
                    proneHasBall, proneIsRetriever, false, myTeam, opponentTeam, isHome);
                candPlayers.add(p);
                candActions.add(PlayerAction.MOVE);
                candScores.add(rawScore * 0.75);
                continue;
            }

            // 1. Ball carrier → MOVE
            if (pCoord.equals(ballCoord)) {
                candPlayers.add(p);
                candActions.add(PlayerAction.MOVE);
                candScores.add(bestActivationSoftmaxScore(
                    p, PlayerAction.MOVE, game, ballCoord, pCoord, true, false, false,
                    myTeam, opponentTeam, isHome));
                continue;
            }

            if (opponentHasBall) {
                FieldCoordinate bcPos = fieldModel.getPlayerCoordinate(ballCarrierPlayer);
                if (bcPos != null) {
                    int dist = chebyshev(pCoord, bcPos);
                    // 2. Blitz ball carrier
                    if (allowBlock && dist > 1 && dist <= p.getMovementWithModifiers() + 1) {
                        double blockProb = computeBlockProbability(p, ballCarrierPlayer, game, myTeam, opponentTeam);
                        candPlayers.add(p);
                        candActions.add(PlayerAction.BLITZ);
                        candScores.add(new ActionScore(blockProb, +0.80, 0.75).softmaxScore());
                    }
                    // 3. Block ball carrier (adjacent)
                    if (allowBlock && dist == 1 && ps.hasTacklezones()) {
                        double blockProb = computeBlockProbability(p, ballCarrierPlayer, game, myTeam, opponentTeam);
                        candPlayers.add(p);
                        candActions.add(PlayerAction.BLOCK);
                        candScores.add(new ActionScore(blockProb, +0.75, 0.70).softmaxScore());
                    }
                }
            }

            // 4. Nearest mover to loose ball
            if (ballIsLoose && ballCoord != null) {
                int dist = chebyshev(pCoord, ballCoord);
                if (dist <= p.getMovementWithModifiers()) {
                    candPlayers.add(p);
                    candActions.add(PlayerAction.MOVE);
                    candScores.add(bestActivationSoftmaxScore(
                        p, PlayerAction.MOVE, game, ballCoord, pCoord, false, true, false,
                        myTeam, opponentTeam, isHome));
                    continue;
                }
            }

            // 5. Block adjacent opponent
            if (allowBlock && ps.hasTacklezones()) {
                Player<?>[] blockTargets = UtilPlayer.findAdjacentBlockablePlayers(game, opponentTeam, pCoord);
                if (blockTargets != null && blockTargets.length > 0) {
                    double bestBlockProb = 0.0;
                    for (Player<?> target : blockTargets) {
                        double bp = computeBlockProbability(p, target, game, myTeam, opponentTeam);
                        if (bp > bestBlockProb) bestBlockProb = bp;
                    }
                    candPlayers.add(p);
                    candActions.add(PlayerAction.BLOCK);
                    candScores.add(new ActionScore(bestBlockProb, +0.50, 0.65).softmaxScore());
                }
            }

            // 6. Receiver (Catch skill) → MOVE
            if (p.hasSkillProperty(NamedProperties.canAttemptCatchInAdjacentSquares)) {
                candPlayers.add(p);
                candActions.add(PlayerAction.MOVE);
                candScores.add(bestActivationSoftmaxScore(
                    p, PlayerAction.MOVE, game, ballCoord, pCoord, false, false, true,
                    myTeam, opponentTeam, isHome));
                continue;
            }

            // 7. Support → MOVE
            candPlayers.add(p);
            candActions.add(PlayerAction.MOVE);
            candScores.add(bestActivationSoftmaxScore(
                p, PlayerAction.MOVE, game, ballCoord, pCoord, false, false, false,
                myTeam, opponentTeam, isHome));

            // 8. FOUL: adjacent prone opponent (only once per turn)
            boolean foulAvailable = allowBlock
                && (!game.getTurnData().isFoulUsed()
                    || java.util.Arrays.stream(myTeam.getPlayers()).anyMatch(
                        pl -> pl.hasSkillProperty(NamedProperties.allowsAdditionalFoul)));
            if (foulAvailable && ps.hasTacklezones()) {
                Player<?>[] proneTargets = UtilPlayer.findAdjacentPronePlayers(game, opponentTeam, pCoord);
                if (proneTargets != null && proneTargets.length > 0) {
                    ActionScore foulScore = computeFoulActionScore(p, proneTargets[0], game, remainingActivations);
                    candPlayers.add(p);
                    candActions.add(PlayerAction.FOUL);
                    candScores.add(foulScore.softmaxScore());
                }
            }
        }

        // 9. End turn
        candPlayers.add(null);
        candActions.add(null);
        double endTurnValue = -1.0 + 1.0 / Math.max(1, remainingActivations);
        candScores.add(new ActionScore(1.0, endTurnValue, 0.30).softmaxScore());

        double[] scores = candScores.stream().mapToDouble(Double::doubleValue).toArray();
        int chosen = argmax ? PolicySampler.argmax(scores) : PolicySampler.sample(scores, T_PLAYER, rng);

        return new PlayerSelection(
            candPlayers.get(chosen),
            candActions.get(chosen),
            scores,
            candPlayers,
            candActions);
    }

    // ── Move target selection ─────────────────────────────────────────────────

    /**
     * Scores all reachable squares via {@link PathProbabilityFinder} and selects
     * a destination for the given acting player.
     *
     * @param isHome true if myTeam attacks toward x=25
     * @param argmax true to always pick the highest-scored square; false to sample
     * @return result; result.chosen==null means end activation
     */
    public static MoveResult selectMoveTarget(Game game, ActingPlayer actingPlayer,
            Team myTeam, Team opponentTeam, boolean isHome, Random rng, boolean argmax) {

        Map<FieldCoordinate, PathProbabilityFinder.PathEntry> pathMap =
            PathProbabilityFinder.findAllPaths(game, actingPlayer);

        Player<?> player = actingPlayer.getPlayer();
        FieldCoordinate playerCoord = game.getFieldModel().getPlayerCoordinate(player);
        FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();

        boolean isBallCarrier = (playerCoord != null && playerCoord.equals(ballCoord));
        boolean isReceiver = !isBallCarrier
            && player.hasSkillProperty(NamedProperties.canAttemptCatchInAdjacentSquares);
        boolean isBallRetriever = !isBallCarrier && !isReceiver && pathMap.containsKey(ballCoord);

        // BLITZ with no moves taken: the player must move (can't just end activation
        // without doing anything), so suppress the END option until they've moved once.
        boolean canEndNow = actingPlayer.getCurrentMove() > 0
            || (actingPlayer.getPlayerAction() != PlayerAction.MOVE
                && actingPlayer.getPlayerAction() != PlayerAction.BLITZ);

        List<FieldCoordinate> candidates = new ArrayList<>(pathMap.keySet());
        int n = candidates.size();
        int total = canEndNow ? n + 1 : n;
        double[] scores = new double[total];
        PathProbabilityFinder.PathEntry[] entries = new PathProbabilityFinder.PathEntry[total];

        double failureCost = roleDiceFailureCost(isBallCarrier, isBallRetriever);

        for (int i = 0; i < n; i++) {
            FieldCoordinate coord = candidates.get(i);
            PathProbabilityFinder.PathEntry entry = pathMap.get(coord);
            ActionScore base = moveBaseActionScore(coord, player, actingPlayer.getCurrentMove(),
                isBallCarrier, isBallRetriever, isReceiver, ballCoord, playerCoord, game,
                myTeam, opponentTeam, isHome);
            double effective = entry.probability * base.value * base.confidence
                - (1.0 - entry.probability) * failureCost;
            scores[i] = 1.0 + effective;
            entries[i] = entry;
        }

        if (canEndNow) {
            entries[n] = null; // sentinel: end action
            if (playerCoord != null && actingPlayer.getCurrentMove() > 0) {
                ActionScore endBase = moveBaseActionScore(playerCoord, player, actingPlayer.getCurrentMove(),
                    isBallCarrier, isBallRetriever, isReceiver, ballCoord, playerCoord, game,
                    myTeam, opponentTeam, isHome);
                scores[n] = 1.0 + endBase.value * endBase.confidence * 0.9;
            } else {
                scores[n] = 1.0 + (-0.05 * 0.50);
            }
        }

        if (total == 0) {
            return new MoveResult(null, candidates, new double[0], false,
                isBallCarrier, isBallRetriever, isReceiver, playerCoord);
        }

        int chosen = argmax ? PolicySampler.argmax(scores) : PolicySampler.sample(scores, T_MOVE, rng);
        return new MoveResult(entries[chosen], candidates, scores, canEndNow,
            isBallCarrier, isBallRetriever, isReceiver, playerCoord);
    }

    // ── Scoring helpers (package-accessible for AiDecisionEngine) ─────────────

    static double bestActivationSoftmaxScore(Player<?> player, PlayerAction action, Game game,
            FieldCoordinate ballCoord, FieldCoordinate playerCoord,
            boolean isBallCarrier, boolean isBallRetriever, boolean isReceiver,
            Team myTeam, Team opponentTeam, boolean isHome) {
        Map<FieldCoordinate, PathProbabilityFinder.PathEntry> pathMap =
            PathProbabilityFinder.findAllPaths(game, player, action);
        if (pathMap.isEmpty()) return 0.0; // trapped — no useful activation score
        double failureCost = roleDiceFailureCost(isBallCarrier, isBallRetriever);
        double bestEffective = 0.0;
        for (Map.Entry<FieldCoordinate, PathProbabilityFinder.PathEntry> e : pathMap.entrySet()) {
            ActionScore base = moveBaseActionScore(
                e.getKey(), player, 0,
                isBallCarrier, isBallRetriever, isReceiver,
                ballCoord, playerCoord, game, myTeam, opponentTeam, isHome);
            double prob = e.getValue().probability;
            double effective = prob * base.value * base.confidence - (1.0 - prob) * failureCost;
            if (effective > bestEffective) bestEffective = effective;
        }
        return 1.0 + bestEffective;
    }

    static ActionScore moveBaseActionScore(FieldCoordinate coord, Player<?> player, int currentMove,
            boolean isBallCarrier, boolean isBallRetriever, boolean isReceiver,
            FieldCoordinate ballCoord, FieldCoordinate playerCoord, Game game,
            Team myTeam, Team opponentTeam, boolean isHome) {

        if (isBallCarrier) {
            if (endzoneDistance(coord, isHome) == 0) {
                return new ActionScore(1.0, +1.0, 1.0);
            }
            double advance = advanceScore(coord, isHome);
            boolean advancing = playerCoord == null
                || advanceScore(coord, isHome) > advanceScore(playerCoord, isHome);
            return advancing
                ? new ActionScore(1.0, advance * 0.6, 0.50)
                : new ActionScore(1.0, advance * 0.3 - 0.1, 0.30);
        }

        if (isBallRetriever) {
            if (ballCoord == null) return new ActionScore(1.0, 0.0, 0.30);
            int dist = chebyshev(coord, ballCoord);
            if (dist == 0) return new ActionScore(1.0, +1.0, 0.95);
            if (dist == 1) return new ActionScore(1.0, +0.80, 0.80);
            return new ActionScore(1.0, Math.max(0.05, 0.7 - 0.1 * dist), 0.55);
        }

        if (isReceiver) {
            int movesRemaining = player.getMovementWithModifiers() - currentMove;
            int edzDist = endzoneDistance(coord, isHome);
            if (edzDist <= movesRemaining) {
                return new ActionScore(1.0, +0.70, 0.75);
            }
            return new ActionScore(1.0, advanceScore(coord, isHome) * 0.5, 0.50);
        }

        // Support player
        boolean adjFriendly = nearFriendlyBallCarrier(coord, ballCoord, game, myTeam);
        boolean adjOpponent = nearOpponent(coord, game, opponentTeam);
        if (adjFriendly) return new ActionScore(1.0, +0.50, 0.60);
        if (adjOpponent) return new ActionScore(1.0, +0.30, 0.50);
        int currentEdz = (playerCoord != null) ? endzoneDistance(playerCoord, isHome) : 25;
        boolean advancing = endzoneDistance(coord, isHome) <= currentEdz;
        return new ActionScore(1.0, advancing ? +0.10 : 0.0, 0.30);
    }

    // ── Public helpers ────────────────────────────────────────────────────────

    public static double computeBlockProbability(Player<?> attacker, Player<?> defender,
            Game game, Team myTeam, Team opponentTeam) {
        FieldCoordinate atkCoord = game.getFieldModel().getPlayerCoordinate(attacker);
        FieldCoordinate defCoord = game.getFieldModel().getPlayerCoordinate(defender);
        if (atkCoord == null || defCoord == null) return 0.33;

        int atkStr = attacker.getStrengthWithModifiers(game);
        int defStr = defender.getStrengthWithModifiers(game);

        boolean noAssists = attacker.hasSkillProperty(NamedProperties.ignoreBlockAssists)
                         || defender.hasSkillProperty(NamedProperties.ignoreBlockAssists);
        int offAssists = 0, defAssists = 0;

        if (!noAssists) {
            for (Player<?> p : UtilPlayer.findAdjacentPlayersWithTacklezones(
                    game, myTeam, defCoord, false)) {
                if (p.equals(attacker)) continue;
                if (effectiveGuard(p, game, opponentTeam)) offAssists++;
                else if (UtilPlayer.findTacklezones(game, p) == 0) offAssists++;
            }
            for (Player<?> p : UtilPlayer.findAdjacentPlayersWithTacklezones(
                    game, opponentTeam, atkCoord, false)) {
                if (p.equals(defender)) continue;
                if (effectiveGuard(p, game, myTeam)) defAssists++;
                else if (UtilPlayer.findTacklezones(game, p) == 0) defAssists++;
            }
        }

        int effAtkStr = Math.max(1, atkStr + offAssists);
        int effDefStr = Math.max(1, defStr + defAssists);

        if      (effAtkStr >= 2 * effDefStr) return 0.70;
        else if (effAtkStr >      effDefStr) return 0.56;
        else if (effAtkStr ==     effDefStr) return 0.33;
        else if (effAtkStr * 2 >= effDefStr) return 0.11;
        else                                 return 0.04;
    }

    public static double roleDiceFailureCost(boolean isBallCarrier, boolean isBallRetriever) {
        if (isBallCarrier)   return 0.50;
        if (isBallRetriever) return 0.40;
        return 0.25;
    }

    public static int chebyshev(FieldCoordinate a, FieldCoordinate b) {
        return Math.max(Math.abs(a.getX() - b.getX()), Math.abs(a.getY() - b.getY()));
    }

    /** Normalized advance score for myTeam. 1.0 = opponent endzone. */
    public static double advanceScore(FieldCoordinate sq, boolean isHome) {
        return isHome ? (double) sq.getX() / 25.0 : (25.0 - sq.getX()) / 25.0;
    }

    /** Distance to opponent endzone (0 = at endzone). */
    public static int endzoneDistance(FieldCoordinate sq, boolean isHome) {
        return isHome ? 25 - sq.getX() : sq.getX();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static ActionScore computeFoulActionScore(Player<?> fouler, Player<?> target,
            Game game, int remainingActivations) {
        boolean hasDirtyPlayer = fouler.hasSkillProperty(NamedProperties.affectsEitherArmourOrInjuryOnFoul);
        boolean hasChainsaw    = fouler.hasSkillProperty(NamedProperties.foulBreaksArmourWithoutRoll);
        int assists = UtilPlayer.findFoulAssists(game, fouler, target);
        int targetArmour = target.getArmourWithModifiers();
        int turnNr = game.getTurnDataHome().getTurnNr();

        double prob = 0.15 + assists * 0.12
            + (hasDirtyPlayer ? 0.25 : 0.0)
            + (hasChainsaw    ? 0.55 : 0.0);
        prob = Math.min(prob, 0.85);

        double value = Math.max(0.10, (targetArmour - 6) * 0.12 + (turnNr >= 7 ? 0.15 : 0.0));
        if (remainingActivations <= 2) value = Math.min(value * 1.4, 0.55);
        value = Math.min(value, 0.55);

        return new ActionScore(prob, value, 0.35);
    }

    private static boolean nearFriendlyBallCarrier(FieldCoordinate sq, FieldCoordinate ballCoord,
            Game game, Team myTeam) {
        if (ballCoord == null) return false;
        Player<?> carrier = game.getFieldModel().getPlayer(ballCoord);
        if (carrier == null || !myTeam.hasPlayer(carrier)) return false;
        FieldCoordinate carrierPos = game.getFieldModel().getPlayerCoordinate(carrier);
        return carrierPos != null && chebyshev(sq, carrierPos) <= 1;
    }

    private static boolean nearOpponent(FieldCoordinate sq, Game game, Team opponentTeam) {
        for (Player<?> opp : opponentTeam.getPlayers()) {
            FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(opp);
            if (pos != null && chebyshev(sq, pos) <= 1) return true;
        }
        return false;
    }

    private static boolean effectiveGuard(Player<?> player, Game game, Team opposingTeam) {
        if (!player.hasSkillProperty(NamedProperties.assistsBlocksInTacklezones)) return false;
        FieldCoordinate pCoord = game.getFieldModel().getPlayerCoordinate(player);
        if (pCoord == null) return false;
        for (Player<?> opp : UtilPlayer.findAdjacentPlayersWithTacklezones(
                game, opposingTeam, pCoord, false)) {
            for (Skill s : opp.getSkills()) {
                if (s.canCancel(NamedProperties.assistsBlocksInTacklezones)) return false;
            }
        }
        return true;
    }
}
