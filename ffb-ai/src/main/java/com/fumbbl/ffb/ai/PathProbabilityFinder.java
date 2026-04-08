package com.fumbbl.ffb.ai;

import com.fumbbl.ffb.FactoryType.Factory;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.FieldCoordinateBounds;
import com.fumbbl.ffb.factory.DodgeModifierFactory;
import com.fumbbl.ffb.factory.common.GoForItModifierFactory;
import com.fumbbl.ffb.mechanics.AgilityMechanic;
import com.fumbbl.ffb.mechanics.Mechanic;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.property.NamedProperties;
import com.fumbbl.ffb.modifiers.DodgeContext;
import com.fumbbl.ffb.modifiers.DodgeModifier;
import com.fumbbl.ffb.modifiers.GoForItContext;
import com.fumbbl.ffb.modifiers.GoForItModifier;
import com.fumbbl.ffb.util.UtilPlayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Computes the maximum-probability path to every square reachable by the acting player
 * in a single Dijkstra pass.
 *
 * Each expansion step uses the actual game factories (DodgeModifierFactory, AgilityMechanic,
 * GoForItModifierFactory) with the explicit intermediate coordinates — no game-state mutation
 * is needed because those factories accept arbitrary fromCoord/toCoord in their context objects.
 */
public class PathProbabilityFinder {

    private static final int MAX_GFI = 2;

    private static final int[][] DIRS = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},           { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };

    // ── Public result type ─────────────────────────────────────────────────────

    public static class PathEntry {
        /** Path from the player's start (exclusive) to this square (inclusive). */
        public final FieldCoordinate[] path;
        /** Cumulative probability of all dodge/GFI rolls along the path. */
        public final double probability;

        PathEntry(FieldCoordinate[] path, double probability) {
            this.path = path;
            this.probability = probability;
        }
    }

    // ── Internal Dijkstra node ─────────────────────────────────────────────────

    private static class PathNode implements Comparable<PathNode> {
        final FieldCoordinate coord;
        final double cumProb;
        final int stepCount;
        final PathNode parent;

        PathNode(FieldCoordinate coord, double cumProb, int stepCount, PathNode parent) {
            this.coord = coord;
            this.cumProb = cumProb;
            this.stepCount = stepCount;
            this.parent = parent;
        }

        @Override
        public int compareTo(PathNode other) {
            return Double.compare(other.cumProb, this.cumProb); // max-heap
        }
    }

    // ── Main entry point ───────────────────────────────────────────────────────

    /**
     * Returns the maximum-probability (path, cumulative-probability) for every square
     * reachable by the acting player from their current position.
     * The result does NOT include the player's starting square.
     */
    public static Map<FieldCoordinate, PathEntry> findAllPaths(Game game, ActingPlayer actingPlayer) {
        Player<?> player = actingPlayer.getPlayer();
        FieldCoordinate startCoord = game.getFieldModel().getPlayerCoordinate(player);
        if (startCoord == null) return Collections.emptyMap();

        Team opposingTeam = UtilPlayer.findOtherTeam(game, player);
        int movementAllowance = player.getMovementWithModifiers();
        int currentMove = actingPlayer.getCurrentMove();
        int maxSteps = movementAllowance - currentMove + MAX_GFI;

        DodgeModifierFactory dodgeFactory = (DodgeModifierFactory) game.getFactory(Factory.DODGE_MODIFIER);
        GoForItModifierFactory gfiFactory = (GoForItModifierFactory) game.getFactory(Factory.GO_FOR_IT_MODIFIER);
        AgilityMechanic agilityMechanic = (AgilityMechanic) game.getFactory(Factory.MECHANIC).forName(Mechanic.Type.AGILITY.name());

        boolean ignoresTZ = player.hasSkillProperty(NamedProperties.ignoreTacklezonesWhenMoving);

        // Precompute GFI probability — same for all steps (only skill modifiers vary, not position)
        Set<GoForItModifier> gfiMods = gfiFactory.findModifiers(
            new GoForItContext(game, player, Collections.emptySet()));
        int gfiModTotal = 0;
        for (GoForItModifier m : gfiMods) gfiModTotal += m.getModifier();
        int gfiMinRoll = Math.max(2, 2 + gfiModTotal);
        double gfiProb = Math.max(1.0 / 6.0, (7.0 - gfiMinRoll) / 6.0);

        Map<FieldCoordinate, Double> bestProb = new HashMap<>();
        Map<FieldCoordinate, PathNode> bestNode = new HashMap<>();
        PriorityQueue<PathNode> queue = new PriorityQueue<>();

        PathNode seed = new PathNode(startCoord, 1.0, 0, null);
        queue.offer(seed);
        bestProb.put(startCoord, 1.0);
        bestNode.put(startCoord, seed);

        while (!queue.isEmpty()) {
            PathNode current = queue.poll();

            // Skip stale entries (a better path to current.coord was already processed)
            Double recorded = bestProb.get(current.coord);
            if (recorded == null || current.cumProb < recorded - 1e-12) continue;

            // Don't expand beyond the maximum reachable steps
            if (current.stepCount >= maxSteps) continue;

            // Dodge check: does leaving current.coord require a dodge?
            boolean needsDodge = !ignoresTZ
                && UtilPlayer.findAdjacentPlayersWithTacklezones(game, opposingTeam, current.coord, false).length > 0;

            for (int[] dir : DIRS) {
                FieldCoordinate toCoord = current.coord.add(dir[0], dir[1]);
                if (!FieldCoordinateBounds.FIELD.isInBounds(toCoord)) continue;
                if (game.getFieldModel().getPlayer(toCoord) != null) continue;

                int newStepCount = current.stepCount + 1;
                int totalMoves = currentMove + newStepCount;

                // Compute step probability
                double stepProb = 1.0;

                if (needsDodge) {
                    // DodgeModifierFactory uses the explicit fromCoord/toCoord in DodgeContext —
                    // no need to move the player in the game model.
                    Set<DodgeModifier> dodgeMods = dodgeFactory.findModifiers(
                        new DodgeContext(game, actingPlayer, current.coord, toCoord));
                    int dodgeMinRoll = agilityMechanic.minimumRollDodge(game, player, dodgeMods);
                    stepProb *= Math.max(1.0 / 6.0, (7.0 - dodgeMinRoll) / 6.0);
                }

                if (totalMoves > movementAllowance) {
                    stepProb *= gfiProb;
                }

                double newCumProb = current.cumProb * stepProb;

                Double prevBest = bestProb.get(toCoord);
                if (prevBest == null || newCumProb > prevBest + 1e-12) {
                    bestProb.put(toCoord, newCumProb);
                    PathNode newNode = new PathNode(toCoord, newCumProb, newStepCount, current);
                    bestNode.put(toCoord, newNode);
                    queue.offer(newNode);
                }
            }
        }

        // Build result map (exclude the starting square)
        Map<FieldCoordinate, PathEntry> result = new HashMap<>();
        for (Map.Entry<FieldCoordinate, PathNode> e : bestNode.entrySet()) {
            FieldCoordinate coord = e.getKey();
            if (coord.equals(startCoord)) continue;
            PathNode node = e.getValue();
            result.put(coord, new PathEntry(reconstructPath(node), node.cumProb));
        }
        return result;
    }

    /**
     * Convenience overload for evaluating a player before they are activated.
     * Creates a mock ActingPlayer with currentMove=0 and the given action.
     */
    public static Map<FieldCoordinate, PathEntry> findAllPaths(
            Game game, Player<?> player, PlayerAction action) {
        // Use an anonymous subclass to return the player directly, bypassing the
        // game.getPlayerById(id) lookup which may not be populated client-side.
        ActingPlayer mock = new ActingPlayer(game) {
            @Override public Player<?> getPlayer() { return player; }
        };
        mock.setPlayerId(player.getId());
        mock.setPlayerAction(action);
        return findAllPaths(game, mock);
    }

    // ── Path reconstruction ────────────────────────────────────────────────────

    private static FieldCoordinate[] reconstructPath(PathNode node) {
        List<FieldCoordinate> path = new ArrayList<>();
        PathNode current = node;
        while (current.parent != null) {
            path.add(current.coord);
            current = current.parent;
        }
        Collections.reverse(path);
        return path.toArray(new FieldCoordinate[0]);
    }
}
