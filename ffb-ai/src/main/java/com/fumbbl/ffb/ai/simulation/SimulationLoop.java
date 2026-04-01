package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.FieldCoordinateBounds;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.ai.strategy.RandomStrategy;
import com.fumbbl.ffb.dialog.DialogPlayerChoiceParameter;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.util.UtilBox;
import com.fumbbl.ffb.net.commands.ClientCommand;
import com.fumbbl.ffb.net.commands.ClientCommandActingPlayer;
import com.fumbbl.ffb.net.commands.ClientCommandEndTurn;
import com.fumbbl.ffb.net.commands.ClientCommandKickoff;
import com.fumbbl.ffb.net.commands.ClientCommandStartGame;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.net.ReceivedCommand;
import com.fumbbl.ffb.server.step.IStep;
import com.fumbbl.ffb.server.step.StepId;
import com.fumbbl.ffb.server.util.UtilServerSetup;

import org.eclipse.jetty.websocket.api.Session;

import java.util.Random;

/**
 * Drives a {@link GameState} to completion by injecting commands directly into
 * the server-side step stack — no network, no threads, no polling delays.
 *
 * <h3>Decision strategy</h3>
 * <ol>
 *   <li>If {@code game.getDialogParameter()} is set, respond via
 *       {@link RandomStrategy#respondToDialog} (most dialogs) or clear display-only
 *       dialogs that the server never waits for.</li>
 *   <li>Otherwise dispatch based on the current {@link StepId}:
 *       SETUP applies the pre-loaded {@link TeamSetup}; KICKOFF sends a random
 *       ball coordinate; INIT_SELECTING always ends the turn (maximises speed).</li>
 * </ol>
 *
 * <h3>Thread safety</h3>
 * Not thread-safe.  Each instance should be used from a single thread.
 */
public class SimulationLoop {

    /** Hard limit to prevent infinite loops if the game gets stuck. */
    private static final int MAX_ITERATIONS = 100_000;

    private final TeamSetup homeSetup;
    private final TeamSetup awaySetup;
    private final CapturingClientCommunication comm = new CapturingClientCommunication();
    private final Random random = new Random();

    public SimulationLoop(TeamSetup homeSetup, TeamSetup awaySetup) {
        this.homeSetup = homeSetup;
        this.awaySetup = awaySetup;
    }

    /** Holds timing and step-profile data for one completed game. */
    public static class GameResult {
        public final long kickoffNs;
        public final long driveNs;
        public final long turns;
        /** Accumulated wall-clock nanoseconds per step+dialog key. */
        public final java.util.Map<String, Long> stepTimeNs;
        /** Hit count per step+dialog key. */
        public final java.util.Map<String, Integer> stepCounts;
        /**
         * Wall-clock nanoseconds for each drive (kickoff end → next kickoff or game end).
         * One entry per drive; the last entry covers the final drive to game end.
         */
        public final java.util.List<Long> perDriveNs;
        /**
         * Wall-clock nanoseconds for each team turn (one INIT_SELECTING to the next,
         * covering the full END_TURN + inducement cleanup sequence in between).
         */
        public final java.util.List<Long> perTurnNs;
        /**
         * Wall-clock nanoseconds for each INIT_SELECTING dispatch only — the cost of
         * the step that begins a player turn. In this simulation the turn is ended
         * immediately at this point, so this is also the per-player-turn cost.
         */
        public final java.util.List<Long> perPlayerTurnNs;

        GameResult(long kickoffNs, long driveNs, long turns,
                   java.util.Map<String, Long> stepTimeNs,
                   java.util.Map<String, Integer> stepCounts,
                   java.util.List<Long> perDriveNs,
                   java.util.List<Long> perTurnNs,
                   java.util.List<Long> perPlayerTurnNs) {
            this.kickoffNs = kickoffNs;
            this.driveNs = driveNs;
            this.turns = turns;
            this.stepTimeNs = stepTimeNs;
            this.stepCounts = stepCounts;
            this.perDriveNs = perDriveNs;
            this.perTurnNs = perTurnNs;
            this.perPlayerTurnNs = perPlayerTurnNs;
        }
    }

    /**
     * Run a game to completion (or until {@link #MAX_ITERATIONS} is reached).
     *
     * @param gameState a freshly created state with StartGame sequence pushed
     * @return elapsed wall-clock time in nanoseconds
     */
    public long runGame(GameState gameState) {
        GameResult r = runGameSplit(gameState);
        return r.kickoffNs + r.driveNs;
    }

    /**
     * Run a game to completion and return a detailed {@link GameResult}.
     *
     * <p>Time is bucketed per iteration by {@link TurnMode}:
     * <ul>
     *   <li>{@link GameResult#kickoffNs} — {@code START_GAME}, {@code SETUP}, {@code KICKOFF}</li>
     *   <li>{@link GameResult#driveNs} — everything else ({@code REGULAR}, {@code BETWEEN_TURNS}, etc.)</li>
     *   <li>{@link GameResult#turns} — one per {@code INIT_SELECTING} step</li>
     *   <li>{@link GameResult#stepTimeNs} — accumulated ns per step+dialog key</li>
     *   <li>{@link GameResult#stepCounts} — hit count per step+dialog key</li>
     * </ul>
     *
     * @param gameState a freshly created state with StartGame sequence pushed
     */
    public GameResult runGameSplit(GameState gameState) {
        long kickoffNs = 0;
        long driveNs = 0;
        long turns = 0;
        Game game = gameState.getGame();

        // Boot the step stack.  INIT_START_GAME requires both home and away coaches
        // to send CLIENT_START_GAME before it advances, so inject it for both teams.
        injectForTeam(gameState, new ClientCommandStartGame(), true);
        injectForTeam(gameState, new ClientCommandStartGame(), false);

        int iterations = 0;
        java.util.Map<String, Integer> stepCounts = new java.util.LinkedHashMap<>();
        java.util.Map<String, Long> stepTimeNs = new java.util.LinkedHashMap<>();
        java.util.List<Long> perDriveNs = new java.util.ArrayList<>();
        java.util.List<Long> perTurnNs = new java.util.ArrayList<>();
        java.util.List<Long> perPlayerTurnNs = new java.util.ArrayList<>();

        // Drive tracking: a drive starts when TurnMode first enters REGULAR after
        // kickoff, and ends when TurnMode next becomes SETUP (new kickoff) or game ends.
        boolean inDrive = false;
        long driveStart = 0;

        // Turn tracking: from one INIT_SELECTING to the next covers the full
        // END_TURN + inducement cleanup sequence.
        long turnStart = -1;

        long iterStart = System.nanoTime();
        while (game.getFinished() == null && ++iterations < MAX_ITERATIONS) {
            IStep currentStep = gameState.getCurrentStep();
            if (currentStep == null) {
                break; // step stack exhausted
            }

            IDialogParameter dialog = game.getDialogParameter();
            StepId stepId = currentStep.getId();
            String key = stepId + (dialog != null ? "+" + dialog.getId() : "");
            stepCounts.merge(key, 1, Integer::sum);

            // Record turn start at the beginning of each INIT_SELECTING iteration.
            if (stepId == StepId.INIT_SELECTING) {
                if (turnStart >= 0) {
                    perTurnNs.add(iterStart - turnStart);
                }
                turnStart = iterStart;
            }

            // INIT_SELECTING never handles player-choice dialogs itself — those are
            // display-only artifacts left by a prior step.  Always drive it via
            // CLIENT_END_TURN so the dialog is cleared by a downstream step.
            if (dialog != null && stepId != StepId.INIT_SELECTING) {
                handleDialog(dialog, game, gameState);
            } else {
                handleStep(stepId, game, gameState);
            }

            long now = System.nanoTime();
            long elapsed = now - iterStart;
            TurnMode tmAfter = game.getTurnMode();
            if (isKickoffPhase(tmAfter)) {
                kickoffNs += elapsed;
            } else {
                driveNs += elapsed;
            }
            stepTimeNs.merge(key, elapsed, Long::sum);

            // Player-turn sample: cost of the INIT_SELECTING dispatch itself.
            if (stepId == StepId.INIT_SELECTING) {
                perPlayerTurnNs.add(elapsed);
                turns++;
            }

            // Drive boundary detection (checked after dispatch so tmAfter is current).
            if (!inDrive && tmAfter == TurnMode.REGULAR) {
                inDrive = true;
                driveStart = now;
            } else if (inDrive && isKickoffPhase(tmAfter)) {
                perDriveNs.add(now - driveStart);
                inDrive = false;
            }

            iterStart = now;
        }

        // Close any open drive / turn at game end.
        long endNow = System.nanoTime();
        if (inDrive) {
            perDriveNs.add(endNow - driveStart);
        }
        if (turnStart >= 0) {
            perTurnNs.add(endNow - turnStart);
        }

        if (iterations >= MAX_ITERATIONS && game.getFinished() == null) {
            IStep stuck = gameState.getCurrentStep();
            System.err.println("[SimulationLoop] Hit MAX_ITERATIONS. Last step: "
                + (stuck != null ? stuck.getId() : "null")
                + ", dialog: " + game.getDialogParameter());
            // Print top-10 most frequent (step+dialog) combinations
            stepCounts.entrySet().stream()
                .sorted(java.util.Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(10)
                .forEach(e -> System.err.println("  " + e.getValue() + "x " + e.getKey()));
        }

        return new GameResult(kickoffNs, driveNs, turns, stepTimeNs, stepCounts,
            perDriveNs, perTurnNs, perPlayerTurnNs);
    }

    private static boolean isKickoffPhase(TurnMode turnMode) {
        return turnMode == TurnMode.START_GAME
            || turnMode == TurnMode.SETUP
            || turnMode == TurnMode.KICKOFF;
    }

    // ── Dialog handling ──────────────────────────────────────────────────────

    private void handleDialog(IDialogParameter dialog, Game game, GameState gameState) {
        switch (dialog.getId()) {
            case KICKOFF_RETURN:
                // The server-side step already returned NEXT_STEP in start();
                // this dialog is display-only.  Just clear it.
                game.setDialogParameter(null);
                break;

            case SETUP_ERROR:
                // Setup was invalid; clear the error and retry — placeReservePlayersIfNeeded
                // will fill any missing reserve players on the next SETUP iteration.
                {
                    com.fumbbl.ffb.dialog.DialogSetupErrorParameter sep = (com.fumbbl.ffb.dialog.DialogSetupErrorParameter) dialog;
                    System.err.println("[SE] " + sep.getTeamId() + ": " + java.util.Arrays.toString(sep.getSetupErrors()));
                }
                game.setDialogParameter(null);
                break;
            case SWARMING_ERROR:
            case INVALID_SOLID_DEFENCE:
                // Error dialogs that don't require a server response.
                // Clear and let the loop retry the waiting step.
                game.setDialogParameter(null);
                break;

            default:
                comm.clearCaptured();
                RandomStrategy.respondToDialog(dialog, game, comm);
                ClientCommand captured = comm.getCapturedCommand();
                if (captured != null) {
                    // Dialogs with an explicit teamId (e.g. MVP nominations, player choice)
                    // must be injected as that team so session-based ownership checks route
                    // the command to the correct home/away nominated list.
                    String dialogTeamId = getDialogTeamId(dialog);
                    if (dialogTeamId != null) {
                        boolean forHome = dialogTeamId.equals(game.getTeamHome().getId());
                        injectForTeam(gameState, captured, forHome);
                    } else {
                        inject(gameState, captured);
                    }
                } else {
                    // RandomStrategy produced nothing (unrecognised dialog) — clear
                    game.setDialogParameter(null);
                }
                break;
        }
    }

    // ── Step handling (no dialog set) ────────────────────────────────────────

    private void handleStep(StepId stepId, Game game, GameState gameState) {
        switch (stepId) {
            case SETUP: {
                // Reset only the current team's available players to reserve so that
                // retries start from a clean slate, without disturbing the other team.
                // (UtilBox.putAllPlayersIntoBox resets both teams — don't use it here.)
                resetCurrentTeamToReserve(game);

                // Apply the pre-loaded field setup for whichever team is currently placing.
                // applyTo() places players at their global field-model coordinates.
                TeamSetup setup = game.isHomePlaying() ? homeSetup : awaySetup;
                setup.applyTo(game);

                // If injuries left some reserve players unplaced (causing validation failures),
                // place them at valid overflow squares via CLIENT_SETUP_PLAYER.
                placeReservePlayersIfNeeded(game, gameState);

                // Pass a null coordinate map so AbstractStep.setPlayerCoordinates() makes
                // no changes — the field model already has correct positions from applyTo().
                // (Passing the field-model coords directly would cause AbstractStep to
                //  double-transform away-team coordinates, moving them to the wrong half.)
                inject(gameState, new ClientCommandEndTurn(TurnMode.SETUP, null));
                break;
            }

            case APPLY_KICKOFF_RESULT: {
                // Some kickoff results (HIGH_KICK, QUICK_SNAP) leave the step waiting
                // for CLIENT_END_TURN with no dialog set.
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;
            }

            case KICKOFF: {
                // Kick to a random square in the opponent half (x=13..24, y=1..13)
                int kx = 13 + random.nextInt(12);
                int ky = 1 + random.nextInt(13);
                inject(gameState, new ClientCommandKickoff(new FieldCoordinate(kx, ky)));
                break;
            }

            case INIT_SELECTING: {
                // Always end the turn immediately — keeps simulation fast and avoids
                // the need to handle per-action sub-steps (MOVE, BLOCK_ROLL, etc.)
                inject(gameState, new ClientCommandEndTurn(game.getTurnMode(), null));
                break;
            }

            default:
                // For any other waiting step: send a null acting-player deselect.
                // This is a safe no-op fallback — if the step doesn't accept it the
                // step just ignores it and keeps waiting (loop will try again next tick).
                inject(gameState, new ClientCommandActingPlayer(null, null, false));
                break;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /**
     * After {@link TeamSetup#applyTo} some players may be missing from the field
     * or the LOS may be under-staffed.  Fixes all constraint violations:
     * <ol>
     *   <li>Total on field must equal {@code min(available, 11)}</li>
     *   <li>Minimum 3 players on the Line of Scrimmage (x=12 home, y=4..10)</li>
     * </ol>
     * Strategy:
     * <ul>
     *   <li>Add RESERVE players to LOS squares first (handles both constraints
     *       simultaneously).</li>
     *   <li>Add remaining RESERVE players to center-field overflow squares.</li>
     *   <li>If LOS is still short after reserves are exhausted, move an existing
     *       STANDING non-LOS player to an empty LOS square.</li>
     * </ul>
     * All coordinates are in home-perspective;
     * {@code UtilServerSetup.setupPlayer} transforms them for the away team.
     */
    private void placeReservePlayersIfNeeded(Game game, GameState gameState) {
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();
        FieldModel fieldModel = game.getFieldModel();

        // Count available players, those already on the field, and those on LOS.
        int available = 0;
        int onField = 0;
        int onLos = 0;
        for (Player<?> player : team.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(player);
            if (!ps.canBeSetUpNextDrive()) continue;
            available++;
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            boolean inHalf = homePlaying
                ? FieldCoordinateBounds.HALF_HOME.isInBounds(coord)
                : FieldCoordinateBounds.HALF_AWAY.isInBounds(coord);
            if (inHalf) {
                onField++;
                boolean inLos = homePlaying
                    ? FieldCoordinateBounds.LOS_HOME.isInBounds(coord)
                    : FieldCoordinateBounds.LOS_AWAY.isInBounds(coord);
                if (inLos) onLos++;
            }
        }

        int minOnLos = 3;
        int losNeeded = (available >= minOnLos) ? Math.max(0, minOnLos - onLos) : Math.max(0, available - onLos);
        int fieldNeeded = Math.max(0, Math.min(available, 11) - onField);

        if (losNeeded == 0 && fieldNeeded == 0) return;

        // LOS squares (home-perspective x=12, y=4..10), middle-out order.
        int[][] losSquares = {
            {12,7}, {12,6}, {12,8}, {12,5}, {12,9}, {12,4}, {12,10}
        };
        // Overflow squares (home-perspective): center field, not LOS, not wide zones.
        int[][] overflowSquares = {
            {5,5}, {5,7}, {5,9}, {6,6}, {6,8},
            {4,6}, {4,8}, {3,6}, {3,8}, {2,5}, {2,9}, {1,7}
        };

        // Phase 1: place RESERVE players — LOS first, then overflow.
        int losIdx = 0;
        int overflowIdx = 0;
        for (Player<?> player : team.getPlayers()) {
            if (losNeeded <= 0 && fieldNeeded <= 0) break;
            PlayerState ps = fieldModel.getPlayerState(player);
            if (!ps.canBeSetUpNextDrive() || ps.getBase() != PlayerState.RESERVE) continue;

            if (losNeeded > 0) {
                while (losIdx < losSquares.length) {
                    int ox = losSquares[losIdx][0];
                    int oy = losSquares[losIdx][1];
                    losIdx++;
                    FieldCoordinate globalCoord = homePlaying
                        ? new FieldCoordinate(ox, oy)
                        : new FieldCoordinate(ox, oy).transform();
                    if (fieldModel.getPlayer(globalCoord) == null) {
                        UtilServerSetup.setupPlayer(gameState, player.getId(), new FieldCoordinate(ox, oy));
                        losNeeded--;
                        fieldNeeded--;
                        break;
                    }
                }
            } else {
                while (overflowIdx < overflowSquares.length) {
                    int ox = overflowSquares[overflowIdx][0];
                    int oy = overflowSquares[overflowIdx][1];
                    overflowIdx++;
                    FieldCoordinate globalCoord = homePlaying
                        ? new FieldCoordinate(ox, oy)
                        : new FieldCoordinate(ox, oy).transform();
                    if (fieldModel.getPlayer(globalCoord) == null) {
                        UtilServerSetup.setupPlayer(gameState, player.getId(), new FieldCoordinate(ox, oy));
                        fieldNeeded--;
                        break;
                    }
                }
            }
        }

        // Phase 2: if LOS is still short after reserves are exhausted, move an
        // existing STANDING non-LOS player to an empty LOS square.
        if (losNeeded > 0) {
            for (Player<?> player : team.getPlayers()) {
                if (losNeeded <= 0) break;
                PlayerState ps = fieldModel.getPlayerState(player);
                if (ps.getBase() != PlayerState.STANDING) continue;
                FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
                boolean alreadyOnLos = homePlaying
                    ? FieldCoordinateBounds.LOS_HOME.isInBounds(coord)
                    : FieldCoordinateBounds.LOS_AWAY.isInBounds(coord);
                if (alreadyOnLos) continue;

                // Move this player to an empty LOS square.
                while (losIdx < losSquares.length) {
                    int ox = losSquares[losIdx][0];
                    int oy = losSquares[losIdx][1];
                    losIdx++;
                    FieldCoordinate globalCoord = homePlaying
                        ? new FieldCoordinate(ox, oy)
                        : new FieldCoordinate(ox, oy).transform();
                    if (fieldModel.getPlayer(globalCoord) == null) {
                        UtilServerSetup.setupPlayer(gameState, player.getId(), new FieldCoordinate(ox, oy));
                        losNeeded--;
                        break;
                    }
                }
            }
        }
    }

    /**
     * Moves all {@code canBeSetUpNextDrive()} players of the current team back
     * into their reserve box — without touching the other team.
     * Called before each SETUP attempt so retries start from a clean slate.
     */
    private static void resetCurrentTeamToReserve(Game game) {
        boolean homePlaying = game.isHomePlaying();
        Team team = homePlaying ? game.getTeamHome() : game.getTeamAway();
        FieldModel fieldModel = game.getFieldModel();
        for (Player<?> player : team.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(player);
            if (ps.canBeSetUpNextDrive()) {
                fieldModel.setPlayerState(player, ps.changeBase(PlayerState.RESERVE));
                UtilBox.putPlayerIntoBox(game, player);
            }
        }
    }

    /**
     * Returns the team ID embedded in dialogs that are team-specific (e.g.
     * {@code DialogPlayerChoiceParameter}).  Returns {@code null} for dialogs
     * that are not team-specific.
     */
    private static String getDialogTeamId(IDialogParameter dialog) {
        if (dialog instanceof DialogPlayerChoiceParameter) {
            return ((DialogPlayerChoiceParameter) dialog).getTeamId();
        }
        return null;
    }

    /** Inject a command as the team that is currently playing (home or away). */
    private static void inject(GameState gameState, ClientCommand cmd) {
        boolean home = gameState.getGame().isHomePlaying();
        Session session = home ? HeadlessFantasyFootballServer.HOME_SESSION
                               : HeadlessFantasyFootballServer.AWAY_SESSION;
        gameState.handleCommand(new ReceivedCommand(cmd, session));
    }

    /** Inject a command on behalf of a specific team (for off-turn dialogs). */
    private static void injectForTeam(GameState gameState, ClientCommand cmd, boolean homeTeam) {
        Session session = homeTeam ? HeadlessFantasyFootballServer.HOME_SESSION
                                   : HeadlessFantasyFootballServer.AWAY_SESSION;
        gameState.handleCommand(new ReceivedCommand(cmd, session));
    }
}
