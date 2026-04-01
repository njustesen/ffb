package com.fumbbl.ffb.ai.client;

import com.fumbbl.ffb.ClientStateId;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.MoveSquare;
import com.fumbbl.ffb.PasswordChallenge;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.PushbackSquare;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.ai.strategy.RandomStrategy;
import com.fumbbl.ffb.client.state.ClientState;
import com.fumbbl.ffb.client.state.logic.LoginLogicModule;
import com.fumbbl.ffb.client.state.logic.LogicModule;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.util.UtilPlayer;

import java.lang.reflect.Field;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Decision engine for the AI client.
 *
 * Runs in a background daemon thread, polling the game state every 100 ms and
 * dispatching the appropriate command to the server.
 *
 * Priority order per tick:
 *   1. Respond to any outstanding dialog (server-sent {@link IDialogParameter}).
 *   2. Perform an active-state action when it is the AI's turn.
 *   3. Handle the login state headlessly (bypassing the Swing dialog).
 */
public class AiDecisionEngine implements Runnable {

    private static final int POLL_INTERVAL_MS = 100;
    private static final int POST_DIALOG_WAIT_MS = 300;
    private static final int POST_ACTION_WAIT_MS = 500;

    private final AiClient client;
    private final String password;
    private final String gameName;
    private final boolean home;
    private final Random random = new Random();

    private volatile boolean running = true;
    private IDialogParameter lastHandledDialog = null;
    private boolean loginAttempted = false;

    public AiDecisionEngine(AiClient client, String password, boolean home) {
        this.client = client;
        this.password = password;
        this.gameName = "LocalGame"; // default; could be made configurable
        this.home = home;
    }

    @Override
    public void run() {
        while (running) {
            try {
                tick();
                Thread.sleep(POLL_INTERVAL_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                running = false;
            } catch (Exception e) {
                // Log but keep running — a single bad tick should not crash the engine
                System.err.println("[AiDecisionEngine] Error in tick: " + e.getMessage());
                e.printStackTrace(System.err);
            }
        }
    }

    private void tick() throws InterruptedException {
        ClientState<? extends LogicModule, ?> state = client.getClientState();

        // ── Login handling ──────────────────────────────────────────────────────
        if (!loginAttempted && state != null && ClientStateId.LOGIN == state.getId()) {
            triggerHeadlessLogin(state);
            return;
        }

        Game game = client.getGame();
        if (game == null) {
            return;
        }

        // ── Dialog response ─────────────────────────────────────────────────────
        IDialogParameter dialogParam = game.getDialogParameter();
        if (dialogParam != null && dialogParam != lastHandledDialog) {
            if (isOurDialog(dialogParam, game)) {
                try {
                    RandomStrategy.respondToDialog(dialogParam, game, client.getCommunication());
                } catch (Exception e) {
                    System.err.println("[AiDecisionEngine] Error responding to dialog " + dialogParam.getId() + ": " + e.getMessage());
                }
                lastHandledDialog = dialogParam;
                Thread.sleep(POST_DIALOG_WAIT_MS);
                return;
            }
        }

        // ── Active-state action ─────────────────────────────────────────────────
        if (state != null) {
            ClientStateId stateId = state.getId();
            // START_GAME requires a confirm from both sides regardless of whose turn it is
            boolean requiresBothSides = stateId == ClientStateId.START_GAME;
            if (requiresBothSides || game.isHomePlaying()) {
                handleActiveState(stateId, game);
                Thread.sleep(POST_ACTION_WAIT_MS);
            }
        }
    }

    /**
     * Bypass the Swing login dialog by directly calling {@link LoginLogicModule}
     * via reflection (the logicModule field is protected in {@code ClientState}).
     */
    private void triggerHeadlessLogin(ClientState<? extends LogicModule, ?> state) {
        loginAttempted = true;
        try {
            // Access the protected logicModule field
            Field logicModuleField = ClientState.class.getDeclaredField("logicModule");
            logicModuleField.setAccessible(true);
            Object lm = logicModuleField.get(state);
            if (!(lm instanceof LoginLogicModule)) {
                return;
            }
            LoginLogicModule loginModule = (LoginLogicModule) lm;

            // Encode the password the same way the Swing dialog does
            String password = this.password;
            byte[] encodedPassword = null;
            int passwordLength = -1;
            if (password != null && !password.isEmpty()) {
                try {
                    encodedPassword = PasswordChallenge.md5Encode(password.getBytes());
                    passwordLength = password.length();
                } catch (NoSuchAlgorithmException e) {
                    System.err.println("[AiDecisionEngine] MD5 encoding failed: " + e.getMessage());
                }
            }

            LoginLogicModule.LoginData loginData = new LoginLogicModule.LoginData(
                gameName,
                encodedPassword,
                passwordLength,
                false  // listGames = false
            );
            loginModule.sendChallenge(loginData);

        } catch (NoSuchFieldException | IllegalAccessException e) {
            System.err.println("[AiDecisionEngine] Could not access LoginLogicModule via reflection: " + e.getMessage());
        }
    }

    /**
     * Determine whether the given dialog belongs to the home (AI) coach.
     *
     * For dialogs that carry a player ID, we check if the player is on the
     * home team.  For dialogs with no player ID we use {@code game.isHomePlaying()}
     * as a heuristic.
     */
    private boolean isOurDialog(IDialogParameter param, Game game) {
        String playerId = extractPlayerId(param);
        if (playerId != null) {
            Player<?> player = game.getPlayerById(playerId);
            if (player == null) {
                return false;
            }
            return game.getTeamHome().hasPlayer(player);
        }
        // No player ID — check teamId if available (e.g. playerChoice dialogs)
        String teamId = extractTeamId(param);
        if (teamId != null) {
            return teamId.equals(game.getTeamHome().getId());
        }
        // No player or team ID — use the active-side heuristic
        return game.isHomePlaying();
    }

    /**
     * Try to extract a player ID from any dialog parameter type that carries one.
     * Returns {@code null} when the dialog has no associated player.
     */
    private String extractPlayerId(IDialogParameter param) {
        if (param == null) {
            return null;
        }
        try {
            java.lang.reflect.Method m = param.getClass().getMethod("getPlayerId");
            return (String) m.invoke(param);
        } catch (NoSuchMethodException ignored) {
            // Dialog does not carry a player ID
        } catch (Exception e) {
            // Reflection failed — treat as no player
        }
        return null;
    }

    /**
     * Try to extract a team ID from dialog parameters that are scoped to a specific team.
     * Returns {@code null} when the dialog has no team ID.
     */
    private String extractTeamId(IDialogParameter param) {
        if (param == null) {
            return null;
        }
        for (String methodName : new String[]{ "getTeamId", "getChoosingTeamId" }) {
            try {
                java.lang.reflect.Method m = param.getClass().getMethod(methodName);
                return (String) m.invoke(param);
            } catch (NoSuchMethodException ignored) {
                // Try next method name
            } catch (Exception e) {
                // Reflection failed — treat as no team
            }
        }
        return null;
    }

    /**
     * Perform a game-state-specific action for the active (home) side.
     */
    private void handleActiveState(ClientStateId stateId, Game game) {
        if (stateId == null) {
            return;
        }
        switch (stateId) {
            case START_GAME:
                // Both coaches have joined; send CLIENT_START_GAME to advance past the start phase
                client.getCommunication().sendStartGame();
                break;

            case SELECT_PLAYER: {
                // 20% chance: end turn immediately
                if (random.nextInt(5) == 0) {
                    client.getCommunication().sendEndTurn(game.getTurnMode());
                    break;
                }
                // Collect home players that are standing and have not yet acted
                List<Player<?>> candidates = new ArrayList<>();
                for (Player<?> p : game.getTeamHome().getPlayers()) {
                    PlayerState ps = game.getFieldModel().getPlayerState(p);
                    if (ps != null && ps.getBase() == PlayerState.STANDING && ps.isActive()) {
                        candidates.add(p);
                    }
                }
                if (candidates.isEmpty()) {
                    client.getCommunication().sendEndTurn(game.getTurnMode());
                    break;
                }
                Player<?> chosen = candidates.get(random.nextInt(candidates.size()));
                PlayerAction[] actions = {
                    PlayerAction.MOVE, PlayerAction.BLOCK,
                    PlayerAction.BLITZ, PlayerAction.FOUL
                };
                PlayerAction action = actions[random.nextInt(actions.length)];
                client.getCommunication().sendActingPlayer(chosen, action, false);
                break;
            }

            case WAIT_FOR_OPPONENT:
                // Opponent's turn — nothing to do
                break;

            case SETUP: {
                // If home players are already on the field, confirm the setup.
                // Otherwise, request the setup list (which triggers the TEAM_SETUP dialog).
                Team homeTeam = game.getTeamHome();
                FieldModel fieldModel = game.getFieldModel();
                if (homeTeam != null && fieldModel != null && hasPlayersOnField(homeTeam, fieldModel)) {
                    client.getCommunication().sendEndTurn(TurnMode.SETUP, homeTeam, fieldModel);
                } else {
                    client.getCommunication().sendTeamSetupLoad(null);
                }
                break;
            }

            case KICKOFF:
                // Kick to a random square in the opponent half (x: 1..12, y: 1..14 for away half)
                int kx = 13 + random.nextInt(12);
                int ky = 1 + random.nextInt(14);
                client.getCommunication().sendKickoff(new FieldCoordinate(kx, ky));
                break;

            case PUSHBACK: {
                PushbackSquare[] squares = game.getFieldModel().getPushbackSquares();
                if (squares != null && squares.length > 0) {
                    PushbackSquare chosen = squares[random.nextInt(squares.length)];
                    // Build a Pushback from the player being pushed back and the chosen square
                    String pushedPlayerId = game.getActingPlayer() != null && game.getActingPlayer().getPlayer() != null
                        ? game.getActingPlayer().getPlayer().getId() : null;
                    if (pushedPlayerId != null) {
                        client.getCommunication().sendPushback(
                            new com.fumbbl.ffb.Pushback(pushedPlayerId, chosen.getCoordinate()));
                    }
                }
                break;
            }

            case TOUCHBACK:
                handleTouchback(game);
                break;

            case SOLID_DEFENCE:
            case KICKOFF_RETURN:
            case HIGH_KICK:
            case QUICK_SNAP:
                // End these special kickoff sub-turns immediately without moving players
                client.getCommunication().sendEndTurn(game.getTurnMode());
                break;

            case MOVE: {
                MoveSquare[] squares = game.getFieldModel().getMoveSquares();
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (actingPlayer == null || squares == null || squares.length == 0 || random.nextBoolean()) {
                    // 50% chance to stop moving, or no squares reachable
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    MoveSquare target = squares[random.nextInt(squares.length)];
                    FieldCoordinate from = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                    client.getCommunication().sendPlayerMove(
                        actingPlayer.getPlayerId(), from,
                        new FieldCoordinate[]{ target.getCoordinate() },
                        null);
                }
                break;
            }

            case BLOCK:
            case BLITZ: {
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (actingPlayer == null || actingPlayer.getPlayer() == null) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                Player<?>[] targets = UtilPlayer.findAdjacentBlockablePlayers(
                    game, game.getTeamAway(), pos);
                if (targets == null || targets.length == 0) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    Player<?> defender = targets[random.nextInt(targets.length)];
                    client.getCommunication().sendBlock(
                        actingPlayer.getPlayerId(), defender,
                        false, false, false, false, false);
                }
                break;
            }

            default:
                // For all other active states (MOVE, BLITZ, BLOCK, FOUL, PASS, etc.),
                // deselect the acting player to end the action / pass to the server.
                client.getCommunication().sendActingPlayer(null, null, false);
                break;
        }
    }

    private void handleTouchback(Game game) {
        // Find any standing home-team player to receive the touchback
        Team homeTeam = game.getTeamHome();
        if (homeTeam == null) {
            return;
        }
        for (Player<?> player : homeTeam.getPlayers()) {
            PlayerState ps = game.getFieldModel().getPlayerState(player);
            FieldCoordinate coord = game.getFieldModel().getPlayerCoordinate(player);
            if (ps != null && coord != null && ps.isActive()) {
                client.getCommunication().sendTouchback(coord);
                return;
            }
        }
        // Fallback: send null coordinate (server should handle gracefully)
        client.getCommunication().sendTouchback(null);
    }

    /**
     * Returns true if at least 3 home-team players have a valid field coordinate
     * (x within [1..FIELD_WIDTH]) — indicating a setup has been loaded.
     */
    private boolean hasPlayersOnField(Team homeTeam, FieldModel fieldModel) {
        int count = 0;
        for (Player<?> player : homeTeam.getPlayers()) {
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            if (coord != null && coord.getX() >= 1 && coord.getX() <= FieldCoordinate.FIELD_WIDTH) {
                count++;
                if (count >= 3) {
                    return true;
                }
            }
        }
        return false;
    }

    public void stop() {
        running = false;
    }
}
