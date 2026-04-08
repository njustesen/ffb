package com.fumbbl.ffb.ai.client;

import com.fumbbl.ffb.ClientStateId;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.FieldCoordinateBounds;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.MoveSquare;
import com.fumbbl.ffb.PasswordChallenge;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.Pushback;
import com.fumbbl.ffb.PushbackSquare;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.dialog.DialogBlockRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPartialReRollParameter;
import com.fumbbl.ffb.dialog.DialogBlockRollPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogId;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionParameter;
import com.fumbbl.ffb.dialog.DialogOpponentBlockSelectionPropertiesParameter;
import com.fumbbl.ffb.dialog.DialogReRollParameter;
import com.fumbbl.ffb.option.GameOptionId;
import com.fumbbl.ffb.option.UtilGameOption;
import com.fumbbl.ffb.ai.ActionScore;
import com.fumbbl.ffb.ai.BoardVisualizer;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.PathProbabilityFinder;
import com.fumbbl.ffb.ai.PolicySampler;
import com.fumbbl.ffb.ai.strategy.RandomStrategy;
import com.fumbbl.ffb.ai.strategy.ScriptedStrategy;
import com.fumbbl.ffb.client.state.ClientState;
import com.fumbbl.ffb.client.state.logic.LoginLogicModule;
import com.fumbbl.ffb.client.state.logic.LogicModule;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.GameResult;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.property.NamedProperties;
import com.fumbbl.ffb.util.UtilPlayer;

import java.lang.reflect.Field;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
    private static final int POST_DIALOG_WAIT_MS = 75;
    private static final int POST_ACTION_WAIT_MS = 100;

    /** Softmax temperatures for various decision types. */
    private static final double T_PLAYER   = 0.50;
    private static final double T_MOVE     = 0.60;
    private static final double T_KICKOFF  = 1.20;

    private final AiClient client;
    private final String password;
    private final String gameName;
    private final boolean home;
    private final boolean useRandom;
    private final Random random = new Random();

    private volatile boolean running = true;
    private IDialogParameter lastHandledDialog = null;
    private boolean loginAttempted = false;
    /** Set to true when HIGH_KICK player has been positioned; reset on SETUP entry. */
    private boolean highKickDone = false;
    private boolean gameResultLogged = false;
    private final MovePolicyState movePolicy = new MovePolicyState();
    private ClientStateId lastLoggedStateId = null;
    private boolean lastLoggedHomePlaying = false;

    public AiDecisionEngine(AiClient client, String password, boolean home) {
        this(client, password, home, false);
    }

    public AiDecisionEngine(AiClient client, String password, boolean home, boolean useRandom) {
        this.client = client;
        this.password = password;
        this.gameName = "LocalGame";
        this.home = home;
        this.useRandom = useRandom;
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
        // Rule: if ANY dialog is active, skip the active-state handler to avoid
        // sending stale/conflicting commands while the server resolves the dialog.
        // Exceptions: (1) error-acknowledgement dialogs where the active state must also run
        // to fix the issue; (2) START_GAME where both sides must independently confirm.
        IDialogParameter dialogParam = game.getDialogParameter();
        if (dialogParam != null) {
            if (dialogParam != lastHandledDialog) {
                boolean ours = isOurDialog(dialogParam, game);
                String dialogExtra = "";
                if (dialogParam instanceof com.fumbbl.ffb.dialog.DialogSetupErrorParameter) {
                    String[] errs = ((com.fumbbl.ffb.dialog.DialogSetupErrorParameter) dialogParam).getSetupErrors();
                    if (errs != null && errs.length > 0) dialogExtra = " errors=" + java.util.Arrays.toString(errs);
                }
                System.out.println("[AI-DIALOG] " + dialogParam.getId() + " ours=" + ours
                    + " homePlaying=" + game.isHomePlaying() + dialogExtra
                    + (useRandom ? " [random]" : " [scripted]"));
                System.out.flush();
                if (ours) {
                    try {
                        if (!useRandom) {
                            visualizeDialogDecision(dialogParam, game);
                        }
                        if (useRandom) {
                            RandomStrategy.respondToDialog(dialogParam, game, client.getCommunication());
                        } else {
                            ScriptedStrategy.respondToDialog(dialogParam, game, client.getCommunication());
                        }
                    } catch (Exception e) {
                        System.err.println("[AiDecisionEngine] Error responding to dialog "
                            + dialogParam.getId() + ": " + e.getMessage());
                    }
                    lastHandledDialog = dialogParam;
                    movePolicy.reset(); // any dice-roll dialog opens a new move window
                    // Log game result immediately when GAME_STATISTICS is confirmed —
                    // the game is over and SPECTATE state may never be reached.
                    if (dialogParam.getId() == DialogId.GAME_STATISTICS) {
                        Thread.sleep(POST_DIALOG_WAIT_MS);
                        logGameResult(game);
                        return;
                    }
                    Thread.sleep(POST_DIALOG_WAIT_MS);
                    if (isErrorAcknowledgement(dialogParam)) {
                        // fall through to active state (e.g. SETUP_ERROR → fix setup)
                    } else {
                        return; // wait for server to clear the dialog
                    }
                } else {
                    // Opponent's dialog: record to suppress re-logging.
                    lastHandledDialog = dialogParam;
                    // START_GAME: both sides must confirm — fall through to active state.
                    // All others: wait for server to clear.
                    if (!isActiveStateContinueDialog(dialogParam)) {
                        return;
                    }
                }
            } else {
                // Already seen/handled this dialog; waiting for server to clear it.
                // For dialogs that need the active state to continue (setup, game start):
                // fall through. For all others: wait.
                if (!isActiveStateContinueDialog(dialogParam)) {
                    return;
                }
            }
        }

        // ── Active-state action ─────────────────────────────────────────────────
        if (state != null) {
            ClientStateId stateId = state.getId();
            boolean homePlaying = game.isHomePlaying();

            // Log state transitions (or when our side starts/stops playing)
            if (stateId != lastLoggedStateId || homePlaying != lastLoggedHomePlaying) {
                lastLoggedStateId = stateId;
                lastLoggedHomePlaying = homePlaying;
                System.out.println("[AI-TICK] state=" + stateId + " homePlaying=" + homePlaying
                    + (useRandom ? " [random]" : " [scripted]"));
                System.out.flush();
            }

            // SPECTATE/REPLAY: always log game result regardless of which side is "playing"
            if (stateId == ClientStateId.SPECTATE || stateId == ClientStateId.REPLAY) {
                logGameResult(game);
                return;
            }

            boolean requiresBothSides = (stateId == ClientStateId.START_GAME);
            if (requiresBothSides || homePlaying) {
                handleActiveState(stateId, game);
                Thread.sleep(POST_ACTION_WAIT_MS);
            }
        }
    }

    // ── Login ────────────────────────────────────────────────────────────────────

    private void triggerHeadlessLogin(ClientState<? extends LogicModule, ?> state) {
        loginAttempted = true;
        try {
            Field logicModuleField = ClientState.class.getDeclaredField("logicModule");
            logicModuleField.setAccessible(true);
            Object lm = logicModuleField.get(state);
            if (!(lm instanceof LoginLogicModule)) {
                return;
            }
            LoginLogicModule loginModule = (LoginLogicModule) lm;
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
                gameName, encodedPassword, passwordLength, false);
            loginModule.sendChallenge(loginData);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            System.err.println("[AiDecisionEngine] Could not access LoginLogicModule: " + e.getMessage());
        }
    }

    // ── Dialog ownership ─────────────────────────────────────────────────────────

    private boolean isOurDialog(IDialogParameter param, Game game) {
        // Both sides must confirm the end-of-game statistics dialog.
        if (param != null && param.getId() == DialogId.GAME_STATISTICS) {
            return true;
        }
        String playerId = extractPlayerId(param);
        if (playerId != null) {
            Player<?> player = game.getPlayerById(playerId);
            if (player == null) return false;
            return game.getTeamHome().hasPlayer(player);
        }
        String teamId = extractTeamId(param);
        if (teamId != null) {
            return teamId.equals(game.getTeamHome().getId());
        }
        return game.isHomePlaying();
    }

    private String extractPlayerId(IDialogParameter param) {
        if (param == null) return null;
        try {
            java.lang.reflect.Method m = param.getClass().getMethod("getPlayerId");
            return (String) m.invoke(param);
        } catch (NoSuchMethodException ignored) {
        } catch (Exception ignored) {
        }
        return null;
    }

    private String extractTeamId(IDialogParameter param) {
        if (param == null) return null;
        for (String methodName : new String[]{"getTeamId", "getChoosingTeamId"}) {
            try {
                java.lang.reflect.Method m = param.getClass().getMethod(methodName);
                return (String) m.invoke(param);
            } catch (NoSuchMethodException ignored) {
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    // ── Active state dispatch ─────────────────────────────────────────────────────

    private void handleActiveState(ClientStateId stateId, Game game) {
        if (stateId == null) return;
        if (useRandom) {
            handleActiveStateRandom(stateId, game);
            return;
        }

        switch (stateId) {

            case START_GAME:
                client.getCommunication().sendStartGame();
                break;

            case SELECT_PLAYER:
                handleSelectPlayer(game);
                break;

            case WAIT_FOR_OPPONENT:
            case WAIT_FOR_SETUP:
                break;

            case SPECTATE:
            case REPLAY:
                logGameResult(game);
                break;

            case SETUP: {
                // Reset high kick state on setup entrance
                highKickDone = false;
                Team homeTeam = game.getTeamHome();
                FieldModel fieldModel = game.getFieldModel();
                if (homeTeam != null && fieldModel != null && hasPlayersOnField(homeTeam, fieldModel)) {
                    // Ensure all available-but-reserve players get field positions
                    // before submitting, to avoid "must field N players" SETUP_ERROR loops.
                    fixReservePlayersForSetup(homeTeam, fieldModel, game);
                    client.getCommunication().sendEndTurn(TurnMode.SETUP, homeTeam, fieldModel);
                } else {
                    client.getCommunication().sendTeamSetupLoad(null);
                }
                break;
            }

            case HIGH_KICK:
                handleHighKick(game);
                break;

            case KICKOFF:
                handleKickoff(game);
                break;

            case SOLID_DEFENCE:
            case KICKOFF_RETURN:
            case QUICK_SNAP:
                client.getCommunication().sendEndTurn(game.getTurnMode());
                break;

            case PUSHBACK:
                handlePushback(game);
                break;

            case TOUCHBACK:
                handleTouchback(game);
                break;

            case MOVE:
                handleMove(game);
                break;

            case BLOCK:
            case BLITZ: {
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (actingPlayer == null || actingPlayer.getPlayer() == null) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                // If the player already blocked (e.g. blitz follow-up left them in BLITZ state
                // with remaining MA), end the action rather than trying to block again.
                if (actingPlayer.hasBlocked() || movePolicy.hasActed()) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                Player<?>[] targets = UtilPlayer.findAdjacentBlockablePlayers(
                    game, game.getTeamAway(), pos);
                if (targets == null || targets.length == 0) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    double[] tScores = new double[targets.length];
                    for (int i = 0; i < targets.length; i++) {
                        tScores[i] = new ActionScore(
                            MoveDecisionEngine.computeBlockProbability(actingPlayer.getPlayer(), targets[i], game,
                                game.getTeamHome(), game.getTeamAway()),
                            +0.70, 0.75).softmaxScore();
                    }
                    savePlayerScoreVisualization(game, targets, tScores, pos,
                        stateId == ClientStateId.BLITZ ? "BLITZ" : "BLOCK");
                    int chosen = PolicySampler.sample(tScores, T_PLAYER, random);
                    movePolicy.recordAction();
                    client.getCommunication().sendBlock(
                        actingPlayer.getPlayerId(), targets[chosen],
                        false, false, false, false, false);
                }
                break;
            }

            case FOUL: {
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (actingPlayer == null || actingPlayer.getPlayer() == null) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                // If the foul already happened (ejection pending), don't try to foul again.
                if (actingPlayer.hasFouled() || movePolicy.hasActed()) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                Player<?>[] targets = UtilPlayer.findAdjacentPronePlayers(
                    game, game.getTeamAway(), pos);
                if (targets == null || targets.length == 0) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    double[] fScores = new double[targets.length];
                    for (int i = 0; i < targets.length; i++) {
                        fScores[i] = computeFoulActionScore(
                            actingPlayer.getPlayer(), targets[i], game, countRemainingActivations(game)).softmaxScore();
                    }
                    savePlayerScoreVisualization(game, targets, fScores, pos, "FOUL");
                    int chosen = PolicySampler.argmax(fScores);
                    movePolicy.recordAction();
                    client.getCommunication().sendFoul(
                        actingPlayer.getPlayerId(), targets[chosen], false);
                }
                break;
            }

            case SELECT_BLITZ_TARGET: {
                if (movePolicy.hasActed()) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                // Score opponents by proximity to ball; visualize and pick via softmax
                FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();
                if (ballCoord == null) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                List<Player<?>> blitzCands = new ArrayList<>();
                List<Double> blitzScores = new ArrayList<>();
                for (Player<?> opp : game.getTeamAway().getPlayers()) {
                    FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(opp);
                    if (pos == null) continue;
                    int dist = MoveDecisionEngine.chebyshev(pos, ballCoord);
                    double proximity = Math.max(0.01, 1.0 - dist * 0.12);
                    blitzCands.add(opp);
                    blitzScores.add(new ActionScore(proximity, +0.75, 0.65).softmaxScore());
                }
                if (blitzCands.isEmpty()) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                double[] bArr = blitzScores.stream().mapToDouble(Double::doubleValue).toArray();
                savePlayerScoreVisualization(game,
                    blitzCands.toArray(new Player<?>[0]), bArr, null, "SELECT_BLITZ_TARGET");
                movePolicy.recordAction();
                int chosen = PolicySampler.sample(bArr, T_PLAYER, random);
                client.getCommunication().sendActingPlayer(blitzCands.get(chosen), PlayerAction.BLITZ, false);
                break;
            }

            case PLACE_BALL: {
                // Safe Pair of Hands — place ball at current acting player coordinate
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (actingPlayer != null && actingPlayer.getPlayer() != null) {
                    FieldCoordinate coord = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                    client.getCommunication().sendTouchback(coord);
                } else {
                    client.getCommunication().sendTouchback(null);
                }
                break;
            }

            case PASS:
            case HAND_OVER:
            case DUMP_OFF:
            case PUNT:
            case PASS_BLOCK:
            case INTERCEPTION:
                client.getCommunication().sendActingPlayer(null, null, false);
                break;

            default:
                // All exotic states: deselect to avoid getting stuck
                client.getCommunication().sendActingPlayer(null, null, false);
                break;
        }
    }

    // ── SELECT_PLAYER ─────────────────────────────────────────────────────────────

    private void handleSelectPlayer(Game game) {
        MoveDecisionEngine.PlayerSelection sel = MoveDecisionEngine.selectPlayer(
            game, game.getTeamHome(), game.getTeamAway(), true, true, random, false);

        FieldCoordinate chosenCoord = sel.player != null
            ? game.getFieldModel().getPlayerCoordinate(sel.player) : null;
        savePlayerSelectionVisualization(game, sel.candidatePlayers, sel.candidateActions,
            sel.rawScores, chosenCoord);

        if (sel.player == null || sel.action == null) {
            client.getCommunication().sendEndTurn(game.getTurnMode());
        } else {
            movePolicy.reset();
            client.getCommunication().sendActingPlayer(sel.player, sel.action, false);
        }
    }

    private ActionScore computeFoulActionScore(Player<?> fouler, Player<?> target, Game game, int remainingActivations) {
        boolean hasDirtyPlayer = fouler.hasSkillProperty(NamedProperties.affectsEitherArmourOrInjuryOnFoul);
        boolean hasChainsaw    = fouler.hasSkillProperty(NamedProperties.foulBreaksArmourWithoutRoll);
        int assists = UtilPlayer.findFoulAssists(game, fouler, target);
        int targetArmour = target.getArmourWithModifiers();
        int turnNr = game.getTurnDataHome().getTurnNr();

        double prob = 0.15 + assists * 0.12
            + (hasDirtyPlayer ? 0.25 : 0.0) + (hasChainsaw ? 0.55 : 0.0);
        prob = Math.min(prob, 0.85);

        double value = Math.max(0.10, (targetArmour - 6) * 0.12 + (turnNr >= 7 ? 0.15 : 0.0));
        if (remainingActivations <= 2) value = Math.min(value * 1.4, 0.55);
        value = Math.min(value, 0.55);

        return new ActionScore(prob, value, 0.35);
    }

    // ── KICKOFF ───────────────────────────────────────────────────────────────────

    private void handleKickoff(Game game) {
        // Score all squares in the opponent's half (x=13..24, y=0..14)
        // Peak near center of opponent half (x=19, y=7)
        List<FieldCoordinate> squares = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        for (int x = 13; x <= 24; x++) {
            for (int y = 0; y <= 14; y++) {
                FieldCoordinate coord = new FieldCoordinate(x, y);
                double dxCenter = x - 19;
                double dyCenter = y - 7;
                // Gaussian: 1.0 at center (x=19,y=7), falls off smoothly
                double score = Math.exp(-0.02 * (dxCenter * dxCenter + dyCenter * dyCenter));
                squares.add(coord);
                scores.add(score);
            }
        }

        double[] scoreArray = scores.stream().mapToDouble(Double::doubleValue).toArray();
        int chosen = PolicySampler.sample(scoreArray, T_KICKOFF, random);
        FieldCoordinate target = squares.get(chosen);

        // Visualize kickoff
        saveKickoffVisualization(squares, scoreArray);

        client.getCommunication().sendKickoff(target);
    }

    // ── HIGH KICK ─────────────────────────────────────────────────────────────────

    private void handleHighKick(Game game) {
        FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();
        if (ballCoord == null) {
            client.getCommunication().sendEndTurn(game.getTurnMode());
            return;
        }
        // Check if a home player is already at the ball coordinate
        Player<?> atBall = game.getFieldModel().getPlayer(ballCoord);
        if (atBall != null && game.getTeamHome().hasPlayer(atBall)) {
            // Best receiver is already in position — end turn
            client.getCommunication().sendEndTurn(game.getTurnMode());
            return;
        }
        if (!highKickDone) {
            // Move best ball carrier to ball position
            Player<?> best = ScriptedStrategy.findBestBallCarrier(game);
            if (best != null) {
                client.getCommunication().sendSetupPlayer(best, ballCoord);
                highKickDone = true;
            } else {
                client.getCommunication().sendEndTurn(game.getTurnMode());
            }
        } else {
            // Already positioned — confirm
            client.getCommunication().sendEndTurn(game.getTurnMode());
        }
    }

    // ── PUSHBACK ──────────────────────────────────────────────────────────────────

    private void handlePushback(Game game) {
        PushbackSquare[] squares = game.getFieldModel().getPushbackSquares();
        if (squares == null || squares.length == 0) {
            return;
        }

        ActingPlayer actingPlayer = game.getActingPlayer();
        if (actingPlayer == null) return;

        boolean weAreAttacker = (actingPlayer.getPlayer() != null)
            && game.getTeamHome().hasPlayer(actingPlayer.getPlayer());

        FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();

        if (weAreAttacker) {
            // Case A: push the opponent — choose worst square for them
            PushbackSquare chosen = choosePushbackForOpponent(squares);
            String pushedId = game.getDefender() != null ? game.getDefender().getId() : null;
            if (pushedId != null) {
                client.getCommunication().sendPushback(new Pushback(pushedId, chosen.getCoordinate()));
            }
        } else {
            // Case B: Side Step — our player chooses their own square
            Player<?> pushedPlayer = game.getDefender();
            boolean isBallCarrier = (pushedPlayer != null) && (ballCoord != null)
                && ballCoord.equals(game.getFieldModel().getPlayerCoordinate(pushedPlayer));
            PushbackSquare chosen = choosePushbackForOurPlayer(squares, isBallCarrier, game);
            String pushedId = (pushedPlayer != null) ? pushedPlayer.getId() : null;
            if (pushedId != null) {
                client.getCommunication().sendPushback(new Pushback(pushedId, chosen.getCoordinate()));
            }
        }
    }

    private PushbackSquare choosePushbackForOpponent(PushbackSquare[] squares) {
        // Priority: off-pitch → closest sideline → furthest from opponent endzone (highest X)
        PushbackSquare bestOffPitch = null;
        PushbackSquare bestSideline = null;
        PushbackSquare bestAdvanced = null;

        int minSidelineY = Integer.MAX_VALUE;
        int maxX = -1;

        for (PushbackSquare sq : squares) {
            FieldCoordinate coord = sq.getCoordinate();
            if (!FieldCoordinateBounds.FIELD.isInBounds(coord)) {
                if (bestOffPitch == null) bestOffPitch = sq;
            } else {
                int sidelineY = Math.min(coord.getY(), 14 - coord.getY());
                if (sidelineY < minSidelineY) {
                    minSidelineY = sidelineY;
                    bestSideline = sq;
                }
                // Push opponent toward their own half (low X for opponent who starts at high X)
                // Home defends x=0, attacks toward x=25 — opponent starts in right half
                // Push toward their own endzone: for away team that's higher X is worse for them
                // Actually, pushing them toward x=0 (home endzone) traps them
                // Simplification: push to highest X (further from home endzone, deeper in their own half)
                // Wait: home team attacks toward x=25. Opponent's endzone is x=0.
                // To push opponent backward (toward their own endzone at x=0): choose LOWEST X
                if (coord.getX() < maxX || maxX < 0) {
                    // Actually we want to push toward opponent's own endzone (x=25 for away team)
                    // Hmm: home team = attacks toward x=25. Away team = attacks toward x=0.
                    // Away team's endzone = x=25. To push opponent backward = toward x=25 (their endzone).
                    // So maximize X for away team's "backward" direction.
                }
                if (coord.getX() > maxX) {
                    maxX = coord.getX();
                    bestAdvanced = sq;
                }
            }
        }

        if (bestOffPitch != null) return bestOffPitch;
        if (bestSideline != null) return bestSideline;
        if (bestAdvanced != null) return bestAdvanced;
        return squares[0];
    }

    private PushbackSquare choosePushbackForOurPlayer(PushbackSquare[] squares, boolean isBallCarrier, Game game) {
        PushbackSquare bestInBounds = null;
        int bestX = -1;
        int bestUncoveredX = -1;
        PushbackSquare bestUncovered = null;

        for (PushbackSquare sq : squares) {
            FieldCoordinate coord = sq.getCoordinate();
            if (!FieldCoordinateBounds.FIELD.isInBounds(coord)) {
                continue; // avoid OOB
            }
            boolean adjacentToOpponent = isAdjacentToOpponent(coord, game);

            if (isBallCarrier) {
                // Ball carrier: maximize advance (lowest endzone distance = highest X for home team)
                if (coord.getX() > bestX) {
                    bestX = coord.getX();
                    bestInBounds = sq;
                }
            } else {
                // Non-ball carrier: prefer not adjacent to opponent
                if (!adjacentToOpponent) {
                    if (coord.getX() > bestUncoveredX) {
                        bestUncoveredX = coord.getX();
                        bestUncovered = sq;
                    }
                } else {
                    if (coord.getX() > bestX) {
                        bestX = coord.getX();
                        bestInBounds = sq;
                    }
                }
            }
        }

        if (bestUncovered != null) return bestUncovered;
        if (bestInBounds != null) return bestInBounds;
        return squares[0];
    }

    private boolean isAdjacentToOpponent(FieldCoordinate coord, Game game) {
        for (Player<?> opp : game.getTeamAway().getPlayers()) {
            FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(opp);
            if (pos != null && MoveDecisionEngine.chebyshev(coord, pos) <= 1) return true;
        }
        return false;
    }

    // ── TOUCHBACK ─────────────────────────────────────────────────────────────────

    private void handleTouchback(Game game) {
        FieldCoordinate coord = ScriptedStrategy.findBestBallCarrierCoord(game);
        client.getCommunication().sendTouchback(coord);
    }

    // ── MOVE ──────────────────────────────────────────────────────────────────────

    private void handleMove(Game game) {
        // Adjacent squares from the server — used only to gate whether movement is available.
        MoveSquare[] adjacentSquares = game.getFieldModel().getMoveSquares();
        ActingPlayer actingPlayer = game.getActingPlayer();

        if (actingPlayer == null || adjacentSquares == null || adjacentSquares.length == 0) {
            client.getCommunication().sendActingPlayer(null, null, false);
            return;
        }

        Player<?> player = actingPlayer.getPlayer();
        FieldCoordinate playerCoord = game.getFieldModel().getPlayerCoordinate(player);
        FieldCoordinate ballCoord = game.getFieldModel().getBallCoordinate();

        // Hard rule: if the player has already moved and no dice roll has occurred
        // since, end the activation immediately (no pathfinding needed).
        if (movePolicy.shouldEndNow()) {
            client.getCommunication().sendActingPlayer(null, null, false);
            return;
        }

        // Delegate to MoveDecisionEngine for consistent behavior with the headless simulation.
        MoveDecisionEngine.MoveResult result = MoveDecisionEngine.selectMoveTarget(
            game, actingPlayer, game.getTeamHome(), game.getTeamAway(), true, random, false);

        // Visualization: softmax probs over all candidate squares
        int n = result.candidates.size();
        if (n > 0) {
            double[] allScores = result.rawScores;
            double[] softmaxProbs = PolicySampler.softmax(allScores, MoveDecisionEngine.T_MOVE);
            Map<FieldCoordinate, Double> colorMap = new LinkedHashMap<>();
            Map<FieldCoordinate, Double> labelMap = new LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                colorMap.put(result.candidates.get(i), softmaxProbs[i]);
                labelMap.put(result.candidates.get(i), softmaxProbs[i]);
            }
            double endPct = result.hasEndOption ? softmaxProbs[n] * 100.0 : 0.0;
            String label = String.format("MOVE — %s [end=%.1f%%]", result.role(), endPct);
            BoardVisualizer.save(client, label, colorMap, labelMap, result.playerCoord);
        }

        if (result.isEndAction()) {
            client.getCommunication().sendActingPlayer(null, null, false);
        } else {
            FieldCoordinate from = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
            movePolicy.recordMove();
            client.getCommunication().sendPlayerMove(
                actingPlayer.getPlayerId(), from, result.chosen.path, null);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    /**
     * Returns true if at least 3 home-team players have a valid field coordinate.
     */
    private boolean hasPlayersOnField(Team homeTeam, FieldModel fieldModel) {
        int count = 0;
        for (Player<?> player : homeTeam.getPlayers()) {
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            if (coord != null && coord.getX() >= 1 && coord.getX() <= FieldCoordinate.FIELD_WIDTH) {
                count++;
                if (count >= 3) return true;
            }
        }
        return false;
    }

    /**
     * Returns true for dialogs that only need sendConfirm() to acknowledge — after which we
     * should immediately run the active-state handler rather than waiting for the next tick.
     */
    private boolean isErrorAcknowledgement(IDialogParameter param) {
        if (param == null) return false;
        DialogId id = param.getId();
        return id == DialogId.SETUP_ERROR
            || id == DialogId.SWARMING_ERROR
            || id == DialogId.INVALID_SOLID_DEFENCE
            || id == DialogId.PENALTY_SHOOTOUT;
    }

    /**
     * Returns true for dialogs where the active-state handler must continue to run
     * while the dialog is present (it takes both a dialog response AND active-state
     * action to advance past these phases).
     *
     * START_GAME: both sides independently need to call sendStartGame().
     * TEAM_SETUP: after sendTeamSetupLoad(), the SETUP handler must call sendEndTurn().
     */
    private boolean isActiveStateContinueDialog(IDialogParameter param) {
        if (param == null) return false;
        DialogId id = param.getId();
        return id == DialogId.START_GAME || id == DialogId.TEAM_SETUP;
    }

    /**
     * Before calling sendEndTurn(SETUP, ...), ensure that the number of players on the field
     * meets the server's requirement: all available players must be placed if fewer than
     * maxPlayersOnField are on the field and available players exist.
     *
     * This handles the case where a saved setup only specifies positions for a subset of
     * available players (e.g. a 9-player setup when 11 are available).
     *
     * Players already at valid field positions are left in place.  Surplus reserve players
     * (beyond maxPlayersOnField) are NOT placed, so a 16-player roster with 11 max on field
     * will still only place 11.
     */
    private void fixReservePlayersForSetup(Team homeTeam, FieldModel fieldModel, Game game) {
        int maxOnField = UtilGameOption.getIntOption(game, GameOptionId.MAX_PLAYERS_ON_FIELD);

        // First pass: count players already on the field and collect occupied squares
        int onField = 0;
        java.util.Set<FieldCoordinate> occupied = new java.util.HashSet<>();
        for (Player<?> player : homeTeam.getPlayers()) {
            PlayerState ps = fieldModel.getPlayerState(player);
            if (!ps.canBeMovedDuringSetup()) continue;
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            if (coord != null && !coord.isBoxCoordinate()
                    && FieldCoordinateBounds.HALF_HOME.isInBounds(coord)) {
                occupied.add(coord);
                onField++;
            }
        }

        // Count total available players this drive
        int available = 0;
        for (Player<?> player : homeTeam.getPlayers()) {
            if (fieldModel.getPlayerState(player).canBeSetUpNextDrive()) available++;
        }

        // Target: field min(maxOnField, available) players.
        // If already at or above target, don't add (surplus roster players stay on bench).
        int target = Math.min(maxOnField, available);
        if (onField >= target) return;

        // Second pass: place reserve players until target is reached
        for (Player<?> player : homeTeam.getPlayers()) {
            if (onField >= target) break;
            PlayerState ps = fieldModel.getPlayerState(player);
            if (!ps.canBeSetUpNextDrive()) continue;
            FieldCoordinate coord = fieldModel.getPlayerCoordinate(player);
            boolean alreadyOnField = (coord != null && !coord.isBoxCoordinate()
                    && FieldCoordinateBounds.HALF_HOME.isInBounds(coord));
            if (alreadyOnField) continue;

            FieldCoordinate pos = findEmptySetupPosition(occupied);
            if (pos != null) {
                fieldModel.setPlayerCoordinate(player, pos);
                fieldModel.setPlayerState(player, ps.changeBase(PlayerState.STANDING).changeActive(true));
                occupied.add(pos);
                onField++;
            }
        }
    }

    /**
     * Returns an empty field position in the home half, preferring the LOS (x=12, y=4..10)
     * and then retreating inward. Returns null if no position is found.
     */
    private FieldCoordinate findEmptySetupPosition(java.util.Set<FieldCoordinate> occupied) {
        // Try LOS (x=12, y=4..10) — must have at least 3 here
        for (int y = 4; y <= 10; y++) {
            FieldCoordinate c = new FieldCoordinate(12, y);
            if (!occupied.contains(c)) return c;
        }
        // Fill inward toward own endzone
        for (int x = 11; x >= 1; x--) {
            for (int y = 4; y <= 10; y++) {
                FieldCoordinate c = new FieldCoordinate(x, y);
                if (!occupied.contains(c)) return c;
            }
        }
        // Wide zones as last resort (avoid if possible)
        for (int x = 12; x >= 1; x--) {
            for (int y = 0; y <= 3; y++) {
                FieldCoordinate c = new FieldCoordinate(x, y);
                if (!occupied.contains(c)) return c;
            }
            for (int y = 11; y <= 14; y++) {
                FieldCoordinate c = new FieldCoordinate(x, y);
                if (!occupied.contains(c)) return c;
            }
        }
        return null;
    }

    // ── Visualization helpers ──────────────────────────────────────────────────────

    private void savePlayerSelectionVisualization(Game game, List<Player<?>> players,
            List<PlayerAction> actions, double[] scores, FieldCoordinate chosenCoord) {
        try {
            double[] probs = PolicySampler.softmax(scores, T_PLAYER);
            Map<FieldCoordinate, Double> probMap = new LinkedHashMap<>();
            for (int i = 0; i < players.size(); i++) {
                Player<?> p = players.get(i);
                if (p == null) continue;
                FieldCoordinate coord = game.getFieldModel().getPlayerCoordinate(p);
                if (coord != null) {
                    // Sum probabilities if the same square appears multiple times
                    probMap.merge(coord, probs[i], Double::sum);
                }
            }
            if (!probMap.isEmpty()) {
                double endPct = probs[probs.length - 1] * 100.0;
                String vizLabel = String.format("SELECT_PLAYER [end=%.1f%%]", endPct);
                BoardVisualizer.save(client, vizLabel, probMap, chosenCoord);
            }
        } catch (Exception e) {
            // Visualization errors should never crash the engine
        }
    }

    private void saveKickoffVisualization(List<FieldCoordinate> squares, double[] scores) {
        try {
            double[] probs = PolicySampler.softmax(scores, T_KICKOFF);
            Map<FieldCoordinate, Double> probMap = new LinkedHashMap<>();
            for (int i = 0; i < squares.size(); i++) {
                probMap.put(squares.get(i), probs[i]);
            }
            BoardVisualizer.save(client, "KICKOFF", probMap);
        } catch (Exception e) {
            // Visualization errors should never crash the engine
        }
    }

    /**
     * Visualizes a set of opponent players as block/foul/blitz targets.
     * Colors and labels come from softmax of {@code scores}; acting player is highlighted cyan.
     */
    private void savePlayerScoreVisualization(Game game, Player<?>[] targets, double[] scores,
            FieldCoordinate actorCoord, String label) {
        try {
            double[] probs = PolicySampler.softmax(scores, T_PLAYER);
            Map<FieldCoordinate, Double> colorMap = new LinkedHashMap<>();
            Map<FieldCoordinate, Double> labelMap = new LinkedHashMap<>();
            for (int i = 0; i < targets.length; i++) {
                FieldCoordinate coord = game.getFieldModel().getPlayerCoordinate(targets[i]);
                if (coord != null) {
                    colorMap.put(coord, probs[i]);
                    labelMap.put(coord, probs[i]);
                }
            }
            if (!colorMap.isEmpty()) {
                BoardVisualizer.save(client, label, colorMap, labelMap, actorCoord);
            }
        } catch (Exception e) {
            // Visualization errors must not crash engine
        }
    }

    /** Count home-team players that are still active and standing (not yet activated). */
    private int countRemainingActivations(Game game) {
        int count = 0;
        FieldModel fm = game.getFieldModel();
        for (Player<?> p : game.getTeamHome().getPlayers()) {
            PlayerState ps = fm.getPlayerState(p);
            if (ps != null && ps.isActive() && ps.getBase() == PlayerState.STANDING) count++;
        }
        return count;
    }

    /**
     * Produces a board snapshot for relevant dialog decisions (block dice, reroll).
     * Called before ScriptedStrategy.respondToDialog so the image reflects the state
     * at the moment the decision is made.
     */
    private void visualizeDialogDecision(IDialogParameter param, Game game) {
        if (param == null || game == null) return;
        try {
            switch (param.getId()) {
                case BLOCK_ROLL: {
                    DialogBlockRollParameter d = (DialogBlockRollParameter) param;
                    visualizeBlockRoll(d.getBlockRoll(), Math.abs(d.getNrOfDice()), game, true, "BLOCK_ROLL");
                    break;
                }
                case BLOCK_ROLL_PARTIAL_RE_ROLL: {
                    DialogBlockRollPartialReRollParameter d = (DialogBlockRollPartialReRollParameter) param;
                    visualizeBlockRoll(d.getBlockRoll(), Math.abs(d.getNrOfDice()), game, true, "BLOCK_ROLL_PARTIAL");
                    break;
                }
                case BLOCK_ROLL_PROPERTIES: {
                    DialogBlockRollPropertiesParameter d = (DialogBlockRollPropertiesParameter) param;
                    visualizeBlockRoll(d.getBlockRoll(), Math.abs(d.getNrOfDice()), game, true, "BLOCK_ROLL_PROPS");
                    break;
                }
                case OPPONENT_BLOCK_SELECTION: {
                    DialogOpponentBlockSelectionParameter d = (DialogOpponentBlockSelectionParameter) param;
                    if (d.getBlockRolls() != null && !d.getBlockRolls().isEmpty()) {
                        int[] roll = d.getBlockRolls().get(0).getBlockRoll();
                        if (roll != null) {
                            visualizeBlockRoll(roll, roll.length, game, false, "OPP_BLOCK_SEL");
                        }
                    }
                    break;
                }
                case OPPONENT_BLOCK_SELECTION_PROPERTIES: {
                    DialogOpponentBlockSelectionPropertiesParameter d =
                        (DialogOpponentBlockSelectionPropertiesParameter) param;
                    if (d.getBlockRolls() != null && !d.getBlockRolls().isEmpty()) {
                        int[] roll = d.getBlockRolls().get(0).getBlockRoll();
                        if (roll != null) {
                            visualizeBlockRoll(roll, roll.length, game, false, "OPP_BLOCK_SEL_PROPS");
                        }
                    }
                    break;
                }
                case RE_ROLL: {
                    DialogReRollParameter rr = (DialogReRollParameter) param;
                    boolean isFumble = rr.isFumble();
                    double scoreYes = isFumble ? 85 : 15;
                    double scoreNo  = isFumble ? 15 : 85;
                    double[] rrProbs = PolicySampler.softmax(new double[]{scoreYes, scoreNo}, 0.20);
                    String action = rr.getReRolledAction() != null ? rr.getReRolledAction().toString() : "?";
                    String vizLabel = String.format("REROLL %s yes=%.0f%% no=%.0f%%",
                        action, rrProbs[0] * 100.0, rrProbs[1] * 100.0);
                    FieldCoordinate playerCoord = null;
                    if (rr.getPlayerId() != null) {
                        Player<?> p = game.getPlayerById(rr.getPlayerId());
                        if (p != null) playerCoord = game.getFieldModel().getPlayerCoordinate(p);
                    }
                    if (playerCoord != null) {
                        Map<FieldCoordinate, Double> cm = new LinkedHashMap<>();
                        cm.put(playerCoord, rrProbs[0]);
                        BoardVisualizer.save(client, vizLabel, cm, cm, null);
                    }
                    break;
                }
                default:
                    break;
            }
        } catch (Exception e) {
            // Visualization must not crash the engine
        }
    }

    private void visualizeBlockRoll(int[] roll, int n, Game game, boolean attackerPick, String label) {
        if (roll == null || n == 0) return;
        double[] dScores = ScriptedStrategy.scoreDice(roll, n, game, attackerPick);
        double[] probs = PolicySampler.softmax(dScores, 0.10);
        int best = PolicySampler.argmax(dScores);
        StringBuilder dice = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            if (i > 0) dice.append(',');
            dice.append(roll[i]).append(String.format("(%.0f%%)", probs[i] * 100.0));
        }
        dice.append("] best=").append(roll[best]);
        String vizLabel = label + " " + dice;

        // Highlight defender (block target) square in green; actor in cyan
        FieldCoordinate defCoord = (game.getDefender() != null)
            ? game.getFieldModel().getPlayerCoordinate(game.getDefender()) : null;
        FieldCoordinate atkCoord = (game.getActingPlayer() != null && game.getActingPlayer().getPlayer() != null)
            ? game.getFieldModel().getPlayerCoordinate(game.getActingPlayer().getPlayer()) : null;
        // For opponent selection: roles are swapped visually — highlight attacker
        FieldCoordinate highlightSquare = attackerPick ? defCoord : atkCoord;
        FieldCoordinate cyanSquare      = attackerPick ? atkCoord : defCoord;
        if (highlightSquare != null) {
            Map<FieldCoordinate, Double> cm = new LinkedHashMap<>();
            cm.put(highlightSquare, probs[best]);
            BoardVisualizer.save(client, vizLabel, cm, cm, cyanSquare);
        }
    }

    // ── Game result logging ───────────────────────────────────────────────────────

    private void logGameResult(Game game) {
        if (gameResultLogged || game == null) return;
        GameResult result = game.getGameResult();
        String homeTeam = game.getTeamHome() != null ? game.getTeamHome().getName() : "Home";
        String awayTeam = game.getTeamAway() != null ? game.getTeamAway().getName() : "Away";
        int homeScore = (result != null) ? result.getScoreHome() : 0;
        int awayScore = (result != null) ? result.getScoreAway() : 0;
        String outcome;
        if (homeScore > awayScore) outcome = homeTeam + " WIN";
        else if (awayScore > homeScore) outcome = awayTeam + " WIN";
        else outcome = "DRAW";
        System.out.println("[GAME_RESULT] " + homeTeam + " " + homeScore
            + " - " + awayScore + " " + awayTeam + " | " + outcome);
        System.out.flush();
        gameResultLogged = true;
    }

    // ── Random active state dispatch ──────────────────────────────────────────────

    private void handleActiveStateRandom(ClientStateId stateId, Game game) {
        switch (stateId) {
            case START_GAME:
                client.getCommunication().sendStartGame();
                break;

            case WAIT_FOR_OPPONENT:
            case WAIT_FOR_SETUP:
                break;

            case SPECTATE:
            case REPLAY:
                logGameResult(game);
                break;

            case SETUP: {
                highKickDone = false;
                Team homeTeam = game.getTeamHome();
                FieldModel fieldModel = game.getFieldModel();
                if (homeTeam != null && fieldModel != null && hasPlayersOnField(homeTeam, fieldModel)) {
                    fixReservePlayersForSetup(homeTeam, fieldModel, game);
                    client.getCommunication().sendEndTurn(TurnMode.SETUP, homeTeam, fieldModel);
                } else {
                    client.getCommunication().sendTeamSetupLoad(null);
                }
                break;
            }

            case KICKOFF: {
                // Pick a random square in the opponent half
                int x = 13 + random.nextInt(13); // 13..25
                int y = random.nextInt(15);       // 0..14
                client.getCommunication().sendKickoff(new FieldCoordinate(x, y));
                break;
            }

            case SELECT_PLAYER: {
                // Pick a random standing home player with MOVE, or end turn
                Team homeTeam = game.getTeamHome();
                FieldModel fieldModel = game.getFieldModel();
                List<Player<?>> active = new ArrayList<>();
                for (Player<?> p : homeTeam.getPlayers()) {
                    PlayerState ps = fieldModel.getPlayerState(p);
                    FieldCoordinate coord = fieldModel.getPlayerCoordinate(p);
                    if (ps != null && ps.isActive() && ps.getBase() == PlayerState.STANDING && coord != null) {
                        active.add(p);
                    }
                }
                if (active.isEmpty()) {
                    client.getCommunication().sendEndTurn(game.getTurnMode());
                } else {
                    Player<?> chosen = active.get(random.nextInt(active.size()));
                    client.getCommunication().sendActingPlayer(chosen, PlayerAction.MOVE, false);
                }
                break;
            }

            case MOVE: {
                MoveSquare[] squares = game.getFieldModel().getMoveSquares();
                ActingPlayer actingPlayer = game.getActingPlayer();
                if (squares == null || squares.length == 0 || actingPlayer == null) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    // Pick a random move square (prefer non-risky), then send the full path
                    List<MoveSquare> safe = new ArrayList<>();
                    for (MoveSquare sq : squares) {
                        if (!sq.isDodging() && !sq.isGoingForIt()) safe.add(sq);
                    }
                    MoveSquare chosen = safe.isEmpty()
                        ? squares[random.nextInt(squares.length)]
                        : safe.get(random.nextInt(safe.size()));
                    Map<FieldCoordinate, PathProbabilityFinder.PathEntry> pathMap =
                        PathProbabilityFinder.findAllPaths(game, actingPlayer);
                    PathProbabilityFinder.PathEntry entry = pathMap.get(chosen.getCoordinate());
                    FieldCoordinate[] path = (entry != null)
                        ? entry.path
                        : new FieldCoordinate[]{chosen.getCoordinate()};
                    FieldCoordinate from = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                    client.getCommunication().sendPlayerMove(
                        actingPlayer.getPlayerId(), from, path, null);
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
                if (actingPlayer.hasBlocked()) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                    break;
                }
                FieldCoordinate pos = game.getFieldModel().getPlayerCoordinate(actingPlayer.getPlayer());
                Player<?>[] targets = UtilPlayer.findAdjacentBlockablePlayers(game, game.getTeamAway(), pos);
                if (targets == null || targets.length == 0) {
                    client.getCommunication().sendActingPlayer(null, null, false);
                } else {
                    Player<?> defender = targets[random.nextInt(targets.length)];
                    client.getCommunication().sendBlock(
                        actingPlayer.getPlayerId(), defender, false, false, false, false, false);
                }
                break;
            }

            case PUSHBACK: {
                PushbackSquare[] psquares = game.getFieldModel().getPushbackSquares();
                if (psquares != null && psquares.length > 0) {
                    PushbackSquare chosen = psquares[random.nextInt(psquares.length)];
                    String pushedId = game.getDefender() != null ? game.getDefender().getId() : null;
                    if (pushedId != null) {
                        client.getCommunication().sendPushback(new Pushback(pushedId, chosen.getCoordinate()));
                    }
                }
                break;
            }

            case TOUCHBACK: {
                Team homeTeam = game.getTeamHome();
                FieldModel fieldModel = game.getFieldModel();
                for (Player<?> p : homeTeam.getPlayers()) {
                    PlayerState ps = fieldModel.getPlayerState(p);
                    FieldCoordinate coord = fieldModel.getPlayerCoordinate(p);
                    if (ps != null && ps.getBase() == PlayerState.STANDING && coord != null) {
                        client.getCommunication().sendTouchback(coord);
                        return;
                    }
                }
                client.getCommunication().sendTouchback(null);
                break;
            }

            case HIGH_KICK:
                // Just end without repositioning in random mode
                client.getCommunication().sendEndTurn(game.getTurnMode());
                break;

            case SOLID_DEFENCE:
            case KICKOFF_RETURN:
            case QUICK_SNAP:
                client.getCommunication().sendEndTurn(game.getTurnMode());
                break;

            default:
                client.getCommunication().sendActingPlayer(null, null, false);
                break;
        }
    }

    public void stop() {
        running = false;
    }
}
