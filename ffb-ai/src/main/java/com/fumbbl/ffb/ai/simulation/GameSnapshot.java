package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.LeaderState;
import com.fumbbl.ffb.MoveSquare;
import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.inducement.Inducement;
import com.fumbbl.ffb.inducement.InducementType;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.InducementSet;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.model.TurnData;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Shallow snapshot of mutable {@link Game} state at an {@code INIT_SELECTING}
 * phase-1 decision point.
 *
 * <p>Used by the MCTS snapshot loop: take <em>once</em> per
 * {@link com.fumbbl.ffb.ai.mcts.BbMctsSearch#selectActivation} call, then
 * restore in O(number of players) per iteration instead of re-running the
 * full JSON clone (~2–3 ms).
 *
 * <h3>What is captured</h3>
 * <ul>
 *   <li>FieldModel: per-player coordinates + states, ball coordinate/flags,
 *       move squares (residual from previous activation).</li>
 *   <li>TurnData (home + away): all boolean flags, counts, inducement uses.</li>
 *   <li>Game: turnMode, lastTurnMode, homePlaying, half, passCoordinate,
 *       finished, defenderId/Action, throwerId/Action, waitingForOpponent.</li>
 *   <li>GameResult: scoreHome, scoreAway.</li>
 * </ul>
 *
 * <h3>What is NOT captured</h3>
 * Cards, prayers, blood spots, track numbers, dice decorations — these are
 * either display-only or are not changed within a single activation sequence.
 * Extend if needed.
 */
public final class GameSnapshot {

    // ── FieldModel ────────────────────────────────────────────────────────────

    private final Map<String, FieldCoordinate> coordByPlayerId;
    private final Map<String, PlayerState>     stateByPlayerId;
    private final FieldCoordinate ballCoordinate;
    private final boolean         ballInPlay;
    private final boolean         ballMoving;
    private final MoveSquare[]    moveSquares;

    // ── Game ──────────────────────────────────────────────────────────────────

    private final TurnMode       turnMode;
    private final TurnMode       lastTurnMode;
    private final boolean        homePlaying;
    private final int            half;
    private final FieldCoordinate passCoordinate;
    private final Date           finished;
    private final String         defenderId;
    private final PlayerAction   defenderAction;
    private final String         throwerId;
    private final PlayerAction   throwerAction;
    private final boolean        waitingForOpponent;

    // ── TurnData ──────────────────────────────────────────────────────────────

    private final TdSnap tdHome;
    private final TdSnap tdAway;

    // ── GameResult ────────────────────────────────────────────────────────────

    private final int scoreHome;
    private final int scoreAway;

    // ── Construction ──────────────────────────────────────────────────────────

    private GameSnapshot(Game game) {
        FieldModel fm = game.getFieldModel();

        // FieldModel
        coordByPlayerId = new HashMap<>();
        stateByPlayerId = new HashMap<>();
        for (Team team : new Team[]{game.getTeamHome(), game.getTeamAway()}) {
            for (Player<?> p : team.getPlayers()) {
                FieldCoordinate coord = fm.getPlayerCoordinate(p);
                if (coord != null) coordByPlayerId.put(p.getId(), coord);
                PlayerState state = fm.getPlayerState(p);
                if (state != null) stateByPlayerId.put(p.getId(), state);
            }
        }
        ballCoordinate = fm.getBallCoordinate();
        ballInPlay     = fm.isBallInPlay();
        ballMoving     = fm.isBallMoving();
        moveSquares    = fm.getMoveSquares();   // returns array copy

        // Game
        turnMode           = game.getTurnMode();
        lastTurnMode       = game.getLastTurnMode();
        homePlaying        = game.isHomePlaying();
        half               = game.getHalf();
        passCoordinate     = game.getPassCoordinate();
        finished           = game.getFinished();
        defenderId         = game.getDefenderId();
        defenderAction     = game.getDefenderAction();
        throwerId          = game.getThrowerId();
        throwerAction      = game.getThrowerAction();
        waitingForOpponent = game.isWaitingForOpponent();

        // TurnData
        tdHome = new TdSnap(game.getTurnDataHome());
        tdAway = new TdSnap(game.getTurnDataAway());

        // Scores
        scoreHome = game.getGameResult().getScoreHome();
        scoreAway = game.getGameResult().getScoreAway();
    }

    /** Take a snapshot of the given game state. */
    public static GameSnapshot take(Game game) {
        return new GameSnapshot(game);
    }

    /** Restore the game to the snapshotted state. */
    public void restore(Game game) {
        FieldModel fm = game.getFieldModel();

        // FieldModel: restore player positions and states
        for (Team team : new Team[]{game.getTeamHome(), game.getTeamAway()}) {
            for (Player<?> p : team.getPlayers()) {
                FieldCoordinate snappedCoord = coordByPlayerId.get(p.getId());
                FieldCoordinate currentCoord = fm.getPlayerCoordinate(p);
                if (snappedCoord != null) {
                    // Player was on field: move to correct position
                    fm.setPlayerCoordinate(p, snappedCoord);
                } else if (currentCoord != null) {
                    // Player was off field but is now on it: remove them
                    fm.remove(p);
                }
                PlayerState snappedState = stateByPlayerId.get(p.getId());
                if (snappedState != null) {
                    fm.setPlayerState(p, snappedState);
                }
            }
        }

        // Ball
        fm.setBallCoordinate(ballCoordinate);
        fm.setBallInPlay(ballInPlay);
        fm.setBallMoving(ballMoving);

        // Move squares
        fm.clearMoveSquares();
        fm.add(moveSquares);

        // Game
        game.setTurnMode(turnMode);
        game.setLastTurnMode(lastTurnMode);
        game.setHomePlaying(homePlaying);
        game.setHalf(half);
        game.setPassCoordinate(passCoordinate);
        game.setFinished(finished);
        game.setDefenderId(defenderId);
        game.setDefenderAction(defenderAction);
        game.setThrowerId(throwerId);
        game.setThrowerAction(throwerAction);
        game.setWaitingForOpponent(waitingForOpponent);

        // TurnData
        tdHome.restore(game.getTurnDataHome());
        tdAway.restore(game.getTurnDataAway());

        // Scores
        game.getGameResult().getTeamResultHome().setScore(scoreHome);
        game.getGameResult().getTeamResultAway().setScore(scoreAway);

        // ActingPlayer: at INIT_SELECTING phase 1, playerId is null → reset to clean state
        game.getActingPlayer().setPlayerId(null);
    }

    // ── TurnData snapshot ─────────────────────────────────────────────────────

    private static final class TdSnap {

        final int     turnNr, reRolls, singleUseReRolls, reRollsBCOD, reRollsStar, reRollsPUtC;
        final int     apothecaries, wanderingApothecaries, plagueDoctors;
        final boolean blitzUsed, foulUsed, reRollUsed, handOverUsed, passUsed, coachBanned;
        final boolean ttmUsed, ktmUsed, bombUsed, secureTheBallUsed, puntUsed;
        final boolean firstTurnAfterKickoff, turnStarted;
        final LeaderState leaderState;
        /** Snapshot of fUses per inducement type (only types that exist in the set). */
        final Map<InducementType, int[]> inducementSnapshot; // int[]{value, uses}

        TdSnap(TurnData td) {
            turnNr               = td.getTurnNr();
            reRolls              = td.getReRolls();
            singleUseReRolls     = td.getSingleUseReRolls();
            reRollsBCOD          = td.getReRollsBrilliantCoachingOneDrive();
            reRollsStar          = td.getReRollShowStarOneDrive();
            reRollsPUtC          = td.getReRollsPumpUpTheCrowdOneDrive();
            apothecaries         = td.getApothecaries();
            wanderingApothecaries = td.getWanderingApothecaries();
            plagueDoctors        = td.getPlagueDoctors();
            blitzUsed            = td.isBlitzUsed();
            foulUsed             = td.isFoulUsed();
            reRollUsed           = td.isReRollUsed();
            handOverUsed         = td.isHandOverUsed();
            passUsed             = td.isPassUsed();
            coachBanned          = td.isCoachBanned();
            ttmUsed              = td.isTtmUsed();
            ktmUsed              = td.isKtmUsed();
            bombUsed             = td.isBombUsed();
            secureTheBallUsed    = td.isSecureTheBallUsed();
            puntUsed             = td.isPuntUsed();
            firstTurnAfterKickoff = td.isFirstTurnAfterKickoff();
            turnStarted          = td.isTurnStarted();
            leaderState          = td.getLeaderState();

            // Inducements: capture value + uses per type
            inducementSnapshot = new HashMap<>();
            for (Map.Entry<InducementType, Inducement> e : td.getInducementSet().getInducementMapping().entrySet()) {
                inducementSnapshot.put(e.getKey(), new int[]{e.getValue().getValue(), e.getValue().getUses()});
            }
        }

        void restore(TurnData td) {
            td.setTurnNr(turnNr);
            td.setReRolls(reRolls);
            td.setSingleUseReRolls(singleUseReRolls);
            td.setReRollsBrilliantCoachingOneDrive(reRollsBCOD);
            td.setReRollShowStarOneDrive(reRollsStar);
            td.setReRollsPumpUpTheCrowdOneDrive(reRollsPUtC);
            td.setApothecaries(apothecaries);
            td.setWanderingApothecaries(wanderingApothecaries);
            td.setPlagueDoctors(plagueDoctors);
            td.setBlitzUsed(blitzUsed);
            td.setFoulUsed(foulUsed);
            td.setReRollUsed(reRollUsed);
            td.setHandOverUsed(handOverUsed);
            td.setPassUsed(passUsed);
            td.setCoachBanned(coachBanned);
            td.setTtmUsed(ttmUsed);
            td.setKtmUsed(ktmUsed);
            td.setBombUsed(bombUsed);
            td.setSecureTheBallUsed(secureTheBallUsed);
            td.setPuntUsed(puntUsed);
            td.setFirstTurnAfterKickoff(firstTurnAfterKickoff);
            td.setTurnStarted(turnStarted);
            td.setLeaderState(leaderState);

            // Restore inducement uses
            InducementSet iset = td.getInducementSet();
            for (Map.Entry<InducementType, int[]> e : inducementSnapshot.entrySet()) {
                Inducement ind = iset.get(e.getKey()); // returns a copy
                if (ind != null) {
                    ind.setUses(e.getValue()[1]);
                    iset.addInducement(ind); // replaces the entry in the map
                }
            }
        }
    }
}
