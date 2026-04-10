package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.IDialogParameter;
import com.fumbbl.ffb.ai.MoveDecisionEngine;
import com.fumbbl.ffb.ai.strategy.DecisionLog;
import com.fumbbl.ffb.model.ActingPlayer;
import com.fumbbl.ffb.model.Game;

/**
 * Callback interface for collecting per-decision training data during headless simulation.
 *
 * <p>Implementations receive decisions from three sources:
 * <ol>
 *   <li>{@link #onDialog} — a ScriptedStrategy dialog response</li>
 *   <li>{@link #onPlayerSelect} — a MoveDecisionEngine player-selection</li>
 *   <li>{@link #onMoveTarget} — a MoveDecisionEngine move-target selection</li>
 * </ol>
 */
public interface ITrainingDataCollector {

    /**
     * Called immediately after {@code ScriptedStrategy.respondToDialog()} returns.
     *
     * @param dialog    the dialog that was just responded to
     * @param game      current game state (before the command is injected)
     * @param log       scores and chosen indices from the ScriptedStrategy pick calls
     * @param agentMode "SCRIPTED_SAMPLE" or "SCRIPTED_ARGMAX"
     */
    void onDialog(IDialogParameter dialog, Game game, DecisionLog log, String agentMode);

    /**
     * Called after {@code MoveDecisionEngine.selectPlayer()} returns during INIT_SELECTING.
     *
     * @param game      current game state
     * @param sel       the player selection result (contains candidates + scores + chosen)
     * @param agentMode agent mode string
     */
    void onPlayerSelect(Game game, MoveDecisionEngine.PlayerSelection sel, String agentMode);

    /**
     * Called after {@code MoveDecisionEngine.selectMoveTarget()} returns.
     *
     * @param game      current game state
     * @param ap        the acting player
     * @param mr        the move result (contains candidates + scores + chosen)
     * @param agentMode agent mode string
     */
    void onMoveTarget(Game game, ActingPlayer ap, MoveDecisionEngine.MoveResult mr, String agentMode);
}
