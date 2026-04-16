package com.fumbbl.ffb.ai.mcts;

import com.fumbbl.ffb.PlayerAction;
import com.fumbbl.ffb.model.Player;

/**
 * A candidate player activation considered by the Blood Bowl MCTS agent.
 *
 * <p>Represents either an end-turn sentinel ({@link #END_TURN}) or a specific
 * player + action pair that can be injected via {@code ClientCommandActingPlayer}.
 */
public final class BbAction {

    /** Sentinel representing "end the current team's turn". */
    public static final BbAction END_TURN = new BbAction(null, null);

    public final Player<?> player;   // null ⇒ end turn
    public final PlayerAction action; // null ⇒ end turn

    public BbAction(Player<?> player, PlayerAction action) {
        this.player = player;
        this.action = action;
    }

    public boolean isEndTurn() {
        return player == null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BbAction)) return false;
        BbAction other = (BbAction) o;
        if (player == null) return other.player == null;
        if (other.player == null) return false;
        return player.getId().equals(other.player.getId()) && action == other.action;
    }

    @Override
    public int hashCode() {
        if (player == null) return 0;
        int h = player.getId().hashCode();
        return 31 * h + (action != null ? action.hashCode() : 0);
    }

    @Override
    public String toString() {
        if (isEndTurn()) return "END_TURN";
        return action + "(" + player.getId() + ")";
    }
}
