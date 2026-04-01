package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.client.net.ClientCommunication;
import com.fumbbl.ffb.model.FieldModel;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.net.commands.ClientCommand;
import com.fumbbl.ffb.net.commands.ClientCommandEndTurn;

import java.util.HashMap;
import java.util.Map;

/**
 * A {@link ClientCommunication} subclass that captures outbound commands
 * instead of sending them over WebSocket.
 *
 * <p>Used by {@link SimulationLoop} to translate {@link com.fumbbl.ffb.ai.strategy.RandomStrategy}
 * dialog responses into {@link com.fumbbl.ffb.server.net.ReceivedCommand} objects that can
 * be injected directly into the server-side {@link com.fumbbl.ffb.server.GameState}.
 */
class CapturingClientCommunication extends ClientCommunication {

    private ClientCommand capturedCommand;

    CapturingClientCommunication() {
        super(null);
    }

    /** Capture instead of sending over the network. */
    @Override
    protected void send(ClientCommand command) {
        this.capturedCommand = command;
    }

    /**
     * Override to avoid {@code getClient().logWithOutGameId()} NPE.
     * Produces {@link ClientCommandEndTurn} with null coordinate map.
     */
    @Override
    public void sendEndTurn(TurnMode turnMode) {
        this.capturedCommand = new ClientCommandEndTurn(turnMode, null);
    }

    /**
     * Override to avoid {@code getClient().logWithOutGameId()} NPE and
     * {@code playerCoordinates()} being private.  Replicates the private
     * helper's logic using {@link PlayerState#canBeMovedDuringSetup()}.
     */
    @Override
    public void sendEndTurn(TurnMode turnMode, Team team, FieldModel fieldModel) {
        Map<String, FieldCoordinate> coords = new HashMap<>();
        if (team != null && fieldModel != null) {
            for (Player<?> player : team.getPlayers()) {
                PlayerState ps = fieldModel.getPlayerState(player);
                if (ps != null && ps.canBeMovedDuringSetup()) {
                    coords.put(player.getId(), fieldModel.getPlayerCoordinate(player));
                }
            }
        }
        this.capturedCommand = new ClientCommandEndTurn(turnMode, coords);
    }

    ClientCommand getCapturedCommand() {
        return capturedCommand;
    }

    void clearCaptured() {
        capturedCommand = null;
    }
}
