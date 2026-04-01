package com.fumbbl.ffb.ai.simulation;

import com.eclipsesource.json.JsonObject;
import com.fumbbl.ffb.FactoryManager;
import com.fumbbl.ffb.factory.IFactorySource;
import com.fumbbl.ffb.model.Game;

/**
 * Utility for cloning a {@link Game} state via JSON round-trip serialisation.
 *
 * This is the foundation for forward-search-based AI planning.  Currently only
 * the clone operation is implemented.
 *
 * TODO: Add a ServerStub that can execute game Steps on the cloned game so that
 *       the AI can simulate the result of an action without sending commands to
 *       the real server.
 */
public class GameSimulator {

    private final IFactorySource factorySource;
    private final FactoryManager factoryManager;

    public GameSimulator(IFactorySource factorySource, FactoryManager factoryManager) {
        this.factorySource = factorySource;
        this.factoryManager = factoryManager;
    }

    /**
     * Produce a deep clone of {@code source} by serialising it to JSON and
     * deserialising into a fresh {@link Game} instance.
     *
     * @param source the game state to clone
     * @return a new, independent copy of the game state
     */
    public Game cloneGame(Game source) {
        JsonObject json = source.toJsonValue();
        Game clone = new Game(factorySource, factoryManager);
        return clone.initFrom(factorySource, json);
    }
}
