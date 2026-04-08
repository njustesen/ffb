package com.fumbbl.ffb.ai.client;

import com.fumbbl.ffb.client.ClientParameters;
import com.fumbbl.ffb.client.FantasyFootballClientAwt;

import java.io.IOException;

/**
 * Headless AI client that extends the standard AWT client.
 *
 * The window is hidden immediately after construction.  A daemon thread
 * running {@link AiDecisionEngine} drives all in-game decisions.
 */
public class AiClient extends FantasyFootballClientAwt {

    private final boolean home;

    public AiClient(ClientParameters parameters, String password, boolean home) throws IOException {
        this(parameters, password, home, false);
    }

    public AiClient(ClientParameters parameters, String password, boolean home, boolean useRandom) throws IOException {
        super(parameters);
        this.home = home;
        // Hide the Swing window — the AI does not need to render anything.
        getUserInterface().setVisible(false);
        // Start the decision loop as a daemon thread.
        AiDecisionEngine engine = new AiDecisionEngine(this, password, home, useRandom);
        Thread engineThread = new Thread(engine, "AI-DecisionEngine");
        engineThread.setDaemon(true);
        engineThread.start();
    }

    public boolean isHome() {
        return home;
    }
}
