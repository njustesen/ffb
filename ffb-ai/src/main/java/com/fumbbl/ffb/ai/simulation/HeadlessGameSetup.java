package com.fumbbl.ffb.ai.simulation;

import com.fumbbl.ffb.TeamSetup;
import com.fumbbl.ffb.TurnMode;
import com.fumbbl.ffb.Weather;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Team;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.RosterCache;
import com.fumbbl.ffb.server.TeamCache;
import com.fumbbl.ffb.FactoryType;
import com.fumbbl.ffb.server.factory.SequenceGeneratorFactory;
import com.fumbbl.ffb.server.step.generator.SequenceGenerator;
import com.fumbbl.ffb.server.step.generator.StartGame;
import com.fumbbl.ffb.server.util.UtilServerStartGame;
import com.fumbbl.ffb.server.util.UtilSkillBehaviours;
import com.fumbbl.ffb.util.UtilTeamValue;
import com.fumbbl.ffb.xml.XmlHandler;

import org.xml.sax.InputSource;

import java.io.File;
import java.io.IOException;

/**
 * Factory that creates a fully-initialized {@link GameState} ready for
 * headless simulation — no database, no WebSocket, no Swing.
 *
 * <p>Initialization order mirrors {@code ServerCommandHandlerJoinApproved}
 * (STANDALONE path): options → rules → skill behaviours → teams → StartGame sequence.
 */
public class HeadlessGameSetup {

    /**
     * Create a fresh {@link GameState} with two loaded teams and the StartGame
     * step sequence pushed onto the stack.  Call
     * {@link com.fumbbl.ffb.server.GameState#handleCommand} with a
     * {@code ClientCommandStartGame} to boot the simulation.
     *
     * @param server      pre-constructed headless server (may be reused across games)
     * @param homeTeamId  team ID for the home side (e.g. {@code "teamLizardmanKalimar"})
     * @param awayTeamId  team ID for the away side
     * @param serverDir   root of the {@code ffb-server} module (for {@code rosters/} and {@code teams/})
     */
    public static GameState create(
            HeadlessFantasyFootballServer server,
            String homeTeamId,
            String awayTeamId,
            File serverDir) throws IOException {

        GameState gameState = new GameState(server);

        // 1. Default game options (BB2025 ruleset, etc.)
        UtilServerStartGame.addDefaultGameOptions(gameState);

        // 2. Initialize rule-dependent members and rules
        Game game = gameState.getGame();
        game.setHomePlaying(true);
        game.setTurnMode(TurnMode.START_GAME);
        game.setTesting(true);   // skips replay-save in StepEndGame
        game.getFieldModel().setWeather(Weather.NICE);
        gameState.initRulesDependentMembers();
        game.initializeRules();
        UtilSkillBehaviours.registerBehaviours(game, server.getDebugLog());

        // 3. Load teams and rosters from XML (no DB)
        TeamCache teamCache = new TeamCache();
        teamCache.init(new File(serverDir, "teams"), server);
        RosterCache rosterCache = new RosterCache();
        rosterCache.init(new File(serverDir, "rosters"));

        Team homeTeam = teamCache.getTeamById(homeTeamId, game);
        homeTeam.updateRoster(rosterCache.getRosterForTeam(homeTeam, game), game.getRules());
        homeTeam.setTeamValue(UtilTeamValue.findTeamValue(homeTeam));

        Team awayTeam = teamCache.getTeamById(awayTeamId, game);
        awayTeam.updateRoster(rosterCache.getRosterForTeam(awayTeam, game), game.getRules());
        awayTeam.setTeamValue(UtilTeamValue.findTeamValue(awayTeam));

        // 5. Register teams on the game (sets game.fTeamHome / fTeamAway)
        server.getGameCache().addTeamToGame(gameState, homeTeam, true);
        server.getGameCache().addTeamToGame(gameState, awayTeam, false);

        // 6. Push StartGame sequence directly (bypassing ownership checks and DB
        //    marker loading that UtilServerStartGame.startGame() would otherwise do)
        SequenceGeneratorFactory seqFactory = game.getFactory(FactoryType.Factory.SEQUENCE_GENERATOR);
        ((StartGame) seqFactory.forName(SequenceGenerator.Type.StartGame.name()))
            .pushSequence(new SequenceGenerator.SequenceParams(gameState));

        return gameState;
    }

    /**
     * Parse a team setup XML file into a {@link TeamSetup}.
     *
     * @param game      any {@link Game} whose rules are initialized (used for XML context)
     * @param setupFile the XML file, e.g. {@code ffb-server/setups/setup_lizardman_Kalimar.xml}
     */
    public static TeamSetup loadTeamSetup(Game game, File setupFile) {
        TeamSetup setup = new TeamSetup();
        XmlHandler.parse(game, new InputSource(setupFile.toURI().toString()), setup);
        return setup;
    }
}
