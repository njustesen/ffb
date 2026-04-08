package com.fumbbl.ffb.ai;

import com.fumbbl.ffb.ai.client.AiClient;
import com.fumbbl.ffb.client.ClientParameters;

/**
 * Entry point for the FFB AI agent.
 *
 * Usage:
 *   java -cp ... com.fumbbl.ffb.ai.AiMain [options]
 *
 * Options:
 *   -coach    &lt;name&gt;   Coach name (default: BattleLore)
 *   -password &lt;pwd&gt;   Password in plain text (default: test)
 *   -server   &lt;host&gt;  Server host (default: localhost)
 *   -port     &lt;port&gt;  Server port (default: 22227)
 *   -home              If present, create the game; otherwise join existing
 */
public class AiMain {

    public static void main(String[] args) {
        String coach = "BattleLore";
        String password = "test";
        String server = "localhost";
        int port = 22227;
        boolean home = false;
        boolean useRandom = false;
        String teamId = null;
        String teamName = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-coach":
                    coach = args[++i];
                    break;
                case "-password":
                    password = args[++i];
                    break;
                case "-server":
                    server = args[++i];
                    break;
                case "-port":
                    port = Integer.parseInt(args[++i]);
                    break;
                case "-home":
                    home = true;
                    break;
                case "-random":
                    useRandom = true;
                    break;
                case "-teamId":
                    teamId = args[++i];
                    break;
                case "-teamName":
                    teamName = args[++i];
                    break;
                default:
                    System.err.println("Unknown argument: " + args[i]);
                    break;
            }
        }

        // Build client args; include team if provided so LoginLogicModule includes it in the join.
        java.util.List<String> clientArgList = new java.util.ArrayList<>(java.util.Arrays.asList(
            "-player",
            "-coach", coach,
            "-server", server,
            "-port", String.valueOf(port)
        ));
        if (teamId != null) {
            clientArgList.add("-teamId");
            clientArgList.add(teamId);
        }
        if (teamName != null) {
            clientArgList.add("-teamName");
            clientArgList.add(teamName);
        }
        String[] clientArgs = clientArgList.toArray(new String[0]);

        ClientParameters parameters = ClientParameters.createValidParams(clientArgs);
        if (parameters == null) {
            System.err.println("Invalid client parameters.");
            System.err.println(ClientParameters.USAGE);
            System.exit(1);
        }

        try {
            AiClient client = new AiClient(parameters, password, home, useRandom);
            client.startClient();
            // Block the main thread so the JVM does not exit
            synchronized (AiMain.class) {
                AiMain.class.wait();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
