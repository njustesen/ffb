package com.fumbbl.ffb.ai.simulation;

import com.eclipsesource.json.JsonArray;
import com.eclipsesource.json.JsonObject;
import com.eclipsesource.json.JsonValue;
import com.fumbbl.ffb.json.LZString;
import com.fumbbl.ffb.json.UtilJson;
import com.fumbbl.ffb.net.commands.ClientCommandReplay;

import javax.websocket.ClientEndpoint;
import javax.websocket.CloseReason;
import javax.websocket.ContainerProvider;
import javax.websocket.OnClose;
import javax.websocket.OnError;
import javax.websocket.OnMessage;
import javax.websocket.OnOpen;
import javax.websocket.Session;
import javax.websocket.WebSocketContainer;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Downloads a single FUMBBL replay via WebSocket and saves it as a
 * gzip-compressed JSON file ({@code .ffbr}).
 *
 * <p>Connects to {@code ws://fumbbl.com:22223/command}, sends a
 * {@link ClientCommandReplay} (LZ-compressed, binary frame), and accumulates
 * all {@code ServerCommandReplay} chunks until {@code lastCommand=true}.
 *
 * <pre>{@code
 * Usage:
 *   FumbblReplayDownloader [--probe-only] <replayId> [<outputPath>]
 *                          [--rules BB2025] [--after YYYY-MM-DD] [--before YYYY-MM-DD]
 * }</pre>
 *
 * <p>Exit codes: {@code 0} = success, {@code 1} = not found/error,
 * {@code 2} = filtered (rules or date mismatch).
 *
 * <p>On success, prints a one-line JSON metadata object to stdout so the
 * Python orchestrator can populate its manifest.
 */
public class FumbblReplayDownloader {

    private static final String HOST = "fumbbl.com";
    private static final int PORT = 22223;

    public static void main(String[] args) throws Exception {
        Logger.getLogger("").setLevel(Level.WARNING);
        Logger.getLogger("org.eclipse.jetty").setLevel(Level.OFF);

        boolean probeOnly = false;
        long replayId = -1;
        String outputPath = null;
        String rulesFilter = "BB2025";
        String afterDate = null;
        String beforeDate = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--probe-only": probeOnly = true; break;
                case "--rules":     rulesFilter = args[++i]; break;
                case "--after":     afterDate   = args[++i]; break;
                case "--before":    beforeDate  = args[++i]; break;
                default:
                    if (!args[i].startsWith("-")) {
                        if (replayId < 0) {
                            replayId = Long.parseLong(args[i]);
                        } else if (outputPath == null) {
                            outputPath = args[i];
                        }
                    }
            }
        }

        if (replayId < 0) {
            System.err.println("Usage: FumbblReplayDownloader [--probe-only] <replayId> [<outputPath>] [options]");
            System.exit(1);
        }
        if (!probeOnly && outputPath == null) {
            System.err.println("Output path required (omit only with --probe-only)");
            System.exit(1);
        }

        BlockingQueue<JsonValue> queue = new LinkedBlockingQueue<>();

        WebSocketContainer container = ContainerProvider.getWebSocketContainer();
        container.setDefaultMaxSessionIdleTimeout(120_000L);
        container.setDefaultMaxTextMessageBufferSize(8 * 1024 * 1024);
        container.setDefaultMaxBinaryMessageBufferSize(16 * 1024 * 1024);

        URI uri = new URI("ws", null, HOST, PORT, "/command", null, null);
        RawJsonEndpoint endpoint = new RawJsonEndpoint(queue);

        Session session;
        try {
            session = container.connectToServer(endpoint, uri);
        } catch (Exception e) {
            System.exit(1);
            return;
        }

        if (!session.isOpen()) {
            System.exit(1);
        }

        // Send ClientCommandReplay — LZ-compressed as a binary frame (required by FUMBBL server)
        ClientCommandReplay cmd = new ClientCommandReplay(replayId, 0, "Unknown");
        String cmdJson = cmd.toJsonValue().toString();
        String compressed = LZString.compressToUTF16(cmdJson);
        byte[] bytes = compressed.getBytes(StandardCharsets.UTF_8);
        session.getAsyncRemote().sendBinary(ByteBuffer.wrap(bytes));

        // Probe-only: short timeout; full download: longer timeout per chunk
        int firstTimeoutSecs = probeOnly ? 8 : 20;
        int chunkTimeoutSecs = probeOnly ? 8 : 60;

        List<JsonValue> allCommands = new ArrayList<>();
        String gameStarted = "unknown";
        String gameRules   = "unknown";
        String homeRace    = "unknown";
        String awayRace    = "unknown";
        // gameStateJson saved so it can be prepended to allCommands after filters pass
        JsonValue gameStateJson = null;
        boolean metadataExtracted = false;
        boolean done = false;
        int timeoutSecs = firstTimeoutSecs;

        while (!done) {
            JsonValue msg = queue.poll(timeoutSecs, TimeUnit.SECONDS);
            timeoutSecs = chunkTimeoutSecs; // subsequent chunks use the longer timeout

            if (msg == null) {
                // Timed out — no more data
                if (allCommands.isEmpty()) {
                    closeQuietly(session);
                    System.exit(1);
                }
                break; // partial replay — save what we have
            }

            JsonObject replayMsg = msg.asObject();
            String netCmdId = replayMsg.getString("netCommandId", "");

            // Handle serverGameState (metadata extraction)
            if ("serverGameState".equals(netCmdId)) {
                if (!metadataExtracted) {
                    metadataExtracted = true;

                    JsonValue gameVal = replayMsg.get("game");
                    if (gameVal != null && !gameVal.isNull()) {
                        JsonObject gameObj = gameVal.asObject();

                        // Date: stored as "yyyy-MM-dd'T'HH:mm:ss.SSS" — take first 10 chars
                        JsonValue sv = gameObj.get("started");
                        if (sv != null && !sv.isNull() && sv.isString()) {
                            String s = sv.asString();
                            gameStarted = s.length() >= 10 ? s.substring(0, 10) : s;
                        }

                        // Rules version: search the gameOptions JSON string
                        JsonValue optsVal = gameObj.get("gameOptions");
                        if (optsVal != null && !optsVal.isNull()) {
                            String opts = optsVal.toString();
                            if      (opts.contains("BB2025")) gameRules = "BB2025";
                            else if (opts.contains("BB2020")) gameRules = "BB2020";
                            else if (opts.contains("BB2016")) gameRules = "BB2016";
                        }

                        // Team races
                        JsonValue th = gameObj.get("teamHome");
                        if (th != null && !th.isNull()) {
                            homeRace = th.asObject().getString("race", "unknown");
                        }
                        JsonValue ta = gameObj.get("teamAway");
                        if (ta != null && !ta.isNull()) {
                            awayRace = ta.asObject().getString("race", "unknown");
                        }
                    }

                    // In probe-only mode: valid if we got game state
                    if (probeOnly) {
                        closeQuietly(session);
                        System.exit(0);
                    }

                    // Apply filters before downloading the rest
                    if (rulesFilter != null && !rulesFilter.equals(gameRules)) {
                        closeQuietly(session);
                        System.exit(2);
                    }
                    if (afterDate != null && gameStarted.compareTo(afterDate) < 0) {
                        closeQuietly(session);
                        System.exit(2);
                    }
                    if (beforeDate != null && gameStarted.compareTo(beforeDate) > 0) {
                        closeQuietly(session);
                        System.exit(2);
                    }

                    // Filters passed — save game state so it can head the command array
                    gameStateJson = msg;
                }
                continue; // game state goes into metadata only (added to allCommands after loop)
            }

            // Handle serverReplay chunks
            if ("serverReplay".equals(netCmdId)) {
                JsonValue commandArrayVal = replayMsg.get("commandArray");
                if (commandArrayVal != null && !commandArrayVal.isNull()) {
                    JsonArray commandArray = commandArrayVal.asArray();
                    for (int i = 0; i < commandArray.size(); i++) {
                        allCommands.add(commandArray.get(i));
                    }
                }

                boolean isLast = replayMsg.getBoolean("lastCommand", false);
                if (isLast) {
                    done = true;
                }
            }
            // Other message types (status, chat, etc.) are silently ignored
        }

        closeQuietly(session);

        if (allCommands.isEmpty()) {
            System.exit(1);
        }

        // Build output: {replayId, source, commandArray}
        // commandArray starts with serverGameState so the parser can reconstruct the game
        JsonObject output = new JsonObject();
        output.add("replayId", replayId);
        output.add("source", "fumbbl");
        JsonArray arr = new JsonArray();
        if (gameStateJson != null) {
            arr.add(gameStateJson);
        }
        for (JsonValue jv : allCommands) {
            arr.add(jv);
        }
        output.add("commandArray", arr);

        byte[] gz = UtilJson.gzip(output);
        File outFile = new File(outputPath);
        File parent = outFile.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(outFile.toPath()))) {
            out.write(gz);
        }

        // Print metadata JSON to stdout for the Python orchestrator
        JsonObject meta = new JsonObject();
        meta.add("id",       replayId);
        meta.add("started",  gameStarted);
        meta.add("rules",    gameRules);
        meta.add("home",     homeRace);
        meta.add("away",     awayRace);
        meta.add("commands", allCommands.size() + (gameStateJson != null ? 1 : 0));
        System.out.println(meta);
        System.exit(0);
    }

    private static void closeQuietly(Session session) {
        try { session.close(); } catch (Exception ignored) {}
    }

    // ── WebSocket endpoint ────────────────────────────────────────────────────

    /**
     * WebSocket endpoint that decodes LZ-compressed binary frames from the
     * FUMBBL server, accumulates multi-frame messages, and enqueues parsed
     * JSON objects for the main thread to process.
     */
    @ClientEndpoint
    public static class RawJsonEndpoint {

        private final BlockingQueue<JsonValue> queue;

        /** Accumulator for multi-frame binary messages. */
        private final java.io.ByteArrayOutputStream frameBuffer = new java.io.ByteArrayOutputStream();

        public RawJsonEndpoint(BlockingQueue<JsonValue> queue) {
            this.queue = queue;
        }

        @OnOpen
        public void onOpen(Session session) {
            // Connected — nothing to do here
        }

        /**
         * Binary messages — the FUMBBL server sends LZ-compressed UTF-16 encoded JSON
         * as potentially multiple binary WebSocket frames. Accumulate until last=true,
         * then decode and enqueue.
         */
        @OnMessage
        public void onBinary(byte[] data, boolean last, Session session) {
            frameBuffer.write(data, 0, data.length);
            if (!last) return; // wait for remaining frames

            byte[] fullBytes = frameBuffer.toByteArray();
            frameBuffer.reset();

            String raw = new String(fullBytes, StandardCharsets.UTF_8);

            // Try LZ decompression first (FUMBBL server uses compressToUTF16)
            String json = LZString.decompressFromUTF16(raw);
            if (json == null || json.isEmpty()) {
                // Not LZ-compressed — try as plain JSON (fallback)
                json = raw;
            }

            try {
                JsonValue jv = JsonValue.readFrom(json);
                JsonObject obj = jv.asObject();
                String id = obj.getString("netCommandId", "");
                // Enqueue game state and replay chunks; ignore unknown types
                if ("serverGameState".equals(id) || "serverReplay".equals(id)) {
                    queue.offer(jv);
                }
            } catch (Exception ignored) {
                // Malformed or non-JSON frame — discard
            }
        }

        /** Text messages — fallback if the server ever sends text frames. */
        @OnMessage
        public void onText(String text) {
            if (text == null || text.isEmpty()) return;
            try {
                JsonValue jv = JsonValue.readFrom(text);
                JsonObject obj = jv.asObject();
                String id = obj.getString("netCommandId", "");
                if ("serverGameState".equals(id) || "serverReplay".equals(id)) {
                    queue.offer(jv);
                }
            } catch (Exception ignored) {}
        }

        @OnError
        public void onError(Session session, Throwable t) {
            // Errors are handled by the main thread via queue timeout
        }

        @OnClose
        public void onClose(Session session, CloseReason reason) {
            // Closed — main thread will detect via queue timeout
        }
    }
}
