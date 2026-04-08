#!/bin/bash
# Start a local AI-vs-AI Blood Bowl game.
# Both sides are controlled by AI agents; no human UI windows open.
# Usage: ./play-ai-vs-ai.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$REPO_DIR/ffb-server"
CLIENT_DIR="$REPO_DIR/ffb-client"
AI_DIR="$REPO_DIR/ffb-ai"
CLIENT_DIST="/tmp/ffb-client-dist"
MARIADB_DATADIR="/opt/homebrew/var/mysql"
MARIADB_SOCKET="/tmp/mysql.sock"
MYSQL="/opt/homebrew/bin/mysql"

export PATH="/opt/homebrew/bin:$PATH"

# ── 1. MariaDB ───────────────────────────────────────────────────────────────

if ! pgrep -x mariadbd > /dev/null 2>&1; then
  echo ">> Starting MariaDB..."
  /opt/homebrew/opt/mariadb/bin/mysqld_safe \
    --datadir="$MARIADB_DATADIR" --socket="$MARIADB_SOCKET" > /dev/null 2>&1 &
  for i in $(seq 1 10); do
    sleep 1
    "$MYSQL" -u root -S "$MARIADB_SOCKET" -e "SELECT 1;" > /dev/null 2>&1 && break
  done
fi

# ── 2. Build if JARs are missing ─────────────────────────────────────────────

NEED_BUILD=false
[ ! -f "$SERVER_DIR/target/FantasyFootballServer.jar" ] && NEED_BUILD=true
[ ! -f "$CLIENT_DIR/target/FantasyFootballClient.jar" ] && NEED_BUILD=true
[ ! -f "$AI_DIR/target/ffb-ai-3.2.0.jar" ] && NEED_BUILD=true

if $NEED_BUILD; then
  echo ">> Building project (this takes ~30s)..."
  cd "$REPO_DIR" && mvn clean install -DskipTests -q
fi

# ── 3. Ensure server lib/ is next to the server JAR ──────────────────────────

if [ ! -d "$SERVER_DIR/target/lib" ]; then
  echo ">> Assembling server lib/..."
  cd "$REPO_DIR" && mvn -pl ffb-server assembly:single -DskipTests -q
  TMP_DIST=$(mktemp -d)
  unzip -q "$SERVER_DIR/target/ffb-server.zip" -d "$TMP_DIST"
  cp -r "$TMP_DIST/lib" "$SERVER_DIR/target/"
  rm -rf "$TMP_DIST"
fi

# ── 4. Create database if it doesn't exist ───────────────────────────────────

DB_EXISTS=$("$MYSQL" -u root -S "$MARIADB_SOCKET" \
  -e "SHOW DATABASES LIKE 'ffblive';" 2>/dev/null | grep -c "ffblive" || true)
if [ "$DB_EXISTS" = "0" ]; then
  echo ">> Creating database ffblive..."
  "$MYSQL" -u root -S "$MARIADB_SOCKET" \
    -e "CREATE DATABASE ffblive CHARACTER SET utf8;"
  "$MYSQL" -u root -S "$MARIADB_SOCKET" \
    -e "CREATE USER IF NOT EXISTS 'root'@'127.0.0.1' IDENTIFIED BY '';
        GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1' WITH GRANT OPTION;
        FLUSH PRIVILEGES;" 2>/dev/null || true
fi

# ── 5. Initialize DB schema if not done yet ───────────────────────────────────

INITIALIZED=$("$MYSQL" -u root -h 127.0.0.1 -P 3306 ffblive \
  -e "SHOW TABLES LIKE 'ffb_coaches';" 2>/dev/null | grep -c "ffb_coaches" || true)
if [ "$INITIALIZED" = "0" ]; then
  echo ">> Initializing database schema..."
  cd "$SERVER_DIR" && java -jar target/FantasyFootballServer.jar \
    standalone initDb -inifile server.ini > /dev/null 2>&1
  echo ">> Setting coach passwords to 'test'..."
  "$MYSQL" -u root -h 127.0.0.1 -P 3306 ffblive \
    -e "UPDATE ffb_coaches SET password=MD5('test');" 2>/dev/null
fi

# ── 6. Start FFB server if not running ───────────────────────────────────────

if ! pgrep -f "FantasyFootballServer" > /dev/null 2>&1; then
  echo ">> Starting FFB server on port 22227..."
  cd "$SERVER_DIR" && java -jar target/FantasyFootballServer.jar \
    standalone -inifile server.ini > /tmp/ffb-server.log 2>&1 &
  sleep 3
  grep -q "running on port" /tmp/ffb-server.log || \
    { echo "ERROR: Server failed to start. Check /tmp/ffb-server.log"; exit 1; }
fi

# ── 7. Unpack client dist if needed ──────────────────────────────────────────

if [ ! -f "$CLIENT_DIST/FantasyFootballClient.jar" ]; then
  echo ">> Unpacking client..."
  rm -rf "$CLIENT_DIST"
  mkdir -p "$CLIENT_DIST"
  unzip -q "$CLIENT_DIR/target/ffb-client.zip" -d "$CLIENT_DIST"
fi

# ── 8. Build AI classpath ────────────────────────────────────────────────────

CP=$(find "$CLIENT_DIST/lib" -name "*.jar" | tr '\n' ':')
AI_JAR="$AI_DIR/target/ffb-ai-3.2.0.jar"
AI_CP="${CP}${CLIENT_DIST}/FantasyFootballClient.jar:${CLIENT_DIST}/FantasyFootballClientResources.jar:${AI_JAR}"

# ── 9. Kill any stale AI agents from previous runs ───────────────────────────

pkill -f "com.fumbbl.ffb.ai.AiMain" 2>/dev/null || true
sleep 1

# ── 10. Launch both AI agents ────────────────────────────────────────────────

echo ">> Launching AI agent 1 (Kalimar)..."
cd "$CLIENT_DIST" && java -cp "$AI_CP" com.fumbbl.ffb.ai.AiMain \
  -coach Kalimar -password test -server localhost -port 22227 -home \
  -teamId teamLizardmanKalimar -teamName "Kalimar's Lizards" \
  > /tmp/ffb-ai-kalimar.log 2>&1 &
AI1_PID=$!

sleep 1

echo ">> Launching AI agent 2 (BattleLore — random)..."
cd "$CLIENT_DIST" && java -cp "$AI_CP" com.fumbbl.ffb.ai.AiMain \
  -coach BattleLore -password test -server localhost -port 22227 \
  -teamId teamHumanBattleLore -teamName "BattleLore's Humans" -random \
  > /tmp/ffb-ai-battlelore.log 2>&1 &
AI2_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Both AI agents are running headlessly."
echo ""
echo "  AI 1 (Kalimar) log:     /tmp/ffb-ai-kalimar.log"
echo "  AI 2 (BattleLore) log:  /tmp/ffb-ai-battlelore.log"
echo "  Server log:             /tmp/ffb-server.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for both agents to finish and then clean up
wait $AI1_PID $AI2_PID 2>/dev/null || true
echo ">> Game over. Stopping server..."
pkill -f "FantasyFootballServer" 2>/dev/null || true
