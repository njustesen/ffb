#!/bin/bash
# Start a local Human vs MCTS-AI Blood Bowl game.
# Usage: ./play-mcts.sh [--mcts-budget N]
#   --mcts-budget N  MCTS rollout iterations per activation (default: 10)
#
# In the human client window: game name "LocalGame", password "test", click Create,
# then pick a team. The MCTS AI will join automatically.

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

# Parse arguments
MCTS_BUDGET=10
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcts-budget) MCTS_BUDGET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

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

# ── 8. Launch human client and MCTS AI ───────────────────────────────────────

CP=$(find "$CLIENT_DIST/lib" -name "*.jar" | tr '\n' ':')
JAVA_CMD="java -cp FantasyFootballClient.jar:FantasyFootballClientResources.jar:${CP} com.fumbbl.ffb.client.FantasyFootballClientAwt"
AI_JAR="$AI_DIR/target/ffb-ai-3.2.0.jar"
AI_CP="${CP}${CLIENT_DIST}/FantasyFootballClient.jar:${CLIENT_DIST}/FantasyFootballClientResources.jar:${AI_JAR}"

echo ">> Launching human client (Kalimar)..."
cd "$CLIENT_DIST" && $JAVA_CMD -player -coach Kalimar -server localhost -port 22227 > /dev/null 2>&1 &

sleep 1

echo ">> Launching MCTS AI agent (BattleLore, budget=$MCTS_BUDGET)..."
pkill -f "com.fumbbl.ffb.ai.AiMain" 2>/dev/null || true
sleep 1
cd "$CLIENT_DIST" && java -cp "$AI_CP" com.fumbbl.ffb.ai.AiMain \
  -coach BattleLore -password test -server localhost -port 22227 \
  -teamId teamHumanBattleLore -teamName "BattleLore's Humans" \
  -mcts-budget "$MCTS_BUDGET" \
  > /tmp/ffb-ai-mcts.log 2>&1 &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Human client (Kalimar) should be open."
echo "  MCTS AI (BattleLore, budget=$MCTS_BUDGET) is running headlessly."
echo ""
echo "  In the human client window:"
echo "    Game name : LocalGame"
echo "    Password  : test"
echo "    Click     : Create"
echo ""
echo "  Then pick a team. The MCTS AI will join automatically."
echo "  MCTS AI log: /tmp/ffb-ai-mcts.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
