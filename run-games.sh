#!/bin/bash
# Run N AI-vs-AI games (scripted Kalimar vs random BattleLore) and report win rates.
# Usage: ./run-games.sh [N]   (default N=5)

set -euo pipefail

N=${1:-5}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$REPO_DIR/ffb-server"
CLIENT_DIR="$REPO_DIR/ffb-client"
AI_DIR="$REPO_DIR/ffb-ai"
CLIENT_DIST="/tmp/ffb-client-dist"
MARIADB_SOCKET="/tmp/mysql.sock"
MYSQL="/opt/homebrew/bin/mysql"

export PATH="/opt/homebrew/bin:$PATH"

# ── MariaDB ───────────────────────────────────────────────────────────────────
if ! pgrep -x mariadbd > /dev/null 2>&1; then
  echo ">> Starting MariaDB..."
  /opt/homebrew/opt/mariadb/bin/mysqld_safe \
    --datadir="/opt/homebrew/var/mysql" --socket="$MARIADB_SOCKET" > /dev/null 2>&1 &
  for i in $(seq 1 10); do sleep 1
    "$MYSQL" -u root -S "$MARIADB_SOCKET" -e "SELECT 1;" > /dev/null 2>&1 && break
  done
fi

# ── Build if needed ───────────────────────────────────────────────────────────
NEED_BUILD=false
[ ! -f "$SERVER_DIR/target/FantasyFootballServer.jar" ] && NEED_BUILD=true
[ ! -f "$AI_DIR/target/ffb-ai-3.2.0.jar" ]            && NEED_BUILD=true
if $NEED_BUILD; then
  echo ">> Building project..."
  cd "$REPO_DIR" && mvn clean install -DskipTests -q
fi

# ── Server assembly ───────────────────────────────────────────────────────────
if [ ! -d "$SERVER_DIR/target/lib" ]; then
  echo ">> Assembling server lib/..."
  cd "$REPO_DIR" && mvn -pl ffb-server assembly:single -DskipTests -q
  TMP_DIST=$(mktemp -d)
  unzip -q "$SERVER_DIR/target/ffb-server.zip" -d "$TMP_DIST"
  cp -r "$TMP_DIST/lib" "$SERVER_DIR/target/"
  rm -rf "$TMP_DIST"
fi

# ── DB setup ─────────────────────────────────────────────────────────────────
DB_EXISTS=$("$MYSQL" -u root -S "$MARIADB_SOCKET" -e "SHOW DATABASES LIKE 'ffblive';" 2>/dev/null | grep -c "ffblive" || true)
if [ "$DB_EXISTS" = "0" ]; then
  echo ">> Creating ffblive database..."
  "$MYSQL" -u root -S "$MARIADB_SOCKET" -e "CREATE DATABASE ffblive CHARACTER SET utf8;"
  "$MYSQL" -u root -S "$MARIADB_SOCKET" \
    -e "CREATE USER IF NOT EXISTS 'root'@'127.0.0.1' IDENTIFIED BY '';
        GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1' WITH GRANT OPTION;
        FLUSH PRIVILEGES;" 2>/dev/null || true
fi

INITIALIZED=$("$MYSQL" -u root -h 127.0.0.1 -P 3306 ffblive \
  -e "SHOW TABLES LIKE 'ffb_coaches';" 2>/dev/null | grep -c "ffb_coaches" || true)
if [ "$INITIALIZED" = "0" ]; then
  echo ">> Initializing schema..."
  cd "$SERVER_DIR" && java -jar target/FantasyFootballServer.jar standalone initDb -inifile server.ini > /dev/null 2>&1
  "$MYSQL" -u root -h 127.0.0.1 -P 3306 ffblive -e "UPDATE ffb_coaches SET password=MD5('test');" 2>/dev/null
fi

# ── Client dist ───────────────────────────────────────────────────────────────
if [ ! -f "$CLIENT_DIST/FantasyFootballClient.jar" ]; then
  echo ">> Unpacking client..."
  rm -rf "$CLIENT_DIST" && mkdir -p "$CLIENT_DIST"
  unzip -q "$CLIENT_DIR/target/ffb-client.zip" -d "$CLIENT_DIST"
fi

CP=$(find "$CLIENT_DIST/lib" -name "*.jar" | tr '\n' ':')
AI_JAR="$AI_DIR/target/ffb-ai-3.2.0.jar"
AI_CP="${CP}${CLIENT_DIST}/FantasyFootballClient.jar:${CLIENT_DIST}/FantasyFootballClientResources.jar:${AI_JAR}"

# ── Run N games ───────────────────────────────────────────────────────────────
SCRIPTED_WINS=0
RANDOM_WINS=0
DRAWS=0
ERRORS=0

for game_num in $(seq 1 "$N"); do
  echo ""
  echo "══════════════ GAME $game_num / $N ══════════════"

  # Kill any stale processes
  pkill -f "com.fumbbl.ffb.ai.AiMain" 2>/dev/null || true
  pkill -f "FantasyFootballServer" 2>/dev/null || true
  sleep 2

  # Start server fresh
  cd "$SERVER_DIR" && java -jar target/FantasyFootballServer.jar \
    standalone -inifile server.ini > /tmp/ffb-server.log 2>&1 &
  SERVER_PID=$!
  sleep 3
  if ! grep -q "running on port" /tmp/ffb-server.log 2>/dev/null; then
    echo "ERROR: Server failed to start for game $game_num"
    ERRORS=$((ERRORS + 1))
    continue
  fi

  # Start scripted agent (Kalimar = home, creates game)
  cd "$CLIENT_DIST" && java -cp "$AI_CP" com.fumbbl.ffb.ai.AiMain \
    -coach Kalimar -password test -server localhost -port 22227 -home \
    -teamId teamLizardmanKalimar -teamName "Kalimar's Lizards" \
    > /tmp/ffb-ai-kalimar.log 2>&1 &
  KALIMAR_PID=$!
  sleep 1

  # Start random agent (BattleLore = away, joins)
  cd "$CLIENT_DIST" && java -cp "$AI_CP" com.fumbbl.ffb.ai.AiMain \
    -coach BattleLore -password test -server localhost -port 22227 \
    -teamId teamHumanBattleLore -teamName "BattleLore's Humans" -random \
    > /tmp/ffb-ai-battlelore.log 2>&1 &
  BATTLELORE_PID=$!

  # Wait for game to finish (up to 10 minutes)
  GAME_DONE=false
  for i in $(seq 1 120); do
    sleep 5
    if grep -q "GAME_RESULT" /tmp/ffb-ai-kalimar.log 2>/dev/null; then
      GAME_DONE=true
      break
    fi
  done

  if ! $GAME_DONE; then
    echo "TIMEOUT: Game $game_num did not finish in time"
    ERRORS=$((ERRORS + 1))
  else
    RESULT_LINE=$(grep "GAME_RESULT" /tmp/ffb-ai-kalimar.log | tail -1)
    echo "$RESULT_LINE"
    if echo "$RESULT_LINE" | grep -q "Kalimar.*WIN\|Lizards.*WIN"; then
      SCRIPTED_WINS=$((SCRIPTED_WINS + 1))
    elif echo "$RESULT_LINE" | grep -q "BattleLore.*WIN\|Humans.*WIN"; then
      RANDOM_WINS=$((RANDOM_WINS + 1))
    else
      DRAWS=$((DRAWS + 1))
    fi
  fi

  # Clean up
  kill $KALIMAR_PID $BATTLELORE_PID $SERVER_PID 2>/dev/null || true
  sleep 2
done

echo ""
echo "════════════════════════════════════════"
echo "  RESULTS after $N games:"
echo "  Scripted (Kalimar) wins: $SCRIPTED_WINS"
echo "  Random (BattleLore) wins: $RANDOM_WINS"
echo "  Draws: $DRAWS"
echo "  Errors/timeouts: $ERRORS"
echo "════════════════════════════════════════"

if [ $N -gt 0 ] && [ $((SCRIPTED_WINS + RANDOM_WINS + DRAWS)) -gt 0 ]; then
  echo "  Win rate: $SCRIPTED_WINS / $((SCRIPTED_WINS + RANDOM_WINS + DRAWS)) completed games"
fi
