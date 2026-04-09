import subprocess
import time
import os
from pathlib import Path
from config import REPO_DIR, SERVER_DIR, server_port

MARIADB_DATADIR = "/opt/homebrew/var/mysql"
MARIADB_SOCKET = "/tmp/mysql.sock"
MYSQL_BIN = os.environ.get("MYSQL_BIN", "/opt/homebrew/bin/mysql")
MYSQLD_SAFE = "/opt/homebrew/opt/mariadb/bin/mysqld_safe"
SERVER_LOG = Path("/tmp/ffb-server.log")


def is_mariadb_running() -> bool:
    result = subprocess.run(["pgrep", "-x", "mariadbd"], capture_output=True)
    if result.returncode == 0:
        return True
    result = subprocess.run(["pgrep", "-x", "mysqld"], capture_output=True)
    return result.returncode == 0


def is_ffb_server_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "FantasyFootballServer"], capture_output=True
    )
    return result.returncode == 0


def start_mariadb() -> dict:
    if is_mariadb_running():
        return {"status": "already_running"}
    env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
    subprocess.Popen(
        [MYSQLD_SAFE, f"--datadir={MARIADB_DATADIR}", f"--socket={MARIADB_SOCKET}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    for _ in range(15):
        time.sleep(1)
        r = subprocess.run(
            [MYSQL_BIN, "-u", "root", "-S", MARIADB_SOCKET, "-e", "SELECT 1;"],
            capture_output=True,
        )
        if r.returncode == 0:
            return {"status": "started"}
    return {"status": "failed", "error": "MariaDB did not respond within 15s"}


def start_ffb_server() -> dict:
    if is_ffb_server_running():
        return {"status": "already_running"}
    jar = SERVER_DIR / "target" / "FantasyFootballServer.jar"
    if not jar.exists():
        return {
            "status": "failed",
            "error": f"JAR not found at {jar}. Run build_project() first.",
        }
    log_file = open(SERVER_LOG, "w")
    subprocess.Popen(
        ["java", "-jar", str(jar), "standalone", "-inifile", "server.ini"],
        cwd=str(SERVER_DIR),
        stdout=log_file,
        stderr=log_file,
    )
    for _ in range(15):
        time.sleep(1)
        if SERVER_LOG.exists():
            content = SERVER_LOG.read_text(errors="replace")
            if "running on port" in content.lower() or f":{server_port()}" in content:
                return {"status": "started", "log": str(SERVER_LOG)}
    return {
        "status": "failed",
        "error": "Server did not log startup message within 15s",
        "log": str(SERVER_LOG),
    }


def stop_ffb_server() -> dict:
    result = subprocess.run(["pkill", "-f", "FantasyFootballServer"], capture_output=True)
    if result.returncode == 0:
        return {"status": "stopped"}
    return {"status": "not_running"}


def server_status() -> dict:
    return {
        "mariadb_running": is_mariadb_running(),
        "ffb_server_running": is_ffb_server_running(),
        "server_port": server_port(),
        "server_log": str(SERVER_LOG),
    }


def build_project(skip_tests: bool = True) -> dict:
    args = ["mvn", "clean", "install"]
    if skip_tests:
        args.append("-DskipTests")
    result = subprocess.run(
        args, cwd=str(REPO_DIR), capture_output=True, text=True, timeout=600
    )
    return {
        "returncode": result.returncode,
        "success": result.returncode == 0,
        "stdout_tail": result.stdout[-3000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }
