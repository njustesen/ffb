import subprocess
import time
from config import REPO_DIR


def run_human_vs_human() -> dict:
    """Starts MariaDB + server + two human GUI clients via play.sh."""
    proc = subprocess.Popen(
        ["bash", str(REPO_DIR / "play.sh")],
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    return {
        "status": "launched",
        "pid": proc.pid,
        "note": "Two client windows should open. Enter game 'LocalGame', password 'test', click Create, then pick a team.",
    }


def run_human_vs_ai() -> dict:
    """Starts MariaDB + server + one human GUI client + AI agent via play.sh --ai."""
    proc = subprocess.Popen(
        ["bash", str(REPO_DIR / "play.sh"), "--ai"],
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    return {
        "status": "launched",
        "pid": proc.pid,
        "note": "Human client opens as Kalimar. AI joins headlessly as BattleLore. AI log: /tmp/ffb-ai.log",
    }


def run_ai_vs_ai() -> dict:
    """Starts MariaDB + server + two headless AI agents via play-ai-vs-ai.sh."""
    proc = subprocess.Popen(
        ["bash", str(REPO_DIR / "play-ai-vs-ai.sh")],
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    return {
        "status": "launched",
        "pid": proc.pid,
        "note": "Both AIs running headlessly. Logs: /tmp/ffb-ai-kalimar.log, /tmp/ffb-ai-battlelore.log",
    }


def run_games_batch(n: int = 5) -> dict:
    """
    Run N sequential AI-vs-AI games via run-games.sh. Blocking.
    Returns win rate statistics.
    """
    result = subprocess.run(
        ["bash", str(REPO_DIR / "run-games.sh"), str(n)],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
        timeout=n * 700,
    )
    return {
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
    }
