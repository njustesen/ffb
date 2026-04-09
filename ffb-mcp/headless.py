import subprocess
from config import REPO_DIR


def run_match_runner(games_per_condition: int = 50) -> dict:
    """
    Run MatchRunner headless benchmark via mvn exec:java.
    No server or DB needed. ~500ms/game after JIT warmup.
    Runs 4 agent pairings x games_per_condition games each.
    """
    args_str = f"{REPO_DIR} {games_per_condition}"
    timeout = games_per_condition * 4 * 10 + 120
    result = subprocess.run(
        [
            "mvn", "-pl", "ffb-ai", "exec:java",
            "-Dexec.mainClass=com.fumbbl.ffb.ai.simulation.MatchRunner",
            f"-Dexec.args={args_str}",
        ],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "success": result.returncode == 0,
        "output": result.stdout,
        "stderr_tail": result.stderr[-3000:] if result.stderr else "",
    }


def run_replay_generator(
    games: int = 100,
    output_dir: str | None = None,
    temperature: float = 0.5,
    races: list[str] | None = None,
    threads: int | None = None,
) -> dict:
    """
    Generate .ffbr replay files via generate-replays.sh (no server/DB needed).
    The script handles building ffb-ai before running.
    Available races: amazon, chaos, chaos_dwarf, dwarf, elf, goblin, high_elf,
      human, lizardman, necromantic, norse, orc, skaven, undead, underworld,
      vampire, wood_elf.
    """
    out = output_dir or str(REPO_DIR / "replays")
    cmd = [
        "bash", str(REPO_DIR / "generate-replays.sh"),
        "--games", str(games),
        "--output", out,
        "--temperature", str(temperature),
    ]
    if races:
        cmd += ["--races", ",".join(races)]
    if threads is not None:
        cmd += ["--threads", str(threads)]

    timeout = games * 15 + 180  # generous: 15s/game + build time
    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "success": result.returncode == 0,
        "output_dir": out,
        "output": result.stdout,
        "stderr_tail": result.stderr[-3000:] if result.stderr else "",
    }
