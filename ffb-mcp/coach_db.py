import subprocess
import os
from config import db_user, db_password, db_host_port_name

MYSQL_BIN = os.environ.get("MYSQL_BIN", "/opt/homebrew/bin/mysql")


def _run_mysql(sql: str) -> str:
    host, port, db = db_host_port_name()
    user = db_user()
    pw = db_password()

    cmd = [MYSQL_BIN, "-u", user, f"-h{host}", f"-P{port}"]
    if pw:
        cmd.append(f"-p{pw}")
    cmd += [db, "-e", sql, "--skip-column-names", "-r"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MySQL error: {result.stderr.strip()}")
    return result.stdout.strip()


def list_coaches() -> list[dict]:
    out = _run_mysql("SELECT name, password FROM ffb_coaches ORDER BY name;")
    coaches = []
    for line in out.splitlines():
        if "\t" in line:
            name, pw = line.split("\t", 1)
            coaches.append({"name": name, "password_hash": pw})
        elif line.strip():
            coaches.append({"name": line.strip()})
    return coaches


def create_coach(name: str, password: str) -> None:
    safe_name = name.replace("'", "''")
    safe_pw = password.replace("'", "''")
    _run_mysql(
        f"INSERT INTO ffb_coaches (name, password) VALUES ('{safe_name}', MD5('{safe_pw}'));"
    )


def delete_coach(name: str) -> None:
    safe_name = name.replace("'", "''")
    _run_mysql(f"DELETE FROM ffb_coaches WHERE name='{safe_name}';")


def set_coach_password(name: str, new_password: str) -> None:
    safe_name = name.replace("'", "''")
    safe_pw = new_password.replace("'", "''")
    _run_mysql(
        f"UPDATE ffb_coaches SET password=MD5('{safe_pw}') WHERE name='{safe_name}';"
    )
