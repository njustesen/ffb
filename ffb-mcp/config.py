import configparser
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = Path(os.environ.get("FFB_REPO_DIR", str(SCRIPT_DIR.parent)))
SERVER_DIR = REPO_DIR / "ffb-server"
INI_PATH = SERVER_DIR / "server.ini"

_ini: dict | None = None


def get_ini() -> dict:
    global _ini
    if _ini is None:
        text = INI_PATH.read_text(encoding="utf-8")
        cp = configparser.RawConfigParser()
        cp.read_string("[server]\n" + text)
        _ini = dict(cp.items("server"))
    return _ini


def server_port() -> int:
    return int(get_ini().get("server.port", "22227"))


def server_base_url() -> str:
    base = get_ini().get("server.base", "http://localhost")
    return f"{base}:{server_port()}"


def admin_password_md5() -> str:
    """Returns the already-MD5-hashed admin password hex string from server.ini."""
    return get_ini()["admin.password"]


def db_user() -> str:
    return get_ini().get("db.user", "root")


def db_password() -> str:
    return get_ini().get("db.password", "")


def db_host_port_name() -> tuple[str, int, str]:
    url = get_ini().get("db.url", "jdbc:mysql://127.0.0.1:3306/ffblive")
    m = re.search(r"mysql://([^:/]+):(\d+)/(\w+)", url)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return "127.0.0.1", 3306, "ffblive"
