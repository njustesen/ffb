import hashlib
import re
import httpx
from config import server_base_url, admin_password_md5, get_ini


def compute_response(challenge_hex: str, md5_pw_hex: str) -> str:
    """
    Replicates PasswordChallenge.createResponse() from ffb-common.
    md5_pw_hex is already the hex-encoded MD5 of the plaintext password
    (stored as-is in server.ini admin.password). Hex-decode to 16 bytes,
    apply 16-byte XOR pads (NOT 64-byte HMAC padding), then double-MD5.
    """
    challenge = bytes.fromhex(challenge_hex)
    pw = bytes.fromhex(md5_pw_hex)
    opad = bytes(b ^ 0x5C for b in pw)
    ipad = bytes(b ^ 0x36 for b in pw)
    inner = hashlib.md5(ipad + challenge).digest()
    return hashlib.md5(opad + inner).hexdigest()


def _get_challenge(endpoint: str) -> str:
    url = f"{server_base_url()}/{endpoint}"
    resp = httpx.get(url, timeout=5.0)
    resp.raise_for_status()
    m = re.search(r"<challenge>([^<]+)</challenge>", resp.text)
    if not m:
        raise RuntimeError(f"No challenge in response from {url}: {resp.text[:200]}")
    return m.group(1)


def admin_request(url_key: str, *substitutions: str) -> str:
    """
    Call an admin endpoint. url_key is a server.ini key like 'admin.url.schedule'.
    substitutions fill $2, $3... ($1 is always the response hash).
    Returns raw response text (XML).
    """
    template = get_ini()[url_key]
    challenge_key = url_key.rsplit(".", 2)[0] + ".url.challenge"
    # Determine challenge endpoint: admin.url.challenge or gamestate.url.challenge
    challenge_endpoint = get_ini().get(challenge_key, "admin/challenge")
    challenge = _get_challenge(challenge_endpoint)
    response = compute_response(challenge, admin_password_md5())
    args = [response] + list(substitutions)
    url_path = template
    for i, val in enumerate(args, start=1):
        url_path = url_path.replace(f"${i}", val)
    full_url = f"{server_base_url()}/{url_path}"
    resp = httpx.get(full_url, timeout=15.0)
    resp.raise_for_status()
    return resp.text


def gamestate_request(url_key: str, *substitutions: str) -> str:
    """
    Call a gamestate endpoint. url_key is a server.ini key like 'gamestate.url.result'.
    """
    template = get_ini()[url_key]
    challenge = _get_challenge(get_ini()["gamestate.url.challenge"])
    response = compute_response(challenge, admin_password_md5())
    args = [response] + list(substitutions)
    url_path = template
    for i, val in enumerate(args, start=1):
        url_path = url_path.replace(f"${i}", val)
    full_url = f"{server_base_url()}/{url_path}"
    resp = httpx.get(full_url, timeout=15.0)
    resp.raise_for_status()
    return resp.text


def parse_game_list_xml(xml_text: str) -> list[dict]:
    """Parse admin list XML into a list of game dicts."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [{"raw": xml_text}]
    games = []
    for game_el in root.iter("game"):
        game = dict(game_el.attrib)
        teams = []
        for team_el in game_el.findall("team"):
            teams.append(dict(team_el.attrib))
        if teams:
            game["teams"] = teams
        games.append(game)
    return games
