import xml.etree.ElementTree as ET
from pathlib import Path
from config import SERVER_DIR

TEAMS_DIR = SERVER_DIR / "teams"
ROSTERS_DIR = SERVER_DIR / "rosters"


def list_teams() -> list[dict]:
    results = []
    for p in sorted(TEAMS_DIR.glob("*.xml")):
        try:
            root = ET.parse(p).getroot()
            results.append({
                "file": p.name,
                "id": root.get("id"),
                "coach": root.findtext("coach"),
                "name": root.findtext("name"),
                "race": root.findtext("race"),
                "rosterId": root.findtext("rosterId"),
                "reRolls": root.findtext("reRolls"),
                "teamRating": root.findtext("teamRating"),
                "playerCount": len(root.findall("player")),
            })
        except ET.ParseError as e:
            results.append({"file": p.name, "error": str(e)})
    return results


def get_team(team_id: str) -> dict | None:
    for p in TEAMS_DIR.glob("*.xml"):
        root = ET.parse(p).getroot()
        if root.get("id") == team_id:
            players = []
            for player in root.findall("player"):
                skills = [s.text for s in player.findall("skillList/skill") if s.text]
                injuries = [i.text for i in player.findall("injuryList/injury") if i.text]
                players.append({
                    "nr": player.get("nr"),
                    "id": player.get("id"),
                    "name": player.findtext("name"),
                    "positionId": player.findtext("positionId"),
                    "gender": player.findtext("gender"),
                    "skills": skills,
                    "injuries": injuries,
                    "status": player.findtext("status"),
                    "spps": _find_spps(player),
                })
            return {
                "id": root.get("id"),
                "file": p.name,
                "coach": root.findtext("coach"),
                "name": root.findtext("name"),
                "race": root.findtext("race"),
                "rosterId": root.findtext("rosterId"),
                "reRolls": root.findtext("reRolls"),
                "fanFactor": root.findtext("fanFactor"),
                "apothecaries": root.findtext("apothecaries"),
                "teamRating": root.findtext("teamRating"),
                "currentTeamValue": root.findtext("currentTeamValue"),
                "teamStrength": root.findtext("teamStrength"),
                "players": players,
            }
    return None


def _find_spps(player_el) -> str | None:
    ps = player_el.find("playerStatistics")
    if ps is not None:
        return ps.get("currentSpps")
    return None


def list_rosters() -> list[dict]:
    results = []
    for p in sorted(ROSTERS_DIR.glob("*.xml")):
        try:
            root = ET.parse(p).getroot()
            positions = []
            for pos in root.findall("position"):
                positions.append({
                    "id": pos.get("id"),
                    "name": pos.findtext("name"),
                    "type": pos.findtext("type"),
                    "quantity": pos.findtext("quantity"),
                    "cost": pos.findtext("cost"),
                    "ma": pos.findtext("movement"),
                    "st": pos.findtext("strength"),
                    "ag": pos.findtext("agility"),
                    "pa": pos.findtext("passing"),
                    "av": pos.findtext("armour"),
                    "skills": [s.text for s in pos.findall("skillList/skill") if s.text],
                })
            results.append({
                "file": p.name,
                "id": root.get("id"),
                "name": root.findtext("name"),
                "type": root.findtext("type"),
                "reRollCost": root.findtext("reRollCost"),
                "positions": positions,
            })
        except ET.ParseError as e:
            results.append({"file": p.name, "error": str(e)})
    return results


def get_roster(roster_id: str) -> dict | None:
    for p in ROSTERS_DIR.glob("*.xml"):
        root = ET.parse(p).getroot()
        if root.get("id") == roster_id:
            positions = []
            for pos in root.findall("position"):
                positions.append({
                    "id": pos.get("id"),
                    "name": pos.findtext("name"),
                    "type": pos.findtext("type"),
                    "quantity": pos.findtext("quantity"),
                    "cost": pos.findtext("cost"),
                    "ma": pos.findtext("movement"),
                    "st": pos.findtext("strength"),
                    "ag": pos.findtext("agility"),
                    "pa": pos.findtext("passing"),
                    "av": pos.findtext("armour"),
                    "skills": [s.text for s in pos.findall("skillList/skill") if s.text],
                })
            return {
                "file": p.name,
                "id": root.get("id"),
                "name": root.findtext("name"),
                "type": root.findtext("type"),
                "reRollCost": root.findtext("reRollCost"),
                "maxReRolls": root.findtext("maxReRolls"),
                "positions": positions,
            }
    return None


def create_team(
    team_id: str,
    coach: str,
    name: str,
    race: str,
    roster_id: str,
    re_rolls: int,
    fan_factor: int,
    apothecaries: int,
    team_rating: int,
    players: list[dict],
) -> Path:
    """
    Create a new team XML in ffb-server/teams/.
    players: list of {nr, name, positionId, skills?: list[str], gender?: str}
    Raises ValueError if team_id already exists.
    """
    for p in TEAMS_DIR.glob("*.xml"):
        if ET.parse(p).getroot().get("id") == team_id:
            raise ValueError(f"Team id '{team_id}' already exists in {p.name}")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '',
        f'<team id="{team_id}">',
        '',
        f'\t<coach>{coach}</coach>',
        f'\t<name>{name}</name>',
        f'\t<race>{race}</race>',
        f'\t<rosterId>{roster_id}</rosterId>',
        f'\t<reRolls>{re_rolls}</reRolls>',
        f'\t<fanFactor>{fan_factor}</fanFactor>',
        f'\t<apothecaries>{apothecaries}</apothecaries>',
        f'\t<teamRating>{team_rating}</teamRating>',
        f'\t<currentTeamValue>{team_rating}</currentTeamValue>',
        f'\t<teamStrength>{team_rating}</teamStrength>',
        '\t<division>[X]</division>',
        '\t<baseIconPath>http://localhost:2224/icons/players/</baseIconPath>',
    ]

    for player in players:
        nr = player["nr"]
        pid = player.get("id", f"{team_id}{nr}")
        pname = player["name"]
        pos_id = player["positionId"]
        gender = player.get("gender", "male")
        skills = player.get("skills", [])
        lines.append('')
        lines.append(f'\t<player nr="{nr}" id="{pid}">')
        lines.append(f'\t\t<name>{pname}</name>')
        lines.append(f'\t\t<gender>{gender}</gender>')
        lines.append(f'\t\t<positionId>{pos_id}</positionId>')
        if skills:
            lines.append('\t\t<skillList>')
            for skill in skills:
                lines.append(f'\t\t\t<skill>{skill}</skill>')
            lines.append('\t\t</skillList>')
        else:
            lines.append('\t\t<skillList></skillList>')
        lines.append('\t\t<injuryList></injuryList>')
        lines.append('\t\t<playerStatistics currentSpps="0"/>')
        lines.append('\t</player>')

    lines.extend(['', '</team>', ''])

    safe_race = race.lower().replace(' ', '_')
    filename = f"team_{safe_race}_{coach}_{team_rating}.xml"
    out_path = TEAMS_DIR / filename
    if out_path.exists():
        import time
        filename = f"team_{safe_race}_{coach}_{team_rating}_{int(time.time())}.xml"
        out_path = TEAMS_DIR / filename

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return out_path
