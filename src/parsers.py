import time
from pathlib import Path

import pandas as pd
import requests

UNDERSTAT_BASE = "https://understat.com"
UNDERSTAT_LEAGUE = "RFPL"
RPL_SEASONS = list(range(2017, 2025))

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate",
    "X-Requested-With": "XMLHttpRequest",
}


def _parse_match_record(match: dict, season: int) -> dict:
    return {
        "match_id": match["id"],
        "season": season,
        "date": match["datetime"],
        "home_team": match["h"]["title"],
        "away_team": match["a"]["title"],
        "home_goals": int(match["goals"]["h"]) if match["goals"]["h"] is not None else None,
        "away_goals": int(match["goals"]["a"]) if match["goals"]["a"] is not None else None,
        "home_xg": float(match["xG"]["h"]) if match["xG"]["h"] is not None else None,
        "away_xg": float(match["xG"]["a"]) if match["xG"]["a"] is not None else None,
        "is_result": bool(match.get("isResult", False)),
    }


def fetch_understat_season(season: int, delay: float = 1.5) -> pd.DataFrame:
    url = f"{UNDERSTAT_BASE}/getLeagueData/{UNDERSTAT_LEAGUE}/{season}"
    headers = {**_HEADERS, "Referer": f"{UNDERSTAT_BASE}/league/{UNDERSTAT_LEAGUE}/{season}"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    records = [_parse_match_record(m, season) for m in data["dates"]]
    time.sleep(delay)
    return pd.DataFrame(records)


def fetch_all_understat(
    seasons: list[int] | None = None,
    raw_dir: Path | str = "data/raw",
    force_reload: bool = False,
) -> pd.DataFrame:
    if seasons is None:
        seasons = RPL_SEASONS

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        cache = raw_dir / f"understat_rpl_{season}.csv"
        if cache.exists() and not force_reload:
            df = pd.read_csv(cache, parse_dates=["date"])
            print(f"  [cache] season {season}: {len(df)} matches")
        else:
            print(f"  [download] season {season} …", end=" ", flush=True)
            df = fetch_understat_season(season)
            df.to_csv(cache, index=False)
            print(f"{len(df)} matches saved → {cache}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


def fetch_fbref_season(season: str = "2023-2024", raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"fbref_rpl_{season.replace('-', '_')}.csv"

    if cache.exists():
        print(f"  [cache] FBref {season}")
        return pd.read_csv(cache)

    try:
        import soccerdata as sd

        fbref = sd.FBref(leagues="RUS-Premier League", seasons=season)
        df = fbref.read_schedule()
        df.to_csv(cache, index=False)
        print(f"  [download] FBref {season}: {len(df)} rows → {cache}")
        return df
    except Exception as exc:
        print(f"  [warning] FBref fetch failed ({exc}). Returning empty DataFrame.")
        return pd.DataFrame()


def fetch_fbref_team_stats(season: str = "2023-2024", raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"fbref_team_stats_{season.replace('-', '_')}.csv"

    if cache.exists():
        print(f"  [cache] FBref team stats {season}")
        return pd.read_csv(cache)

    try:
        import soccerdata as sd

        fbref = sd.FBref(leagues="RUS-Premier League", seasons=season)
        df = fbref.read_team_season_stats(stat_type="standard")
        df.to_csv(cache, index=False)
        print(f"  [download] FBref team stats {season}: {len(df)} rows → {cache}")
        return df
    except Exception as exc:
        print(f"  [warning] FBref team stats failed ({exc}). Returning empty DataFrame.")
        return pd.DataFrame()
