import numpy as np
import pandas as pd

WINDOW = 5
H2H_WINDOW = 5
ELO_START = 1500
ELO_K = 32

RESULT_MAP = {"H": 2, "D": 1, "A": 0}


def _add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"],
        "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A"),
    )
    df["result_encoded"] = df["result"].map(RESULT_MAP)
    return df


def _home_points(result: str) -> int:
    return {"H": 3, "D": 1, "A": 0}[result]


def _away_points(result: str) -> int:
    return {"H": 0, "D": 1, "A": 3}[result]


def _rolling_team_stats(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    rows_home = df.assign(
        team=df["home_team"],
        xg_for=df["home_xg"],
        xg_against=df["away_xg"],
        goals_for=df["home_goals"],
        goals_against=df["away_goals"],
        pts=df["result"].map(_home_points),
        is_home=True,
    )
    rows_away = df.assign(
        team=df["away_team"],
        xg_for=df["away_xg"],
        xg_against=df["home_xg"],
        goals_for=df["away_goals"],
        goals_against=df["home_goals"],
        pts=df["result"].map(_away_points),
        is_home=False,
    )

    long = (
        pd.concat([rows_home, rows_away], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    stats: list[str] = ["xg_for", "xg_against", "goals_for", "goals_against", "pts"]
    rolled = long.groupby("team")[stats].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    rolled.columns = [f"roll_{c}" for c in stats]

    long = pd.concat([long, rolled], axis=1)
    return long


def _pivot_back(long: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    home_stats = long[long["is_home"]].set_index("match_id")[
        ["roll_xg_for", "roll_xg_against", "roll_goals_for", "roll_goals_against", "roll_pts"]
    ].rename(columns=lambda c: f"home_{c}")

    away_stats = long[~long["is_home"]].set_index("match_id")[
        ["roll_xg_for", "roll_xg_against", "roll_goals_for", "roll_goals_against", "roll_pts"]
    ].rename(columns=lambda c: f"away_{c}")

    result = df_orig.copy().set_index("match_id")
    result = result.join(home_stats).join(away_stats)
    return result.reset_index()


def _add_venue_form(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    home_long = df.assign(team=df["home_team"], pts=df["result"].map(_home_points))
    away_long = df.assign(team=df["away_team"], pts=df["result"].map(_away_points))

    df = df.copy()
    home_long_sorted = home_long.sort_values("date").groupby("team", group_keys=False).apply(lambda x: x)
    away_long_sorted = away_long.sort_values("date").groupby("team", group_keys=False).apply(lambda x: x)

    home_long_sorted["home_venue_form"] = home_long_sorted.groupby("team")["pts"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    away_long_sorted["away_venue_form"] = away_long_sorted.groupby("team")["pts"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home_map = home_long_sorted.set_index("match_id")["home_venue_form"]
    away_map = away_long_sorted.set_index("match_id")["away_venue_form"]

    df["home_venue_form"] = df["match_id"].map(home_map)
    df["away_venue_form"] = df["match_id"].map(away_map)
    return df


def _add_h2h(df: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    h2h_home_win: list[float] = []
    h2h_draw: list[float] = []
    h2h_away_win: list[float] = []

    for _, row in df.iterrows():
        past = df[
            (df["date"] < row["date"])
            & (
                ((df["home_team"] == row["home_team"]) & (df["away_team"] == row["away_team"]))
                | ((df["home_team"] == row["away_team"]) & (df["away_team"] == row["home_team"]))
            )
        ].tail(window)

        if len(past) == 0:
            h2h_home_win.append(np.nan)
            h2h_draw.append(np.nan)
            h2h_away_win.append(np.nan)
            continue

        outcomes: list[str] = []
        for _, pmatch in past.iterrows():
            if pmatch["home_team"] == row["home_team"]:
                outcomes.append(pmatch["result"])
            else:
                outcomes.append({"H": "A", "D": "D", "A": "H"}[pmatch["result"]])

        n = len(outcomes)
        h2h_home_win.append(outcomes.count("H") / n)
        h2h_draw.append(outcomes.count("D") / n)
        h2h_away_win.append(outcomes.count("A") / n)

    df["h2h_home_win_rate"] = h2h_home_win
    df["h2h_draw_rate"] = h2h_draw
    df["h2h_away_win_rate"] = h2h_away_win
    return df


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def _add_elo(df: pd.DataFrame, k: float = ELO_K, start: float = ELO_START) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    elo: dict[str, float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo.setdefault(h, start)
        elo.setdefault(a, start)

        elo_h, elo_a = elo[h], elo[a]
        home_elos.append(elo_h)
        away_elos.append(elo_a)

        exp_h = _expected_score(elo_h, elo_a)
        actual_h = {"H": 1.0, "D": 0.5, "A": 0.0}[row["result"]]

        elo[h] = elo_h + k * (actual_h - exp_h)
        elo[a] = elo_a + k * ((1.0 - actual_h) - (1.0 - exp_h))

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def _add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    last_match: dict[str, pd.Timestamp] = {}
    home_rest: list[float] = []
    away_rest: list[float] = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        d: pd.Timestamp = row["date"]

        home_rest.append((d - last_match[h]).days if h in last_match else np.nan)
        away_rest.append((d - last_match[a]).days if a in last_match else np.nan)

        last_match[h] = d
        last_match[a] = d

    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    return df


def build_features(df_matches: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    df = df_matches.copy()
    df = df[df["is_result"]].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = _add_result_column(df)
    long = _rolling_team_stats(df, window=window)
    df = _pivot_back(long, df)
    df = _add_venue_form(df, window=window)
    df = _add_h2h(df, window=H2H_WINDOW)
    df = _add_elo(df)
    df = _add_days_rest(df)
    df["match_week"] = df.groupby("season").cumcount() + 1

    return df
