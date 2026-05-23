from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import FEATURE_COLS

DESCRIPTION = """
Предсказывает исход матча Российской Премьер-лиги (РПЛ).

**Модель:** Random Forest (tuned) · Weighted F1 = 0.52 (val)

**Возможные исходы:**
- `H` — победа хозяев
- `D` — ничья
- `A` — победа гостей

---

**Как использовать:**

1. `GET /teams` — получить список доступных команд
2. `POST /predict_match` — передать названия команд и тур, получить прогноз
3. `POST /predict` — передать все признаки вручную (для продвинутого использования)
"""

app = FastAPI(
    title="Прогноз матчей РПЛ",
    description=DESCRIPTION,
    version="1.0",
)

model = joblib.load(ROOT / "models" / "rf_tuned.pkl")
df = pd.read_csv(ROOT / "data" / "processed" / "all_matches.csv", parse_dates=["date"])

RESULT_DECODE = {2: "H", 1: "D", 0: "A"}
RESULT_RU = {"H": "Победа хозяев", "D": "Ничья", "A": "Победа гостей"}


class FeatureInput(BaseModel):
    home_roll_xg_for: float
    home_roll_xg_against: float
    away_roll_xg_for: float
    away_roll_xg_against: float
    home_roll_goals_for: float
    home_roll_goals_against: float
    away_roll_goals_for: float
    away_roll_goals_against: float
    home_roll_pts: float
    away_roll_pts: float
    home_venue_form: float
    away_venue_form: float
    h2h_home_win_rate: float
    h2h_draw_rate: float
    h2h_away_win_rate: float
    home_elo: float
    away_elo: float
    elo_diff: float
    home_days_rest: float
    away_days_rest: float
    match_week: int


def _team_latest_stats(team: str) -> dict:
    home = df[df["home_team"] == team].sort_values("date")
    away = df[df["away_team"] == team].sort_values("date")

    if home.empty and away.empty:
        raise ValueError(f"Команда '{team}' не найдена в данных")

    use_home = (
        not home.empty
        and (away.empty or home.iloc[-1]["date"] >= away.iloc[-1]["date"])
    )

    if use_home:
        row = home.iloc[-1]
        return {
            "roll_xg_for": row["home_roll_xg_for"],
            "roll_xg_against": row["home_roll_xg_against"],
            "roll_goals_for": row["home_roll_goals_for"],
            "roll_goals_against": row["home_roll_goals_against"],
            "roll_pts": row["home_roll_pts"],
            "venue_form": row["home_venue_form"],
            "elo": row["home_elo"],
        }
    else:
        row = away.iloc[-1]
        return {
            "roll_xg_for": row["away_roll_xg_for"],
            "roll_xg_against": row["away_roll_xg_against"],
            "roll_goals_for": row["away_roll_goals_for"],
            "roll_goals_against": row["away_roll_goals_against"],
            "roll_pts": row["away_roll_pts"],
            "venue_form": row["away_venue_form"],
            "elo": row["away_elo"],
        }


def _h2h_rates(home_team: str, away_team: str) -> tuple[float, float, float]:
    mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    past = df[mask].tail(5)
    if past.empty:
        return 1 / 3, 1 / 3, 1 / 3

    outcomes = []
    for _, row in past.iterrows():
        res = row["result"]
        if row["home_team"] != home_team:
            res = {"H": "A", "D": "D", "A": "H"}[res]
        outcomes.append(res)

    n = len(outcomes)
    return outcomes.count("H") / n, outcomes.count("D") / n, outcomes.count("A") / n


@app.get("/", summary="Статус сервиса")
def health():
    return {"status": "ok", "model": "Random Forest (tuned)", "metric": "Weighted F1 = 0.5217 (val)"}


@app.get(
    "/teams",
    summary="Список команд",
    description="Возвращает все команды РПЛ, которые есть в обучающих данных (сезоны 2017–2024).",
)
def get_teams():
    teams = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
    return {"teams": teams, "count": len(teams)}


@app.post(
    "/predict",
    summary="Прогноз по вектору признаков",
    description=(
        "Принимает все 21 признак вручную и возвращает прогноз. "
        "Используйте `/predict_match`, если хотите просто передать названия команд."
    ),
)
def predict(features: FeatureInput):
    X = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
    pred_code = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    result = RESULT_DECODE[pred_code]
    return {
        "prediction": result,
        "prediction_ru": RESULT_RU[result],
        "probabilities": {
            "A (победа гостей)": round(float(proba[0]), 4),
            "D (ничья)": round(float(proba[1]), 4),
            "H (победа хозяев)": round(float(proba[2]), 4),
        },
    }


@app.post(
    "/predict_match",
    summary="Прогноз по названиям команд",
    description=(
        "Основной эндпоинт. Передайте `home_team` и `away_team` "
        "(английские названия из `/teams`) и номер тура. "
        "Признаки для модели берутся из последних матчей каждой команды в данных."
    ),
)
def predict_match(home_team: str, away_team: str, match_week: int = 20):
    try:
        h = _team_latest_stats(home_team)
        a = _team_latest_stats(away_team)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    hw, dw, aw = _h2h_rates(home_team, away_team)
    elo_diff = h["elo"] - a["elo"]

    features = {
        "home_roll_xg_for": h["roll_xg_for"],
        "home_roll_xg_against": h["roll_xg_against"],
        "away_roll_xg_for": a["roll_xg_for"],
        "away_roll_xg_against": a["roll_xg_against"],
        "home_roll_goals_for": h["roll_goals_for"],
        "home_roll_goals_against": h["roll_goals_against"],
        "away_roll_goals_for": a["roll_goals_for"],
        "away_roll_goals_against": a["roll_goals_against"],
        "home_roll_pts": h["roll_pts"],
        "away_roll_pts": a["roll_pts"],
        "home_venue_form": h["venue_form"],
        "away_venue_form": a["venue_form"],
        "h2h_home_win_rate": hw,
        "h2h_draw_rate": dw,
        "h2h_away_win_rate": aw,
        "home_elo": h["elo"],
        "away_elo": a["elo"],
        "elo_diff": elo_diff,
        "home_days_rest": 7.0,
        "away_days_rest": 7.0,
        "match_week": match_week,
    }

    X = pd.DataFrame([features])[FEATURE_COLS]
    pred_code = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    result = RESULT_DECODE[pred_code]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": result,
        "prediction_ru": RESULT_RU[result],
        "probabilities": {
            "A (победа гостей)": round(float(proba[0]), 4),
            "D (ничья)": round(float(proba[1]), 4),
            "H (победа хозяев)": round(float(proba[2]), 4),
        },
    }
