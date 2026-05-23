from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import FEATURE_COLS

st.set_page_config(page_title="Прогноз матчей РПЛ", page_icon="⚽", layout="centered")

TEAM_RU = {
    "Akron": "Акрон",
    "Amkar": "Амкар",
    "Anzhi Makhachkala": "Анжи",
    "Arsenal Tula": "Арсенал Тула",
    "Baltika": "Балтика",
    "CSKA Moscow": "ЦСКА",
    "Dinamo Moscow": "Динамо Москва",
    "Dynamo Makhachkala": "Динамо Махачкала",
    "FC Krasnodar": "Краснодар",
    "FC Orenburg": "Оренбург",
    "FC Rostov": "Ростов",
    "FC Rotor Volgograd": "Ротор",
    "FC Tambov": "Тамбов",
    "FC Ufa": "Уфа",
    "FC Yenisey Krasnoyarsk": "Енисей",
    "FK Akhmat": "Ахмат",
    "Fakel": "Факел",
    "Khimki": "Химки",
    "Krylya Sovetov Samara": "Крылья Советов",
    "Lokomotiv Moscow": "Локомотив",
    "Nizhny Novgorod": "Нижний Новгород",
    "PFC Sochi": "Сочи",
    "Rubin Kazan": "Рубин",
    "SKA-Khabarovsk": "СКА-Хабаровск",
    "Spartak Moscow": "Спартак",
    "Torpedo Moscow": "Торпедо",
    "Tosno": "Тосно",
    "Ural": "Урал",
    "Zenit St. Petersburg": "Зенит",
}
TEAM_EN = {v: k for k, v in TEAM_RU.items()}

RESULT_LABEL = {"H": "Победа хозяев", "D": "Ничья", "A": "Победа гостей"}
RESULT_COLOR = {"H": "#27ae60", "D": "#e67e22", "A": "#e74c3c"}
RESULT_ICON  = {"H": "🏠", "D": "🤝", "A": "✈️"}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "rf_tuned.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "processed" / "all_matches.csv", parse_dates=["date"])


model = load_model()
df = load_data()

all_teams_en = sorted(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))
all_teams_ru = [TEAM_RU.get(t, t) for t in all_teams_en]
all_teams_ru_sorted = sorted(all_teams_ru)


def team_latest_stats(team_en: str) -> dict:
    home_rows = df[df["home_team"] == team_en].sort_values("date")
    away_rows = df[df["away_team"] == team_en].sort_values("date")
    use_home = (
        not home_rows.empty
        and (away_rows.empty or home_rows.iloc[-1]["date"] >= away_rows.iloc[-1]["date"])
    )
    if use_home:
        row = home_rows.iloc[-1]
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
        row = away_rows.iloc[-1]
        return {
            "roll_xg_for": row["away_roll_xg_for"],
            "roll_xg_against": row["away_roll_xg_against"],
            "roll_goals_for": row["away_roll_goals_for"],
            "roll_goals_against": row["away_roll_goals_against"],
            "roll_pts": row["away_roll_pts"],
            "venue_form": row["away_venue_form"],
            "elo": row["away_elo"],
        }


def h2h_rates(home_en: str, away_en: str) -> tuple[float, float, float]:
    mask = (
        ((df["home_team"] == home_en) & (df["away_team"] == away_en))
        | ((df["home_team"] == away_en) & (df["away_team"] == home_en))
    )
    past = df[mask].tail(5)
    if past.empty:
        return 1 / 3, 1 / 3, 1 / 3
    outcomes = []
    for _, row in past.iterrows():
        res = row["result"]
        if row["home_team"] != home_en:
            res = {"H": "A", "D": "D", "A": "H"}[res]
        outcomes.append(res)
    n = len(outcomes)
    return outcomes.count("H") / n, outcomes.count("D") / n, outcomes.count("A") / n


st.title("⚽ Прогноз матча РПЛ")
st.caption("Модель: Random Forest · данные Understat 2017–2024 · Weighted F1 = 0.52")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Хозяева**")
    home_ru = st.selectbox("Хозяева", all_teams_ru_sorted, label_visibility="collapsed",
                           index=all_teams_ru_sorted.index("Зенит"))
with col2:
    st.markdown("**Гости**")
    away_options = [t for t in all_teams_ru_sorted if t != home_ru]
    away_ru = st.selectbox("Гости", away_options, label_visibility="collapsed",
                           index=away_options.index("ЦСКА") if "ЦСКА" in away_options else 0)

match_week = st.slider("Тур", min_value=1, max_value=34, value=20)

st.divider()

if st.button("Предсказать исход", type="primary", use_container_width=True):
    home_en = TEAM_EN.get(home_ru, home_ru)
    away_en = TEAM_EN.get(away_ru, away_ru)

    h = team_latest_stats(home_en)
    a = team_latest_stats(away_en)
    hw, dw, aw = h2h_rates(home_en, away_en)
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
    result = {2: "H", 1: "D", 0: "A"}[pred_code]

    color = RESULT_COLOR[result]
    icon  = RESULT_ICON[result]
    label = RESULT_LABEL[result]

    st.markdown(
        f"<div style='text-align:center; padding:16px; border-radius:10px; "
        f"background:{color}22; border: 2px solid {color}'>"
        f"<span style='font-size:2rem'>{icon}</span><br>"
        f"<span style='font-size:1.5rem; font-weight:700; color:{color}'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Победа хозяев", f"{proba[2]:.0%}")
    c2.metric("Ничья",         f"{proba[1]:.0%}")
    c3.metric("Победа гостей", f"{proba[0]:.0%}")

    st.write("")

    proba_df = pd.DataFrame({
        "Исход": ["Победа хозяев", "Ничья", "Победа гостей"],
        "Вероятность": [round(float(proba[2]), 3), round(float(proba[1]), 3), round(float(proba[0]), 3)],
    }).set_index("Исход")
    st.bar_chart(proba_df)

    with st.expander("Детали: Elo и форма команд"):
        dc1, dc2 = st.columns(2)
        dc1.metric(f"Elo {home_ru}", f"{h['elo']:.0f}")
        dc2.metric(f"Elo {away_ru}", f"{a['elo']:.0f}", delta=f"{-elo_diff:+.0f} к хозяевам")
        dc1.metric("Форма (очки)", f"{h['roll_pts']:.2f}")
        dc2.metric("Форма (очки)", f"{a['roll_pts']:.2f}")
        dc1.metric("xG (за)", f"{h['roll_xg_for']:.2f}")
        dc2.metric("xG (за)", f"{a['roll_xg_for']:.2f}")
