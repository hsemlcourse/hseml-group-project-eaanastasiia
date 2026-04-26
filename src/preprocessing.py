from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42

TRAIN_END_SEASON = 2021
VAL_SEASON = 2022
TEST_START_SEASON = 2023

FEATURE_COLS = [
    "home_roll_xg_for",
    "home_roll_xg_against",
    "away_roll_xg_for",
    "away_roll_xg_against",
    "home_roll_goals_for",
    "home_roll_goals_against",
    "away_roll_goals_for",
    "away_roll_goals_against",
    "home_roll_pts",
    "away_roll_pts",
    "home_venue_form",
    "away_venue_form",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_days_rest",
    "away_days_rest",
    "match_week",
]

TARGET_COL = "result_encoded"


def drop_incomplete_rows(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    nan_frac = df[FEATURE_COLS].isna().mean(axis=1)
    mask = nan_frac <= threshold
    dropped = (~mask).sum()
    if dropped:
        print(f"  [drop] {dropped} rows with >{threshold:.0%} missing feature values")
    return df[mask].reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset="match_id", keep="first").reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        print(f"  [dedup] removed {removed} duplicate match_id rows")
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    # NaN values arise at the start of seasons when rolling windows have insufficient history.
    imputer = SimpleImputer(strategy="median")
    df = df.copy()
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    return df


def clip_outliers(df: pd.DataFrame, cols: list[str] | None = None, z_thresh: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    if cols is None:
        cols = FEATURE_COLS
    for col in cols:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma == 0:
            continue
        lo, hi = mu - z_thresh * sigma, mu + z_thresh * sigma
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_clipped:
            print(f"  [clip] {col}: {n_clipped} values clipped to [{lo:.2f}, {hi:.2f}]")
        df[col] = df[col].clip(lo, hi)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "result_encoded" not in df.columns:
        df["result_encoded"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] <= TRAIN_END_SEASON].reset_index(drop=True)
    val = df[df["season"] == VAL_SEASON].reset_index(drop=True)
    test = df[df["season"] >= TEST_START_SEASON].reset_index(drop=True)

    total = len(df)
    print(
        f"  [split] train={len(train)} ({len(train)/total:.1%})  "
        f"val={len(val)} ({len(val)/total:.1%})  "
        f"test={len(test)} ({len(test)/total:.1%})"
    )
    return train, val, test


def run_preprocessing(
    df_features: pd.DataFrame,
    processed_dir: Path | str = "data/processed",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    print("Preprocessing pipeline started")
    df = remove_duplicates(df_features)
    df = drop_incomplete_rows(df, threshold=0.5)
    df = clip_outliers(df)
    df = impute_features(df)
    df = encode_target(df)
    train, val, test = temporal_split(df)

    if save:
        train.to_csv(processed_dir / "train.csv", index=False)
        val.to_csv(processed_dir / "val.csv", index=False)
        test.to_csv(processed_dir / "test.csv", index=False)
        df.to_csv(processed_dir / "all_matches.csv", index=False)
        print(f"  [save] datasets written to {processed_dir}/")

    print("Preprocessing pipeline finished")
    return train, val, test


if __name__ == "__main__":
    import sys

    data_path = Path("data/processed/all_matches.csv")
    if not data_path.exists():
        print("Run notebooks/00_data_collection.ipynb first to generate all_matches.csv")
        sys.exit(1)

    df = pd.read_csv(data_path, parse_dates=["date"])
    train, val, test = run_preprocessing(df, save=False)
    print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")
