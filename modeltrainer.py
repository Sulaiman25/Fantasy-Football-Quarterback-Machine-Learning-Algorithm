# Weight Calculator Pipeline for NFL QB Offseason Factors (2022 -> 2023)
#
# What this does:
# - Loads four CSV files:
#     1) 2022 QB stats (per player)
#     2) 2023 QB stats (per player)
#     3) Offseason change data: UNCERTAIN factors (per player)
#     4) Offseason change data: HIGH-CERTAINTY factors (per player)
# - Cleans players (drops rookies, FAs, role-changes) using your rules:
#     * Use FPPG instead of totals
#     * Require >= 8 games in BOTH seasons
#     * Require (passing_yards >= 3000 OR rushing_yards >= 500) in BOTH seasons
# - Merges the four sources
# - Computes CHANGE_FPPG = FPPG_2023 - FPPG_2022
# - Runs a Ridge regression to estimate weights (impact of each factor on CHANGE_FPPG)
# - Saves two files:
#     * /mnt/data/clean_merged_for_weights.csv
#     * /mnt/data/learned_offseason_weights.csv
#
# How to use:
# - Put your four CSVs in /mnt/data (or change paths below)
# - Ensure your two offseason CSVs use "PLAYER" as the player name column
#   and columns exactly named per your spec:
#       UNCERTAIN: ["PLAYER","HEAD_COACH","OC","SCHEME_FIT","OFF_FIELD"]
#       HIGH-CERT: ["PLAYER","WEAPONS","O_LINE","INDOOR_GAMES","SCHEDULE","RETURN_FROM_INJURY"]
# - Run this cell. It will print a summary and show the resulting weight table.

import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# --------------------------
# CONFIG: file paths (edit these to your filenames)
# --------------------------
from pathlib import Path

BASE = Path(__file__).resolve().parent  # folder where this script lives

PATH_2022     = BASE / "2022_qb_dummy_stats.csv"
PATH_2023     = BASE / "2023_qb_dummy_stats.csv"
PATH_UNCERTAIN = BASE / "2022_2023_dummydata_no_certainty.csv"
PATH_HIGHCERT  = BASE / "2022_2023_dummydata_high_certainty.csv"

# --------------------------
# Helpers
# --------------------------

def normalize_player_name(s: str) -> str:
    """Strip team abbrev in parentheses and trim whitespace."""
    if pd.isna(s):
        return s
    # Remove anything in parentheses like " (KC)"
    s = re.sub(r"\s*\([^)]+\)", "", str(s)).strip()
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s

def find_fppg_column(df: pd.DataFrame):
    """Find an FPPG column in the common formats."""
    candidates = ["FPTS/G", "FPPG", "FPTS_per_game", "FPTS_G"]
    for c in df.columns:
        if c in candidates:
            return c
    # fallback: look for something like 'FPTS' and divide by G if both exist
    if "FPTS" in df.columns and "G" in df.columns and df["G"].notna().all():
        # create a temp FPPG column
        df["_FPPG_TEMP_"] = pd.to_numeric(df["FPTS"], errors="coerce") / pd.to_numeric(df["G"], errors="coerce")
        return "_FPPG_TEMP_"
    raise ValueError("Could not find an FPPG column. Expected one of FPTS/G, FPPG, FPTS_per_game, FPTS_G (or provide FPTS and G).")

def detect_passing_and_rushing_yards(df: pd.DataFrame):
    """
    Try to identify passing yards and rushing yards columns.
    Many fantasy CSVs duplicate column names (e.g., 'YDS' twice). Pandas may suffix them as 'YDS' and 'YDS.1'.
    Heuristic:
      - Passing yards column will generally have a larger mean (~2000-5000)
      - Rushing yards column will generally have a smaller mean (~0-800)
    """
    # Gather all YDS-like columns
    yds_cols = [c for c in df.columns if c.upper().startswith("YDS")]
    if not yds_cols:
        # try exact 'YDS' and 'YDS_RUSH'
        if "YDS" in df.columns and "YDS_RUSH" in df.columns:
            return "YDS", "YDS_RUSH"
        raise ValueError("Could not find yards columns. Expected YDS columns (e.g., 'YDS' and 'YDS.1') or 'YDS'/'YDS_RUSH'.")
    # Score columns by mean
    means = {c: pd.to_numeric(df[c], errors="coerce").dropna().mean() for c in yds_cols}
    # Passing yards likely the max mean
    pass_col = max(means, key=means.get)
    # Rushing yards likely the min mean
    rush_col = min(means, key=means.get)
    return pass_col, rush_col

def load_and_prepare_stats(path: str, season_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names (strip spaces, remove BOM)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    # Try to find the player name column robustly
    name_col = None
    candidates_exact = ["Player", "PLAYER", "Name", "QB"]
    for c in candidates_exact:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        # fuzzy: any column containing 'player' or 'name'
        for c in df.columns:
            lc = c.lower()
            if "player" in lc or "name" in lc:
                name_col = c
                break
    if name_col is None:
        # LAST resort: if a TEAM column exists, use it (warn!)
        if "TEAM" in df.columns or "Team" in df.columns:
            name_col = "TEAM" if "TEAM" in df.columns else "Team"
            print(f"WARNING [{season_label}]: Using '{name_col}' as PLAYER. "
                  f"If this holds TEAM names, merges to PLAYER-based sheets may fail.")
        else:
            raise ValueError(f"{season_label}: Could not find a player name column. "
                             f"Got columns: {list(df.columns)}")

    # Normalize player values (strip team in parentheses)
    def normalize_player_name(s: str) -> str:
        if pd.isna(s): return s
        s = re.sub(r"\s*\([^)]+\)", "", str(s)).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    df[name_col] = df[name_col].apply(normalize_player_name)
    df = df.rename(columns={name_col: "PLAYER"})

    # FPPG detection (create from FPTS/G or FPTS/G = FPTS / G)
    candidates_fppg = ["FPTS/G", "FPPG", "FPTS_per_game", "FPTS_G"]
    fppg_col = None
    for c in candidates_fppg:
        if c in df.columns:
            fppg_col = c
            break
    if fppg_col is None:
        if "FPTS" in df.columns and "G" in df.columns:
            df["_FPPG_TEMP_"] = pd.to_numeric(df["FPTS"], errors="coerce") / pd.to_numeric(df["G"], errors="coerce")
            fppg_col = "_FPPG_TEMP_"
        else:
            raise ValueError(f"{season_label}: Need FPPG or (FPTS and G). Columns: {list(df.columns)}")

    df[fppg_col] = pd.to_numeric(df[fppg_col], errors="coerce")

    if "G" not in df.columns:
        raise ValueError(f"{season_label}: Expected a 'G' (games) column.")
    df["G"] = pd.to_numeric(df["G"], errors="coerce")

    # Detect passing/rushing yards even if duplicated names (YDS / YDS.1)
    yds_cols = [c for c in df.columns if c.upper().startswith("YDS")]
    if len(yds_cols) >= 2:
        means = {c: pd.to_numeric(df[c], errors="coerce").dropna().mean() for c in yds_cols}
        pass_col = max(means, key=means.get)
        rush_col = min(means, key=means.get)
    else:
        # fallbacks commonly seen
        if "YDS" in df.columns and "YDS_RUSH" in df.columns:
            pass_col, rush_col = "YDS", "YDS_RUSH"
        else:
            # try common explicit names
            candidates_pass = ["PASS_YDS", "PYDS", "PASSING_YDS"]
            candidates_rush = ["RUSH_YDS", "RYDS", "RUSHING_YDS"]
            pass_col = next((c for c in candidates_pass if c in df.columns), None)
            rush_col = next((c for c in candidates_rush if c in df.columns), None)
            if pass_col is None or rush_col is None:
                raise ValueError(f"{season_label}: Could not identify passing/rushing yards. Columns: {list(df.columns)}")

    df[pass_col] = pd.to_numeric(df[pass_col], errors="coerce")
    df[rush_col] = pd.to_numeric(df[rush_col], errors="coerce")

    out = df[["PLAYER", fppg_col, "G", pass_col, rush_col]].copy()
    out = out.rename(columns={
        fppg_col: f"FPPG_{season_label}",
        "G": f"G_{season_label}",
        pass_col: f"PASS_YDS_{season_label}",
        rush_col: f"RUSH_YDS_{season_label}",
    })

    # Keep the row with the most games if duplicates
    out = out.sort_values(by=[f"G_{season_label}"], ascending=False).drop_duplicates(subset=["PLAYER"], keep="first")
    return out
def load_offseason_changes(path_uncertain: str, path_highcert: str) -> pd.DataFrame:
    u = pd.read_csv(path_uncertain)
    h = pd.read_csv(path_highcert)
    # Normalize player column name
    def norm(df):
        if "PLAYER" in df.columns:
            df["PLAYER"] = df["PLAYER"].apply(normalize_player_name)
            return df
        elif "Player" in df.columns:
            df = df.rename(columns={"Player": "PLAYER"})
            df["PLAYER"] = df["PLAYER"].apply(normalize_player_name)
            return df
        else:
            raise ValueError("Offseason CSV is missing 'PLAYER' column.")
    u = norm(u)
    h = norm(h)
    # Merge
    merged = pd.merge(u, h, on="PLAYER", how="inner")
    return merged

# --------------------------
# Load data
# --------------------------
try:
    s22 = load_and_prepare_stats(PATH_2022, "2022")
    s23 = load_and_prepare_stats(PATH_2023, "2023")
    changes = load_offseason_changes(PATH_UNCERTAIN, PATH_HIGHCERT)
except Exception as e:
    print("ERROR while loading files:", e)
    raise

# --------------------------
# Eligibility filtering
# --------------------------
df = pd.merge(s22, s23, on="PLAYER", how="inner")

# Rules:
# 1) Games >= 8 in BOTH years
mask_games = (df["G_2022"] >= 8) & (df["G_2023"] >= 8)

# 2) Role thresholds in BOTH years:
#    (PASS_YDS >= 3000) OR (RUSH_YDS >= 500)
mask_role_2022 = (df["PASS_YDS_2022"] >= 3000) | (df["RUSH_YDS_2022"] >= 500)
mask_role_2023 = (df["PASS_YDS_2023"] >= 3000) | (df["RUSH_YDS_2023"] >= 500)

eligible = df[mask_games & mask_role_2022 & mask_role_2023].copy()

# --------------------------
# Merge with offseason change factors
# --------------------------
eligible = pd.merge(eligible, changes, on="PLAYER", how="inner")

# --------------------------
# Target: CHANGE_FPPG
# --------------------------
eligible["CHANGE_FPPG"] = eligible["FPPG_2023"] - eligible["FPPG_2022"]

# --------------------------
# Define features
# --------------------------
feature_cols = [
    # Uncertain
    "HEAD_COACH", "OC", "SCHEME_FIT", "OFF_FIELD",
    # High-certainty
    "WEAPONS", "O_LINE", "INDOOR_GAMES", "SCHEDULE", "RETURN_FROM_INJURY",
]

# Coerce to numeric safely
for c in feature_cols:
    eligible[c] = pd.to_numeric(eligible[c], errors="coerce")

eligible = eligible.dropna(subset=feature_cols + ["CHANGE_FPPG"]).copy()

# --------------------------
# Train Ridge regression (standardized features)
# --------------------------
X = eligible[feature_cols].values
y = eligible["CHANGE_FPPG"].values

# If there aren't enough rows, skip training gracefully
weights_df = None
if len(eligible) >= len(feature_cols) + 1:
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True, random_state=42))
    ])
    model.fit(X, y)
    coefs = model.named_steps["ridge"].coef_
    intercept = model.named_steps["ridge"].intercept_
    weights_df = pd.DataFrame({
        "FACTOR": feature_cols,
        "WEIGHT_FPPG_CHANGE": np.round(coefs, 3)
    })
    weights_df.loc[len(weights_df)] = ["INTERCEPT", round(intercept, 3)]
else:
    weights_df = pd.DataFrame({
        "FACTOR": ["INSUFFICIENT_ELIGIBLE_ROWS"],
        "WEIGHT_FPPG_CHANGE": [np.nan]
    })

# --------------------------
# Save outputs
# --------------------------
out_merged = "/mnt/data/clean_merged_for_weights.csv"
out_weights = "/mnt/data/learned_offseason_weights.csv"

eligible.to_csv(out_merged, index=False)
weights_df.to_csv(out_weights, index=False)

# Show to the user

print(f"Saved merged data -> {out_merged}")
print(f"Saved weights -> {out_weights}")
