import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re

# ---------- helpers ----------
def _strip_percents(df, cols):
    # Convert things like "64.3%" -> 64.3 (not None)
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.replace(r'[^\d\.\-]', '', regex=True)
            )
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r'[^\d\.\-]', '', regex=True)
            )
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _exact_player_match(df, player_name):
    # Try exact (case-insensitive) match on the whole 'Player' cell first
    p_clean = player_name.strip().lower()
    exact = df[df['Player'].astype(str).str.strip().str.lower() == p_clean]
    if not exact.empty:
        return exact.iloc[[0]]

    # Fallback: match on name without team tag if user typed only the name
    # e.g., "Lamar Jackson" vs "Lamar Jackson (BAL)"
    # Extract just the name part before the team in parentheses.
    name_only = df['Player'].astype(str).str.replace(r'\s*\(.*\)$', '', regex=True)
    fallback = df[name_only.str.strip().str.lower() == p_clean]
    if not fallback.empty:
        return fallback.iloc[[0]]

    # Last resort: safe contains on word boundary
    safe = df[df['Player'].astype(str).str.contains(rf'\b{re.escape(player_name.strip())}\b', case=False, na=False)]
    if not safe.empty:
        return safe.iloc[[0]]

    return pd.DataFrame()

# ---------- Function 1 ----------
def initial_point_determination(csv_path, player_name):
    df = pd.read_csv(csv_path)

    features = ['Cmp', 'ATT', 'PCT', 'PASS YDS', 'PASS TD', 'INT', 'RUSH YDS', 'RUSH TD', 'FL']
    target = 'FPTS/G'

    # Guard: ensure required columns exist
    missing = [c for c in features + [target, 'Player'] if c not in df.columns]
    if missing:
        return None, f"Missing columns in stats CSV: {missing}"

    # Clean training data
    df = _strip_percents(df, ['PCT'])
    df = _to_numeric(df, [c for c in features if c != 'PCT'])
    df[target] = pd.to_numeric(df[target], errors='coerce')

    X = df[features]
    y = df[target]

    mask = X.notnull().all(axis=1) & y.notnull()
    if mask.sum() < 10:
        return None, "Not enough clean rows to train."

    X_train, X_test, y_train, y_test = train_test_split(X[mask], y[mask], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    # Find player row robustly
    player_row = _exact_player_match(df, player_name)
    if player_row.empty:
        return None, f"QB '{player_name}' not found."

    X_player = player_row[features].copy()
    # Already cleaned the columns globally, but re-coerce just in case
    X_player = X_player.apply(pd.to_numeric, errors='coerce')

    if X_player.isnull().any().any():
        return None, f"Not enough clean data to predict for {player_name}."

    prediction = float(model.predict(X_player)[0])
    return prediction, mae

# ---------- Function 2 ----------
def situation_adjuster(player_name, raw_ppg, situation_csv_path, starter_points=15.0):
    # Normalize column names and fix the known typo
    canonical_weights = {
        'HEAD COACH': 0.10,
        'OC': 0.04,
        'WEAPONS': 0.05,
        'O LINE': 0.05,
        'SCHEDULE': 0.04,
        'INDOOR HOME GAMES': 0.02,   # expects 0/1 (or a small count scaled), see note below
        'SCHEME FIT': 0.04,
        'RETURN FROM INJURY': 0.12,
        'OFF FIELD': 0.02
    }

    df = pd.read_csv(situation_csv_path)
    if 'Player' not in df.columns:
        return None, "Situation CSV missing 'Player' column."

    # Unify the typo if present
    if 'RETURN FROM INURY' in df.columns and 'RETURN FROM INJURY' not in df.columns:
        df = df.rename(columns={'RETURN FROM INURY': 'RETURN FROM INJURY'})

    row = _exact_player_match(df, player_name)
    if row.empty:
        return None, f"Context for '{player_name}' not found in situation CSV."

    # Optional additive starter bump (if you include a STARTER column with 0/1)
    starter_bump = 0.0
    if 'STARTER' in df.columns:
        try:
            starter_flag = int(row.iloc[0]['STARTER'])
            if starter_flag == 1:
                starter_bump = starter_points
        except Exception:
            pass

    # Compute multiplicative adjustment
    adj_total = 0.0
    for col, weight in canonical_weights.items():
        if col in row.columns:
            try:
                val = float(row.iloc[0][col])
                # If "INDOOR HOME GAMES" is a count (e.g., 9), scale to 0/1-ish
                if col == 'INDOOR HOME GAMES' and val > 1:
                    val = min(val / 9.0, 1.0)  # 9 domes ~ full credit; tweak if you want
                # Clamp inputs to [-1, 1] to avoid wild swings
                val = max(-1.0, min(1.0, val))
                adj_total += val * weight
            except Exception:
                continue

    adjusted_ppg = (raw_ppg + starter_bump) * (1 + adj_total)
    return adjusted_ppg, adj_total

# ---------- MAIN ----------
def main():
    stats_csv = "/Users/sulaimankhan/Downloads/PythonProjects/FantasyFootballModel/QbStats.csv"
    situation_csv = "/Users/sulaimankhan/Downloads/PythonProjects/FantasyFootballModel/QBsituation.csv"

    while True:
        name = input("\nEnter a QB's name (or type 'exit' to quit): ").strip()
        if name.lower() == 'exit':
            print("Goodbye!")
            break

        raw_ppg, raw_err = initial_point_determination(stats_csv, name)
        if raw_ppg is None:
            print(raw_err)
            continue

        adjusted_ppg, adj = situation_adjuster(name, raw_ppg, situation_csv)
        if adjusted_ppg is None:
            print(adj)
            continue

        print(f"\n⚡ Raw projection for {name}: {raw_ppg:.1f} FPTS/G (MAE ±{raw_err:.1f})")
        print(f"➕ Starter bump (if any): +15.0 pre-multiplier" if 'STARTER' in pd.read_csv(situation_csv).columns else "➕ Starter bump: (not used)")
        print(f"🔄 Situation multiplier: {adj * 100:+.1f}%")
        print(f"🏈 Final projection for {name}: {adjusted_ppg:.1f} FPTS/G")

if __name__ == "__main__":
    main()
