import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === Function 1: Predict raw fantasy points from stats ===
def initial_point_determination(csv_path, player_name):
    df = pd.read_csv(csv_path)
    features = ['Cmp', 'ATT', 'PCT', 'PASS YDS', 'PASS TD', 'INT', 'RUSH YDS', 'RUSH TD', 'FL']
    target = 'FPTS/G'

    X = df[features]
    y = df[target]

    # Clean training data
    X = X.astype(str).applymap(lambda val: val if not str(val).strip().endswith('%') else None)
    X = X.replace(r"[^\d\.\-]", "", regex=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate error
    mae = mean_absolute_error(y_test, model.predict(X_test))

    # Find player
    player_row = df[df['Player'].str.contains(player_name, case=False, na=False)]
    if player_row.empty:
        return None, f"QB '{player_name}' not found."

    X_player = player_row[features]
    X_player = X_player.astype(str).applymap(lambda val: val if not str(val).strip().endswith('%') else None)
    X_player = X_player.replace(r"[^\d\.\-]", "", regex=True)
    X_player = X_player.apply(pd.to_numeric, errors='coerce')

    if X_player.isnull().any().any():
        return None, f"Not enough clean data to predict for {player_name}."

    prediction = model.predict(X_player)[0]
    return prediction, mae


# === Function 2: Adjust the prediction based on situation factors ===
def situation_adjuster(player_name, raw_ppg, situation_csv_path):
    weights = {
        'HEAD COACH': 0.10,
        'OC': 0.04,
        'WEAPONS': 0.05,
        'O LINE': 0.05,
        'SCHEDULE': 0.04,
        'INDOOR HOME GAMES': 0.02,
        'SCHEME FIT': 0.04,
        'RETURN FROM INJURY': 0.12,
        'OFF FIELD': 0.02
    }

    df = pd.read_csv(situation_csv_path)
    row = df[df['Player'].str.contains(player_name, case=False, na=False)]
    if row.empty:
        return None, f"Context for '{player_name}' not found in situation CSV."

    adj_total = 0.0
    for col, weight in weights.items():
        if col in row.columns:
            val = row.iloc[0][col]
            try:
                val = int(val)
                adj_total += val * weight
            except:
                continue

    adjusted_ppg = raw_ppg * (1 + adj_total)
    return adjusted_ppg, adj_total


# === MAIN FUNCTION ===
# === Streamlit Frontend ===

import streamlit as st

st.set_page_config(page_title="Fantasy Football QB Projections", layout="centered")
st.title("🏈 Fantasy Football QB Projection Tool")

st.write("Enter a quarterback's name to see their projected fantasy points per game.")

player_name = st.text_input("QB Name")

stats_csv = "QbStats.csv"
situation_csv = "QBsituation.csv"

if st.button("Get Prediction"):
    raw_ppg, raw_err = initial_point_determination(stats_csv, player_name)

    if raw_ppg is None:
        st.error(raw_err)
    else:
        adjusted_ppg, adj_percent = situation_adjuster(player_name, raw_ppg, situation_csv)

        if adjusted_ppg is None:
            st.warning(adj_percent)
        else:
            st.success(f"⚡ Raw Prediction: {raw_ppg:.1f} FPTS/G (±{raw_err:.1f} MAE)")
            st.info(f"🔄 Situation Adjustment: {adj_percent * 100:+.1f}%")
            st.markdown(f"### 🏁 Final Projection: **{adjusted_ppg:.1f} FPTS/G**")

