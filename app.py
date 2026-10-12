import os
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
app = Flask(__name__)

DATA_PATH = os.environ.get("IPL_DATA_PATH", "ipl_data.csv")
TARGET_COL = None
FEATURE_COLS = []
CATEGORICAL_COLS = []
NUMERIC_COLS = []
DATA = None
RF_PIPELINE = None
XGB_PIPELINE = None
METRICS = {}
TRAINING_ERROR = None

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    for c in df.columns:
        cl = str(c).strip().lower().replace("_", " ").replace("-", " ")
        if any(name.lower() in cl for name in candidates):
            return c
    return None

def normalize_team(value):
    if pd.isna(value):
        return value
    return TEAM_ALIASES.get(str(value), str(value))

def load_and_train():
    global DATA, TARGET_COL, FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS
    global RF_PIPELINE, XGB_PIPELINE, METRICS, TRAINING_ERROR
    if not os.path.exists(DATA_PATH):
        TRAINING_ERROR = f"Dataset file not found: {DATA_PATH}. Place ipl_data.csv in the project root."
        return
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty or len(df.columns) < 4:
            raise ValueError("The CSV has too few columns to train a model.")

        # Follow the feature/target convention used in the uploaded notebook:
        # columns from index 2 through the penultimate column are features; last is target.
        target_col = df.columns[-1]
        feature_cols = list(df.columns[2:-1])
        if not feature_cols:
            raise ValueError("No model features found. Expected features in columns 3 through the penultimate column.")

        model_df = df[feature_cols + [target_col]].copy()
        model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
        model_df = model_df.dropna(subset=[target_col])
        for col in feature_cols:
            if model_df[col].dtype == "object":
                model_df[col] = model_df[col].fillna("Unknown").astype(str)
            else:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df[col] = model_df[col].fillna(model_df[col].median())
        if len(model_df) < 10:
            raise ValueError("Not enough valid rows in the CSV to train the models.")

        X = model_df[feature_cols]
        y = model_df[target_col]
        cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        # Column-based encoder is safer than assuming all first five columns are categorical.
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", encoder, cat_cols),
                ("numeric", "passthrough", num_cols),
            ],
            remainder="drop"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=32
        )
        rf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1))
        ])
        xgb = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ("numeric", "passthrough", num_cols),
                ], remainder="drop"
            )),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42, n_jobs=2
            ))
        ])
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        xgb_pred = xgb.predict(X_test)

        def score(y_true, pred):
            return {
                "r2": round(float(r2_score(y_true, pred)), 4),
                "mae": round(float(mean_absolute_error(y_true, pred)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 3),
            }

        DATA = df
        TARGET_COL = target_col
        FEATURE_COLS = feature_cols
        CATEGORICAL_COLS = cat_cols
        NUMERIC_COLS = num_cols
        RF_PIPELINE = rf
        XGB_PIPELINE = xgb
        METRICS = {"random_forest": score(y_test, rf_pred), "xgboost": score(y_test, xgb_pred)}
        TRAINING_ERROR = None
    except Exception as exc:
        TRAINING_ERROR = str(exc)

def get_ui_columns():
    if DATA is None:
        return None
    df = DATA
    return {
        "batting": find_col(df, ["batting_team", "bat_team", "batting team", "team_batting"]),
        "bowling": find_col(df, ["bowling_team", "bowl_team", "bowling team", "team_bowling"]),
        "venue": find_col(df, ["venue", "stadium", "ground"]),
        "over": find_col(df, ["current_over", "overs", "over"]),
        "wickets": find_col(df, ["current_wickets", "wickets", "wicket"]),
        "runs": find_col(df, ["current_runs", "runs", "score", "total_runs", "total"]),
        "date": find_col(df, ["date", "season", "match_date"]),
        "winner": find_col(df, ["winner", "winning_team", "match_winner"]),
    }

def available_values(col):
    if DATA is None or not col or col not in DATA.columns:
        return []
    vals = DATA[col].dropna().astype(str).unique().tolist()
    return sorted(vals)[:250]

def build_model_row(payload):
    # Begin with representative values for every trained feature, then override
    # columns that match the fields provided by the dashboard.
    row = {}
    for col in FEATURE_COLS:
        series = DATA[col]
        if series.dtype == "object":
            mode = series.dropna().astype(str).mode()
            row[col] = mode.iloc[0] if len(mode) else "Unknown"
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            row[col] = float(numeric.median()) if numeric.notna().any() else 0.0

    cols = get_ui_columns() or {}
    for field, key in [("batting", "batting_team"), ("bowling", "bowling_team"), ("venue", "venue")]:
        col = cols.get(field)
        if col and col in row:
            row[col] = payload.get(key, row[col])

    for field, key in [("over", "current_over"), ("wickets", "current_wickets"), ("runs", "current_runs")]:
        col = cols.get(field)
        if col and col in row:
            row[col] = payload.get(key, row[col])

    # Some common feature names might not have been identified by the exact matcher.
    aliases = {
        "batting_team": ["batting_team", "bat_team", "batting team"],
        "bowling_team": ["bowling_team", "bowl_team", "bowling team"],
        "venue": ["venue", "stadium", "ground"],
        "current_over": ["current_over", "overs", "over"],
        "current_wickets": ["current_wickets", "wickets", "wicket"],
        "current_runs": ["current_runs", "runs", "score", "total_runs", "total"],
    }
    for payload_key, names in aliases.items():
        for col in FEATURE_COLS:
            norm = str(col).strip().lower().replace("_", " ")
            if any(name.replace("_", " ") in norm for name in names):
                row[col] = payload.get(payload_key, row[col])

    # Make types align with training data.
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            row[col] = str(row[col])
        else:
            try:
                row[col] = float(row[col])
            except (TypeError, ValueError):
                row[col] = 0.0
    return pd.DataFrame([row], columns=FEATURE_COLS)

def historical_stats(payload):
    cols = get_ui_columns() or {}
    result = {
        "matches_played": "—",
        "batting_team_wins": "—",
        "bowling_team_wins": "—",
        "average_first_innings_score": "—",
        "recent_matches": [],
    }
    if DATA is None:
        return result
    batting_col, bowling_col = cols.get("batting"), cols.get("bowling")
    venue_col, winner_col = cols.get("venue"), cols.get("winner")
    if not batting_col or not bowling_col:
        return result
    bat, bowl = payload.get("batting_team"), payload.get("bowling_team")
    match_df = DATA[
        (DATA[batting_col].astype(str) == str(bat)) &
        (DATA[bowling_col].astype(str) == str(bowl))
    ].copy()
    if venue_col and payload.get("venue"):
        venue_matches = match_df[match_df[venue_col].astype(str) == str(payload["venue"])]
        if not venue_matches.empty:
            match_df = venue_matches
    result["matches_played"] = int(len(match_df))
    if winner_col and winner_col in match_df.columns:
        result["batting_team_wins"] = int((match_df[winner_col].astype(str) == str(bat)).sum())
        result["bowling_team_wins"] = int((match_df[winner_col].astype(str) == str(bowl)).sum())
    score_col = cols.get("runs")
    if score_col and score_col in match_df.columns:
        scores = pd.to_numeric(match_df[score_col], errors="coerce").dropna()
        if not scores.empty:
            result["average_first_innings_score"] = round(float(scores.mean()), 1)
    date_col = cols.get("date")
    display_cols = [date_col, venue_col, score_col, winner_col]
    match_df = match_df.tail(8)
    for _, row in match_df.iterrows():
        result["recent_matches"].append({
            "date_or_season": str(row[date_col]) if date_col else "—",
            "venue": str(row[venue_col]) if venue_col else "—",
            "batting_team_score": str(row[score_col]) if score_col else "—",
            "bowling_team_score": "—",
            "winner": str(row[winner_col]) if winner_col and pd.notna(row[winner_col]) else "—",
        })
    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/options")
def api_options():
    if DATA is None:
        return jsonify({"error": TRAINING_ERROR or "Model training did not complete. Check ipl_data.csv."}), 503
    cols = get_ui_columns()
    teams = sorted(set(available_values(cols["batting"]) + available_values(cols["bowling"])))
    return jsonify({"teams": teams, "venues": available_values(cols["venue"]), "metrics": METRICS})

@app.post("/api/predict")
def api_predict():
    if DATA is None or RF_PIPELINE is None or XGB_PIPELINE is None:
        return jsonify({"error": TRAINING_ERROR or "Models are not ready."}), 503
    payload = request.get_json(silent=True) or {}
    required = ["batting_team", "bowling_team", "venue", "current_over", "current_wickets", "current_runs"]
    if any(payload.get(key) in (None, "") for key in required):
        return jsonify({"error": "Please provide all match inputs, including current runs."}), 400
    if payload["batting_team"] == payload["bowling_team"]:
        return jsonify({"error": "Batting and bowling teams must be different."}), 400
    try:
        model_row = build_model_row(payload)
        rf_score = max(float(payload["current_runs"]), float(RF_PIPELINE.predict(model_row)[0]))
        xgb_score = max(float(payload["current_runs"]), float(XGB_PIPELINE.predict(model_row)[0]))
        return jsonify({
            "predictions": {"random_forest": round(rf_score), "xgboost": round(xgb_score)},
            "metrics": METRICS,
            "stats": historical_stats(payload)
        })
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

if __name__ == "__main__":
    load_and_train()
    if TRAINING_ERROR:
        print("WARNING:", TRAINING_ERROR)
    else:
        print("Models trained. Evaluation:", METRICS)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
