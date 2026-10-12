# IPL Score Predictor Dashboard (Flask + Random Forest + XGBoost)

## Files
- `app.py` — Flask app, trains Random Forest and XGBoost from `ipl_data.csv`, and exposes API endpoints.
- `templates/index.html` — dashboard page.
- `static/css/style.css` — styling.
- `static/javascript/dashboard.js` — frontend behaviour.
- `requirements.txt` — Python packages.

## Run locally
1. Install Python 3.10 or newer.
2. Put `ipl_data.csv` in this same folder (download it from the dataset repository you shared).
3. Open a terminal in this folder.
4. Run `python -m venv .venv` then activate it:
   - Windows: `.venv\\Scripts\\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Run `pip install -r requirements.txt`.
6. Run `python app.py`.
7. Open `http://127.0.0.1:5000` in your browser.

## Important model note
Your uploaded notebook and `xgboost.py` use `data.iloc[:, 2:-1]` as features and the last column as the target. This app follows that same convention when training. The app maps dashboard values into matching feature columns when their names look like batting team, bowling team, venue, over, wickets, or runs. Any additional model features not entered on the dashboard are filled with representative values (mode for text, median for numeric), so prediction quality may be limited if the original model expects player-level features or other information. For a scientifically valid model, retrain using features that are available at prediction time.

The historical-statistics section is approximate and depends on your CSV containing recognizable columns for batting team, bowling team, venue, runs/score, and winner. If your dataset is ball-by-ball rather than match-summary data, its statistics should be aggregated first.

The frontend uses Chart.js from a CDN, so an internet connection is needed for the chart.


### Simplified algorithm comparison
The dashboard compares only the predicted final score from Random Forest and XGBoost. R², MAE, and RMSE analytics have been removed from the page. After entering match details, the comparison chart and table show both predicted scores and their absolute difference.
