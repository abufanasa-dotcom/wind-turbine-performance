from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

project_dir = Path(__file__).resolve().parent
data_path = (
    project_dir / "data" / "processed"
    / "turbine1_2022_review_flags.csv"
)

df = pd.read_csv(data_path)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

passes_screen = (
    df["passes_initial_screen"]
    .astype(str).str.lower().eq("true")
)

required = ["wind_speed_mps", "wind_direction_deg", "power_kw"]
valid = (
    np.isfinite(df[required]).all(axis=1)
    & df["wind_speed_mps"].ge(0)
    & df["wind_direction_deg"].between(0, 360)
)

data = df.loc[passes_screen & valid].copy()

# Encode direction as a circular variable.
angle = np.deg2rad(data["wind_direction_deg"])
data["direction_sin"] = np.sin(angle)
data["direction_cos"] = np.cos(angle)

data["wind_bin"] = np.floor(
    data["wind_speed_mps"] / 0.5
).astype(int)

train = data.loc[
    data["timestamp_utc"] < "2022-09-01"
].copy()

validation = data.loc[
    (data["timestamp_utc"] >= "2022-09-01")
    & (data["timestamp_utc"] < "2022-11-01")
].copy()

# Rebuild the same baseline using training data only.
curve = train.groupby("wind_bin")["power_kw"].agg(
    median_power_kw="median",
    training_rows="size",
)
curve = curve.loc[curve["training_rows"] >= 30]

validation["baseline_kw"] = validation["wind_bin"].map(
    curve["median_power_kw"]
)

# Fixed starting settings, without tuning on validation results.
settings = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42,
)

# Compare wind speed alone against wind speed plus direction.
feature_sets = {
    "ML_speed": ["wind_speed_mps"],
    "ML_speed_direction": [
        "wind_speed_mps",
        "direction_sin",
        "direction_cos",
    ],
}

for name, features in feature_sets.items():
    model = HistGradientBoostingRegressor(**settings)
    model.fit(train[features], train["power_kw"])
    validation[name] = model.predict(validation[features])

# Evaluate all models on exactly the same rows.
common = validation.dropna(subset=["baseline_kw"]).copy()
common["month"] = common["timestamp_utc"].dt.strftime("%Y-%m")

if common.empty:
    raise RuntimeError("No common validation rows.")

prediction_columns = [
    "baseline_kw",
    "ML_speed",
    "ML_speed_direction",
]

def score(frame, prediction_column):
    error = frame[prediction_column] - frame["power_kw"]
    return {
        "MAE_kW": error.abs().mean(),
        "Bias_kW": error.mean(),
    }

overall = pd.DataFrame({
    name: score(common, name)
    for name in prediction_columns
}).T

baseline_mae = overall.loc["baseline_kw", "MAE_kW"]
overall["MAE_reduction_pct"] = (
    100 * (baseline_mae - overall["MAE_kW"]) / baseline_mae
    if baseline_mae > 0 else np.nan
)

print("\n--- MODEL COMPARISON ---")
print("Training rows:", len(train))
print("Common validation rows:", len(common))
print(overall.round(2).to_string())

monthly_rows = []
for month, group in common.groupby("month"):
    for name in prediction_columns:
        monthly_rows.append({
            "month": month,
            "model": name,
            "rows": len(group),
            **score(group, name),
        })

monthly = pd.DataFrame(monthly_rows)

print("\n--- MONTHLY MODEL COMPARISON ---")
print(monthly.round(2).to_string(index=False))

report_dir = project_dir / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

overall.to_csv(report_dir / "model_comparison.csv")
monthly.to_csv(report_dir / "monthly_model_comparison.csv", index=False)
common.to_csv(report_dir / "model_validation_predictions.csv", index=False)

print("\nComparison reports saved.")
# Freeze the chosen model and its existing settings.
# Train on January–August only, as before.
selected_features = [
    "wind_speed_mps",
    "direction_sin",
    "direction_cos",
]

selected_model = HistGradientBoostingRegressor(**settings)
selected_model.fit(train[selected_features], train["power_kw"])

# Additional temporal evaluation: November–December 2022.
holdout = data.loc[
    (data["timestamp_utc"] >= "2022-11-01")
    & (data["timestamp_utc"] < "2023-01-01")
].copy()

holdout["baseline_kw"] = holdout["wind_bin"].map(
    curve["median_power_kw"]
)
holdout["selected_ml_kw"] = selected_model.predict(
    holdout[selected_features]
)

# Compare both models on the same supported rows.
holdout_common = holdout.dropna(subset=["baseline_kw"]).copy()

if holdout_common.empty:
    raise RuntimeError("No supported rows in the holdout period.")

holdout_scores = pd.DataFrame({
    name: score(holdout_common, name)
    for name in ["baseline_kw", "selected_ml_kw"]
}).T

reference_mae = holdout_scores.loc["baseline_kw", "MAE_kW"]
holdout_scores["MAE_reduction_pct"] = (
    100 * (reference_mae - holdout_scores["MAE_kW"]) / reference_mae
    if reference_mae > 0 else np.nan
)

print("\n--- NOVEMBER–DECEMBER EVALUATION ---")
print("Eligible rows:", len(holdout))
print("Common evaluated rows:", len(holdout_common))
print(
    "Baseline coverage:",
    f"{100 * len(holdout_common) / len(holdout):.2f}%"
)
print(holdout_scores.round(2).to_string())

holdout_common["month"] = (
    holdout_common["timestamp_utc"].dt.strftime("%Y-%m")
)

monthly_holdout_rows = []

for month, group in holdout_common.groupby("month"):
    for name in ["baseline_kw", "selected_ml_kw"]:
        monthly_holdout_rows.append({
            "month": month,
            "model": name,
            "rows": len(group),
            **score(group, name),
        })

monthly_holdout = pd.DataFrame(monthly_holdout_rows)

print("\n--- MONTHLY HOLDOUT RESULTS ---")
print(monthly_holdout.round(2).to_string(index=False))

# Preserve results and the trained model.
holdout.to_csv(
    report_dir / "holdout_predictions.csv",
    index=False,
)
holdout_scores.to_csv(report_dir / "holdout_scores.csv")
monthly_holdout.to_csv(
    report_dir / "holdout_monthly_scores.csv",
    index=False,
)

import joblib

model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": selected_model,
        "features": selected_features,
        "training_period": "2022-01-01 to 2022-08-31",
        "target": "Power (kW)",
        "purpose": "Historical performance estimation",
    },
    model_dir / "power_model_v1.joblib",
)

print("\nEvaluation reports and model saved.")