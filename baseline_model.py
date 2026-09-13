from pathlib import Path
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parent
file_path = (
    project_dir / "data" / "processed"
    / "turbine1_2022_review_flags.csv"
)

df = pd.read_csv(file_path)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

# Accept Boolean values or their CSV text representation.
passes_screen = (
    df["passes_initial_screen"]
    .astype(str).str.lower().eq("true")
)

# Basic physical checks; keep zero and negative power readings.
valid_inputs = (
    np.isfinite(df["wind_speed_mps"])
    & np.isfinite(df["power_kw"])
    & df["wind_speed_mps"].ge(0)
)

data = df.loc[passes_screen & valid_inputs].copy()

# Each bin covers 0.5 m/s: [0, 0.5), [0.5, 1.0), etc.
bin_width = 0.5
data["wind_bin"] = np.floor(
    data["wind_speed_mps"] / bin_width
).astype(int)

train = data.loc[
    data["timestamp_utc"] < "2022-09-01"
].copy()

validation = data.loc[
    (data["timestamp_utc"] >= "2022-09-01")
    & (data["timestamp_utc"] < "2022-11-01")
].copy()

# Learn the reference curve using training data only.
curve = train.groupby("wind_bin")["power_kw"].agg(
    median_power_kw="median",
    training_rows="size",
)

# Initial support rule: at least 30 training rows per bin.
# This is a development setting, not an industry standard.
supported_curve = curve.loc[curve["training_rows"] >= 30]

validation["predicted_kw"] = validation["wind_bin"].map(
    supported_curve["median_power_kw"]
)

evaluated = validation.dropna(subset=["predicted_kw"]).copy()

if evaluated.empty:
    raise RuntimeError("No validation rows have supported predictions.")

# Positive bias means overestimating measured power.
error = evaluated["predicted_kw"] - evaluated["power_kw"]
mae = error.abs().mean()
bias = error.mean()
rated_power_kw = 2050

print("\n--- BASELINE VALIDATION ---")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Evaluated rows:", len(evaluated))
print(
    "Prediction coverage:",
    f"{100 * len(evaluated) / len(validation):.2f}%"
)
print(f"MAE: {mae:.2f} kW")
print(f"MAE / rated power: {100 * mae / rated_power_kw:.2f}%")
print(f"Bias (predicted - measured): {bias:.2f} kW")

print("\n--- MONTHLY VALIDATION ---")
evaluated["absolute_error_kw"] = error.abs()
evaluated["error_kw"] = error
evaluated["month"] = evaluated["timestamp_utc"].dt.strftime("%Y-%m")

print(
    evaluated.groupby("month").agg(
        rows=("power_kw", "size"),
        mae_kw=("absolute_error_kw", "mean"),
        bias_kw=("error_kw", "mean"),
    ).round(2).to_string()
)

report_dir = project_dir / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

curve.to_csv(report_dir / "baseline_training_curve.csv")
validation.to_csv(
    report_dir / "baseline_validation_predictions.csv",
    index=False,
)

print("\nBaseline reports saved.")