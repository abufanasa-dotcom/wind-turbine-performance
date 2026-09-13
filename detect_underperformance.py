from pathlib import Path
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parent
report_dir = project_dir / "reports"

validation = pd.read_csv(
    report_dir / "model_validation_predictions.csv"
)
holdout = pd.read_csv(
    report_dir / "holdout_predictions.csv"
)

# Positive deficit means measured power is below predicted power.
validation["deficit_kw"] = (
    validation["ML_speed_direction"] - validation["power_kw"]
)

# Fix the threshold using validation data only.
# A 99th-percentile threshold is a starting review rule.
threshold_kw = max(
    0.0,
    float(validation["deficit_kw"].quantile(0.99)),
)

holdout["timestamp_utc"] = pd.to_datetime(
    holdout["timestamp_utc"], utc=True
)
holdout = holdout.sort_values("timestamp_utc").reset_index(drop=True)

holdout["deficit_kw"] = (
    holdout["selected_ml_kw"] - holdout["power_kw"]
)

# Stay within the same wind-bin support used in model comparison.
supported = holdout["baseline_kw"].notna()
finite_values = np.isfinite(
    holdout[["selected_ml_kw", "power_kw"]]
).all(axis=1)

holdout["above_threshold"] = (
    supported
    & finite_values
    & holdout["deficit_kw"].gt(threshold_kw)
)

# A missing or excluded interval must break the sequence.
consecutive = holdout["timestamp_utc"].diff().eq(
    pd.Timedelta(minutes=10)
)

new_group = (
    ~consecutive
    | holdout["above_threshold"].ne(
        holdout["above_threshold"].shift()
    )
)
holdout["group_id"] = new_group.cumsum()

events = (
    holdout.loc[holdout["above_threshold"]]
    .groupby("group_id")
    .agg(
        first_sample_utc=("timestamp_utc", "min"),
        last_sample_utc=("timestamp_utc", "max"),
        intervals=("deficit_kw", "size"),
        mean_deficit_kw=("deficit_kw", "mean"),
        max_deficit_kw=("deficit_kw", "max"),
        deficit_sum_kw=("deficit_kw", "sum"),
        mean_wind_mps=("wind_speed_mps", "mean"),
    )
)

# Keep runs of at least three consecutive intervals.
events = events.loc[events["intervals"] >= 3].copy()

events["represented_minutes"] = events["intervals"] * 10
events["estimated_gap_kwh"] = events["deficit_sum_kw"] / 6

events = (
    events.drop(columns="deficit_sum_kw")
    .sort_values("estimated_gap_kwh", ascending=False)
    .reset_index()
)

events.insert(
    0, "review_event_id", np.arange(1, len(events) + 1)
)

print("\n--- UNDERPERFORMANCE REVIEW ---")
print(f"Validation-derived threshold: {threshold_kw:.2f} kW")
print("Minimum consecutive intervals: 3")
print("Eligible holdout rows:", len(holdout))
print("Supported rows:", int(supported.sum()))
print("Review events:", len(events))

if events.empty:
    print("No events met the fixed review rule.")
else:
    print("\n--- TOP 10 EVENTS BY ESTIMATED ENERGY GAP ---")
    display_columns = [
        "review_event_id",
        "first_sample_utc",
        "last_sample_utc",
        "represented_minutes",
        "mean_deficit_kw",
        "estimated_gap_kwh",
        "mean_wind_mps",
    ]
print(events[display_columns].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
events.to_csv(
    report_dir / "underperformance_review_events.csv",
    index=False,
)
holdout.to_csv(
    report_dir / "holdout_review_flags.csv",
    index=False,
)

print("\nReview reports saved.")