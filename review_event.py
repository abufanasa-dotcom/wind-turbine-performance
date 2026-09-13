from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

project_dir = Path(__file__).resolve().parent
report_dir = project_dir / "reports"

events = pd.read_csv(
    report_dir / "underperformance_review_events.csv"
)
predictions = pd.read_csv(
    report_dir / "holdout_review_flags.csv"
)
measurements = pd.read_csv(
    project_dir / "data" / "processed"
    / "turbine1_2022_review_flags.csv"
)

if events.empty:
    raise RuntimeError("No review events found.")

for frame in [predictions, measurements]:
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True
    )

# Select the largest estimated energy-gap event.
event = events.sort_values(
    "estimated_gap_kwh", ascending=False
).iloc[0]

first = pd.to_datetime(event["first_sample_utc"], utc=True)
last = pd.to_datetime(event["last_sample_utc"], utc=True)

window_start = first - pd.Timedelta(hours=2)
window_end = last + pd.Timedelta(hours=2)

window = measurements.loc[
    measurements["timestamp_utc"].between(window_start, window_end)
].copy()

window = window.merge(
    predictions[["timestamp_utc", "selected_ml_kw"]],
    on="timestamp_utc",
    how="left",
    validate="one_to_one",
).sort_values("timestamp_utc")

# Restore a regular time grid so gaps are not joined by plotted lines.
time_grid = pd.date_range(
    window_start, window_end, freq="10min"
)
window = window.set_index("timestamp_utc").reindex(time_grid)

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    constrained_layout=True,
)

axes[0].plot(
    window.index, window["power_kw"],
    label="Measured power", marker=".", linewidth=1.3,
)
axes[0].plot(
    window.index, window["selected_ml_kw"],
    label="Model estimate", marker=".", linewidth=1.3,
)
axes[0].plot(
    window.index, window["power_setpoint_kw"],
    label="Power setpoint", linestyle="--", alpha=0.8,
)
axes[0].set_ylabel("Power (kW)")
axes[0].legend()

axes[1].plot(
    window.index, window["wind_speed_mps"],
    color="seagreen", marker=".",
)
axes[1].set_ylabel("Wind speed (m/s)")
axes[1].set_xlabel("Sample timestamp (UTC)")

for ax in axes:
    # Highlight sample timestamps, not confirmed physical event boundaries.
    ax.axvspan(first, last, color="orange", alpha=0.15)
    ax.grid(alpha=0.25)

axes[1].xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M", tz=first.tzinfo)
)

fig.suptitle(
    f"Kelmarsh 1 — review event on {first:%Y-%m-%d}\n"
    "Shading: first to last flagged sample"
)

figure_dir = report_dir / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

figure_path = figure_dir / "02_largest_event_review.png"
fig.savefig(figure_path, dpi=180)

print("\n--- FLAGGED SAMPLE DETAILS ---")
print(
    window.loc[first:last, [
        "wind_speed_mps",
        "power_kw",
        "selected_ml_kw",
        "power_setpoint_kw",
    ]].to_string(float_format=lambda value: f"{value:.2f}")
)
print("\nFigure saved to:", figure_path)
# Inspect operational messages around the plotted event.
status_files = list(
    (project_dir / "data" / "raw").rglob(
        "Status_Kelmarsh_1_*.csv"
    )
)

if len(status_files) != 1:
    raise RuntimeError("Expected one turbine 1 status file.")

status = pd.read_csv(
    status_files[0],
    skiprows=9,
    encoding="utf-8-sig",
    dtype=str,
)
status.columns = status.columns.str.strip()

for source, destination in [
    ("Timestamp start", "start_utc"),
    ("Timestamp end", "end_utc"),
]:
    status[destination] = pd.to_datetime(
        status[source].str.strip().replace("-", pd.NA),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
        utc=True,
    )

# Include bounded events overlapping the window,
# plus messages whose recorded start falls inside it.
overlaps_window = (
    status["start_utc"].le(window_end)
    & status["end_utc"].gt(window_start)
)
starts_in_window = status["start_utc"].between(
    window_start, window_end
)

context = status.loc[
    overlaps_window | starts_in_window,
    [
        "start_utc",
        "end_utc",
        "Status",
        "Code",
        "Message",
        "IEC category",
    ],
].sort_values("start_utc")

print("\n--- OPERATIONAL CONTEXT ---")
print("Window:", window_start, "to", window_end)

if context.empty:
    print("No matching operational records in this window.")
else:
    print(context.to_string(index=False))

context.to_csv(
    report_dir / "largest_event_operational_context.csv",
    index=False,
)
plt.show()