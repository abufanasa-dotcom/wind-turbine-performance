from pathlib import Path
import pandas as pd

# Locate the project folder and the raw data.
project_dir = Path(__file__).resolve().parent
raw_dir = project_dir / "data" / "raw"

# Find the measurement file for turbine 1.
files = list(raw_dir.rglob("Turbine_Data_Kelmarsh_1_*.csv"))

if len(files) != 1:
    raise RuntimeError(
        f"Expected one turbine 1 file, found {len(files)}: {files}"
    )

file_path = files[0]

# Skip the nine description lines; keep the column header.
df = pd.read_csv(file_path, skiprows=9, low_memory=False)

# Remove the leading '#' and spaces from column names.
df.columns = [
    name.lstrip("#").strip()
    for name in df.columns
]

print(f"\nFile: {file_path.name}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nFirst 5 rows — first 6 columns:")
print(df.iloc[:5, :6].to_string(index=False))

print("\nAll column names:")
for number, name in enumerate(df.columns, start=1):
    print(f"{number}. {name}")

print("\nTop 10 columns by missing values (%):")
missing_percent = df.isna().mean().mul(100).sort_values(ascending=False)
print(missing_percent.head(10).round(2).to_string())
# Show candidate columns for the performance analysis.
keywords = ("power", "direction", "temperature", "status", "availability")

candidate_columns = [
    name for name in df.columns
    if name == "Wind speed (m/s)"
    or any(word in name.lower() for word in keywords)
]

print("\nCandidate columns and missing values:")
for name in candidate_columns:
    missing = df[name].isna().mean() * 100
    print(f"{name} | missing: {missing:.2f}%")
    # 1. Parse timestamps using the UTC timezone stated in the file.
timestamps = pd.to_datetime(
    df["Date and time"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
    utc=True,
)

valid_times = timestamps.dropna()

# Expected timestamps: every 10 minutes throughout 2022.
expected_times = pd.date_range(
    start="2022-01-01",
    end="2023-01-01",
    freq="10min",
    inclusive="left",
    tz="UTC",
)

observed_times = pd.DatetimeIndex(valid_times.unique())

print("\n--- TIME CHECK ---")
print("Invalid timestamps:", timestamps.isna().sum())
print("Duplicate timestamps:", valid_times.duplicated().sum())
print("Timestamps in ascending order:", valid_times.is_monotonic_increasing)
print("First timestamp:", valid_times.min())
print("Last timestamp:", valid_times.max())
print("Missing expected timestamps:", len(expected_times.difference(observed_times)))
print("Unexpected timestamps:", len(observed_times.difference(expected_times)))

# 2. Inspect numeric values without modifying the original dataframe.
columns_to_check = [
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Power (kW)",
    "Turbine Power setpoint (kW)",
    "Data Availability",
]

numeric_data = df[columns_to_check].apply(pd.to_numeric, errors="coerce")

conversion_failures = (
    df[columns_to_check].notna() & numeric_data.isna()
).sum()

print("\n--- NON-NUMERIC VALUES ---")
print(conversion_failures.to_string())

print("\n--- NUMERIC SUMMARY ---")
print(
    numeric_data.describe()
    .loc[["count", "min", "mean", "max"]]
    .T.round(2)
    .to_string()
)

# 3. Count rows with all three essential measurements.
core_columns = [
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Power (kW)",
]

core_data = numeric_data[core_columns]
finite_core = core_data.notna().all(axis=1) & ~core_data.isin(
    [float("inf"), float("-inf")]
).any(axis=1)

print("\n--- CORE DATA CHECK ---")
print("Rows with all three finite measurements:", finite_core.sum())
print("Rows missing a core value or containing infinity:", (~finite_core).sum())
print("Negative wind speeds:", (numeric_data["Wind speed (m/s)"] < 0).sum())
print(
    "Wind directions outside 0–360:",
    (
        (numeric_data["Wind direction (°)"] < 0)
        | (numeric_data["Wind direction (°)"] > 360)
    ).sum(),
)
print("Negative power readings:", (numeric_data["Power (kW)"] < 0).sum())
import matplotlib.pyplot as plt

# Keep complete, finite measurements for visualization only.
plot_data = numeric_data.loc[finite_core].copy()

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# Panel 1: all complete measurements.
axes[0].scatter(
    plot_data["Wind speed (m/s)"],
    plot_data["Power (kW)"],
    s=3,
    alpha=0.15,
    color="steelblue",
    rasterized=True,
)
axes[0].set_title("All complete measurements")

# Panel 2: highlight negative power readings.
negative_power = plot_data["Power (kW)"] < 0

axes[1].scatter(
    plot_data["Wind speed (m/s)"],
    plot_data["Power (kW)"],
    s=3,
    alpha=0.08,
    color="gray",
    rasterized=True,
)

axes[1].scatter(
    plot_data.loc[negative_power, "Wind speed (m/s)"],
    plot_data.loc[negative_power, "Power (kW)"],
    s=5,
    alpha=0.4,
    color="darkorange",
    label="Negative power",
    rasterized=True,
)
axes[1].set_title("Negative power highlighted")
axes[1].legend()

for ax in axes:
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Wind speed (m/s)")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Measured power (kW)")
fig.suptitle("Kelmarsh turbine 1 — 2022")

# Save the figure for the project documentation.
figure_dir = project_dir / "reports" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

figure_path = figure_dir / "01_raw_power_curve.png"
fig.savefig(figure_path, dpi=180)

print("\nFigure saved to:", figure_path)
print(
    "Negative power readings among plotted rows:",
    negative_power.sum(),
)
# Inspect power setpoints without filtering the data.
print("\n--- MOST COMMON POWER SETPOINTS (rounded kW) ---")
setpoints = numeric_data["Turbine Power setpoint (kW)"]

print(
    setpoints.round()
    .value_counts()
    .head(12)
    .to_string()
)

print("\n--- DATA AVAILABILITY DISTRIBUTION ---")
print(
    numeric_data["Data Availability"]
    .value_counts(dropna=False)
    .head(12)
    .to_string()
)

# Diagnostic window around the visible 1,640 kW band.
# These bounds are for inspection, not training filters.
band_mask = (
    numeric_data["Wind speed (m/s)"].between(10, 14)
    & numeric_data["Power (kW)"].between(1600, 1680)
)

print("\n--- INSPECTION OF THE 1,640 kW BAND ---")
print("Number of rows:", band_mask.sum())

print(
    numeric_data.loc[
        band_mask,
        [
            "Wind speed (m/s)",
            "Power (kW)",
            "Turbine Power setpoint (kW)",
            "Data Availability",
        ],
    ]
    .describe()
    .round(2)
    .to_string()
)

# Read the beginning of the status file before choosing a CSV parser.
status_files = list(raw_dir.rglob("Status_Kelmarsh_1_*.csv"))

print("\n--- STATUS FILE PREVIEW ---")
if len(status_files) != 1:
    print("Expected one status file; found:", len(status_files))
else:
    print("File:", status_files[0].name)

    with status_files[0].open(
        "r", encoding="utf-8-sig"
    ) as status_file:
        for _ in range(15):
            line = status_file.readline()
            if not line:
                break
            print(line.rstrip())
            # Check whether availability flags match complete measurements.
availability_ok = numeric_data["Data Availability"].eq(1)

print("\n--- AVAILABILITY AGREEMENT ---")
print(
    pd.crosstab(
        availability_ok.rename("availability_is_1"),
        finite_core.rename("core_measurements_complete"),
        dropna=False,
    ).to_string()
)

# Read the status file: nine description lines precede its header.
if len(status_files) != 1:
    raise RuntimeError("Expected exactly one turbine 1 status file.")

events = pd.read_csv(
    status_files[0],
    skiprows=9,
    encoding="utf-8-sig",
    dtype=str,
)

events.columns = events.columns.str.strip()

# Preserve raw values and parse dates into separate columns.
for source, destination in [
    ("Timestamp start", "start_utc"),
    ("Timestamp end", "end_utc"),
]:
    events[destination] = pd.to_datetime(
        events[source].str.strip().replace("-", pd.NA),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
        utc=True,
    )

print("\n--- EVENT TYPES ---")
print(events["Status"].value_counts(dropna=False).to_string())

print("\n--- IEC CATEGORIES ---")
print(events["IEC category"].value_counts(dropna=False).to_string())

print("\n--- EVENT TIMESTAMP CHECK ---")
print("Rows:", len(events))
print("Missing or unparsed starts:", events["start_utc"].isna().sum())
print("Missing or unparsed ends:", events["end_utc"].isna().sum())
print(
    "Ends earlier than starts:",
    (events["end_utc"] < events["start_utc"]).sum(),
)

print("\n--- END TIME AVAILABILITY BY EVENT TYPE ---")
print(
    pd.crosstab(
        events["Status"].fillna("Missing status"),
        events["end_utc"].notna().rename("has_parsed_end"),
    ).to_string()
)

print("\n--- MOST COMMON EVENT MESSAGES ---")
print(
    events.groupby(
        ["Status", "Code", "Message"],
        dropna=False,
    )
    .size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

# Preliminary energy consistency check.
# Assumption: each power value represents a 10-minute mean.
power = numeric_data["Power (kW)"]
finite_power = power.notna() & ~power.isin(
    [float("inf"), float("-inf")]
)

energy_from_power = power.loc[finite_power].sum() / 6

# Reference copied from the status file header.
header_energy_kwh = 5_165_695

print("\n--- PRELIMINARY ENERGY CHECK ---")
print("Intervals without finite power:", (~finite_power).sum())
print(f"Energy from available power readings: {energy_from_power:,.1f} kWh")
print(f"Status header production: {header_energy_kwh:,.1f} kWh")
print(
    "Difference relative to header:",
    f"{100 * (energy_from_power - header_energy_kwh) / header_energy_kwh:.2f}%",
)
import numpy as np

# Keep only events with a positive, explicitly recorded duration.
bounded_events = events.loc[
    events["end_utc"].notna()
    & (events["end_utc"] > events["start_utc"])
].copy()

print("\n--- ZERO-DURATION EVENTS ---")
print(
    events.loc[
        events["end_utc"].eq(events["start_utc"]),
        "Status",
    ].value_counts().to_string()
)

def flag_event_overlap(event_rows, timestamp_is_start):
    """Flag SCADA intervals overlapping at least one event."""
    if timestamp_is_start:
        interval_start = timestamps
        interval_end = timestamps + pd.Timedelta(minutes=10)
    else:
        interval_start = timestamps - pd.Timedelta(minutes=10)
        interval_end = timestamps

    flags = np.zeros(len(df), dtype=bool)

    for start, end in event_rows[
        ["start_utc", "end_utc"]
    ].itertuples(index=False, name=None):
        # Half-open intervals: touching boundaries alone do not overlap.
        overlaps = (interval_start < end) & (interval_end > start)
        flags |= overlaps.to_numpy()

    return pd.Series(flags, index=df.index)


event_groups = {
    "Stop": bounded_events["Status"].eq("Stop"),
    "Curtailment": bounded_events["Status"].eq("Curtailment"),
    "Communication": bounded_events["Status"].eq("Communication"),
    "Warning": bounded_events["Status"].eq("Warning"),
    "Wind sensor warnings": (
        bounded_events["Status"].eq("Warning")
        & bounded_events["Code"].str.strip().isin(["6635", "6525"])
    ),
}

overlap_flags = {}
summary_rows = []

for name, event_mask in event_groups.items():
    selected_events = bounded_events.loc[event_mask]

    start_flags = flag_event_overlap(selected_events, True)
    end_flags = flag_event_overlap(selected_events, False)

    overlap_flags[name] = {
        "start": start_flags,
        "end": end_flags,
    }

    summary_rows.append({
        "Event group": name,
        "Bounded events": len(selected_events),
        "Rows if timestamp=start": int(start_flags.sum()),
        "Rows if timestamp=end": int(end_flags.sum()),
        "Rows with different flags": int((start_flags != end_flags).sum()),
    })

print("\n--- EVENT OVERLAP COMPARISON ---")
print(pd.DataFrame(summary_rows).to_string(index=False))

print("\n--- CURTAILMENT DETAILS ---")
print(
    events.loc[
        events["Status"].eq("Curtailment"),
        [
            "Timestamp start",
            "Timestamp end",
            "Code",
            "Message",
            "IEC category",
        ],
    ].to_string(index=False)
)
# Build an auditable table: preserve rows and add review flags.
review = pd.DataFrame({
    "timestamp_utc": timestamps,
    "wind_speed_mps": numeric_data["Wind speed (m/s)"],
    "wind_direction_deg": numeric_data["Wind direction (°)"],
    "power_kw": numeric_data["Power (kW)"],
    "power_setpoint_kw": numeric_data["Turbine Power setpoint (kW)"],
    "core_complete": finite_core,
    "availability_ok": availability_ok,
})

# Flag overlap under either timestamp interpretation.
for group, column in [
    ("Stop", "stop_overlap"),
    ("Curtailment", "curtailment_overlap"),
    ("Communication", "communication_overlap"),
    ("Warning", "warning_overlap"),
    ("Wind sensor warnings", "wind_sensor_warning_overlap"),
]:
    review[column] = (
        overlap_flags[group]["start"]
        | overlap_flags[group]["end"]
    )

# Preliminary event-screening rule.
# General warnings remain visible but are not all excluded automatically.
exclusion_columns = [
    "stop_overlap",
    "curtailment_overlap",
    "communication_overlap",
    "wind_sensor_warning_overlap",
]

review["passes_initial_screen"] = (
    review["core_complete"]
    & review["availability_ok"]
    & ~review[exclusion_columns].any(axis=1)
)

print("\n--- INITIAL SCREENING ---")
print("Total rows:", len(review))
print("Pass initial screen:", review["passes_initial_screen"].sum())
print("Do not pass:", (~review["passes_initial_screen"]).sum())

# Inspect the surviving operating conditions.
candidate_rows = review.loc[review["passes_initial_screen"]]

print("\n--- REMAINING NON-POSITIVE POWER ---")
print("Power <= 0 kW:", candidate_rows["power_kw"].le(0).sum())

print("\n--- REMAINING SETPOINTS (rounded kW) ---")
print(
    candidate_rows["power_setpoint_kw"]
    .round()
    .value_counts(dropna=False)
    .head(12)
    .to_string()
)

# Test several candidate thresholds; do not choose one automatically.
print("\n--- SETPOINT THRESHOLD SENSITIVITY ---")
for threshold in [2000, 2030, 2050]:
    selected = candidate_rows["power_setpoint_kw"].ge(threshold)
    print(
        f"Setpoint >= {threshold} kW: "
        f"{selected.sum():,} rows"
    )

print("\n--- MONTHLY COUNTS AFTER INITIAL SCREEN ---")
print(
    candidate_rows.groupby(
        candidate_rows["timestamp_utc"].dt.strftime("%Y-%m")
    ).size().to_string()
)

# Save flags for documentation and subsequent analysis.
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

review_path = processed_dir / "turbine1_2022_review_flags.csv"
review.to_csv(review_path, index=False)

print("\nReview table saved to:", review_path)
plt.show()