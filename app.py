from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Wind Performance Review",
    layout="wide",
)

root = Path(__file__).resolve().parent
reports = root / "reports"

required_files = {
    "scores": reports / "holdout_scores.csv",
    "events": reports / "underperformance_review_events.csv",
    "predictions": reports / "holdout_review_flags.csv",
    "measurements": (
        root / "data" / "processed"
        / "turbine1_2022_review_flags.csv"
    ),
}

missing = [str(path) for path in required_files.values() if not path.exists()]
if missing:
    st.error("Missing project files:")
    st.write(missing)
    st.stop()

scores = pd.read_csv(required_files["scores"], index_col=0)
events = pd.read_csv(required_files["events"])
predictions = pd.read_csv(required_files["predictions"])
measurements = pd.read_csv(required_files["measurements"])

for frame in [predictions, measurements]:
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True
    )

st.title("Wind Turbine Performance Review")
st.caption(
    "Kelmarsh turbine 1 · Historical SCADA analysis · 2022 · UTC"
)

st.write(
    "Compare measured power with a wind-speed-and-direction model, "
    "then inspect sustained deviations."
)

a, b, c = st.columns(3)
a.metric(
    "Evaluation MAE",
    f"{scores.loc['selected_ml_kw', 'MAE_kW']:.2f} kW",
)
b.metric(
    "MAE reduction vs baseline",
    f"{scores.loc['selected_ml_kw', 'MAE_reduction_pct']:.2f}%",
)
c.metric("Events requiring review", len(events))

with st.expander("Evaluation method and limits"):
    st.write(
        "Training: January–August. Model selection and alert threshold: "
        "September–October. Additional temporal evaluation: November–December. "
        "All periods are from 2022; external-year validation is pending."
    )
    st.write(
        "Reported model scores use the same screened, baseline-supported rows. "
        "MAE reduction is not an accuracy percentage."
    )
    st.write(
        "Alerts indicate model deviations, not confirmed faults. "
        "Estimated energy gaps are not verified recoverable losses."
    )
    st.write(
        "Timestamp start/end convention is unresolved. Event screening "
        "excludes overlap under either interpretation."
    )

st.subheader("Model evaluation")
st.dataframe(scores.round(2))

st.subheader("Review events")
st.dataframe(events.drop(columns=["group_id"], errors="ignore"))

if events.empty:
    st.info("No events met the review rule.")
    st.stop()

selected_id = st.selectbox(
    "Choose an event",
    events["review_event_id"].tolist(),
)
event = events.loc[events["review_event_id"].eq(selected_id)].iloc[0]

first = pd.to_datetime(event["first_sample_utc"], utc=True)
last = pd.to_datetime(event["last_sample_utc"], utc=True)
start = first - pd.Timedelta(hours=2)
end = last + pd.Timedelta(hours=2)

window = measurements.loc[
    measurements["timestamp_utc"].between(start, end)
].merge(
    predictions[["timestamp_utc", "selected_ml_kw"]],
    on="timestamp_utc",
    how="left",
    validate="one_to_one",
)

window = (
    window.set_index("timestamp_utc")
    .reindex(pd.date_range(start, end, freq="10min"))
)
window.index.name = "timestamp_utc"

st.write(
    f"Flagged sample timestamps: {first:%Y-%m-%d %H:%M} "
    f"to {last:%H:%M} UTC"
)

left, right = st.columns(2)
left.metric(
    "Represented measurement time",
    f"{int(event['represented_minutes'])} min",
)
right.metric(
    "Estimated gap within flagged intervals",
    f"{event['estimated_gap_kwh']:.2f} kWh",
)

power_chart = go.Figure()
for column, label in [
    ("power_kw", "Measured power"),
    ("selected_ml_kw", "Model estimate"),
    ("power_setpoint_kw", "Power setpoint"),
]:
    power_chart.add_trace(go.Scatter(
        x=window.index,
        y=window[column],
        name=label,
        mode="lines+markers",
        connectgaps=False,
    ))

power_chart.update_layout(
    yaxis_title="Power (kW)",
    xaxis_title="Sample timestamp (UTC)",
    hovermode="x unified",
)
st.plotly_chart(power_chart)

wind_chart = go.Figure(go.Scatter(
    x=window.index,
    y=window["wind_speed_mps"],
    mode="lines+markers",
    name="Wind speed",
    connectgaps=False,
))
wind_chart.update_layout(
    yaxis_title="Wind speed (m/s)",
    xaxis_title="Sample timestamp (UTC)",
)
st.plotly_chart(wind_chart)

st.subheader("Engineering assessment")
st.write(
    "Review candidate — cause not established. "
    "Inspect operating records, measurement quality, and model limitations."
)

# Operational-context review was completed for the largest event only.
largest_id = events.loc[
    events["estimated_gap_kwh"].idxmax(), "review_event_id"
]
if selected_id == largest_id:
    st.info(
        "For the largest event, the reviewed ±2-hour window contained "
        "no matching bounded events or messages starting in that window. "
        "Earlier messages without end times remain unresolved."
    )
else:
    st.caption("Individual operational-context review is pending.")

st.download_button(
    "Download event table (CSV)",
    data=events.to_csv(index=False).encode("utf-8"),
    file_name="underperformance_review_events.csv",
    mime="text/csv",
)

st.caption(
    "Source: Cubico Sustainable Investments — Kelmarsh dataset, "
    "Zenodo record 16807551, CC BY 4.0. "
    "Independent portfolio analysis; no affiliation implied."
)