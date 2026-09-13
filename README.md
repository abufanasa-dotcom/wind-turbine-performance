# Wind Turbine Performance Review

[Open the interactive dashboard](https://abufanasa-dotcom-wind-turbine-performance-app-vtq3j5.streamlit.app/)

Independent portfolio project by Ahmed Abufanas. Built with AI-assisted development and step-by-step engineering review.

Analyze historical SCADA measurements from Kelmarsh turbine 1, estimate power from wind conditions, and prioritize sustained deviations for human review. The Streamlit dashboard displays saved analysis results; it does not ingest live telemetry or run online predictions.

## Current status

Local Windows prototype demonstrated by the author. Data checks, model comparisons, event review, dashboard selection, CSV download, and methodology display were exercised. A fresh-environment rebuild, automated regression tests, external-year evaluation, and public deployment have not yet been verified.

## Data and attribution

Source: Charlie Plumley and Roberta Takeuchi, *Kelmarsh wind farm data*, v4, Cubico Sustainable Investments, Zenodo (2025).

- Dataset: https://zenodo.org/records/16807551
- DOI: https://doi.org/10.5281/zenodo.16807551
- Data license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- This project transforms the source data through screening, feature engineering, modelling, and aggregation. No affiliation or endorsement is implied. The data license does not automatically license the project code.

The current analysis uses 2022 only: 52,560 ten-minute timestamps and 303 source columns for turbine 1. The Senvion MM92 has rated power 2,050 kW, rotor diameter 92 m, and hub height 78.5 m.

Download these files from the dataset page:

| File | Purpose |
| --- | --- |
| `Kelmarsh_WT_static.csv` | Turbine specifications |
| `Kelmarsh_WT_dataSignalMapping.csv` | Signal definitions |
| `Kelmarsh_SCADA_2022_4457.zip` | 2022 measurements and event records |

Place the three downloads in `data/raw/`, then extract the ZIP into a subfolder there. Keep original files unchanged. The scripts search recursively for one `Turbine_Data_Kelmarsh_1_*.csv` and one `Status_Kelmarsh_1_*.csv`. Adding other years to this search tree requires updating the file selection first.

## Environment

Author-reported working environment: Windows, PowerShell, VS Code, Python **3.14.7** (terminal screenshot). The uploaded `requirements.txt` records the installed environment; package availability and installation on another machine have not been independently verified.

Selected versions from that file:

| Package | Version |
| --- | --- |
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| plotly | 7.0.0 |
| streamlit | 1.63.0 |
| joblib | 1.6.0 |

## Setup and execution

Open the project root in VS Code. Run these commands in a PowerShell terminal at the project root. If the existing `.venv` already works, reuse it instead of recreating it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the existing project scripts in this order:

```powershell
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe baseline_model.py
.\.venv\Scripts\python.exe compare_models.py
.\.venv\Scripts\python.exe detect_underperformance.py
.\.venv\Scripts\python.exe review_event.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Close Matplotlib figure windows to let plotting scripts finish. Leave the Streamlit terminal running while using the dashboard. An optional first-run email prompt accepts a blank response. Open the Local URL shown in the terminal, usually `http://localhost:8501`. This address refers to the local computer, not a published portfolio URL. Stop the server with Ctrl+C.

## Project files

| Path | Role |
| --- | --- |
| `inspect_data.py` | Source inspection, timestamp checks, event screening, review flags, initial plot |
| `baseline_model.py` | Wind-speed-bin reference and development evaluation |
| `compare_models.py` | ML comparison, additional temporal evaluation, model serialization |
| `detect_underperformance.py` | Validation-derived threshold and persistent-deviation grouping |
| `review_event.py` | Largest-event visualization and operational-context inspection |
| `app.py` | Local dashboard reading saved CSV results |
| `requirements.txt` | Captured environment dependencies |
| `data/raw/` | Original source files and extracted data |
| `data/processed/turbine1_2022_review_flags.csv` | All original timestamps with derived values and screening flags |
| `reports/` | Evaluation tables, predictions, event tables, and context |
| `reports/figures/` | Saved analysis plots |
| `models/power_model_v1.joblib` | Trained model plus feature and training-period metadata |
| `exports/` | Copies downloaded from the dashboard |

The README accompanies these scripts; it is not a standalone implementation package. Only load model files from trusted sources.

## Data quality and screening

- All expected 2022 timestamps are present, ordered, and unique. No invalid timestamps were found.
- 52,006 rows contain finite wind speed, wind direction, and power; 554 do not. For this file, that completeness flag exactly matches `Data Availability = 1`.
- Power alone is missing in 550 intervals. Integrating available power as ten-minute means gives 5,160,034 kWh, versus 5,165,695 kWh in the event-file header: approximately -0.11%. This supports the unit interpretation but is not a complete reconciliation.
- The signal mapping labels Power with kWh while the measurement header says kW. The discrepancy is recorded; the analysis provisionally uses the measurement-header unit and ten-minute mean interpretation.
- Bounded Stop, Curtailment, Communication, and selected wind-sensor-warning events (codes 6635 and 6525) are used for screening. Any overlap under either timestamp-start or timestamp-end interpretation excludes the row. General warnings remain flagged but are not universally excluded.
- Informational messages without end timestamps are not automatically extended forward. Missing records do not establish normal operation.
- Initial screening retains 50,099 rows. It is an event-screened reference population, not a certified healthy-operation dataset. Non-positive power is retained.
- Low power-setpoint values alone are not classified as curtailment: the signal varies with operating conditions. Potential-power estimates and measured-power-derived features are excluded from model inputs.

## Models and temporal evaluation

Training: January–August 2022, 33,928 screened rows. Development/model selection: September–October. Additional evaluation: November–December. Whole-year exploratory inspection occurred before modelling, so the last period is not described as a fully untouched external test.

Baseline: median power in 0.5 m/s wind-speed bins, requiring at least 30 training observations per bin. ML: `HistGradientBoostingRegressor`, absolute-error loss, learning rate 0.05, 200 iterations, 15 maximum leaf nodes, minimum 50 samples per leaf, L2 regularization 1.0, early stopping disabled, random seed 42.

The selected model uses wind speed and sine/cosine encoding of wind direction. It estimates power for contemporaneous measured wind conditions; it is not a next-day forecast. Temperature is not an input in this version.

| Development model | MAE (kW) | Bias (kW) |
| --- | ---: | ---: |
| Baseline | 56.21 | +30.77 |
| ML: speed | 50.94 | +30.44 |
| ML: speed and direction | 49.95 | +28.67 |

Development comparison uses 8,003 common rows out of 8,005 eligible rows. Direction improved aggregate MAE, but not September's MAE.

| November–December model | MAE (kW) | Bias (kW) | MAE reduction vs baseline |
| --- | ---: | ---: | ---: |
| Baseline | 56.17 | +17.10 | — |
| Selected ML | 48.79 | +13.40 | 13.14% |

Additional evaluation uses 8,137 common rows out of 8,166 eligible rows (99.64% baseline coverage). November ML MAE: 49.47 kW; December: 48.05 kW. Positive bias means overprediction. MAE reduction is not prediction accuracy or a financial saving. No confidence interval or statistical-significance claim has been established.

## Deviation review

Deficit = predicted power minus measured power. The development residuals' 99th percentile sets a fixed threshold of approximately 277.47 kW. At least three consecutive ten-minute exceedances are required. Missing, excluded, or below-threshold intervals break a run. Analysis is restricted to baseline-supported bins; it does not cover all operating periods.

Five candidate events were identified. The selected-event energy gap is the sum of positive deficits multiplied by 1/6 hour. It is a model-relative estimate, not a confirmed recoverable loss. The threshold is not an industry standard, a confidence bound, or a verified 1% fault false-alarm rate.

Largest event: six flagged samples on 17 November 2022, from 02:10 through 03:00 UTC. Six intervals represent 60 minutes, although first-to-last sample timestamps span 50 minutes. Physical interval boundaries remain uncertain. Mean deficit and estimated gap are approximately 350.88 kW and 350.88 kWh. Setpoints range from 2,038.75 to 2,060 kW; they do not show a low ceiling directly explaining the discrepancy.

No matching bounded events or messages starting within the reviewed 00:10–05:00 UTC window were found. Earlier open-ended messages remain unresolved. Cause: **not established**. Individual operational-context review of the other four candidates is pending.

## Evidence and remaining work

Author screenshots show successful event selection (event 2: 50 minutes, 276.96 kWh), CSV export with five events, and expansion of methodology notes. The displayed dashboard uses precomputed predictions, not the serialized model at runtime.

Before stronger deployment or performance claims: verify installation in a fresh environment; add meaningful automated tests for interval overlap, sequence breaks, and energy units; validate on a later year without retuning; investigate residuals and alternative physical explanations; review remaining candidates. The current dashboard's largest-event context note reflects the manually reviewed dataset and must be updated if inputs change.

## Interview walkthrough

1. Define the operational question and identify the published source.
2. Explain screening, the timestamp ambiguity, and why low setpoints were not blindly removed.
3. Compare the baseline and selected model on identical chronological evaluation rows.
4. Open a candidate event and distinguish observed measurements from model estimates.
5. State the unresolved cause and the evidence needed for an operational decision.

Describe the work as an independent engineering/data-analysis portfolio project. Do not present it as employment at Cubico, a production monitoring deployment, or validated predictive maintenance.
