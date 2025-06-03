# data_mimic_sb
# Energy Community Data Processing Pipeline

This repository contains a small prototype used to transform building and time series data for energy community modelling.  Three processing steps are implemented:

1. **Process Buildings** – cleans raw building information and applies simple solar and battery scenarios.
2. **Process Time Series** – converts time series exports into a compact per‑building format.
3. **Generate Assignments** – groups buildings into lines/clusters and outputs assignment files.

The pipeline can be run interactively via Streamlit or through a command line interface.

## Prerequisites

- Python 3.8+
- Packages listed in `requirements.txt`:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Launch the graphical interface which guides you through uploading data and running each step:

```bash
streamlit run app.py
```

Uploads can include the raw building and time series CSVs or previously processed files to skip steps.
Processed results can be downloaded from the app once each step completes.

## Running from the Command Line

The CLI provides an interactive text menu and reads configuration paths from `main_mimic.py`.
Update the paths at the top of that file to point to your input and output CSVs, then execute:

```bash
python main_mimic.py
```

Choose the desired menu option to run one or more pipeline steps.

## Input File Formats

### Buildings CSV

The building input file must contain at least:

- **`building_id`** &ndash; unique identifier (alternative columns such as `OBJECTID` or `fid` are automatically detected).
- **`lat`** and **`lon`** &ndash; latitude and longitude of each building. Several common column names are recognised (`latitude`, `Longitude`, `x`, `y`, etc.).
- **`building_function`** (optional but recommended) &ndash; used for solar and battery scenario assignment. Missing values default to `default`.
- Additional columns like `area`, `label` or `peak_load_kW` are preserved if present.

### Time Series CSV

Time series files are expected to contain:

- A building identifier column (`BuildingID` or `building_id`).
- A **`VariableName`** column indicating the energy category. A column named
  **`Energy`** is also accepted and treated the same way.
- One column per time step (e.g. `01/01 00:00:00`, `01/01 01:00:00`, ...).

The scripts map variable names such as `Electricity:Facility [J](Daily)` or `Heating:EnergyTransfer [J](Hourly)` to internal categories.
Any unrecognised variables result in synthetic placeholder profiles.

## Output Files

Each pipeline step produces CSV output:

1. **Process Buildings** → `buildings_processed.csv`
2. **Process Time Series** → `time_series_processed.csv`
3. **Generate Assignments** →
   - `building_assignments_generated.csv` &ndash; mapping of each building to a line ID
   - `buildings_processed_with_lines.csv` &ndash; processed building data including the assigned line

File paths can be customised in `main_mimic.py` (for the CLI) or downloaded directly from the Streamlit interface.
