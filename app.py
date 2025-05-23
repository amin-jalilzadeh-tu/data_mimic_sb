import streamlit as st
import pandas as pd
import numpy as np
import json
import io

# Import the refactored processing functions
# This assumes the script is run from the 'mimic' directory
# and the processing scripts are in a 'processing' subdirectory.
try:
    from processing.data_processing import process_building_data
    from processing.timeseries_processing import process_timeseries_data
    from processing.assignment_generation import generate_assignments
except ImportError:
    st.error("Could not import processing modules. Make sure the script is run from the 'mimic' directory and your processing scripts are inside a 'processing' sub-folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Cluster GNN Data Pipeline",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Default Scenario Configurations ---
# These are the same as in your main_mimic.py, serving as editable defaults
DEFAULT_SOLAR_SCENARIOS = {
    "default": {"penetration_rate": 0.10, "types": [{"name": "default_pv_small", "kWp_range": [2.0, 4.0], "weight": 0.6}, {"name": "default_pv_medium", "kWp_range": [4.1, 8.0], "weight": 0.4}]},
    "residential": {"penetration_rate": 0.25, "types": [{"name": "res_pv_standard", "kWp_range": [3.0, 7.0], "weight": 0.7}, {"name": "res_pv_large", "kWp_range": [7.1, 15.0], "weight": 0.3}]},
    "non_residential": {"penetration_rate": 0.40, "types": [{"name": "comm_pv_medium", "kWp_range": [10.0, 50.0], "weight": 0.5}, {"name": "comm_pv_large", "kWp_range": [50.1, 250.0], "weight": 0.3}, {"name": "comm_pv_xlarge", "kWp_range": [250.1, 1000.0], "weight": 0.2}]}
}

DEFAULT_BATTERY_SCENARIOS = {
    "default": {"penetration_rate_if_solar": 0.15, "types": [{"name": "default_batt_small", "kWh_range": [4.0, 8.0], "kW_range": [2.0, 4.0], "weight": 1.0}]},
    "residential": {"penetration_rate_if_solar": 0.30, "types": [{"name": "res_batt_daily", "kWh_range": [5.0, 15.0], "kW_range": [2.5, 7.0], "weight": 0.8}, {"name": "res_batt_backup", "kWh_range": [15.1, 30.0], "kW_range": [5.0, 10.0], "weight": 0.2}]},
    "non_residential": {"penetration_rate_if_solar": 0.50, "types": [{"name": "comm_batt_demand_peak", "kWh_range": [20.0, 100.0], "kW_range": [10.0, 50.0], "weight": 0.6}, {"name": "comm_batt_storage_large", "kWh_range": [100.1, 500.0], "kW_range": [50.1, 200.0], "weight": 0.4}]}
}

# --- Helper Functions ---
@st.cache_data # Cache data loading
def load_csv(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def df_to_csv_bytes(df):
    """Converts a DataFrame to CSV bytes for downloading."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')

# --- Main App UI ---
st.title("🏙️ Energy Community Data Processing Pipeline")
st.markdown("An interactive tool to process building and time series data for energy system modeling.")

# --- Sidebar for Configuration and Control ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. File Upload
    st.subheader("1. Upload Data Files")
    uploaded_buildings_file = st.file_uploader("Upload Extracted Buildings CSV", type="csv", help="CSV file containing building metadata (e.g., area, function, lat/lon).")
    uploaded_timeseries_file = st.file_uploader("Upload Time Series CSV", type="csv", help="CSV file with energy consumption data (e.g., from EnergyPlus).")
    
    # Load data into dataframes
    df_buildings_raw = load_csv(uploaded_buildings_file)
    df_timeseries_raw = load_csv(uploaded_timeseries_file)
    
    # 2. Assignment Configuration
    st.subheader("2. Assignment Configuration")
    num_lines = st.number_input("Number of Energy Lines/Clusters", min_value=1, value=6, help="The number of groups to cluster buildings into.")

    # 3. Scenario Configuration (using expander and json editor)
    st.subheader("3. Scenario Configuration")
    with st.expander("Edit Solar & Battery Scenarios (Advanced)", expanded=False):
        st.markdown("Edit the JSON below to define technology penetration rates and types.")
        solar_scenarios_json = st.json_editor(DEFAULT_SOLAR_SCENARIOS, height=300)
        battery_scenarios_json = st.json_editor(DEFAULT_BATTERY_SCENARIOS, height=300)

    # 4. Execution
    st.subheader("4. Run Pipeline")
    run_button = st.button("Process All Data", type="primary", use_container_width=True)

# --- Main Panel for Status and Results ---

# Initialize session state to store results
if 'pipeline_complete' not in st.session_state:
    st.session_state.pipeline_complete = False
if 'results' not in st.session_state:
    st.session_state.results = {}


if run_button:
    # Reset state on new run
    st.session_state.pipeline_complete = False
    st.session_state.results = {}

    # Validation
    if uploaded_buildings_file is None or uploaded_timeseries_file is None:
        st.error("Please upload both the buildings and time series CSV files.")
    else:
        st.info("Pipeline started! Processing may take a few moments...")
        progress_bar = st.progress(0, "Starting...")
        status_text = st.empty()

        try:
            # --- Step 1: Process Building Data ---
            status_text.text("Step 1/3: Processing building data with scenarios...")
            progress_bar.progress(10, "Processing building data...")
            df_buildings_processed = process_building_data(
                buildings_df=df_buildings_raw.copy(), # Pass a copy to avoid modifying cached raw data
                solar_scenarios=solar_scenarios_json,
                battery_scenarios=battery_scenarios_json
            )
            progress_bar.progress(40, "Building data processed.")
            
            # --- Step 2: Process Time Series Data ---
            status_text.text("Step 2/3: Processing time series data...")
            df_timeseries_processed = process_timeseries_data(
                ts_df=df_timeseries_raw.copy(), # Pass a copy
                buildings_df=df_buildings_processed.copy()
            )
            progress_bar.progress(70, "Time series processed.")

            # --- Step 3: Generate Assignments ---
            status_text.text("Step 3/3: Generating building assignments...")
            df_assignments, df_buildings_with_lines = generate_assignments(
                buildings_df=df_buildings_processed.copy(),
                num_lines=num_lines
            )
            progress_bar.progress(100, "Pipeline complete!")
            status_text.success("✅ Pipeline finished successfully! Results are ready below.")

            # Store results in session state
            st.session_state.results = {
                'buildings_processed': df_buildings_processed,
                'timeseries_processed': df_timeseries_processed,
                'assignments': df_assignments,
                'buildings_with_lines': df_buildings_with_lines
            }
            st.session_state.pipeline_complete = True

        except Exception as e:
            status_text.error(f"An error occurred during processing: {e}")
            st.exception(e) # Display full traceback for debugging
            st.session_state.pipeline_complete = False


# --- Display Results ---
if st.session_state.pipeline_complete:
    st.header("📊 Results")
    
    results = st.session_state.results
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary & Downloads", "Map View", "Processed Building Data", "Processed Time Series"])

    with tab1:
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Buildings Processed", len(results['buildings_with_lines']))
        col2.metric("Lines/Clusters Generated", results['assignments']['line_id'].nunique())
        col3.metric("Time Series Categories", results['timeseries_processed']['Energy'].nunique())
        
        st.subheader("Download Processed Files")
        st.download_button(
            label="⬇️ Download Buildings with Lines CSV",
            data=df_to_csv_bytes(results['buildings_with_lines']),
            file_name='buildings_processed_with_lines.csv',
            mime='text/csv',
            use_container_width=True
        )
        st.download_button(
            label="⬇️ Download Building Assignments CSV",
            data=df_to_csv_bytes(results['assignments']),
            file_name='building_assignments.csv',
            mime='text/csv',
            use_container_width=True
        )
        st.download_button(
            label="⬇️ Download Processed Time Series CSV",
            data=df_to_csv_bytes(results['timeseries_processed']),
            file_name='time_series_processed.csv',
            mime='text/csv',
            use_container_width=True
        )

    with tab2:
        st.subheader("Building Assignments Map")
        map_df = results['buildings_with_lines'][['lat', 'lon', 'line_id']].copy()
        
        # Check if line_id is numeric or can be converted to category for coloring
        if pd.api.types.is_numeric_dtype(map_df['line_id']):
             map_df['color'] = map_df['line_id'] # Use numeric line_id for color scale if it is
        else:
            # Create a color map for categorical line_id
            unique_lines = map_df['line_id'].unique()
            colors = st.get_option("charts.PALETTE") # Use streamlit's color palette
            color_map = {line: colors[i % len(colors)] for i, line in enumerate(unique_lines)}
            map_df['color'] = map_df['line_id'].map(color_map)

        st.map(map_df, latitude='lat', longitude='lon', color='color', size=50)


    with tab3:
        st.subheader("Processed Building Data with Line Assignments")
        st.dataframe(results['buildings_with_lines'])

    with tab4:
        st.subheader("Processed Time Series Data (in Joules per interval)")
        st.dataframe(results['timeseries_processed'])
else:
    st.info("Configure your settings in the sidebar and click 'Process All Data' to begin.")
