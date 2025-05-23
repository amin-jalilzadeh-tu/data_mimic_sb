import streamlit as st
import pandas as pd
import numpy as np
import json
import io

# Import the refactored processing functions
try:
    from processing.data_processing import process_building_data
    from processing.timeseries_processing import process_timeseries_data
    from processing.assignment_generation import generate_assignments
except ImportError:
    st.error("Could not import processing modules. Make sure this script is in the 'mimic' directory and your processing scripts are inside a 'processing' sub-folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Community Data Pipeline",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Default Scenario Configurations ---
# These serve as the initial structure and default values for the UI
DEFAULT_SOLAR_SCENARIOS = {
    "default": {"penetration_rate": 0.10, "types": [{"name": "default_pv_small", "kWp_range": [2.0, 4.0], "weight": 0.6}, {"name": "default_pv_medium", "kWp_range": [4.1, 8.0], "weight": 0.4}]},
    "residential": {"penetration_rate": 0.25, "types": [{"name": "res_pv_standard", "kWp_range": [3.0, 7.0], "weight": 0.7}, {"name": "res_pv_large", "kWp_range": [7.1, 15.0], "weight": 0.3}]},
    "non_residential": {"penetration_rate": 0.40, "types": [{"name": "comm_pv_medium", "kWp_range": [10.0, 50.0], "weight": 0.5}, {"name": "comm_pv_large", "kWp_range": [50.1, 250.0], "weight": 0.3}, {"name": "comm_pv_xlarge", "kWp_range": [250.1, 1000.0], "weight": 0.2}]}
}

DEFAULT_BATTERY_SCENARIOS = {
    "default": {"penetration_rate_if_solar": 0.15, "overall_penetration_rate": 0.02, "types": [{"name": "default_batt_small", "kWh_range": [4.0, 8.0], "kW_range": [2.0, 4.0], "weight": 1.0}]},
    "residential": {"penetration_rate_if_solar": 0.30, "overall_penetration_rate": 0.05, "types": [{"name": "res_batt_daily", "kWh_range": [5.0, 15.0], "kW_range": [2.5, 7.0], "weight": 0.8}, {"name": "res_batt_backup", "kWh_range": [15.1, 30.0], "kW_range": [5.0, 10.0], "weight": 0.2}]},
    "non_residential": {"penetration_rate_if_solar": 0.50, "overall_penetration_rate": 0.10, "types": [{"name": "comm_batt_demand_peak", "kWh_range": [20.0, 100.0], "kW_range": [10.0, 50.0], "weight": 0.6}, {"name": "comm_batt_storage_large", "kWh_range": [100.1, 500.0], "kW_range": [50.1, 200.0], "weight": 0.4}]}
}

# --- Helper Functions ---
@st.cache_data
def load_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    return None

def df_to_csv_bytes(df):
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
    uploaded_buildings_file = st.file_uploader("Upload Extracted Buildings CSV", type="csv")
    uploaded_timeseries_file = st.file_uploader("Upload Time Series CSV", type="csv")
    
    df_buildings_raw = load_csv(uploaded_buildings_file)
    df_timeseries_raw = load_csv(uploaded_timeseries_file)
    
    # 2. Assignment Configuration
    st.subheader("2. Assignment Configuration")
    num_lines_for_assignment = st.number_input("Number of Energy Lines/Clusters", min_value=1, value=6)

    # 3. Scenario Configuration - User Friendly Widgets
    st.subheader("3. Scenario Configuration")

    # Initialize scenario dicts in session state if they don't exist
    # This helps maintain user edits if they navigate away or if there's a rerun
    if 'ui_solar_scenarios' not in st.session_state:
        st.session_state.ui_solar_scenarios = {} # Will be populated by UI
    if 'ui_battery_scenarios' not in st.session_state:
        st.session_state.ui_battery_scenarios = {}

    # --- Solar Scenarios UI ---
    with st.expander("☀️ Solar Scenarios", expanded=True):
        for bldg_func_key, default_scenario_data in DEFAULT_SOLAR_SCENARIOS.items():
            with st.container(): # Use container for better visual grouping
                st.markdown(f"**{bldg_func_key.capitalize()} Buildings**")
                pen_rate_key = f"solar_pen_rate_{bldg_func_key}"
                pen_rate = st.slider(
                    f"Penetration Rate", 
                    min_value=0.0, max_value=1.0, 
                    value=float(default_scenario_data.get("penetration_rate", 0.1)), 
                    step=0.01, 
                    key=pen_rate_key,
                    help=f"Fraction of {bldg_func_key} buildings to get solar."
                )
                
                st.session_state.ui_solar_scenarios.setdefault(bldg_func_key, {})['penetration_rate'] = pen_rate
                st.session_state.ui_solar_scenarios[bldg_func_key].setdefault('types', [])

                for i, type_data in enumerate(default_scenario_data.get("types", [])):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Type {i+1}: {type_data.get('name', 'Unnamed')}*")
                    cols = st.columns([2,2,1]) # For kWp_range min, max, weight
                    
                    kwp_min_key = f"solar_kwp_min_{bldg_func_key}_{i}"
                    kwp_max_key = f"solar_kwp_max_{bldg_func_key}_{i}"
                    weight_key = f"solar_weight_{bldg_func_key}_{i}"

                    kwp_min = cols[0].number_input(f"kWp Min", value=float(type_data.get("kWp_range", [0,0])[0]), step=0.1, key=kwp_min_key, format="%.1f")
                    kwp_max = cols[1].number_input(f"kWp Max", value=float(type_data.get("kWp_range", [0,0])[1]), step=0.1, key=kwp_max_key, format="%.1f")
                    weight = cols[2].number_input(f"Weight", value=float(type_data.get("weight", 1.0)), min_value=0.0, step=0.1, key=weight_key, format="%.1f")
                    
                    # Update session state for this type
                    if len(st.session_state.ui_solar_scenarios[bldg_func_key]['types']) <= i:
                        st.session_state.ui_solar_scenarios[bldg_func_key]['types'].append({})
                    
                    st.session_state.ui_solar_scenarios[bldg_func_key]['types'][i] = {
                        "name": type_data.get('name', f'type_{i+1}'), # Keep original name or generate one
                        "kWp_range": [kwp_min, kwp_max],
                        "weight": weight
                    }
                st.divider()


    # --- Battery Scenarios UI ---
    with st.expander("🔋 Battery Scenarios", expanded=True):
        for bldg_func_key, default_scenario_data in DEFAULT_BATTERY_SCENARIOS.items():
            with st.container():
                st.markdown(f"**{bldg_func_key.capitalize()} Buildings**")
                
                pen_rate_solar_key = f"batt_pen_solar_{bldg_func_key}"
                pen_rate_solar = st.slider(
                    f"Penetration if Solar", 
                    min_value=0.0, max_value=1.0, 
                    value=float(default_scenario_data.get("penetration_rate_if_solar", 0.15)), 
                    step=0.01, 
                    key=pen_rate_solar_key,
                    help=f"Fraction of {bldg_func_key} buildings *with solar* to also get a battery."
                )
                
                pen_rate_overall_key = f"batt_pen_overall_{bldg_func_key}"
                pen_rate_overall = st.slider(
                    f"Overall Penetration", 
                    min_value=0.0, max_value=1.0, 
                    value=float(default_scenario_data.get("overall_penetration_rate", 0.02)), 
                    step=0.01, 
                    key=pen_rate_overall_key,
                    help=f"Additional fraction of *all* {bldg_func_key} buildings to get a battery (regardless of solar)."
                )

                st.session_state.ui_battery_scenarios.setdefault(bldg_func_key, {})['penetration_rate_if_solar'] = pen_rate_solar
                st.session_state.ui_battery_scenarios[bldg_func_key]['overall_penetration_rate'] = pen_rate_overall
                st.session_state.ui_battery_scenarios[bldg_func_key].setdefault('types', [])

                for i, type_data in enumerate(default_scenario_data.get("types", [])):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Type {i+1}: {type_data.get('name', 'Unnamed')}*")
                    cols_kwh = st.columns(2)
                    kwh_min_key = f"batt_kwh_min_{bldg_func_key}_{i}"
                    kwh_max_key = f"batt_kwh_max_{bldg_func_key}_{i}"
                    kwh_min = cols_kwh[0].number_input(f"kWh Min", value=float(type_data.get("kWh_range", [0,0])[0]), step=0.1, key=kwh_min_key, format="%.1f")
                    kwh_max = cols_kwh[1].number_input(f"kWh Max", value=float(type_data.get("kWh_range", [0,0])[1]), step=0.1, key=kwh_max_key, format="%.1f")
                    
                    cols_kw_weight = st.columns([1,1,1]) # For kW_range min, max, weight
                    kw_min_key = f"batt_kw_min_{bldg_func_key}_{i}"
                    kw_max_key = f"batt_kw_max_{bldg_func_key}_{i}"
                    batt_weight_key = f"batt_weight_{bldg_func_key}_{i}"
                    kw_min = cols_kw_weight[0].number_input(f"kW Min", value=float(type_data.get("kW_range", [0,0])[0]), step=0.1, key=kw_min_key, format="%.1f")
                    kw_max = cols_kw_weight[1].number_input(f"kW Max", value=float(type_data.get("kW_range", [0,0])[1]), step=0.1, key=kw_max_key, format="%.1f")
                    batt_weight = cols_kw_weight[2].number_input(f"Weight", value=float(type_data.get("weight", 1.0)), min_value=0.0, step=0.1, key=batt_weight_key, format="%.1f")

                    if len(st.session_state.ui_battery_scenarios[bldg_func_key]['types']) <= i:
                        st.session_state.ui_battery_scenarios[bldg_func_key]['types'].append({})
                    
                    st.session_state.ui_battery_scenarios[bldg_func_key]['types'][i] = {
                        "name": type_data.get('name', f'type_{i+1}'),
                        "kWh_range": [kwh_min, kwh_max],
                        "kW_range": [kw_min, kw_max],
                        "weight": batt_weight
                    }
                st.divider()


    # 4. Execution
    st.subheader("4. Run Pipeline")
    run_button = st.button("Process All Data", type="primary", use_container_width=True)

# --- Main Panel for Status and Results ---
if 'pipeline_complete' not in st.session_state: st.session_state.pipeline_complete = False
if 'results' not in st.session_state: st.session_state.results = {}

if run_button:
    st.session_state.pipeline_complete = False
    st.session_state.results = {}

    if uploaded_buildings_file is None or uploaded_timeseries_file is None:
        st.error("Please upload both the buildings and time series CSV files.")
    else:
        # Reconstruct scenario dictionaries from session state (UI inputs)
        # This ensures the latest UI edits are used
        current_solar_scenarios = {}
        for bldg_func, data in DEFAULT_SOLAR_SCENARIOS.items():
            current_solar_scenarios[bldg_func] = {
                'penetration_rate': st.session_state.get(f"solar_pen_rate_{bldg_func}", data['penetration_rate']),
                'types': []
            }
            for i, type_data_default in enumerate(data.get('types', [])):
                current_solar_scenarios[bldg_func]['types'].append({
                    'name': type_data_default.get('name'),
                    'kWp_range': [
                        st.session_state.get(f"solar_kwp_min_{bldg_func}_{i}", type_data_default['kWp_range'][0]),
                        st.session_state.get(f"solar_kwp_max_{bldg_func}_{i}", type_data_default['kWp_range'][1])
                    ],
                    'weight': st.session_state.get(f"solar_weight_{bldg_func}_{i}", type_data_default['weight'])
                })
        
        current_battery_scenarios = {}
        for bldg_func, data in DEFAULT_BATTERY_SCENARIOS.items():
            current_battery_scenarios[bldg_func] = {
                'penetration_rate_if_solar': st.session_state.get(f"batt_pen_solar_{bldg_func}", data['penetration_rate_if_solar']),
                'overall_penetration_rate': st.session_state.get(f"batt_pen_overall_{bldg_func}", data.get('overall_penetration_rate', 0.0)),
                'types': []
            }
            for i, type_data_default in enumerate(data.get('types', [])):
                 current_battery_scenarios[bldg_func]['types'].append({
                    'name': type_data_default.get('name'),
                    'kWh_range': [
                        st.session_state.get(f"batt_kwh_min_{bldg_func}_{i}", type_data_default['kWh_range'][0]),
                        st.session_state.get(f"batt_kwh_max_{bldg_func}_{i}", type_data_default['kWh_range'][1])
                    ],
                    'kW_range': [
                        st.session_state.get(f"batt_kw_min_{bldg_func}_{i}", type_data_default['kW_range'][0]),
                        st.session_state.get(f"batt_kw_max_{bldg_func}_{i}", type_data_default['kW_range'][1])
                    ],
                    'weight': st.session_state.get(f"batt_weight_{bldg_func}_{i}", type_data_default['weight'])
                })


        st.info("Pipeline started! Processing may take a few moments...")
        progress_bar = st.progress(0, "Starting...")
        status_text = st.empty()

        try:
            status_text.text("Step 1/3: Processing building data...")
            progress_bar.progress(10)
            df_buildings_processed = process_building_data(
                buildings_df=df_buildings_raw.copy(),
                solar_scenarios=current_solar_scenarios,
                battery_scenarios=current_battery_scenarios
            )
            progress_bar.progress(40)
            
            status_text.text("Step 2/3: Processing time series data...")
            df_timeseries_processed = process_timeseries_data(
                ts_df=df_timeseries_raw.copy(),
                buildings_df=df_buildings_processed.copy()
            )
            progress_bar.progress(70)

            status_text.text("Step 3/3: Generating building assignments...")
            df_assignments, df_buildings_with_lines = generate_assignments(
                buildings_df=df_buildings_processed.copy(),
                num_lines=num_lines_for_assignment # Use the variable from UI
            )
            progress_bar.progress(100)
            status_text.success("✅ Pipeline finished successfully!")

            st.session_state.results = {
                'buildings_processed_raw_step1': df_buildings_processed, # Store intermediate for clarity
                'timeseries_processed': df_timeseries_processed,
                'assignments': df_assignments,
                'buildings_with_lines': df_buildings_with_lines
            }
            st.session_state.pipeline_complete = True

        except Exception as e:
            status_text.error(f"An error occurred: {e}")
            st.exception(e)
            st.session_state.pipeline_complete = False

if st.session_state.pipeline_complete:
    st.header("📊 Results")
    results = st.session_state.results
    tab1, tab2, tab3, tab4 = st.tabs(["Summary & Downloads", "Map View", "Processed Building Data", "Processed Time Series"])

    with tab1:
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Buildings Processed", len(results['buildings_with_lines']))
        col2.metric("Lines/Clusters Generated", results['assignments']['line_id'].nunique() if not results['assignments'].empty else 0)
        col3.metric("Time Series Categories", results['timeseries_processed']['Energy'].nunique() if not results['timeseries_processed'].empty else 0)
        
        st.subheader("Download Processed Files")
        st.download_button("⬇️ Buildings with Lines CSV", df_to_csv_bytes(results['buildings_with_lines']), 'buildings_processed_with_lines.csv', 'text/csv', use_container_width=True)
        st.download_button("⬇️ Building Assignments CSV", df_to_csv_bytes(results['assignments']), 'building_assignments.csv', 'text/csv', use_container_width=True)
        st.download_button("⬇️ Processed Time Series CSV", df_to_csv_bytes(results['timeseries_processed']), 'time_series_processed.csv', 'text/csv', use_container_width=True)

    with tab2:
        st.subheader("Building Assignments Map")
        if not results['buildings_with_lines'].empty and 'lat' in results['buildings_with_lines'].columns and 'lon' in results['buildings_with_lines'].columns:
            map_df = results['buildings_with_lines'][['lat', 'lon', 'line_id']].copy()
            map_df.dropna(subset=['lat', 'lon'], inplace=True) # Ensure no NaNs for map
            
            if not map_df.empty:
                unique_lines = map_df['line_id'].unique()
                # Hardcoded palette for broader compatibility if st.get_option is problematic
                palette = ["#FF4B4B", "#4B7BFF", "#4BFF7B", "#FFFF4B", "#FF4BFF", "#4BFFFF", "#FFA500", "#8A2BE2", "#008080", "#D2691E"]
                color_map = {line: palette[i % len(palette)] for i, line in enumerate(unique_lines)}
                map_df['color'] = map_df['line_id'].map(color_map)
                st.map(map_df, latitude='lat', longitude='lon', color='color', size=50)
            else:
                st.warning("No valid data to display on the map (check lat/lon).")
        else:
            st.warning("Processed building data for map is not available or missing lat/lon columns.")


    with tab3:
        st.subheader("Processed Building Data with Line Assignments")
        st.dataframe(results['buildings_with_lines'])

    with tab4:
        st.subheader("Processed Time Series Data (in Joules per interval)")
        st.dataframe(results['timeseries_processed'])
else:
    st.info("Configure your settings in the sidebar and click 'Process All Data' to begin.")
