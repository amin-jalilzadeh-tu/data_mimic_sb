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
    st.error("Could not import processing modules. Ensure this script is in the 'mimic' directory and your processing scripts are inside a 'processing' sub-folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Community Data Pipeline",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Default Scenario Configurations ---
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
@st.cache_data(ttl=3600)
def load_csv_data_from_bytes(uploaded_file_bytes, filename="uploaded_file"):
    if uploaded_file_bytes is not None:
        try:
            return pd.read_csv(io.BytesIO(uploaded_file_bytes))
        except Exception as e:
            st.error(f"Error reading CSV ({filename}): {e}")
            return None
    return None

def df_to_csv_bytes(df):
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')

# --- Initialize Session State ---
# For storing data between steps and UI states
for key, default_val in [
    ('raw_buildings_bytes', None), ('raw_timeseries_bytes', None),
    ('processed_buildings_uploaded_bytes', None), ('processed_timeseries_uploaded_bytes', None),
    ('df_buildings_processed', None), ('df_timeseries_processed', None),
    ('df_assignments', None), ('df_buildings_with_lines', None),
    ('step1_complete_internal', False), ('processed_buildings_available', False), 
    ('step2_complete_internal', False), ('processed_timeseries_available', False),
    ('step3_complete', False),
    ('status_message', ""), ('status_type', "info")
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- Main App UI ---
st.title("🏙️ Energy Community Data Processing Pipeline")
st.markdown("An interactive tool to process building and time series data for energy system modeling.")

# --- Sidebar for Configuration and Control ---
with st.sidebar:
    st.header("⚙️ Configuration & Inputs")

    # 1. File Upload
    st.subheader("1. Upload Data Files")
    st.markdown("**Raw Input Files (for running steps from scratch):**")
    raw_buildings_uploader = st.file_uploader("Upload Raw Buildings CSV (for Step 1)", type="csv", key="raw_buildings_uploader")
    raw_timeseries_uploader = st.file_uploader("Upload Raw Time Series CSV (for Step 2)", type="csv", key="raw_timeseries_uploader")
    
    st.markdown("---")
    st.markdown("**Intermediate Input Files (Optional - to skip steps):**")
    processed_buildings_uploader = st.file_uploader("Upload Processed Buildings CSV (Skips Step 1)", type="csv", key="processed_buildings_uploader")
    processed_timeseries_uploader = st.file_uploader("Upload Processed Time Series CSV (Skips Step 2)", type="csv", key="processed_timeseries_uploader")


    # Logic to handle uploaded file bytes and update availability flags
    if raw_buildings_uploader is not None:
        st.session_state.raw_buildings_bytes = raw_buildings_uploader.getvalue()
    if raw_timeseries_uploader is not None:
        st.session_state.raw_timeseries_bytes = raw_timeseries_uploader.getvalue()

    if processed_buildings_uploader is not None:
        if st.session_state.get('processed_buildings_filename') != processed_buildings_uploader.name: # New file uploaded
            st.session_state.processed_buildings_uploaded_bytes = processed_buildings_uploader.getvalue()
            st.session_state.processed_buildings_available = True
            st.session_state.step1_complete_internal = False 
            st.session_state.df_buildings_processed = None 
            st.session_state.processed_buildings_filename = processed_buildings_uploader.name
            st.success("Uploaded 'Processed Buildings CSV' will be used.")
    elif st.session_state.step1_complete_internal:
        st.session_state.processed_buildings_available = True
    # else: st.session_state.processed_buildings_available = False # Handled by button logic

    if processed_timeseries_uploader is not None:
        if st.session_state.get('processed_timeseries_filename') != processed_timeseries_uploader.name:
            st.session_state.processed_timeseries_uploaded_bytes = processed_timeseries_uploader.getvalue()
            st.session_state.processed_timeseries_available = True
            st.session_state.step2_complete_internal = False
            st.session_state.df_timeseries_processed = None
            st.session_state.processed_timeseries_filename = processed_timeseries_uploader.name
            st.success("Uploaded 'Processed Time Series CSV' will be used.")
    elif st.session_state.step2_complete_internal:
        st.session_state.processed_timeseries_available = True
    # else: st.session_state.processed_timeseries_available = False
    
    st.subheader("2. Assignment Configuration")
    num_lines_for_assignment = st.number_input("Number of Energy Lines/Clusters", min_value=1, value=6, key="num_lines")

    st.subheader("3. Scenario Configuration")
    current_solar_scenarios = {}
    current_battery_scenarios = {}
    # (Scenario UI code - assuming it's populated as before)
    with st.expander("☀️ Solar Scenarios", expanded=False):
        for bldg_func_key, default_scenario_data in DEFAULT_SOLAR_SCENARIOS.items():
            current_solar_scenarios.setdefault(bldg_func_key, {})
            current_solar_scenarios[bldg_func_key].setdefault('types', [])
            with st.container():
                st.markdown(f"**{bldg_func_key.capitalize()} Buildings**")
                pen_rate = st.slider(f"Penetration Rate##solar_pen_{bldg_func_key}", 0.0, 1.0, float(default_scenario_data.get("penetration_rate", 0.1)), 0.01)
                current_solar_scenarios[bldg_func_key]['penetration_rate'] = pen_rate
                for i, type_data in enumerate(default_scenario_data.get("types", [])):
                    st.markdown(f"*{type_data.get('name', 'Unnamed')}*")
                    cols = st.columns([2,2,1]); kwp_min = cols[0].number_input(f"kWp Min##solar_kwp_min_{bldg_func_key}_{i}", value=float(type_data.get("kWp_range", [0,0])[0]), step=0.1, format="%.1f"); kwp_max = cols[1].number_input(f"kWp Max##solar_kwp_max_{bldg_func_key}_{i}", value=float(type_data.get("kWp_range", [0,0])[1]), step=0.1, format="%.1f"); weight = cols[2].number_input(f"Weight##solar_w_{bldg_func_key}_{i}", value=float(type_data.get("weight", 1.0)), min_value=0.0, step=0.1, format="%.1f")
                    current_solar_scenarios[bldg_func_key]['types'].append({"name": type_data.get('name'), "kWp_range": [kwp_min, kwp_max], "weight": weight})
                st.divider()
    with st.expander("🔋 Battery Scenarios", expanded=False):
        for bldg_func_key, default_scenario_data in DEFAULT_BATTERY_SCENARIOS.items():
            current_battery_scenarios.setdefault(bldg_func_key, {}); current_battery_scenarios[bldg_func_key].setdefault('types', [])
            with st.container():
                st.markdown(f"**{bldg_func_key.capitalize()} Buildings**"); pen_rate_solar = st.slider(f"Penetration if Solar##batt_pen_solar_{bldg_func_key}", 0.0, 1.0, float(default_scenario_data.get("penetration_rate_if_solar", 0.15)), 0.01); pen_rate_overall = st.slider(f"Overall Penetration##batt_pen_overall_{bldg_func_key}", 0.0, 1.0, float(default_scenario_data.get("overall_penetration_rate", 0.02)), 0.01)
                current_battery_scenarios[bldg_func_key]['penetration_rate_if_solar'] = pen_rate_solar; current_battery_scenarios[bldg_func_key]['overall_penetration_rate'] = pen_rate_overall
                for i, type_data in enumerate(default_scenario_data.get("types", [])):
                    st.markdown(f"*{type_data.get('name', 'Unnamed')}*"); cols_kwh = st.columns(2); kwh_min = cols_kwh[0].number_input(f"kWh Min##batt_kwh_min_{bldg_func_key}_{i}", value=float(type_data.get("kWh_range", [0,0])[0]), step=0.1, format="%.1f"); kwh_max = cols_kwh[1].number_input(f"kWh Max##batt_kwh_max_{bldg_func_key}_{i}", value=float(type_data.get("kWh_range", [0,0])[1]), step=0.1, format="%.1f")
                    cols_kw_weight = st.columns([1,1,1]); kw_min = cols_kw_weight[0].number_input(f"kW Min##batt_kw_min_{bldg_func_key}_{i}", value=float(type_data.get("kW_range", [0,0])[0]), step=0.1, format="%.1f"); kw_max = cols_kw_weight[1].number_input(f"kW Max##batt_kw_max_{bldg_func_key}_{i}", value=float(type_data.get("kW_range", [0,0])[1]), step=0.1, format="%.1f"); batt_weight = cols_kw_weight[2].number_input(f"Weight##batt_w_{bldg_func_key}_{i}", value=float(type_data.get("weight", 1.0)), min_value=0.0, step=0.1, format="%.1f")
                    current_battery_scenarios[bldg_func_key]['types'].append({"name": type_data.get('name'), "kWh_range": [kwh_min, kwh_max], "kW_range": [kw_min, kw_max], "weight": batt_weight})
                st.divider()

    # 4. Execution Buttons
    st.subheader("4. Run Pipeline Steps")
    
    if st.button("▶️ Run Step 1: Process Buildings", use_container_width=True, key="run_step1"):
        if st.session_state.raw_buildings_bytes is None:
            st.session_state.status_message = "Upload Raw Buildings CSV for Step 1."; st.session_state.status_type = "error"
        else:
            df_buildings_raw = load_csv_data_from_bytes(st.session_state.raw_buildings_bytes, "Raw Buildings")
            if df_buildings_raw is not None:
                with st.spinner("Running Step 1..."):
                    try:
                        st.session_state.df_buildings_processed = process_building_data(df_buildings_raw.copy(), current_solar_scenarios, current_battery_scenarios)
                        st.session_state.step1_complete_internal = True; st.session_state.processed_buildings_available = True
                        st.session_state.processed_buildings_uploaded_bytes = None # Prioritize internal
                        st.session_state.step2_complete_internal = False; st.session_state.df_timeseries_processed = None; st.session_state.processed_timeseries_available = False; st.session_state.processed_timeseries_uploaded_bytes = None;
                        st.session_state.step3_complete = False; st.session_state.df_assignments = None; st.session_state.df_buildings_with_lines = None
                        st.session_state.status_message = "✅ Step 1: Building data processed!"; st.session_state.status_type = "success"
                    except Exception as e: st.session_state.status_message = f"Error in Step 1: {e}"; st.session_state.status_type = "error"; st.exception(e)
            else: st.session_state.status_message = "Could not load Raw Buildings CSV."; st.session_state.status_type = "error"

    # Determine if processed buildings data is ready for step 2
    is_processed_buildings_ready_for_step2 = st.session_state.processed_buildings_available

    step2_disabled_reason = ""
    if st.session_state.raw_timeseries_bytes is None: step2_disabled_reason = "Upload Raw Time Series CSV."
    elif not is_processed_buildings_ready_for_step2: step2_disabled_reason = "Run Step 1 or Upload Processed Buildings CSV."
    
    if st.button("▶️ Run Step 2: Process Time Series", use_container_width=True, key="run_step2", disabled=bool(step2_disabled_reason)):
        df_input_buildings_s2 = None; source_msg_s2 = ""
        if st.session_state.processed_buildings_uploaded_bytes:
            df_input_buildings_s2 = load_csv_data_from_bytes(st.session_state.processed_buildings_uploaded_bytes, "Uploaded Processed Buildings")
            if df_input_buildings_s2 is not None: source_msg_s2 = "Using uploaded processed buildings for Step 2."
        elif st.session_state.df_buildings_processed is not None:
            df_input_buildings_s2 = st.session_state.df_buildings_processed
            source_msg_s2 = "Using Step 1 output for Step 2."
        
        if df_input_buildings_s2 is None: st.session_state.status_message = "Processed building data not found for Step 2."; st.session_state.status_type = "error"
        else:
            df_timeseries_raw = load_csv_data_from_bytes(st.session_state.raw_timeseries_bytes, "Raw Time Series")
            if df_timeseries_raw is not None:
                st.info(source_msg_s2)
                with st.spinner("Running Step 2..."):
                    try:
                        st.session_state.df_timeseries_processed = process_timeseries_data(df_timeseries_raw.copy(), df_input_buildings_s2.copy())
                        st.session_state.step2_complete_internal = True; st.session_state.processed_timeseries_available = True
                        st.session_state.processed_timeseries_uploaded_bytes = None # Prioritize internal
                        st.session_state.step3_complete = False; st.session_state.df_assignments = None; st.session_state.df_buildings_with_lines = None
                        st.session_state.status_message = "✅ Step 2: Time series data processed!"; st.session_state.status_type = "success"
                    except Exception as e: st.session_state.status_message = f"Error in Step 2: {e}"; st.session_state.status_type = "error"; st.exception(e)
            else: st.session_state.status_message = "Could not load Raw Time Series CSV."; st.session_state.status_type = "error"
    elif step2_disabled_reason: st.caption(f"Step 2 disabled: {step2_disabled_reason}")

    # Determine if processed buildings data is ready for step 3
    is_processed_buildings_ready_for_step3 = st.session_state.processed_buildings_available

    step3_disabled_reason = ""
    if not is_processed_buildings_ready_for_step3 : step3_disabled_reason = "Run Step 1 or Upload Processed Buildings CSV."
    # Step 3 doesn't directly need processed_timeseries_available for *its own execution*, but Run All does.

    if st.button("▶️ Run Step 3: Generate Assignments", use_container_width=True, key="run_step3", disabled=bool(step3_disabled_reason)):
        df_input_buildings_s3 = None; source_msg_s3 = ""
        if st.session_state.processed_buildings_uploaded_bytes:
            df_input_buildings_s3 = load_csv_data_from_bytes(st.session_state.processed_buildings_uploaded_bytes, "Uploaded Processed Buildings")
            if df_input_buildings_s3 is not None: source_msg_s3 = "Using uploaded processed buildings data for Step 3."
        elif st.session_state.df_buildings_processed is not None:
            df_input_buildings_s3 = st.session_state.df_buildings_processed
            source_msg_s3 = "Using Step 1 output for Step 3."

        if df_input_buildings_s3 is None: st.session_state.status_message = "Processed building data not found for Step 3."; st.session_state.status_type = "error"
        else:
            st.info(source_msg_s3)
            with st.spinner("Running Step 3..."):
                try:
                    assignments_df, buildings_with_lines_df = generate_assignments(df_input_buildings_s3.copy(), num_lines_for_assignment)
                    st.session_state.df_assignments = assignments_df; st.session_state.df_buildings_with_lines = buildings_with_lines_df
                    st.session_state.step3_complete = True
                    st.session_state.status_message = "✅ Step 3: Assignments generated!"; st.session_state.status_type = "success"
                except Exception as e: st.session_state.status_message = f"Error in Step 3: {e}"; st.session_state.status_type = "error"; st.exception(e)
    elif step3_disabled_reason: st.caption(f"Step 3 disabled: {step3_disabled_reason}")

    if st.button("🚀 Run All Steps (from raw data)", type="primary", use_container_width=True, key="run_all"):
        if st.session_state.raw_buildings_bytes is None or st.session_state.raw_timeseries_bytes is None:
            st.session_state.status_message = "Upload both Raw Buildings and Raw Time Series CSVs to run all steps."; st.session_state.status_type = "error"
        else:
            df_buildings_raw = load_csv_data_from_bytes(st.session_state.raw_buildings_bytes, "Raw Buildings")
            df_timeseries_raw = load_csv_data_from_bytes(st.session_state.raw_timeseries_bytes, "Raw Time Series")
            if df_buildings_raw is not None and df_timeseries_raw is not None:
                st.session_state.status_message = "Pipeline started (All Steps)..."; st.session_state.status_type = "info"; progress_bar = st.progress(0)
                try:
                    st.session_state.status_message = "Running Step 1..."; st.session_state.df_buildings_processed = process_building_data(df_buildings_raw.copy(), current_solar_scenarios, current_battery_scenarios)
                    st.session_state.step1_complete_internal = True; st.session_state.processed_buildings_available = True; st.session_state.processed_buildings_uploaded_bytes = None; progress_bar.progress(33)
                    st.session_state.status_message = "Running Step 2..."; st.session_state.df_timeseries_processed = process_timeseries_data(df_timeseries_raw.copy(), st.session_state.df_buildings_processed.copy())
                    st.session_state.step2_complete_internal = True; st.session_state.processed_timeseries_available = True; st.session_state.processed_timeseries_uploaded_bytes = None; progress_bar.progress(66)
                    st.session_state.status_message = "Running Step 3..."; st.session_state.df_assignments, st.session_state.df_buildings_with_lines = generate_assignments(st.session_state.df_buildings_processed.copy(), num_lines_for_assignment)
                    st.session_state.step3_complete = True; progress_bar.progress(100)
                    st.session_state.status_message = "✅ All steps completed successfully!"; st.session_state.status_type = "success"
                except Exception as e: st.session_state.status_message = f"Error during 'Run All Steps': {e}"; st.session_state.status_type = "error"; st.exception(e)
            else: st.session_state.status_message = "Could not load raw CSVs for 'Run All Steps'."; st.session_state.status_type = "error"

# --- Main Panel for Status and Results ---
if st.session_state.status_message:
    if st.session_state.status_type == "success": st.success(st.session_state.status_message, icon="🎉")
    elif st.session_state.status_type == "error": st.error(st.session_state.status_message, icon="🔥")
    elif st.session_state.status_type == "warning": st.warning(st.session_state.status_message, icon="⚠️")
    else: st.info(st.session_state.status_message, icon="ℹ️")

st.header("📊 Results & Outputs")
results_available_display = False
tabs_to_create_titles = []
tab_content_map = {} # To store content for each tab

# Check for available processed buildings data (either uploaded or from Step 1)
df_display_buildings = None
buildings_source_info = ""
if st.session_state.processed_buildings_uploaded_bytes and st.session_state.processed_buildings_available:
    df_display_buildings = load_csv_data_from_bytes(st.session_state.processed_buildings_uploaded_bytes, "Uploaded Processed Buildings")
    if df_display_buildings is not None: buildings_source_info = " (from uploaded file)"
elif st.session_state.df_buildings_processed is not None:
    df_display_buildings = st.session_state.df_buildings_processed
    if df_display_buildings is not None: buildings_source_info = " (from Step 1 execution)"

if df_display_buildings is not None:
    tabs_to_create_titles.append("🏛️ Processed Buildings")
    tab_content_map["🏛️ Processed Buildings"] = {
        "df": df_display_buildings, 
        "download_name": "buildings_processed.csv",
        "caption": f"Displaying: Processed Building Data{buildings_source_info}"
    }
    results_available_display = True

# Check for available processed time series data (either uploaded or from Step 2)
df_display_timeseries = None
timeseries_source_info = ""
if st.session_state.processed_timeseries_uploaded_bytes and st.session_state.processed_timeseries_available:
    df_display_timeseries = load_csv_data_from_bytes(st.session_state.processed_timeseries_uploaded_bytes, "Uploaded Processed Time Series")
    if df_display_timeseries is not None: timeseries_source_info = " (from uploaded file)"
elif st.session_state.df_timeseries_processed is not None:
    df_display_timeseries = st.session_state.df_timeseries_processed
    if df_display_timeseries is not None: timeseries_source_info = " (from Step 2 execution)"

if df_display_timeseries is not None:
    tabs_to_create_titles.append("📈 Processed Time Series")
    tab_content_map["📈 Processed Time Series"] = {
        "df": df_display_timeseries, 
        "download_name": "time_series_processed.csv",
        "caption": f"Displaying: Processed Time Series Data{timeseries_source_info}"
    }
    results_available_display = True

if st.session_state.df_assignments is not None and st.session_state.df_buildings_with_lines is not None:
    tabs_to_create_titles.append("🗺️ Assignments & Map")
    tab_content_map["🗺️ Assignments & Map"] = {
        "assignments_df": st.session_state.df_assignments,
        "buildings_with_lines_df": st.session_state.df_buildings_with_lines
    }
    results_available_display = True

if not results_available_display:
    tabs_to_create_titles.append("👋 Welcome")
    tab_content_map["👋 Welcome"] = {"message": "Upload files and run pipeline steps from the sidebar to see results here. You can also upload intermediate processed files to skip steps."}

# Create tabs
active_tabs = st.tabs(tabs_to_create_titles)

for i, title in enumerate(tabs_to_create_titles):
    with active_tabs[i]:
        content = tab_content_map.get(title)
        if "df" in content: # For tabs displaying a single main DataFrame
            st.subheader(title.split(" ", 1)[1]) # Remove icon for subheader
            if content.get("caption"): st.caption(content["caption"])
            st.dataframe(content["df"])
            st.download_button(f"⬇️ Download {title.split(' ', 1)[1]}", df_to_csv_bytes(content["df"]), content["download_name"], 'text/csv', use_container_width=True)
        
        elif title == "🗺️ Assignments & Map":
            st.subheader("Building Assignments")
            st.dataframe(content["assignments_df"])
            st.download_button("⬇️ Download Assignments", df_to_csv_bytes(content["assignments_df"]), 'building_assignments.csv', 'text/csv', use_container_width=True)
            
            st.subheader("Buildings with Line IDs")
            st.dataframe(content["buildings_with_lines_df"])
            st.download_button("⬇️ Download Buildings with Lines", df_to_csv_bytes(content["buildings_with_lines_df"]), 'buildings_with_lines.csv', 'text/csv', use_container_width=True)

            if not content["buildings_with_lines_df"].empty and 'lat' in content["buildings_with_lines_df"].columns:
                map_df = content["buildings_with_lines_df"][['lat', 'lon', 'line_id']].copy().dropna(subset=['lat', 'lon'])
                if not map_df.empty:
                    unique_lines = map_df['line_id'].unique(); palette = ["#FF4B4B", "#4B7BFF", "#4BFF7B", "#FFFF4B", "#FF4BFF", "#4BFFFF", "#FFA500", "#8A2BE2"]
                    color_map = {line: palette[i % len(palette)] for i, line in enumerate(unique_lines)}
                    map_df['color'] = map_df['line_id'].map(color_map)
                    st.map(map_df, latitude='lat', longitude='lon', color='color', size=50)
                else: st.warning("No valid data for map (check lat/lon).")
            else: st.warning("Building data for map not available or missing required columns.")
        
        elif title == "👋 Welcome":
            st.info(content["message"], icon="👋")

