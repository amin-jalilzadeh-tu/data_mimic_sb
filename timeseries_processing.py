import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
import csv

# These are defaults if not overridden by more specific logic or input data
EXPECTED_CATEGORIES = {
    "heating": {"positive": True}, # Load
    "cooling": {"positive": True}, # Load
    "facility": {"positive": True}, # Load (general electricity use)
    "generation": {"positive": True}, # Source (e.g., PV, reduces net load)
    "battery_charge": {"positive": False}, # Can be +/- (+ve = charging = load, -ve = discharging = source)
    "total_electricity": {"positive": False} # Calculated net load
}

# Define the calculation for total_electricity based on available categories
# Positive values add to load, negative values reduce load
# Since we are keeping units as Joules, these factors remain the same.
TOTAL_ELEC_CALC = {
    "facility": 1,
    "heating": 1,
    "cooling": 1,
    "generation": -1, # PV generation reduces net load
    "battery_charge": 1 # Charging adds to load, discharging (negative value) reduces load
}

def process_timeseries_data(real_timeseries_path, processed_buildings_path, processed_timeseries_output_path):
    """
    Loads raw time series data, processes it based on building info, and saves a cleaned version.
    SKIPS UNIT CONVERSION TO KW - data remains in Joules.
    Accepts paths as arguments.
    """
    print(f"Starting Time Series Data Processing (NO UNIT CONVERSION TO KW)...")
    print(f"Time Series Input: {real_timeseries_path}")
    print(f"Processed Buildings Input: {processed_buildings_path}")
    print(f"Output: {processed_timeseries_output_path} (values will be in Joules per interval)")

    # --- Load Data ---
    try:
        try:
            ts_df = pd.read_csv(real_timeseries_path, sep=None, engine='python', on_bad_lines='warn')
        except Exception as e_sep:
            print(f"Could not auto-detect separator for time series CSV, trying comma: {e_sep}")
            ts_df = pd.read_csv(real_timeseries_path, on_bad_lines='warn') 

        print(f"Loaded time series data with shape: {ts_df.shape}")
        if ts_df.empty:
            print("Error: Loaded time series data is empty.")
            raise ValueError("Time series input file is empty or could not be read properly.")
    except FileNotFoundError:
        print(f"Error: Real time series file not found at {real_timeseries_path}")
        raise
    except Exception as e:
        print(f"Error loading time series CSV from {real_timeseries_path}: {e}")
        raise

    building_flags = {} 
    all_building_ids_from_processed_bldgs = []
    try:
        buildings_df = pd.read_csv(processed_buildings_path)
        buildings_df['building_id'] = buildings_df['building_id'].astype(str)
        all_building_ids_from_processed_bldgs = buildings_df['building_id'].unique().tolist()
        if 'has_solar' in buildings_df.columns and 'has_battery' in buildings_df.columns:
            # Also get solar_capacity_kWp and battery_power_kW for scaling synthetic profiles
            cols_to_get = ['has_solar', 'has_battery']
            if 'solar_capacity_kWp' in buildings_df.columns: cols_to_get.append('solar_capacity_kWp')
            if 'battery_power_kW' in buildings_df.columns: cols_to_get.append('battery_power_kW') # For battery charge/discharge rate
            
            building_flags = buildings_df.set_index('building_id')[cols_to_get].to_dict('index')
            print(f"Loaded asset flags (and capacities if available) for {len(building_flags)} buildings from {processed_buildings_path}.")
        else:
            print(f"Warning: 'has_solar' or 'has_battery' not found in {processed_buildings_path}. Synthetic profiles will assume no solar/battery or use generic capacities.")
            for b_id in all_building_ids_from_processed_bldgs:
                building_flags[b_id] = {'has_solar': False, 'has_battery': False, 'solar_capacity_kWp': 0, 'battery_power_kW': 0}
    except FileNotFoundError:
        print(f"Warning: Processed building file {processed_buildings_path} not found. Synthetic profiles will assume no solar/battery.")
    except Exception as e:
        print(f"Error loading processed building CSV from {processed_buildings_path}: {e}")

    # --- Identify Time Columns ---
    variable_col = None
    if 'VariableName' in ts_df.columns:
        variable_col = 'VariableName'
    elif 'Energy' in ts_df.columns:
        variable_col = 'Energy'

    id_cols_count = 0
    if 'BuildingID' in ts_df.columns or 'building_id' in ts_df.columns:
        id_cols_count += 1
    if variable_col:
        id_cols_count += 1
    if id_cols_count == 0: 
        for i, col_type in enumerate(ts_df.dtypes):
            if pd.api.types.is_numeric_dtype(col_type): id_cols_count = i; break
        if id_cols_count == 0 and not ts_df.empty : id_cols_count = 1
        # print(f"Guessed {id_cols_count} ID columns based on data types.")

    potential_time_cols = ts_df.columns[id_cols_count:]
    time_cols = [
        col for col in potential_time_cols 
        if any(char in str(col) for char in ['/', ':', '-']) or 
           str(col).replace('.', '', 1).replace('-', '', 1).isdigit() or 
           any(keyword in str(col).lower() for keyword in ['interval', 'timestep', 'time', 'hour', 'minute'])
    ]

    if not time_cols:
        if len(ts_df.columns) > id_cols_count:
             time_cols = ts_df.columns[id_cols_count:].tolist()
        else:
             raise ValueError("Could not identify time columns in time series data.")
    num_timesteps = len(time_cols)
    print(f"Identified {num_timesteps} time columns. First few: {time_cols[:min(5, len(time_cols))]}...")

    # --- Variable Mapping and Filtering ---
    ts_df_filtered = pd.DataFrame()
    if variable_col == 'VariableName':
        variable_map = {
            'Electricity:Facility [J](Daily) ': 'facility',
            'Heating:EnergyTransfer [J](Hourly)': 'heating',
            'Cooling:EnergyTransfer [J](Hourly)': 'cooling',
        }
        print(f"Attempting to filter for variables based on variable_map: {list(variable_map.keys())}")

        available_variables_in_csv = ts_df['VariableName'].unique()
        valid_keys_in_map = {k: v for k, v in variable_map.items() if k in available_variables_in_csv}

        if not valid_keys_in_map:
            print("\n*** Warning: None of the specified VariableNames in variable_map were found in the CSV file! ***")
            print("Available VariableNames in CSV were:")
            for var_name in available_variables_in_csv:
                print(f"- '{var_name}'")
            print("Proceeding with synthetic data for all categories for all buildings.")
            ts_df_filtered = pd.DataFrame(columns=ts_df.columns.tolist() + ['Energy', 'Interval'])
        else:
            print(f"Found matches in CSV for these VariableNames: {list(valid_keys_in_map.keys())}")
            ts_df_filtered = ts_df[ts_df['VariableName'].isin(valid_keys_in_map.keys())].copy()
            ts_df_filtered['Energy'] = ts_df_filtered['VariableName'].map(valid_keys_in_map)

            def get_interval(var_name_str):
                if "(Hourly)" in var_name_str:
                    return "Hourly"
                if "(Daily)" in var_name_str:
                    return "Daily"
                return "Unknown"

            ts_df_filtered['Interval'] = ts_df_filtered['VariableName'].apply(get_interval)
    elif variable_col == 'Energy':
        print("Using 'Energy' column from input CSV directly.")
        ts_df_filtered = ts_df.copy()
        valid_keys_in_map = None
    else:
        raise ValueError("'VariableName' or 'Energy' column is missing from the time series input CSV.")

    if variable_col == 'VariableName' and ts_df_filtered.empty and valid_keys_in_map:
        print("Warning: No data rows remained after filtering with valid variable names. Profiles will be synthetic.")
    
    original_id_col_name_in_ts = None
    if 'BuildingID' in ts_df.columns: original_id_col_name_in_ts = 'BuildingID'
    elif 'building_id' in ts_df.columns: original_id_col_name_in_ts = 'building_id'
    else: 
        for common_id_name in ['ID', 'Name', 'Site:Location', 'Building Name', ts_df.columns[0]]: 
            if common_id_name in ts_df.columns:
                original_id_col_name_in_ts = common_id_name; break
    if not original_id_col_name_in_ts:
        raise ValueError("Building ID column missing in time series data.")

    all_building_ids_from_ts = ts_df[original_id_col_name_in_ts].astype(str).unique()
    # print(f"Found {len(all_building_ids_from_ts)} unique building IDs in the time series data using column '{original_id_col_name_in_ts}'.")

    if not ts_df_filtered.empty:
        id_col_to_rename_in_filtered = None
        if 'BuildingID' in ts_df_filtered.columns: id_col_to_rename_in_filtered = 'BuildingID'
        elif 'building_id' in ts_df_filtered.columns: pass 
        elif original_id_col_name_in_ts in ts_df_filtered.columns: id_col_to_rename_in_filtered = original_id_col_name_in_ts
        
        if id_col_to_rename_in_filtered and id_col_to_rename_in_filtered != 'building_id':
            ts_df_filtered.rename(columns={id_col_to_rename_in_filtered: 'building_id'}, inplace=True)
        
        if 'building_id' in ts_df_filtered.columns:
            ts_df_filtered['building_id'] = ts_df_filtered['building_id'].astype(str)
    # else: print("Note: Filtered time series data is empty.")

    # --- UNIT CONVERSION SKIPPED ---
    print("Skipping direct unit conversion to kW. Data will remain in original Joules per interval.")
    # Values in time columns are directly used as Joules.
    # For "Daily" variables, if they represent a daily total spread over hourly columns,
    # they might need to be divided by 24 if an *average hourly Joules* is desired.
    # Current assumption: the values are as-is for their reported interval.
    if not ts_df_filtered.empty:
        for col_name in time_cols: 
            if col_name in ts_df_filtered.columns: 
                 ts_df_filtered[col_name] = pd.to_numeric(ts_df_filtered[col_name], errors='coerce')
        ts_df_filtered.fillna(0.0, inplace=True) # Fill any NaNs from coercion or original data
    # else: print("Filtered time series data is empty, no values to process.")


    # --- Pivot Data ---
    pivoted_data_from_ts = pd.DataFrame() 
    if not ts_df_filtered.empty and 'building_id' in ts_df_filtered.columns and 'Energy' in ts_df_filtered.columns:
        print("Pivoting filtered time series data...")
        try:
            pivoted_data_from_ts = ts_df_filtered.pivot_table(
                index='building_id',
                columns='Energy',
                values=time_cols, 
                aggfunc='sum' 
            )
            print("Pivot successful.")
        except Exception as e:
            print(f"Error during pivot: {e}")
            pivoted_data_from_ts = pd.DataFrame() 
    else:
        print("Skipping pivot as filtered time series data is empty or missing key columns.")

    # --- Generate Profiles ---
    final_data_dict = {} 
    if all_building_ids_from_processed_bldgs:
        ids_to_process_for_profiles = all_building_ids_from_processed_bldgs
        print(f"Processing time series for {len(ids_to_process_for_profiles)} building IDs from '{processed_buildings_path}'.")
    elif all_building_ids_from_ts:
        ids_to_process_for_profiles = all_building_ids_from_ts
    else:
        print("Error: No building IDs available. Cannot generate time series profiles.")
        time_headers_for_empty = [f"T{i+1:03d}" for i in range(num_timesteps if num_timesteps > 0 else 8760)]
        csv_header_for_empty = ["building_id", "Energy"] + time_headers_for_empty
        with open(processed_timeseries_output_path, 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(csv_header_for_empty)
        return 

    J_PER_KWH = 3.6e6 # Joules per kWh, for scaling synthetic data if needed
    HOURS_PER_DAY = 24

    print("Generating/Processing time series profiles (values in Joules per interval)...")
    for b_id_str in ids_to_process_for_profiles: 
        final_data_dict[b_id_str] = {}
        b_asset_info = building_flags.get(b_id_str, {'has_solar': False, 'has_battery': False, 'solar_capacity_kWp': 0, 'battery_power_kW':0})
        has_solar_for_bldg = b_asset_info.get('has_solar', False)
        has_battery_for_bldg = b_asset_info.get('has_battery', False)
        # Get capacities for scaling synthetic profiles (kWp for solar, kW for battery power)
        solar_capacity_kwp = b_asset_info.get('solar_capacity_kWp', 0 if has_solar_for_bldg else 0) # Default to 0 if not found
        battery_power_kw = b_asset_info.get('battery_power_kW', 0 if has_battery_for_bldg else 0)


        building_exists_in_pivoted_ts = b_id_str in pivoted_data_from_ts.index

        for category_name, props in EXPECTED_CATEGORIES.items():
            if category_name == "total_electricity": continue 
            profile_values = None # This will store Joules
            
            # Determine the interval for this category from the filtered data if available
            # This is tricky because pivot_data_from_ts loses the 'Interval' column.
            # We need to infer it or assume. For now, assume hourly for synthetic.
            # If data comes from CSV, its original interval is preserved (as Joules for that interval).
            category_interval = "Hourly" # Default assumption for synthetic data
            if building_exists_in_pivoted_ts and category_name in pivoted_data_from_ts.columns.get_level_values('Energy'):
                # Try to get original interval if possible (difficult after pivot)
                # For now, we assume the pivoted data is already in correct Joules for its original interval
                pass


            if building_exists_in_pivoted_ts and category_name in pivoted_data_from_ts.columns.get_level_values('Energy'):
                try:
                    if not time_cols: pass
                    elif (category_name, time_cols[0]) in pivoted_data_from_ts.columns:
                         profile_series = pivoted_data_from_ts.loc[b_id_str, (category_name, slice(None))]
                         profile_values = profile_series.fillna(0.0).values.tolist() 
                except : pass 
            
            if profile_values is None: # Synthetic generation in Joules
                # For synthetic data, assume an hourly interval for these J values
                # 1 kW for 1 hour = 3.6 MJ = 3.6e6 J
                # For daily values, it would be 1kW * 24h = 24 kWh = 24 * 3.6e6 J
                
                # Example: average 1 kW for facility = 3.6e6 J/hour
                # Example: average 0.5 kW for heating = 1.8e6 J/hour
                # Example: PV generation: if 3 kWp panel, avg output might be 0.5-1 kW. 1kW = 3.6e6 J/hour
                # Example: Battery charge/discharge: if 2kW power, then +/- 2 * 3.6e6 J/hour

                if category_name == "generation":
                    # Synthetic solar generation (Joules per hour)
                    # Rough estimate: peak kWp * some capacity factor * J_PER_KWH / hours (if kWp is daily avg)
                    # Or, simpler: kWp * (random factor for sun) * 3.6e6 (if kWp is instantaneous potential)
                    # Let's use solar_capacity_kwp (which is in kW)
                    # avg_hourly_solar_output_kw = solar_capacity_kwp * random.uniform(0.1, 0.5) # Assume some sun
                    # profile_values = [round(max(0, random.gauss(avg_hourly_solar_output_kw, avg_hourly_solar_output_kw*0.5)) * J_PER_KWH , 0) if has_solar_for_bldg else 0.0 for _ in range(num_timesteps)]
                    # Simpler:
                    base_gen_j = (solar_capacity_kwp if solar_capacity_kwp > 0 else 2.0) * J_PER_KWH * 0.3 # avg 30% output of peak
                    profile_values = [round(max(0, random.gauss(base_gen_j, base_gen_j * 0.8)), 0) if has_solar_for_bldg else 0.0 for _ in range(num_timesteps)]

                elif category_name == "battery_charge":
                    # Synthetic battery charge/discharge (Joules per hour)
                    # Use battery_power_kw
                    base_batt_j = (battery_power_kw if battery_power_kw > 0 else 2.0) * J_PER_KWH
                    profile_values = [round(random.uniform(-base_batt_j, base_batt_j) * 0.5, 0) if has_battery_for_bldg else 0.0 for _ in range(num_timesteps)] # charge/discharge up to 50% of power rating

                elif category_name == "heating": 
                    profile_values = [round(max(0, random.gauss(0.8 * J_PER_KWH, 0.4 * J_PER_KWH)), 0) for _ in range(num_timesteps)] 
                elif category_name == "cooling": 
                    profile_values = [round(max(0, random.gauss(0.4 * J_PER_KWH, 0.2 * J_PER_KWH)), 0) for _ in range(num_timesteps)] 
                elif category_name == "facility": 
                    # If facility data from CSV was (Daily), its Joules are daily totals.
                    # Synthetic facility should also be daily total if we want consistency, then spread or kept as daily.
                    # For simplicity, let's make synthetic facility also an hourly Joules value for now.
                    profile_values = [round(max(0.05 * J_PER_KWH, random.gauss(0.3 * J_PER_KWH, 0.1 * J_PER_KWH)), 0) for _ in range(num_timesteps)] 
                else: profile_values = [0.0] * num_timesteps

            if len(profile_values) != num_timesteps:
                current_len = len(profile_values)
                if current_len < num_timesteps: profile_values.extend([0.0] * (num_timesteps - current_len))
                else: profile_values = profile_values[:num_timesteps]
            final_data_dict[b_id_str][category_name] = profile_values

    print("Calculating 'total_electricity' (in Joules per interval)...")
    for b_id_str in final_data_dict: 
        total_elec_profile_np = np.zeros(num_timesteps, dtype=float)
        for category_name_calc, factor in TOTAL_ELEC_CALC.items():
            if category_name_calc in final_data_dict[b_id_str]:
                profile_data_list = final_data_dict[b_id_str][category_name_calc]
                if isinstance(profile_data_list, (list, np.ndarray)) and len(profile_data_list) == num_timesteps:
                     try:
                          profile_array_np = np.array(profile_data_list, dtype=float)
                          total_elec_profile_np += profile_array_np * factor
                     except ValueError: pass
        final_data_dict[b_id_str]["total_electricity"] = [round(v, 0) for v in total_elec_profile_np.tolist()] # Round Joules to whole numbers

    print("Formatting output data for CSV (values in Joules per interval)...")
    output_rows_list = []
    if num_timesteps == 8760 : 
        time_headers_output = time_cols # Use original time column names if 8760
    else: 
        time_headers_output = [f"T{i+1:04d}" for i in range(num_timesteps)] 
        if num_timesteps > 0 : print(f"Warning: num_timesteps is {num_timesteps}. Using generic time headers.")

    csv_header_output = ["building_id", "Energy"] + time_headers_output

    for b_id_str_out, categories_data_out in final_data_dict.items():
        for category_name_out, profile_values_out in categories_data_out.items():
            if category_name_out in EXPECTED_CATEGORIES: 
                 if len(profile_values_out) != num_timesteps: 
                      current_len_out = len(profile_values_out)
                      if current_len_out < num_timesteps: profile_values_out.extend([0.0] * (num_timesteps - current_len_out))
                      else: profile_values_out = profile_values_out[:num_timesteps]
                 output_rows_list.append([str(b_id_str_out), category_name_out] + profile_values_out) 

    print(f"Saving processed time series data to: {processed_timeseries_output_path}")
    try:
        with open(processed_timeseries_output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(csv_header_output)
            writer.writerows(output_rows_list)
        print("Time series data processing complete.")
    except Exception as e:
        print(f"Error writing processed time series CSV: {e}")
        raise

if __name__ == "__main__":
    test_dir = "test_timeseries_data_v4_no_conversion"
    os.makedirs(test_dir, exist_ok=True)
    test_real_timeseries_path = os.path.join(test_dir, "dummy_merged_as_is_v4.csv")
    test_processed_buildings_path = os.path.join(test_dir, "dummy_buildings_processed_for_ts_v4.csv") 
    test_processed_timeseries_output_path = os.path.join(test_dir, "dummy_time_series_processed_v4.csv")

    if not os.path.exists(test_processed_buildings_path):
        pd.DataFrame({
            'building_id': ['bldg_ts_1', 'bldg_ts_2'], 'has_solar': [True, False], 'has_battery': [True, False],
            'solar_capacity_kWp':[3,0], 'battery_power_kW':[2,0], # Added capacities
            'lat': [0,0],'lon':[0,0],'peak_load_kW':[0,0],
            'battery_capacity_kWh':[0,0],'label':['A','B'],
            'line_id':['L1','L1'],'building_function':['residential','commercial']
        }).to_csv(test_processed_buildings_path, index=False)

    if not os.path.exists(test_real_timeseries_path):
        num_dummy_ts_timesteps = 24 
        time_cols_dummy_ts = [(datetime(2023,1,1) + timedelta(hours=i)).strftime("%m/%d %H:%M:%S") for i in range(num_dummy_ts_timesteps)]
        dummy_ts_data_list = [
            {'BuildingID': 'bldg_ts_1', 'VariableName': 'Electricity:Facility [J](Daily) ', **{t:random.randint(1e8,5e8) for t in time_cols_dummy_ts}}, # Larger Joules
            {'BuildingID': 'bldg_ts_1', 'VariableName': 'Heating:EnergyTransfer [J](Hourly)', **{t:random.randint(1e7,5e7) for t in time_cols_dummy_ts}},
            {'BuildingID': 'bldg_ts_2', 'VariableName': 'Cooling:EnergyTransfer [J](Hourly)', **{t:random.randint(1e7,3e7) for t in time_cols_dummy_ts}},
        ]
        pd.DataFrame(dummy_ts_data_list).to_csv(test_real_timeseries_path, index=False)

    print(f"\nRunning timeseries_processing.py (NO CONVERSION) directly for testing in '{test_dir}'.")
    try:
        process_timeseries_data(test_real_timeseries_path, test_processed_buildings_path, test_processed_timeseries_output_path)
        print(f"\nTest finished. Check: {test_processed_timeseries_output_path}")
        # if os.path.exists(test_processed_timeseries_output_path):
            # output_check_df = pd.read_csv(test_processed_timeseries_output_path)
            # print("Sample of output (Joules):")
            # print(output_check_df.head())
    except Exception as e:
        print(f"Error during direct test run: {e}")
        import traceback
        traceback.print_exc()
