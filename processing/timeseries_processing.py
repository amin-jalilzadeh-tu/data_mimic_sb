import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

EXPECTED_CATEGORIES = {"heating": {}, "cooling": {}, "facility": {}, "generation": {}, "battery_charge": {}, "total_electricity": {}}
TOTAL_ELEC_CALC = {"facility": 1, "heating": 1, "cooling": 1, "generation": -1, "battery_charge": 1}

def process_timeseries_data(ts_df: pd.DataFrame, buildings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes time series data from a DataFrame, using building data for context.
    SKIPS UNIT CONVERSION TO KW. Returns a processed DataFrame.
    """
    print("Starting Time Series Data Processing (NO UNIT CONVERSION)...")

    building_flags = {}
    all_building_ids_from_processed_bldgs = []
    if not buildings_df.empty:
        all_building_ids_from_processed_bldgs = buildings_df['building_id'].unique().tolist()
        cols_to_get = ['has_solar', 'has_battery']
        if 'solar_capacity_kWp' in buildings_df.columns: cols_to_get.append('solar_capacity_kWp')
        if 'battery_power_kW' in buildings_df.columns: cols_to_get.append('battery_power_kW')
        building_flags = buildings_df.set_index('building_id')[cols_to_get].to_dict('index')
        print(f"Loaded asset flags for {len(building_flags)} buildings.")

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
        if id_cols_count == 0 and not ts_df.empty: id_cols_count = 1
    time_cols = ts_df.columns[id_cols_count:].tolist()
    num_timesteps = len(time_cols)
    print(f"Identified {num_timesteps} time columns.")

    ts_df_filtered = pd.DataFrame()
    if variable_col == 'VariableName':
        variable_map = {
            'Electricity:Facility [J](Daily) ': 'facility',
            'Heating:EnergyTransfer [J](Hourly)': 'heating',
            'Cooling:EnergyTransfer [J](Hourly)': 'cooling',
        }

        available_vars = ts_df['VariableName'].unique()
        valid_keys = {k: v for k, v in variable_map.items() if k in available_vars}
        print(f"Found matches for: {list(valid_keys.keys())}")

        if valid_keys:
            ts_df_filtered = ts_df[ts_df['VariableName'].isin(valid_keys.keys())].copy()
            ts_df_filtered['Energy'] = ts_df_filtered['VariableName'].map(valid_keys)
            # Ensure time columns are numeric
            for col in time_cols:
                ts_df_filtered[col] = pd.to_numeric(ts_df_filtered[col], errors='coerce')
            ts_df_filtered.fillna(0.0, inplace=True)
        else:
            print("Warning: No matching variables found. All time series profiles will be synthetic.")
    elif variable_col == 'Energy':
        print("Using 'Energy' column from input DataFrame directly.")
        ts_df_filtered = ts_df.copy()
    else:
        raise ValueError("'VariableName' or 'Energy' column missing from time series file.")

    original_id_col = 'BuildingID' if 'BuildingID' in ts_df.columns else 'building_id'
    if original_id_col not in ts_df.columns: original_id_col = ts_df.columns[0]
    
    if not ts_df_filtered.empty:
        if original_id_col != 'building_id': ts_df_filtered.rename(columns={original_id_col: 'building_id'}, inplace=True)
        ts_df_filtered['building_id'] = ts_df_filtered['building_id'].astype(str)

    pivoted_data = pd.DataFrame()
    if not ts_df_filtered.empty and 'building_id' in ts_df_filtered.columns:
        pivoted_data = ts_df_filtered.pivot_table(index='building_id', columns='Energy', values=time_cols, aggfunc='sum')
        print("Pivot successful.")
    
    ids_to_process = all_building_ids_from_processed_bldgs if all_building_ids_from_processed_bldgs else ts_df[original_id_col].astype(str).unique().tolist()
    
    final_data_dict = {}
    J_PER_KWH = 3.6e6
    
    for b_id in ids_to_process:
        final_data_dict[b_id] = {}
        info = building_flags.get(b_id, {})
        has_solar = info.get('has_solar', False)
        has_battery = info.get('has_battery', False)
        solar_cap = info.get('solar_capacity_kWp', 2.0)
        batt_pow = info.get('battery_power_kW', 2.0)
        
        for cat in EXPECTED_CATEGORIES:
            if cat == "total_electricity": continue
            profile = None
            if b_id in pivoted_data.index and cat in pivoted_data.columns.get_level_values('Energy'):
                profile = pivoted_data.loc[b_id, (slice(None), cat)].fillna(0.0).values.tolist()

            if profile is None: # Synthetic data in Joules
                if cat == "generation":
                    base_gen_j = solar_cap * J_PER_KWH * 0.3
                    profile = [round(max(0, random.gauss(base_gen_j, base_gen_j * 0.8)), 0) if has_solar else 0.0 for _ in range(num_timesteps)]
                elif cat == "battery_charge":
                    base_batt_j = batt_pow * J_PER_KWH
                    profile = [round(random.uniform(-base_batt_j, base_batt_j) * 0.5, 0) if has_battery else 0.0 for _ in range(num_timesteps)]
                elif cat == "heating": profile = [round(max(0, random.gauss(0.8 * J_PER_KWH, 0.4 * J_PER_KWH)), 0) for _ in range(num_timesteps)]
                elif cat == "cooling": profile = [round(max(0, random.gauss(0.4 * J_PER_KWH, 0.2 * J_PER_KWH)), 0) for _ in range(num_timesteps)]
                elif cat == "facility": profile = [round(max(0.05 * J_PER_KWH, random.gauss(0.3 * J_PER_KWH, 0.1 * J_PER_KWH)), 0) for _ in range(num_timesteps)]
                else: profile = [0.0] * num_timesteps
            
            final_data_dict[b_id][cat] = profile

    for b_id in final_data_dict:
        total_elec = np.zeros(num_timesteps, dtype=float)
        for cat, factor in TOTAL_ELEC_CALC.items():
            if cat in final_data_dict[b_id]:
                total_elec += np.array(final_data_dict[b_id][cat], dtype=float) * factor
        final_data_dict[b_id]["total_electricity"] = [round(v, 0) for v in total_elec.tolist()]

    output_rows = []
    headers = ["building_id", "Energy"] + time_cols
    for b_id, data in final_data_dict.items():
        for cat, profile in data.items():
            if len(profile) == num_timesteps:
                output_rows.append([str(b_id), cat] + profile)

    print("Time series processing complete.")
    return pd.DataFrame(output_rows, columns=headers)
