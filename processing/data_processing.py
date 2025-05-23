import pandas as pd
import numpy as np
import random
import os

def get_weighted_random_type(types_config):
    if not types_config or not isinstance(types_config, list): return None
    type_options = [t for t in types_config if isinstance(t, dict) and "weight" in t and isinstance(t['weight'], (int, float)) and t['weight'] > 0]
    if not type_options:
        if types_config and isinstance(types_config[0], dict): return types_config[0]
        return None
    weights = [t['weight'] for t in type_options]
    return random.choices(type_options, weights=weights, k=1)[0]

def process_building_data(buildings_df: pd.DataFrame, solar_scenarios=None, battery_scenarios=None) -> pd.DataFrame:
    """
    Processes building data from a DataFrame and applies scenarios.
    Returns a processed DataFrame.
    """
    print("Starting Building Data Processing...")

    # --- Preprocessing & Column Renaming (Building ID, Lat, Lon) ---
    if 'ogc_fid' in buildings_df.columns: buildings_df.rename(columns={'ogc_fid': 'building_id'}, inplace=True)
    elif 'building_id' not in buildings_df.columns:
        alt_id_cols = ['OBJECTID', 'fid', 'id', 'ID']
        for alt_id in alt_id_cols:
            if alt_id in buildings_df.columns:
                print(f"Using '{alt_id}' as 'building_id'.")
                buildings_df.rename(columns={alt_id: 'building_id'}, inplace=True)
                break
    if 'building_id' not in buildings_df.columns:
        print("Creating 'building_id' from index.")
        buildings_df['building_id'] = 'bldg_' + buildings_df.index.astype(str)

    buildings_df['building_id'] = buildings_df['building_id'].astype(str)

    lat_col, lon_col = None, None
    common_lat_cols = ['lat', 'latitude', 'Latitude', 'LATITUDE', 'Y', 'y', 'centr_y', 'y_coord', 'Lat', 'Lon']
    common_lon_cols = ['lon', 'longitude', 'Longitude', 'LONGITUDE', 'X', 'x', 'centr_x', 'x_coord', 'Lng', 'Long']
    for col in common_lat_cols:
        if col in buildings_df.columns: lat_col = col; break
    for col in common_lon_cols:
        if col in buildings_df.columns: lon_col = col; break

    if not lat_col or not lon_col: raise ValueError("Latitude/Longitude columns not found in the uploaded buildings file.")
    if lat_col != 'lat': buildings_df.rename(columns={lat_col: 'lat'}, inplace=True)
    if lon_col != 'lon': buildings_df.rename(columns={lon_col: 'lon'}, inplace=True)
        
    buildings_df['lat'] = pd.to_numeric(buildings_df['lat'], errors='coerce')
    buildings_df['lon'] = pd.to_numeric(buildings_df['lon'], errors='coerce')
    buildings_df.dropna(subset=['lat', 'lon', 'building_id'], inplace=True)
    
    print(f"{len(buildings_df)} buildings remaining after cleaning.")
    if buildings_df.empty: return buildings_df

    if 'building_function' not in buildings_df.columns:
        buildings_df['building_function'] = 'default'
    else:
        buildings_df['building_function'] = buildings_df['building_function'].astype(str).fillna('default').str.lower()
    print("Normalized 'building_function' column.")

    # --- Fill/Generate Missing Columns ---
    if 'peak_load_kW' not in buildings_df.columns:
        def estimate_peak_load(row):
            try: area = float(row.get('area', 100))
            except: area = 100.0
            func = row.get('building_function', 'default')
            load_factor = random.uniform(0.05, 0.15) if func == 'residential' else random.uniform(0.1, 0.3)
            base_load = 5 if func == 'residential' else 10
            return round(max(base_load, area * load_factor), 1)
        buildings_df['peak_load_kW'] = buildings_df.apply(estimate_peak_load, axis=1)
    else:
        buildings_df['peak_load_kW'] = pd.to_numeric(buildings_df['peak_load_kW'], errors='coerce').fillna(0)

    if solar_scenarios and isinstance(solar_scenarios, dict):
        print("Applying solar scenarios...")
        def assign_solar_scenario(row):
            bldg_func_key = str(row.get('building_function', 'default')).lower()
            scenario = solar_scenarios.get(bldg_func_key, solar_scenarios.get('default'))
            has_solar, solar_kwp = False, 0.0
            if scenario and isinstance(scenario, dict) and random.random() < scenario.get('penetration_rate', 0):
                has_solar = True
                chosen_type = get_weighted_random_type(scenario.get('types'))
                if chosen_type and 'kWp_range' in chosen_type and len(chosen_type['kWp_range']) == 2:
                    solar_kwp = round(random.uniform(*chosen_type['kWp_range']), 1)
            return pd.Series([has_solar, solar_kwp])
        buildings_df[['has_solar', 'solar_capacity_kWp']] = buildings_df.apply(assign_solar_scenario, axis=1)
    else:
        buildings_df[['has_solar', 'solar_capacity_kWp']] = False, 0.0

    if battery_scenarios and isinstance(battery_scenarios, dict):
        print("Applying battery scenarios...")
        def assign_battery_scenario(row):
            bldg_func_key = str(row.get('building_function', 'default')).lower()
            scenario = battery_scenarios.get(bldg_func_key, battery_scenarios.get('default'))
            has_battery, battery_kwh, battery_kw = False, 0.0, 0.0
            if scenario and isinstance(scenario, dict):
                if row.get('has_solar', False) and 'penetration_rate_if_solar' in scenario:
                    if random.random() < scenario['penetration_rate_if_solar']: has_battery = True
                if not has_battery and 'overall_penetration_rate' in scenario:
                    if random.random() < scenario['overall_penetration_rate']: has_battery = True
                if has_battery:
                    chosen_type = get_weighted_random_type(scenario.get('types'))
                    if chosen_type:
                        if 'kWh_range' in chosen_type and len(chosen_type['kWh_range']) == 2:
                            battery_kwh = round(random.uniform(*chosen_type['kWh_range']), 1)
                        if 'kW_range' in chosen_type and len(chosen_type['kW_range']) == 2:
                            battery_kw = round(random.uniform(*chosen_type['kW_range']), 1)
            return pd.Series([has_battery, battery_kwh, battery_kw])
        buildings_df[['has_battery', 'battery_capacity_kWh', 'battery_power_kW']] = buildings_df.apply(assign_battery_scenario, axis=1)
    else:
        buildings_df[['has_battery', 'battery_capacity_kWh', 'battery_power_kW']] = False, 0.0, 0.0

    if 'label' not in buildings_df.columns:
        if 'meestvoorkomendelabel' in buildings_df.columns and buildings_df['meestvoorkomendelabel'].notna().any():
             buildings_df['label'] = buildings_df['meestvoorkomendelabel'].fillna(random.choice(['A', 'B', 'C', 'D']))
        else:
             buildings_df['label'] = [random.choice(['A', 'B', 'C', 'D']) for _ in range(len(buildings_df))]
    else:
        buildings_df['label'] = buildings_df['label'].astype(str).fillna('A')

    buildings_df['line_id'] = 'NOT_ASSIGNED_YET'
    print("Building data processing complete.")
    return buildings_df
