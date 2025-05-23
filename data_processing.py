import pandas as pd
import numpy as np
import random
import os

# --- Helper function for weighted random choice ---
def get_weighted_random_type(types_config):
    """
    Selects a type from a list of type configurations based on their weights.
    Each type_config in types_list should be a dict with 'name', 'weight', and range keys.
    Returns None if no valid types or weights are found.
    """
    if not types_config or not isinstance(types_config, list):
        return None
    
    # Filter for valid type dictionaries that contain a 'weight'
    type_options = [t for t in types_config if isinstance(t, dict) and "weight" in t and isinstance(t['weight'], (int, float)) and t['weight'] > 0]
    
    if not type_options: # No valid types with positive weights
        # Fallback: if there are types defined but weights are missing/invalid, pick the first one
        if types_config and isinstance(types_config[0], dict):
            # print("Warning: No valid weights found for types, picking first type.")
            return types_config[0]
        return None

    weights = [t['weight'] for t in type_options]
    chosen_type = random.choices(type_options, weights=weights, k=1)[0]
    return chosen_type

# --- Main processing function ---
def process_building_data(real_buildings_path, processed_buildings_output_path,
                          solar_scenarios=None, battery_scenarios=None):
    """
    Loads raw building data, processes it, and saves a cleaned version.
    Applies solar and battery scenarios if provided.
    """
    print(f"Starting Building Data Processing...")
    print(f"Input: {real_buildings_path}")
    print(f"Output: {processed_buildings_output_path}")
    if solar_scenarios:
        print("Solar scenarios provided and will be applied.")
    else:
        print("No solar scenarios provided; using basic/fallback logic for solar.")
    if battery_scenarios:
        print("Battery scenarios provided and will be applied.")
    else:
        print("No battery scenarios provided; using basic/fallback logic for battery.")

    # --- Load Data ---
    try:
        buildings_df = pd.read_csv(real_buildings_path)
        print(f"Loaded {len(buildings_df)} buildings.")
    except FileNotFoundError:
        print(f"Error: Real building file not found at {real_buildings_path}")
        raise
    except Exception as e:
        print(f"Error loading building CSV from {real_buildings_path}: {e}")
        raise

    # --- Preprocessing & Column Renaming (Building ID, Lat, Lon) ---
    # Building ID
    if 'ogc_fid' in buildings_df.columns:
        buildings_df.rename(columns={'ogc_fid': 'building_id'}, inplace=True)
    elif 'building_id' not in buildings_df.columns:
        alt_id_cols = ['OBJECTID', 'fid', 'id', 'ID'] # Common alternatives
        found_alt_id = False
        for alt_id in alt_id_cols:
            if alt_id in buildings_df.columns:
                print(f"Warning: 'building_id' not found. Using '{alt_id}' as 'building_id'.")
                buildings_df.rename(columns={alt_id: 'building_id'}, inplace=True)
                found_alt_id = True
                break
        if not found_alt_id:
            print("Warning: No standard 'building_id' or common alternative found. Creating 'building_id' from DataFrame index.")
            buildings_df['building_id'] = 'bldg_' + buildings_df.index.astype(str)

    if 'building_id' not in buildings_df.columns: 
         raise ValueError("CRITICAL ERROR: 'building_id' could not be established even after fallbacks.")
    buildings_df['building_id'] = buildings_df['building_id'].astype(str)

    # Latitude and Longitude
    lat_col, lon_col = None, None
    common_lat_cols = ['lat', 'latitude', 'Latitude', 'LATITUDE', 'Y', 'y', 'centr_y', 'y_coord', 'Lat', 'Lon']
    common_lon_cols = ['lon', 'longitude', 'Longitude', 'LONGITUDE', 'X', 'x', 'centr_x', 'x_coord', 'Lng', 'Long']
    for col_name in common_lat_cols:
        if col_name in buildings_df.columns: lat_col = col_name; break
    for col_name in common_lon_cols:
        if col_name in buildings_df.columns: lon_col = col_name; break
    
    if not lat_col or not lon_col:
        print(f"Error: Latitude or Longitude column not found. Searched common names like {common_lat_cols} and {common_lon_cols}.")
        print(f"Available columns in input CSV: {buildings_df.columns.tolist()}")
        raise ValueError("Latitude/Longitude columns missing. Please ensure your input CSV has recognizable lat/lon columns.")
    
    if lat_col != 'lat': buildings_df.rename(columns={lat_col: 'lat'}, inplace=True)
    if lon_col != 'lon': buildings_df.rename(columns={lon_col: 'lon'}, inplace=True)
        
    # Convert lat/lon to numeric, coercing errors, then drop NaNs for essential columns
    buildings_df['lat'] = pd.to_numeric(buildings_df['lat'], errors='coerce')
    buildings_df['lon'] = pd.to_numeric(buildings_df['lon'], errors='coerce')
    buildings_df.dropna(subset=['lat', 'lon', 'building_id'], inplace=True)
    
    print(f"{len(buildings_df)} buildings remaining after dropping rows with missing essential info (building_id, lat, lon).")
    if buildings_df.empty:
        print("Warning: No buildings remain after dropping NaNs. Output CSV will be empty or processing might fail if subsequent steps expect data.")
        # Decide if an empty CSV with headers should be written or an error raised.
        # For now, allow empty DataFrame to proceed, but subsequent steps might fail.

    # --- Ensure 'building_function' column exists for scenario application ---
    if 'building_function' not in buildings_df.columns:
        print("Warning: 'building_function' column not found in input CSV.")
        if solar_scenarios or battery_scenarios:
             print("Solar/Battery scenarios are provided but rely on 'building_function'. All buildings will use 'default' scenario settings.")
        buildings_df['building_function'] = 'default' # Create a default column if missing
    else:
        # Normalize building_function: convert to string, fill NaNs with 'default', and maybe lowercase for consistency
        buildings_df['building_function'] = buildings_df['building_function'].astype(str).fillna('default').str.lower()
        print("Normalized 'building_function' column (to string, NaNs to 'default', lowercase). Ensure scenario keys match.")


    # --- Fill/Generate Missing Columns ---

    # 1. Estimate peak_load_kW
    print("Processing 'peak_load_kW'...")
    if 'peak_load_kW' not in buildings_df.columns:
        print("  'peak_load_kW' column missing, generating based on 'area' and 'building_function'.")
        def estimate_peak_load(row):
            area = row.get('area', 100) # Default area if 'area' column is missing or NaN
            # Ensure area is numeric
            try: area = float(area)
            except (ValueError, TypeError): area = 100.0

            func = row.get('building_function', 'default') # Uses the potentially cleaned func
            
            load_factor = random.uniform(0.05, 0.15) if func == 'residential' else random.uniform(0.1, 0.3)
            base_load = 5 if func == 'residential' else 10
            return round(max(base_load, area * load_factor), 1)
        buildings_df['peak_load_kW'] = buildings_df.apply(estimate_peak_load, axis=1)
    else:
        print("  'peak_load_kW' column exists. Converting to numeric and filling NaNs with 0.")
        buildings_df['peak_load_kW'] = pd.to_numeric(buildings_df['peak_load_kW'], errors='coerce').fillna(0)

    # 2. Generate Solar Info based on Scenarios
    print("Processing solar columns...")
    if solar_scenarios and isinstance(solar_scenarios, dict):
        print("  Applying solar scenarios based on 'building_function'.")
        def assign_solar_scenario(row):
            # Ensure building_function matches keys in solar_scenarios (e.g. lowercase)
            bldg_func_key = str(row.get('building_function', 'default')).lower()
            scenario = solar_scenarios.get(bldg_func_key, solar_scenarios.get('default')) # Fallback to default scenario
            
            has_solar = False
            solar_kwp = 0.0

            if scenario and isinstance(scenario, dict):
                if random.random() < scenario.get('penetration_rate', 0):
                    has_solar = True
                    chosen_type = get_weighted_random_type(scenario.get('types'))
                    if chosen_type and 'kWp_range' in chosen_type and isinstance(chosen_type['kWp_range'], list) and len(chosen_type['kWp_range']) == 2:
                        min_kwp, max_kwp = chosen_type['kWp_range']
                        solar_kwp = round(random.uniform(min_kwp, max_kwp), 1)
                    # else: print(f"Warning: Invalid or missing kWp_range for solar type in scenario for {bldg_func_key}")
            # else: print(f"Warning: No valid solar scenario found for building_function '{bldg_func_key}' or default.")
            return pd.Series([has_solar, solar_kwp])

        buildings_df[['has_solar', 'solar_capacity_kWp']] = buildings_df.apply(assign_solar_scenario, axis=1)
    else: 
        print("  No valid solar scenarios provided or scenarios not a dict. Using basic/fallback logic for solar.")
        if 'has_solar' not in buildings_df.columns: # Only generate if column doesn't exist
            print("    Generating basic 'has_solar' and 'solar_capacity_kWp'.")
            def assign_solar_basic(row): 
                bouwjaar = row.get('bouwjaar', 1980) # Assuming 'bouwjaar' might exist for basic logic
                if pd.isna(bouwjaar): bouwjaar = 1980
                try: bouwjaar = int(bouwjaar)
                except: bouwjaar = 1980 # Default if conversion fails
                prob_solar = max(0, min(1, 0.1 + 0.6 * ((bouwjaar - 1980) / (2024 - 1980)))) # Example logic
                has_solar = random.random() < prob_solar
                solar_kwp = round(random.uniform(2.0, 15.0), 1) if has_solar else 0.0
                return pd.Series([has_solar, solar_kwp])
            buildings_df[['has_solar', 'solar_capacity_kWp']] = buildings_df.apply(assign_solar_basic, axis=1)
        else: # If columns exist, ensure correct types and fill NaNs
             print("    'has_solar'/'solar_capacity_kWp' columns exist. Ensuring types and filling NaNs.")
             buildings_df['has_solar'] = buildings_df['has_solar'].map(
                 {'True': True, 'False': False, True: True, False: False, 'true': True, 'false': False, 1: True, 0: False, 1.0: True, 0.0: False}
             ).fillna(False).astype(bool)
             buildings_df['solar_capacity_kWp'] = pd.to_numeric(buildings_df['solar_capacity_kWp'], errors='coerce').fillna(0.0)


    # 3. Generate Battery Info based on Scenarios
    print("Processing battery columns...")
    if battery_scenarios and isinstance(battery_scenarios, dict):
        print("  Applying battery scenarios based on 'building_function'.")
        def assign_battery_scenario(row):
            bldg_func_key = str(row.get('building_function', 'default')).lower()
            scenario = battery_scenarios.get(bldg_func_key, battery_scenarios.get('default'))
            
            has_battery = False
            battery_kwh = 0.0
            battery_kw = 0.0

            if scenario and isinstance(scenario, dict):
                # Primary: penetration if solar exists
                if row.get('has_solar', False) and 'penetration_rate_if_solar' in scenario:
                    if random.random() < scenario['penetration_rate_if_solar']:
                        has_battery = True
                
                # Secondary: overall penetration rate (if not already assigned battery)
                if not has_battery and 'overall_penetration_rate' in scenario:
                    if random.random() < scenario['overall_penetration_rate']:
                        has_battery = True
                
                if has_battery:
                    chosen_type = get_weighted_random_type(scenario.get('types'))
                    if chosen_type:
                        if 'kWh_range' in chosen_type and isinstance(chosen_type['kWh_range'], list) and len(chosen_type['kWh_range']) == 2:
                            min_kwh, max_kwh = chosen_type['kWh_range']
                            battery_kwh = round(random.uniform(min_kwh, max_kwh), 1)
                        if 'kW_range' in chosen_type and isinstance(chosen_type['kW_range'], list) and len(chosen_type['kW_range']) == 2:
                            min_kw, max_kw = chosen_type['kW_range']
                            battery_kw = round(random.uniform(min_kw, max_kw), 1)
            return pd.Series([has_battery, battery_kwh, battery_kw])

        buildings_df[['has_battery', 'battery_capacity_kWh', 'battery_power_kW']] = buildings_df.apply(assign_battery_scenario, axis=1)
    else: 
        print("  No valid battery scenarios provided. Using basic/fallback logic for battery.")
        if 'has_battery' not in buildings_df.columns:
            print("    Generating basic 'has_battery', 'battery_capacity_kWh', 'battery_power_kW'.")
            def assign_battery_basic(row):
                has_solar_flag = row.get('has_solar', False) # Depends on solar being processed first
                has_battery = has_solar_flag and (random.random() < 0.3) # Example: 30% of solar homes
                battery_kwh = round(random.uniform(5.0, 30.0), 1) if has_battery else 0.0
                battery_kw = round(random.uniform(3.0, 10.0), 1) if has_battery else 0.0
                return pd.Series([has_battery, battery_kwh, battery_kw])
            buildings_df[['has_battery', 'battery_capacity_kWh', 'battery_power_kW']] = buildings_df.apply(assign_battery_basic, axis=1)
        else: # If columns exist, ensure correct types and fill NaNs
            print("    'has_battery'/'battery_capacity_kWh'/'battery_power_kW' columns exist. Ensuring types and filling NaNs.")
            buildings_df['has_battery'] = buildings_df['has_battery'].map(
                {'True': True, 'False': False, True: True, False: False, 'true': True, 'false': False, 1: True, 0: False, 1.0: True, 0.0: False}
            ).fillna(False).astype(bool)
            buildings_df['battery_capacity_kWh'] = pd.to_numeric(buildings_df['battery_capacity_kWh'], errors='coerce').fillna(0.0)
            buildings_df['battery_power_kW'] = pd.to_numeric(buildings_df['battery_power_kW'], errors='coerce').fillna(0.0)


    # 4. Generate Label (remains the same, or could be scenario-driven too if needed)
    print("Processing 'label' column...")
    if 'label' not in buildings_df.columns:
        print("  'label' column missing, generating.")
        if 'meestvoorkomendelabel' in buildings_df.columns and buildings_df['meestvoorkomendelabel'].notna().any():
             buildings_df['label'] = buildings_df['meestvoorkomendelabel'].fillna(random.choice(['A', 'B', 'C', 'D']))
             print("    Used 'meestvoorkomendelabel' for 'label', filled NaNs randomly.")
        else:
             buildings_df['label'] = [random.choice(['A', 'B', 'C', 'D']) for _ in range(len(buildings_df))]
             print("    Generated random 'label'.")
    else:
        print("  'label' column exists. Filling NaNs with 'A'.")
        buildings_df['label'] = buildings_df['label'].astype(str).fillna('A') # Ensure string type

    # 5. Add/Update placeholder line_id
    print("Adding/Updating 'line_id' column (will be properly assigned in Step 3 if run).")
    buildings_df['line_id'] = 'NOT_ASSIGNED_YET' # Overwrite or create for this processing step

    # --- Final Check & Save ---
    # Define all columns that are expected to be in the output
    required_cols = [
        'building_id', 'lat', 'lon', 'peak_load_kW', 'has_solar',
        'solar_capacity_kWp', 'has_battery', 'battery_capacity_kWh',
        'battery_power_kW', 'label', 'line_id', 'building_function' # building_function is now key
    ]
    # Add other useful columns that might exist in the input and should be carried over
    optional_cols_to_carry = ['area', 'height', 'bouwjaar', 'age_range', 'meestvoorkomendelabel']
    
    final_output_columns = []
    # Add required columns first, ensuring they exist (create with defaults if not)
    for col in required_cols:
        if col not in buildings_df.columns:
            print(f"Warning: Expected column '{col}' is missing after processing. Creating with default value.")
            if col in ['lat', 'lon', 'peak_load_kW', 'solar_capacity_kWp', 'battery_capacity_kWh', 'battery_power_kW']:
                buildings_df[col] = 0.0
            elif col in ['has_solar', 'has_battery']:
                buildings_df[col] = False
            elif col == 'building_function':
                 buildings_df[col] = 'default'
            else: # building_id, label, line_id
                buildings_df[col] = 'UNKNOWN_OR_DEFAULT'
        final_output_columns.append(col)

    # Add optional columns if they exist in the DataFrame and are not already included
    for col in optional_cols_to_carry:
        if col in buildings_df.columns and col not in final_output_columns:
            final_output_columns.append(col)
    
    # Ensure no duplicate columns (though the above logic should prevent it)
    final_output_columns = sorted(list(set(final_output_columns)), key=final_output_columns.index)


    print(f"Final columns for output: {final_output_columns}")
    print(f"Saving processed building data to: {processed_buildings_output_path}")
    try:
        # If buildings_df is empty, to_csv will create an empty file with headers (if columns are specified)
        if buildings_df.empty:
            print("Warning: Building DataFrame is empty. Output file will have headers but no data.")
            # Create an empty DataFrame with the desired columns to ensure headers are written
            empty_df_with_headers = pd.DataFrame(columns=final_output_columns)
            empty_df_with_headers.to_csv(processed_buildings_output_path, index=False)
        else:
            buildings_df[final_output_columns].to_csv(processed_buildings_output_path, index=False)
        print("Building data processing complete.")
    except KeyError as e:
        print(f"KeyError during save: {e}. This means a column in final_output_columns is not in buildings_df.")
        print(f"DataFrame columns available: {buildings_df.columns.tolist()}")
        print(f"Attempted output columns: {final_output_columns}")
        print("Attempting to save all available columns in the DataFrame as a fallback.")
        buildings_df.to_csv(processed_buildings_output_path, index=False) # Fallback
        print("Fallback save complete (all available columns). Please check the output CSV.")
    except Exception as ex:
        print(f"An unexpected error occurred during saving the processed building data: {ex}")
        raise


if __name__ == "__main__":
    # --- Dummy data and scenario for direct testing ---
    test_real_buildings_path = "dummy_extracted_buildings_for_scenario.csv"
    test_processed_output_path = "dummy_buildings_processed_scenario.csv"

    # More detailed dummy scenarios for testing
    dummy_solar_scenarios = {
        "default": {"penetration_rate": 0.05, "types": [{"name": "s_def", "kWp_range": [1,2], "weight":1}]},
        "residential": {
            "penetration_rate": 0.6, # 60% of residential
            "types": [
                {"name": "s_res_small", "kWp_range": [2.5, 4.0], "weight":0.4},
                {"name": "s_res_medium", "kWp_range": [4.1, 7.0], "weight":0.6}
            ]
        },
        "commercial": {
            "penetration_rate": 0.75, # 75% of commercial
             "types": [
                {"name": "s_com_large", "kWp_range": [10, 50], "weight":0.7},
                {"name": "s_com_xlarge", "kWp_range": [50.1, 200], "weight":0.3}
            ]
        },
        "industrial": {"penetration_rate": 0.2, "types": [{"name": "s_ind", "kWp_range": [100,500], "weight":1}]}
        # 'unknown' building_function will use 'default'
    }
    dummy_battery_scenarios = {
        "default": {"penetration_rate_if_solar": 0.1, "types": [{"name": "b_def", "kWh_range": [2,3], "kW_range": [1,2], "weight":1}]},
        "residential": {
            "penetration_rate_if_solar": 0.5, # 50% of solar-equipped residential
            "overall_penetration_rate": 0.1, # Additional 10% of all residential (even without solar)
            "types": [
                {"name": "b_res_daily", "kWh_range": [5,10], "kW_range": [2.5,5], "weight":0.8},
                {"name": "b_res_backup", "kWh_range": [10.1,20], "kW_range": [5,7], "weight":0.2}
            ]
        },
        "commercial": {
            "penetration_rate_if_solar": 0.4,
            "types": [{"name": "b_com", "kWh_range": [20,100], "kW_range": [10,50], "weight":1}]
        }
        # 'industrial' and 'unknown' will use 'default' battery scenario
    }

    if not os.path.exists(test_real_buildings_path):
        print(f"Creating dummy input file for scenario testing: {test_real_buildings_path}")
        dummy_data_list = []
        building_functions = ['residential', 'commercial', 'industrial', 'unknown', None, 'Residential', 'COMMERCIAL']
        for i in range(20): # Create 20 dummy buildings
            dummy_data_list.append({
                'ID': f'test_bldg_{i+1}', # Using 'ID' to test alternative building_id column name
                'Lat': 52.0 + i*0.005,    # Using 'Lat' to test alternative latitude column name
                'X': 5.0 + i*0.005,       # Using 'X' (often longitude) to test alternative
                'area': random.randint(50, 1000),
                'building_function': random.choice(building_functions),
                'bouwjaar': random.randint(1950, 2023),
                'meestvoorkomendelabel': random.choice(['A', 'B', 'C', 'D', None])
            })
        pd.DataFrame(dummy_data_list).to_csv(test_real_buildings_path, index=False)

    print(f"\nRunning data_processing.py directly for scenario testing with dummy files.")
    try:
        process_building_data(
            test_real_buildings_path,
            test_processed_output_path,
            solar_scenarios=dummy_solar_scenarios,
            battery_scenarios=dummy_battery_scenarios
        )
        print(f"\nTest finished. Check content of: {test_processed_output_path}")
        
        # Optional: Load and print some stats from the output for verification
        if os.path.exists(test_processed_output_path):
            output_df = pd.read_csv(test_processed_output_path)
            if not output_df.empty:
                print("\n--- Sample of Processed Data ---")
                print(output_df[['building_id', 'building_function', 'has_solar', 'solar_capacity_kWp', 'has_battery', 'battery_capacity_kWh', 'battery_power_kW']].head())
                print(f"\nTotal buildings processed: {len(output_df)}")
                print(f"Solar stats: {output_df['has_solar'].sum()} buildings have solar. Average kWp (if solar): {output_df[output_df['has_solar']]['solar_capacity_kWp'].mean():.2f}")
                print(f"Battery stats: {output_df['has_battery'].sum()} buildings have batteries. Average kWh (if battery): {output_df[output_df['has_battery']]['battery_capacity_kWh'].mean():.2f}")
                print("\nCounts per building_function:")
                print(output_df['building_function'].value_counts())
                print("\nSolar counts per building_function:")
                print(output_df[output_df['has_solar']]['building_function'].value_counts())
                print("\nBattery counts per building_function:")
                print(output_df[output_df['has_battery']]['building_function'].value_counts())
            else:
                print("Processed output file is empty.")
        else:
            print("Processed output file was not created.")

    except Exception as e:
        print(f"Error during direct test run of data_processing.py: {e}")
        import traceback
        traceback.print_exc()
