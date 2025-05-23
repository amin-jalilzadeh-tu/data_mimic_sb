import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math
import csv
import os

def distance_lat_lon_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in kilometers."""
    R = 6371.0 # Earth radius in kilometers
    try:
        # Ensure inputs are valid numbers before conversion
        lat1_f, lon1_f, lat2_f, lon2_f = float(lat1), float(lon1), float(lat2), float(lon2)
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1_f, lon1_f, lat2_f, lon2_f])
    except (ValueError, TypeError) as e:
        # print(f"Warning: Invalid coordinates for distance calculation: ({lat1},{lon1}) to ({lat2},{lon2}). Error: {e}. Returning infinity.")
        return float('inf') # Return infinity for invalid inputs

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def generate_assignments(buildings_input_path, assignments_output_path, buildings_output_path_with_lines, num_lines):
    """
    Generates building assignments based on clustering and updates building data with line IDs.
    Accepts paths and num_lines as arguments.
    """
    print(f"Starting Assignment Generation...")
    print(f"Buildings Input: {buildings_input_path}")
    print(f"Assignments Output: {assignments_output_path}")
    print(f"Updated Buildings Output (with new line_id): {buildings_output_path_with_lines}")
    print(f"Target Number of Lines/Clusters: {num_lines}")

    # 1. Load Processed Building Data
    if not os.path.exists(buildings_input_path):
        print(f"Error: Input building file not found at {buildings_input_path}")
        raise FileNotFoundError(f"Input building file for assignment not found: {buildings_input_path}")
    try:
        buildings_df = pd.read_csv(buildings_input_path)
        # Ensure necessary columns exist
        required_cols_for_assignment = ['building_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols_for_assignment if col not in buildings_df.columns]
        if missing_cols:
             print(f"Error: Input building file '{buildings_input_path}' is missing required columns for assignment: {missing_cols}.")
             raise ValueError(f"Missing required columns: {missing_cols} in building input file.")
        
        # Convert lat/lon to numeric, coercing errors and then dropping NaNs
        buildings_df['lat'] = pd.to_numeric(buildings_df['lat'], errors='coerce')
        buildings_df['lon'] = pd.to_numeric(buildings_df['lon'], errors='coerce')
        
        initial_count = len(buildings_df)
        # Also ensure building_id is not NaN before converting to str, though it should be handled by data_processing
        buildings_df.dropna(subset=['building_id'], inplace=True) 
        buildings_df['building_id'] = buildings_df['building_id'].astype(str)
        buildings_df.dropna(subset=['lat', 'lon'], inplace=True) # Drop after lat/lon conversion
        
        if len(buildings_df) < initial_count:
            print(f"Dropped {initial_count - len(buildings_df)} buildings due to missing/invalid lat, lon, or building_id before clustering.")

        print(f"Loaded {len(buildings_df)} buildings with valid coordinates for clustering.")

        if buildings_df.empty:
            print("Error: No buildings with valid coordinates available for clustering after cleaning.")
            # Write empty output files to prevent downstream errors if that's the desired behavior.
            # For assignments:
            pd.DataFrame(columns=["building_id", "line_id", "distance_km"]).to_csv(assignments_output_path, index=False)
            # For updated buildings: use columns from input if possible, or a minimal set
            try:
                empty_updated_df_cols = pd.read_csv(buildings_input_path, nrows=0).columns.tolist()
                if 'line_id' not in empty_updated_df_cols: empty_updated_df_cols.append('line_id')
            except: # Fallback if input can't be read for columns
                empty_updated_df_cols = required_cols_for_assignment + ['line_id']
            pd.DataFrame(columns=empty_updated_df_cols).to_csv(buildings_output_path_with_lines, index=False)
            print("Empty assignment and updated building files created.")
            return # Stop further processing if no data

        current_num_lines = int(num_lines) # Ensure it's an int
        if len(buildings_df) < current_num_lines:
            print(f"Warning: Number of buildings with valid coordinates ({len(buildings_df)}) is less than the desired number of lines ({current_num_lines}).")
            current_num_lines = max(1, len(buildings_df)) # Need at least 1 cluster if there are buildings
            print(f"Adjusted NUM_LINES for clustering to: {current_num_lines}")
        num_lines = current_num_lines


    except Exception as e:
        print(f"Error loading or processing building CSV from {buildings_input_path} for assignment: {e}")
        raise

    # 2. Perform K-Means Clustering
    print(f"Performing K-Means clustering to create {num_lines} groups (lines)...")
    coords_for_kmeans = buildings_df[['lat', 'lon']].values
    
    if len(coords_for_kmeans) == 0: # Should be caught by buildings_df.empty check, but as a safeguard
        print("Error: No coordinates available to perform K-Means clustering.")
        # Write empty files as above
        pd.DataFrame(columns=["building_id", "line_id", "distance_km"]).to_csv(assignments_output_path, index=False)
        pd.DataFrame(columns=buildings_df.columns.tolist() + (['line_id'] if 'line_id' not in buildings_df.columns else [])).to_csv(buildings_output_path_with_lines, index=False)
        return

    # n_init='auto' is default in newer scikit-learn, suppresses warning
    kmeans_model = KMeans(n_clusters=num_lines, random_state=42, n_init='auto')
    buildings_df['cluster_label'] = kmeans_model.fit_predict(coords_for_kmeans)
    print("K-Means clustering complete.")

    # 3. Get Cluster Centroids (representing line connection points or centers)
    cluster_centroids = kmeans_model.cluster_centers_ # Array of [lat, lon] for each cluster center
    centroid_id_map = {i: cluster_centroids[i] for i in range(num_lines)} # Map cluster label to centroid coords

    # 4. Assign Line IDs and Calculate Distances
    print("Assigning line IDs based on clusters and calculating distances to centroids...")
    assignments_list = []
    # Generate line IDs like L0001, L0002, ...
    cluster_to_line_id_map = {cluster_idx: f"L{cluster_idx + 1:04d}" for cluster_idx in range(num_lines)}

    # Use .copy() when adding 'line_id' to avoid SettingWithCopyWarning if buildings_df is a slice
    # However, buildings_df here should be the full DataFrame loaded.
    # Direct assignment is usually fine after .dropna() if it returns a new DataFrame.
    # To be safe, ensure we operate on the main df or a proper copy for modification.
    
    # The 'line_id' column should have been created in data_processing.py, here we update it.
    if 'line_id' not in buildings_df.columns:
        buildings_df['line_id'] = None # Create if somehow missing

    for index, building_row in buildings_df.iterrows():
        cluster_label = building_row['cluster_label']
        building_unique_id = building_row['building_id']
        bldg_lat, bldg_lon = building_row['lat'], building_row['lon']

        assigned_line_id_val = cluster_to_line_id_map[cluster_label]
        centroid_coords = centroid_id_map[cluster_label]
        centroid_lat_val, centroid_lon_val = centroid_coords[0], centroid_coords[1]

        distance_to_centroid_km = distance_lat_lon_km(bldg_lat, bldg_lon, centroid_lat_val, centroid_lon_val)

        assignments_list.append({
            "building_id": building_unique_id,
            "line_id": assigned_line_id_val,
            "distance_km": round(distance_to_centroid_km, 5) if distance_to_centroid_km != float('inf') else None 
        })
        # Update the line_id in the main buildings DataFrame
        buildings_df.loc[index, 'line_id'] = assigned_line_id_val

    print(f"Assigned buildings to {len(cluster_to_line_id_map)} generated lines.")

    # 5. Create building_assignments_generated.csv
    print(f"Saving building-to-line assignments to: {assignments_output_path}")
    try:
        assignments_df = pd.DataFrame(assignments_list)
        assignments_df.to_csv(assignments_output_path, index=False, columns=["building_id", "line_id", "distance_km"])
        print("Assignments CSV created successfully.")
    except Exception as e:
        print(f"Error writing assignments CSV to {assignments_output_path}: {e}")
        raise


    # 6. Save Updated Building DataFrame (with the new line_id)
    print(f"Saving updated building data (with new line_id) to: {buildings_output_path_with_lines}")
    try:
        # Columns to save: all from the input building_df, ensuring 'line_id' is updated.
        # 'cluster_label' can be dropped if not needed in the final output.
        output_df_for_buildings = buildings_df.copy()
        if 'cluster_label' in output_df_for_buildings.columns:
            output_df_for_buildings = output_df_for_buildings.drop(columns=['cluster_label'])
        
        # Ensure the column order is reasonable, e.g., 'line_id' near other IDs or key attributes.
        # This is optional, but can make the CSV more readable.
        # Example: bring 'line_id' after 'label' if 'label' exists.
        cols = output_df_for_buildings.columns.tolist()
        if 'line_id' in cols and 'label' in cols:
            cols.insert(cols.index('label') + 1, cols.pop(cols.index('line_id')))
            output_df_for_buildings = output_df_for_buildings[cols]
            
        output_df_for_buildings.to_csv(buildings_output_path_with_lines, index=False)
        print("Updated building CSV (with new line_id) saved successfully.")
    except Exception as e:
        print(f"Error saving updated building CSV to {buildings_output_path_with_lines}: {e}")
        print(f"DataFrame columns at time of saving: {output_df_for_buildings.columns.tolist()}")
        raise

    print("Assignment generation process finished.")

# This block is for testing the script directly
if __name__ == "__main__":
    test_dir_assign = "test_assignment_data"
    os.makedirs(test_dir_assign, exist_ok=True)

    # Assumes dummy_buildings_processed_scenario.csv was created by data_processing.py's test block
    test_buildings_input_assign = os.path.join(test_dir_assign, "dummy_buildings_for_assignment.csv") 
    test_assignments_output_assign = os.path.join(test_dir_assign, "dummy_building_assignments_generated.csv")
    test_buildings_with_lines_output_assign = os.path.join(test_dir_assign, "dummy_buildings_processed_with_lines.csv")
    test_num_lines_assign = 3

    # Create a dummy input file for assignment testing if it doesn't exist
    if not os.path.exists(test_buildings_input_assign):
        print(f"Creating dummy input for assignment generation test: {test_buildings_input_assign}")
        dummy_data_assign_list = []
        for i in range(15): # Create 15 dummy buildings
            dummy_data_assign_list.append({
                'building_id': f'assign_bldg_{i+1}',
                'lat': 52.2 + random.uniform(-0.05, 0.05) + (i//5)*0.1, # Create some spatial grouping
                'lon': 5.2 + random.uniform(-0.05, 0.05) + (i%5)*0.1,
                'peak_load_kW': random.randint(5,50),
                'has_solar': random.choice([True, False]),
                'solar_capacity_kWp': random.uniform(0,10) if True else 0, # Simplified
                'has_battery': random.choice([True, False]),
                'battery_capacity_kWh': random.uniform(0,20) if True else 0,
                'battery_power_kW': random.uniform(0,5) if True else 0,
                'label': random.choice(['A','B','C']),
                'line_id': 'NOT_ASSIGNED_YET', # This will be overwritten
                'building_function': random.choice(['residential', 'commercial', 'default']),
                'area': random.randint(50,300)
            })
        pd.DataFrame(dummy_data_assign_list).to_csv(test_buildings_input_assign, index=False)

    print(f"\nRunning assignment_generation.py directly for testing with dummy files in '{test_dir_assign}'.")
    try:
        generate_assignments(
            test_buildings_input_assign,
            test_assignments_output_assign,
            test_buildings_with_lines_output_assign,
            test_num_lines_assign
        )
        print(f"\nTest finished. Check content of: \n - {test_assignments_output_assign}\n - {test_buildings_with_lines_output_assign}")
        # if os.path.exists(test_assignments_output_assign):
        #     print("\nSample Assignments:")
        #     print(pd.read_csv(test_assignments_output_assign).head())
        # if os.path.exists(test_buildings_with_lines_output_assign):
        #     print("\nSample Updated Buildings (with line_id):")
        #     print(pd.read_csv(test_buildings_with_lines_output_assign)[['building_id', 'line_id', 'lat', 'lon']].head())
    except Exception as e:
        print(f"Error during direct test run of assignment_generation.py: {e}")
        import traceback
        traceback.print_exc()
