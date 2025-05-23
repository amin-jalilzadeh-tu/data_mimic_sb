import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math

def distance_lat_lon_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in kilometers."""
    R = 6371.0
    try:
        lat1_f, lon1_f, lat2_f, lon2_f = map(float, [lat1, lon1, lat2, lon2])
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1_f, lon1_f, lat2_f, lon2_f])
    except (ValueError, TypeError):
        return float('inf')
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_assignments(buildings_df: pd.DataFrame, num_lines: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates building assignments from a DataFrame.
    Returns two DataFrames: (assignments_df, buildings_with_lines_df).
    """
    print("Starting Assignment Generation...")
    
    if buildings_df.empty or not all(col in buildings_df.columns for col in ['building_id', 'lat', 'lon']):
        print("Warning: Input DataFrame for assignment is empty or missing required columns (building_id, lat, lon).")
        empty_assign = pd.DataFrame(columns=["building_id", "line_id", "distance_km"])
        return empty_assign, buildings_df

    # Drop any remaining NaNs in coordinate columns
    buildings_df.dropna(subset=['lat', 'lon'], inplace=True)
    if buildings_df.empty:
        print("Warning: No buildings with valid coordinates for clustering.")
        empty_assign = pd.DataFrame(columns=["building_id", "line_id", "distance_km"])
        return empty_assign, buildings_df


    if len(buildings_df) < num_lines:
        print(f"Warning: Number of buildings ({len(buildings_df)}) is less than requested lines ({num_lines}). Adjusting.")
        num_lines = max(1, len(buildings_df))

    print(f"Performing K-Means clustering for {num_lines} groups...")
    coords = buildings_df[['lat', 'lon']].values
    
    # Suppress KMeans memory leak warning on Windows by setting n_init explicitly
    try:
        kmeans = KMeans(n_clusters=num_lines, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
    except Exception as e:
        # Fallback for older scikit-learn or other issues
        print(f"KMeans with n_init=10 failed: {e}. Trying with n_init='auto'.")
        kmeans = KMeans(n_clusters=num_lines, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(coords)

    buildings_df['cluster_label'] = cluster_labels
    centroids = kmeans.cluster_centers_
    
    print("Assigning line IDs and calculating distances...")
    assignments_list = []
    cluster_to_line_id_map = {i: f"L{i + 1:04d}" for i in range(num_lines)}
    
    buildings_with_lines_df = buildings_df.copy()

    for index, row in buildings_with_lines_df.iterrows():
        cluster = row['cluster_label']
        assigned_line_id = cluster_to_line_id_map.get(cluster, "L_ERROR")
        
        # Update line_id in the DataFrame
        buildings_with_lines_df.loc[index, 'line_id'] = assigned_line_id
        
        centroid_lat, centroid_lon = centroids[cluster]
        distance_km = distance_lat_lon_km(row['lat'], row['lon'], centroid_lat, centroid_lon)
        
        assignments_list.append({
            "building_id": row['building_id'],
            "line_id": assigned_line_id,
            "distance_km": round(distance_km, 5) if distance_km != float('inf') else None
        })

    assignments_df = pd.DataFrame(assignments_list)
    
    # Drop temporary cluster label column from final output
    if 'cluster_label' in buildings_with_lines_df.columns:
        buildings_with_lines_df.drop(columns=['cluster_label'], inplace=True)
        
    print("Assignment generation complete.")
    return assignments_df, buildings_with_lines_df
