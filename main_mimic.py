import os
import sys
import data_processing # Assuming it's in the same directory or Python path
import timeseries_processing
import assignment_generation

# --- Configuration ---
CONFIG = {
    # Step 1: Data Processing (buildings)
    "REAL_BUILDINGS_PATH": r"D:\Documents\E_Plus_2030_py\output\399a658c-b28a-4871-966a-10cf1bfedce0\extracted_idf_buildings.csv",
    "PROCESSED_BUILDINGS_OUTPUT_PATH": r"D:\Documents\energy_cluster_gnn\mimic\buildings_processed.csv",

    # Step 2: Time Series Processing
    "REAL_TIMESERIES_PATH": r"D:\Documents\E_Plus_2030_py\output\399a658c-b28a-4871-966a-10cf1bfedce0\results\merged_as_is.csv",
    "PROCESSED_TIMESERIES_OUTPUT_PATH": r"D:\Documents\energy_cluster_gnn\mimic\time_series_processed.csv",

    # Step 3: Assignment Generation
    "ASSIGNMENTS_OUTPUT_PATH": r"D:\Documents\energy_cluster_gnn\mimic\building_assignments_generated.csv",
    "BUILDINGS_PROCESSED_WITH_LINES_OUTPUT_PATH": r"D:\Documents\energy_cluster_gnn\mimic\buildings_processed_with_lines.csv",
    "NUM_LINES_FOR_ASSIGNMENT": 6,

    # --- Scenario Configurations for Solar and Battery ---
    "SOLAR_SCENARIOS": {
        # Default scenario if building_function is not specified or not found in other keys
        "default": {
            "penetration_rate": 0.10,  # 10% of these buildings get solar
            "types": [
                {"name": "default_pv_small", "kWp_range": [2.0, 4.0], "weight": 0.6},
                {"name": "default_pv_medium", "kWp_range": [4.1, 8.0], "weight": 0.4}
            ]
        },
        "residential": {
            "penetration_rate": 0.25,  # 25% of residential buildings
            "types": [
                {"name": "res_pv_standard", "kWp_range": [3.0, 7.0], "weight": 0.7}, # kWp = kiloWatt-peak (solar panel capacity)
                {"name": "res_pv_large", "kWp_range": [7.1, 15.0], "weight": 0.3}
            ]
        },
        "non_residential": { # Example: commercial, industrial, office
            "penetration_rate": 0.40,  # 40% of non-residential buildings
            "types": [
                {"name": "comm_pv_medium", "kWp_range": [10.0, 50.0], "weight": 0.5},
                {"name": "comm_pv_large", "kWp_range": [50.1, 250.0], "weight": 0.3},
                {"name": "comm_pv_xlarge", "kWp_range": [250.1, 1000.0], "weight": 0.2} # e.g., large rooftops
            ]
        },
        # Add more building_function keys as needed, e.g., "industrial", "office"
        # Ensure building_function values in your CSV match these keys (case-sensitive)
        # If a building_function from your CSV isn't listed here, it will use the "default" scenario.
    },

    "BATTERY_SCENARIOS": {
        # Default scenario
        "default": {
            "penetration_rate_if_solar": 0.15,  # 15% of buildings *that already have solar* under this category
            # "overall_penetration_rate": 0.02, # Optional: 2% of all buildings in this category get a battery, regardless of solar
            "types": [
                {"name": "default_batt_small", "kWh_range": [4.0, 8.0], "kW_range": [2.0, 4.0], "weight": 1.0}
            ]
        },
        "residential": {
            "penetration_rate_if_solar": 0.30,  # 30% of residential buildings with solar get a battery
            # "overall_penetration_rate": 0.05, # 5% of all residential buildings get a battery
            "types": [
                {"name": "res_batt_daily", "kWh_range": [5.0, 15.0], "kW_range": [2.5, 7.0], "weight": 0.8}, # kWh = kiloWatt-hour (battery energy capacity)
                {"name": "res_batt_backup", "kWh_range": [15.1, 30.0], "kW_range": [5.0, 10.0], "weight": 0.2}  # kW = kiloWatt (battery power capacity)
            ]
        },
        "non_residential": {
            "penetration_rate_if_solar": 0.50,  # 50% of non-residential buildings with solar get a battery
            # "overall_penetration_rate": 0.10, # 10% of all non-residential buildings get a battery
            "types": [
                {"name": "comm_batt_demand_peak", "kWh_range": [20.0, 100.0], "kW_range": [10.0, 50.0], "weight": 0.6},
                {"name": "comm_batt_storage_large", "kWh_range": [100.1, 500.0], "kW_range": [50.1, 200.0], "weight": 0.4}
            ]
        }
    }
}

def check_file_exists(filepath, step_name, is_input=True):
    """Checks if a file exists and prints an appropriate message."""
    if is_input and not os.path.exists(filepath):
        print(f"Error for {step_name}: Input file not found: {filepath}")
        print("Please ensure the file exists or run the previous step that generates it.")
        return False
    return True

def run_step1_data_processing():
    """Runs the building data processing step."""
    print("\n--- Running Step 1: Building Data Processing ---")
    if not check_file_exists(CONFIG["REAL_BUILDINGS_PATH"], "Step 1"):
        return False
    
    output_dir = os.path.dirname(CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        # Pass the scenario configurations to the processing function
        data_processing.process_building_data(
            real_buildings_path=CONFIG["REAL_BUILDINGS_PATH"],
            processed_buildings_output_path=CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"],
            solar_scenarios=CONFIG.get("SOLAR_SCENARIOS"), # Use .get() for graceful handling if key is missing
            battery_scenarios=CONFIG.get("BATTERY_SCENARIOS")
        )
        print("--- Step 1: Building Data Processing Finished ---")
        return True
    except Exception as e:
        print(f"Error during Step 1: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False

def run_step2_timeseries_processing():
    """Runs the time series data processing step."""
    print("\n--- Running Step 2: Time Series Data Processing ---")
    if not check_file_exists(CONFIG["REAL_TIMESERIES_PATH"], "Step 2 (Real Time Series)"):
        return False
    if not check_file_exists(CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"], "Step 2 (Processed Buildings Input)"):
        print("This file is generated by Step 1. Please run Step 1 first if it's missing.")
        return False

    output_dir = os.path.dirname(CONFIG["PROCESSED_TIMESERIES_OUTPUT_PATH"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        timeseries_processing.process_timeseries_data(
            real_timeseries_path=CONFIG["REAL_TIMESERIES_PATH"],
            processed_buildings_path=CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"],
            processed_timeseries_output_path=CONFIG["PROCESSED_TIMESERIES_OUTPUT_PATH"]
        )
        print("--- Step 2: Time Series Data Processing Finished ---")
        return True
    except Exception as e:
        print(f"Error during Step 2: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_step3_assignment_generation():
    """Runs the building assignment generation step."""
    print("\n--- Running Step 3: Assignment Generation ---")
    if not check_file_exists(CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"], "Step 3 (Processed Buildings Input)"):
        print("This file is generated by Step 1. Please run Step 1 first if it's missing.")
        return False

    output_dir_assignments = os.path.dirname(CONFIG["ASSIGNMENTS_OUTPUT_PATH"])
    if output_dir_assignments and not os.path.exists(output_dir_assignments):
        os.makedirs(output_dir_assignments)
        print(f"Created directory: {output_dir_assignments}")
        
    output_dir_buildings_with_lines = os.path.dirname(CONFIG["BUILDINGS_PROCESSED_WITH_LINES_OUTPUT_PATH"])
    if output_dir_buildings_with_lines and not os.path.exists(output_dir_buildings_with_lines):
        os.makedirs(output_dir_buildings_with_lines)
        print(f"Created directory: {output_dir_buildings_with_lines}")

    try:
        assignment_generation.generate_assignments(
            buildings_input_path=CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"],
            assignments_output_path=CONFIG["ASSIGNMENTS_OUTPUT_PATH"],
            buildings_output_path_with_lines=CONFIG["BUILDINGS_PROCESSED_WITH_LINES_OUTPUT_PATH"],
            num_lines=CONFIG["NUM_LINES_FOR_ASSIGNMENT"]
        )
        print("--- Step 3: Assignment Generation Finished ---")
        return True
    except Exception as e:
        print(f"Error during Step 3: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\nData Processing Pipeline Menu:")
        print("1. Run Step 1: Building Data Processing")
        print("2. Run Step 2: Time Series Data Processing (requires Step 1 output)")
        print("3. Run Step 3: Assignment Generation (requires Step 1 output)")
        print("4. Run all steps (1, 2, 3)")
        print("5. Run Step 1 and 2")
        print("6. Run Step 2 and 3 (requires Step 1 output to exist)")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            run_step1_data_processing()
        elif choice == '2':
            run_step2_timeseries_processing()
        elif choice == '3':
            run_step3_assignment_generation()
        elif choice == '4':
            print("\n--- Running All Steps ---")
            if run_step1_data_processing():
                if run_step2_timeseries_processing():
                    run_step3_assignment_generation()
            print("--- All Steps Execution Attempted ---")
        elif choice == '5':
            print("\n--- Running Step 1 and 2 ---")
            if run_step1_data_processing():
                run_step2_timeseries_processing()
            print("--- Step 1 and 2 Execution Attempted ---")
        elif choice == '6':
            print("\n--- Running Step 2 and 3 ---")
            if not check_file_exists(CONFIG["PROCESSED_BUILDINGS_OUTPUT_PATH"], "Prerequisite for Step 2 & 3"):
                print("Cannot proceed with Step 2 and 3 as Step 1 output is missing.")
            elif run_step2_timeseries_processing():
                run_step3_assignment_generation()
            print("--- Step 2 and 3 Execution Attempted ---")
        elif choice == '0':
            print("Exiting pipeline.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    mimic_dir = r"D:\Documents\energy_cluster_gnn\mimic"
    if not os.path.exists(mimic_dir):
        try:
            os.makedirs(mimic_dir)
            print(f"Created base output directory: {mimic_dir}")
        except OSError as e:
            print(f"Error creating directory {mimic_dir}: {e}.")

    main_menu()
