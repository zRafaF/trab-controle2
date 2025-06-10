import pandas as pd
import matplotlib.pyplot as plt
import sys
import os # Import os for path manipulation

def plot_temperature_data(csv_files):
    """
    Reads temperature data from multiple CSV files and plots them on a single graph.

    Args:
        csv_files (list): A list of paths to the CSV files to be plotted.
    """
    plt.figure(figsize=(12, 7)) # Set the figure size for better readability
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style for the plot

    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            continue

        try:
            # Read the CSV file into a pandas DataFrame
            # Parse 'Timestamp' column as datetime objects
            df = pd.read_csv(file_path, parse_dates=['Timestamp'])

            # Ensure the necessary columns exist
            if 'Timestamp' not in df.columns or 'Temperature (C)' not in df.columns:
                print(f"Skipping {file_path}: Missing 'Timestamp' or 'Temperature (C)' column.")
                continue

            # Extract data
            timestamps = df['Timestamp']
            temperatures = df['Temperature (C)']

            # Create a label for the plot based on the filename
            # e.g., "fan_0_percent.csv" becomes "Fan 0%"
            label = os.path.basename(file_path).replace('.csv', '').replace('_', ' ').title()

            # Plot the data
            plt.plot(timestamps, temperatures, label=label)

        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty or has no data.")
        except Exception as e:
            print(f"An error occurred while reading or plotting {file_path}: {e}")

    # Add plot titles and labels
    plt.title('Incubator Temperature Response at Different Fan Settings', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(fontsize=10) # Display the legend with labels
    plt.grid(True) # Show grid lines
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show() # Display the plot

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_csv1> [path_to_csv2 ...] ")
        print("Example: python plot_data.py fan_0_percent.csv fan_50_percent.csv fan_100_percent.csv")
        sys.exit(1)

    # All arguments after the script name are treated as CSV file paths
    csv_files_to_plot = sys.argv[1:]
    plot_temperature_data(csv_files_to_plot)
