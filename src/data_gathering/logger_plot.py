import serial
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Import FuncAnimation for real-time plotting
import threading # Import threading for running serial read in a separate thread
import collections # For using deque to limit data points in plot

# --- Global Data Storage and Flags ---
# Use a deque (double-ended queue) to store a limited number of data points for plotting.
# This prevents the plot from becoming too slow with an ever-growing list.
MAX_PLOT_POINTS = 200 # Display last 200 data points on the plot
plot_timestamps = collections.deque(maxlen=MAX_PLOT_POINTS)
plot_temperatures = collections.deque(maxlen=MAX_PLOT_POINTS)

# Flag to control the serial reading thread
running = True

# Store the start time for relative elapsed time in plots
script_start_time = time.time()

# --- Matplotlib Plot Setup ---
# Set up the figure and axes for the plot outside the function
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'r-', label='Temperature (°C)') # 'r-' for red line

# Set initial plot labels and title
ax.set_xlabel('Time Elapsed (seconds)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Real-time Temperature Reading from Arduino')
ax.grid(True)
ax.legend()

# --- Serial Reading Function (to run in a separate thread) ---
def serial_reader_thread(port, baud_rate, output_filename):
    """
    Function to continuously read data from the Arduino serial port and log it.
    This function will run in a separate thread to not block the main plotting loop.
    """
    global running, plot_timestamps, plot_temperatures, script_start_time
    ser = None # Initialize serial object to None for finally block

    try:
        # Establish serial connection
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on port {port} at {baud_rate} baud.")
        print(f"Logging data to: {output_filename}")

        with open(output_filename, 'a') as log_file:
            # Write a header if the file is newly created or empty
            if log_file.tell() == 0:
                 log_file.write("Timestamp,Voltage (mV),Temperature (C)\n")

            while running: # Loop as long as the 'running' flag is True
                serial_data_line = ser.readline().decode('utf-8').strip()

                if serial_data_line:
                    parts = serial_data_line.split('\t')

                    # Skip header from Arduino
                    if "Voltage (mV)" in parts[0] and "Temperature (C)" in parts[1]:
                        print(f"Skipping header from Arduino: {serial_data_line}")
                        continue

                    try:
                        voltage_mv = float(parts[0])
                        temperature_c = float(parts[1])

                        current_time = time.time()
                        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                        elapsed_time = current_time - script_start_time # Time since script started

                        print(f"[{timestamp_str}] Voltage: {voltage_mv:.2f} mV, Temp: {temperature_c:.2f} C")

                        log_file.write(f"{timestamp_str},{voltage_mv:.2f},{temperature_c:.2f}\n")
                        log_file.flush() # Ensure data is written to disk

                        # Add data to the plotting deques
                        plot_timestamps.append(elapsed_time)
                        plot_temperatures.append(temperature_c)

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: '{serial_data_line}' - {e}")
                time.sleep(0.01) # Small delay to prevent busy-waiting
    except serial.SerialException as e:
        print(f"Serial port error in thread: {e}")
        print(f"Please ensure '{port}' is correct and the Arduino is connected/not in use.")
    except FileNotFoundError:
        print(f"Error: Could not create or open file '{output_filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred in thread: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Serial connection closed in thread.")
        running = False # Signal the main thread to stop if an error occurs here

# --- Matplotlib Animation Update Function ---
def update_plot(frame):
    """
    This function is called by FuncAnimation to update the plot with new data.
    """
    # Set the data for the plot line
    line.set_data(list(plot_timestamps), list(plot_temperatures))

    # Dynamically adjust x-axis limits
    if plot_timestamps:
        min_time = plot_timestamps[0]
        max_time = plot_timestamps[-1]
        # Add a small buffer to the x-axis limits
        ax.set_xlim(min_time - 1, max_time + 1)
    else:
        ax.set_xlim(0, 10) # Default if no data yet

    # Dynamically adjust y-axis limits
    if plot_temperatures:
        min_temp = min(plot_temperatures)
        max_temp = max(plot_temperatures)
        # Add a small buffer to the y-axis limits (e.g., 5 degrees)
        ax.set_ylim(min_temp - 5, max_temp + 5)
    else:
        ax.set_ylim(0, 100) # Default if no data yet (0-100 C range)

    # Autoscale view is already handled by set_xlim and set_ylim

    return line, # FuncAnimation requires returning the artists that were modified

# --- Main Execution Block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <output_file_name>")
        print("Example: python arduino_logger.py temperature_log.csv")
        sys.exit(1)

    output_file = sys.argv[1]

    # --- Configuration for Arduino Connection ---
    # IMPORTANT: Change 'COM5' to your Arduino's actual serial port!
    # On Windows, it's typically 'COMX' (e.g., 'COM3', 'COM5').
    # On Linux/macOS, it's typically '/dev/ttyUSB0' or '/dev/ttyACM0'.
    arduino_port = 'COM6'
    # The baud rate must match the Serial.begin() value in your Arduino code
    arduino_baud_rate = 9600
    # --------------------------------------------

    # Start the serial reading in a separate thread
    # The `daemon=True` ensures the thread will close when the main program exits
    serial_thread = threading.Thread(target=serial_reader_thread,
                                     args=(arduino_port, arduino_baud_rate, output_file),
                                     daemon=True)
    serial_thread.start()

    # Set up the animation. This function will continuously call 'update_plot'.
    # interval=100 means update every 100 milliseconds (10 frames per second).
    # blit=True means only draw the parts of the plot that have changed (can improve performance).
    ani = FuncAnimation(fig, update_plot, interval=100, blit=True)

    # Show the plot. This call blocks the main thread and keeps the plot window open.
    # The serial_reader_thread will continue to run in the background.
    try:
        plt.show()
    except KeyboardInterrupt: # Catch Ctrl+C to cleanly stop
        print("\nPlotting stopped by user.")
    finally:
        running = False # Signal the serial reading thread to stop
        serial_thread.join(timeout=2) # Wait for the thread to finish gracefully (with timeout)
        print("Script finished.")

