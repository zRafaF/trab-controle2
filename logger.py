import serial
import sys
import time

def read_and_log_arduino(port, baud_rate, output_filename):
    """
    Connects to an Arduino via serial, reads data, and logs it to a file.

    Args:
        port (str): The serial port the Arduino is connected to (e.g., 'COM5', '/dev/ttyUSB0').
        baud_rate (int): The baud rate for serial communication (must match Arduino's Serial.begin()).
        output_filename (str): The name of the file to log the data to.
    """
    try:
        # Establish serial connection with the Arduino
        # The timeout ensures the read() method doesn't block indefinitely
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on port {port} at {baud_rate} baud.")
        print(f"Logging data to: {output_filename}")

        # Open the output file in append mode ('a') or write mode ('w')
        # 'w' will overwrite the file each time the script runs.
        # 'a' will append to the file if it exists, otherwise create it.
        # For a log file, 'a' is generally preferred.
        with open(output_filename, 'a') as log_file:
            # Optionally, write a header to the log file if it's new
            # To ensure the header isn't duplicated on subsequent runs if using 'a' mode,
            # you might check if the file is empty first, but for simplicity, we'll
            # just write it if the file is opened in 'w' or is newly created in 'a'.
            # The Arduino already prints a header to its serial monitor, so we'll mirror that.
            if log_file.tell() == 0: # Check if file is empty (only for 'a' mode)
                 log_file.write("Timestamp,Voltage (mV),Temperature (C)\n")


            while True:
                # Read a line from the serial port.
                # readLine() reads until a newline character ('\n') is found.
                # .decode('utf-8') converts bytes to a string.
                # .strip() removes leading/trailing whitespace, including '\r\n'.
                line = ser.readline().decode('utf-8').strip()

                if line: # Process only if a non-empty line is received
                    # The Arduino sends "Voltage (mV) \t Temperature (C)"
                    # We can split this by the tab character
                    parts = line.split('\t')

                    # Check if the line is the header from Arduino
                    if "Voltage (mV)" in parts[0] and "Temperature (C)" in parts[1]:
                        print(f"Skipping header: {line}")
                        continue # Skip the header and wait for actual data

                    try:
                        # Extract voltage and temperature
                        voltage_mv = float(parts[0])
                        temperature_c = float(parts[1])

                        # Get current timestamp for logging
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                        # Print to console for real-time monitoring
                        print(f"[{timestamp}] Voltage: {voltage_mv:.2f} mV, Temp: {temperature_c:.2f} C")

                        # Write to log file
                        log_file.write(f"{timestamp},{voltage_mv:.2f},{temperature_c:.2f}\n")
                        log_file.flush() # Ensure data is written to disk immediately

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: '{line}' - {e}")
                        # This might happen if an incomplete line or unexpected data is received.

                time.sleep(0.1) # Small delay to prevent busy-waiting

    except serial.SerialException as e:
        print(f"Serial port error: {e}")
        print(f"Please ensure '{port}' is correct and the Arduino is connected.")
        print("On Linux/macOS, ports often look like /dev/ttyUSB0 or /dev/ttyACM0.")
    except FileNotFoundError:
        print(f"Error: Could not create or open file '{output_filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <output_file_name>")
        print("Example: python arduino_logger.py temperature_log.csv")
        sys.exit(1)

    # The output filename is the first command-line argument
    output_file = sys.argv[1]

    # Set your Arduino's serial port here.
    # On Windows, it's typically 'COMX' (e.g., 'COM5').
    # On Linux/macOS, it's typically '/dev/ttyUSB0' or '/dev/ttyACM0'.
    arduino_port = 'COM6' # Change this to your Arduino's actual port!
    arduino_baud_rate = 9600 # Must match Serial.begin() in Arduino code

    read_and_log_arduino(arduino_port, arduino_baud_rate, output_file)
