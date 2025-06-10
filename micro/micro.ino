// Define the analog pin where the LM35 sensor is connected
const int LM35_PIN = A0;

// Define the onboard LED pin. Most Arduino Nanos have it on pin 13.
const int ONBOARD_LED = 13; // LED_BUILTIN can also be used for convenience

// Define the voltage reference for the Arduino Nano (5V)
// The analogRead function on a 5V Arduino typically maps 0-5V to 0-1023.
// We'll use 5000 for millivolts.
const float ANALOG_REFERENCE_MV = 5000.0; // 5V = 5000mV

// Define the resolution of the ADC (Analog-to-Digital Converter)
// Arduino's 10-bit ADC has 2^10 = 1024 possible values (0 to 1023)
const float ADC_RESOLUTION = 1024.0;

// Define the LM35 sensitivity: 10 mV per degree Celsius
const float LM35_SENSITIVITY_MV_PER_C = 10.0;

// Variable to keep track of the LED state
bool ledState = LOW;

void setup() {
  // Initialize serial communication at 9600 baud rate.
  // This speed is common for debugging and sending data to a computer.
  Serial.begin(9600);
  Serial.println("Starting LM35 Temperature Sensor Reading...");
  Serial.println("Voltage (mV) \t Temperature (C)"); // Header for serial monitor and plotter

  // Set the LED pin as an OUTPUT
  pinMode(ONBOARD_LED, OUTPUT);
}

void loop() {
  // Read the analog value from the LM35 sensor pin.
  // This value will be between 0 and 1023.
  int analogReading = analogRead(LM35_PIN);

  // Convert the analog reading to voltage in millivolts (mV).
  // Formula: (analog_reading * reference_voltage_mV) / ADC_resolution
  float voltageMv = (analogReading * ANALOG_REFERENCE_MV) / ADC_RESOLUTION;

  // Convert the voltage in mV to temperature in Celsius.
  // The LM35 outputs 10 mV per degree Celsius, so Temperature (C) = Voltage (mV) / 10.
  float temperatureC = voltageMv / LM35_SENSITIVITY_MV_PER_C;

  // Print the voltage and temperature to the serial monitor.
  // Using a tab character (\t) to separate values allows the Serial Plotter
  // to interpret them as distinct data points for two different plots.
  Serial.print(voltageMv);
  Serial.print("\t"); // Tab character for separation
  Serial.println(temperatureC);

  // Toggle the onboard LED
  ledState = !ledState; // Invert the current state (LOW becomes HIGH, HIGH becomes LOW)
  digitalWrite(ONBOARD_LED, ledState); // Write the new state to the LED pin

  // Wait for 1 second before taking the next reading.
  delay(1000);
}
