import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters ---

# Plant Transfer Function: Gp(s) = (-57.6964 * e^-6.4081s) / (104.6271s + 1)
# K is in units of (°C / fan_fraction). Where fan_fraction is 0 to 1.
# But our control signal is 0 to 100%, so we will divide it by 100 in the model.
K = -57.6964  # Process Gain
tau = 104.6271  # Time Constant (seconds)
theta = 6.4081  # Dead Time (seconds)

# Simulation setup
sim_time = 600  # Total simulation time in seconds
dt = 0.1  # Time step for the simulation (s)
n = int(sim_time / dt)  # Number of simulation steps

# --- 2. Initial Conditions and Inputs ---

# The system starts at its steady state with 0% fan.
initial_temp = 153.0
# The steady state input corresponding to the initial temp.
initial_fan_percent = 0.0

# We will simulate two scenarios:
# 1. Fan stays at 0%
# 2. Fan switches to 100%
fan_input_0_percent = np.full(n, 0.0)
fan_input_100_percent = np.full(n, 100.0)


# --- 3. Simulation Function ---
# This function simulates the plant for a given fan input profile.
def simulate_plant(fan_input):
    """
    Simulates the FOPDT model of the incubator.

    The differential equation for a FOPDT model in terms of deviation variables
    (y = PV - PV_ss, u = U - U_ss) is:
        tau * dy/dt + y = K * u(t - theta)

    Substituting back the physical variables (PV, U) and rearranging:
        d(PV)/dt = (1/tau) * [K * (U(t-theta) - U_ss) - (PV - PV_ss)]

    Since U_ss = 0, this simplifies to:
        d(PV)/dt = (1/tau) * [K * U(t-theta) - (PV - PV_ss)]
    """
    pv = np.full(n, initial_temp)  # Process Variable (temperature)

    # Dead time buffer for the control output (fan signal)
    delay_steps = int(np.ceil(theta / dt))
    input_history = np.full(delay_steps, initial_fan_percent)

    for i in range(1, n):
        # Get the fan input from the past (accounting for dead time)
        delayed_fan_input = input_history[0]

        # Convert fan percent (0-100) to fraction (0-1) for the model's gain `K`
        u_fraction = delayed_fan_input / 100.0

        # Corrected plant model equation
        dpv_dt = (1 / tau) * (K * u_fraction - (pv[i - 1] - initial_temp))
        pv[i] = pv[i - 1] + dpv_dt * dt

        # Update the dead time buffer: shift and add the newest input
        input_history = np.roll(input_history, -1)
        input_history[-1] = fan_input[i]

    return pv


# --- 4. Run Simulations ---
pv_response_0_fan = simulate_plant(fan_input_0_percent)
pv_response_100_fan = simulate_plant(fan_input_100_percent)

# --- 5. Plotting the Results ---
t = np.linspace(0, sim_time, n)
plt.figure(figsize=(12, 7))

# Plot the two scenarios
plt.plot(
    t, pv_response_0_fan, label="Fan at 0% (Open Loop)", color="orange", linestyle="--"
)
plt.plot(t, pv_response_100_fan, label="Fan at 100% (Open Loop)", color="blue")

# Calculate and show the predicted final temperature for 100% fan
final_temp_predicted = initial_temp + K
plt.axhline(
    y=final_temp_predicted,
    color="red",
    linestyle=":",
    label=f"Predicted Final Temp at 100% Fan ({final_temp_predicted:.2f}°C)",
)

plt.title("Open-Loop System Response", fontsize=16)
plt.xlabel("Time (seconds)")
plt.ylabel("Incubator Temperature (°C)")
plt.legend()
plt.grid(True)
plt.ylim(90, 160)
plt.show()
