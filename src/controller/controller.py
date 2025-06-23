import numpy as np
import matplotlib.pyplot as plt

# --- 1. System and Simulation Parameters ---

# Plant Transfer Function: Gp(s) = (-57.6964 * e^-6.4081s) / (104.6271s + 1)
K = -57.6964  # Process Gain (°C / fan_fraction)
tau = 104.6271  # Time Constant (seconds)
theta = 6.4081  # Dead Time (seconds)

# Simulation setup
sim_time = 1000  # Total simulation time in seconds
dt = 0.1  # Time step for the simulation (s)
n = int(sim_time / dt)  # Number of simulation steps


# --- 2. PID Controller Tuning Sets ---
# We will define several sets of PID gains to compare their performance.
# Each tuple is in the format (Kp, Ki, Kd, 'Name')
tuning_sets = [
    (-40.0, -1.0, -20.0, "Very Aggressive"),
    (-10.0, -0.2, -50.0, "Aggressive"),
    (-4.0, -0.05, -40.0, "Balanced"),
    # (-2.0, -0.02, -20.0, "Conservative"),
]

# --- 3. Setpoint and Initial Conditions ---
# These are constant for all simulation runs.
initial_temp = 153.0
initial_fan_percent = 0.0

setpoint_1 = 120.0
setpoint_2 = 100.0
setpoint_change_time = 500  # Time (s) to switch to the second setpoint

sp = np.full(n, setpoint_1)
change_index = int(setpoint_change_time / dt)
sp[change_index:] = setpoint_2

t = np.linspace(0, sim_time, n)  # Time vector


# --- 4. Simulation Function ---
# We encapsulate the simulation logic in a function to easily run it multiple times.
def run_simulation(Kp, Ki, Kd):
    """
    Runs a full PID simulation for a given set of tuning parameters.
    Returns the process variable (pv) and controller output (co) arrays.
    """
    # Initialization for this specific run
    pv = np.full(n, initial_temp)
    co = np.zeros(n)
    e = np.zeros(n)
    integral = 0
    prev_error = 0
    fan_input_history = np.full(int(np.ceil(theta / dt)), initial_fan_percent)
    co[0] = initial_fan_percent

    # Main Simulation Loop
    for i in range(1, n):
        # Controller Logic
        e[i] = sp[i] - pv[i - 1]
        p_term = Kp * e[i]

        is_saturated = (
            co[i - 1] >= 100 and p_term + Ki * (integral + e[i] * dt) > 100
        ) or (co[i - 1] <= 0 and p_term + Ki * (integral + e[i] * dt) < 0)

        if not is_saturated:
            integral += e[i] * dt
        i_term = Ki * integral

        derivative = (e[i] - prev_error) / dt
        d_term = Kd * derivative

        raw_co = p_term + i_term + d_term
        prev_error = e[i]
        co[i] = np.clip(raw_co, 0, 100)

        # Plant Simulation
        delayed_fan_input = fan_input_history[0]
        u_fraction = delayed_fan_input / 100.0
        dpv_dt = (1 / tau) * (K * u_fraction - (pv[i - 1] - initial_temp))
        pv[i] = pv[i - 1] + dpv_dt * dt

        # Update dead time buffer
        fan_input_history = np.roll(fan_input_history, -1)
        fan_input_history[-1] = co[i]

    return pv, co


# --- 5. Run Simulations and Store Results ---
results = []
print("--- Running Simulations for Different Tunings ---")
for Kp, Ki, Kd, name in tuning_sets:
    print(f"Tuning: {name} (Kp={Kp}, Ki={Ki}, Kd={Kd})")
    pv_result, co_result = run_simulation(Kp, Ki, Kd)
    results.append({"name": name, "pv": pv_result, "co": co_result})
print("---------------------------------------------")


# --- 6. Plotting the Comparison ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.suptitle("PID Tuning Comparison", fontsize=16)

# Plot 1: Temperature vs. Setpoint
ax1.plot(t, sp, label="Setpoint (SP)", linestyle="--", color="k", linewidth=2)
for res in results:
    ax1.plot(t, res["pv"], label=f"PV ({res['name']})", linewidth=2)
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("System Response Comparison")
ax1.legend()
ax1.grid(True)

# Plot 2: Control Output (Fan Signal)
for res in results:
    ax2.plot(t, res["co"], label=f"CO ({res['name']})", linewidth=2)
ax2.set_ylabel("Fan Output (%)")
ax2.set_xlabel("Time (seconds)")
ax2.set_title("Controller Output Comparison")
ax2.set_ylim(-5, 105)
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
