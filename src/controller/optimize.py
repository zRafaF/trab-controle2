import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time

# --- 1. System and Simulation Parameters ---

# Plant Transfer Function: Gp(s) = (-57.6964 * e^-6.4081s) / (104.6271s + 1)
K = -57.6964  # Process Gain (°C / fan_fraction)
tau = 104.6271  # Time Constant (seconds)
theta = 6.4081  # Dead Time (seconds)

# Simulation setup
sim_time = 1000  # Total simulation time in seconds
dt = 0.1  # Time step for the simulation (s)
n = int(sim_time / dt)  # Number of simulation steps

# --- 2. Setpoint and Initial Conditions ---
initial_temp = 153.0
initial_fan_percent = 0.0
setpoint_1 = 120.0
setpoint_2 = 100.0
setpoint_change_time = 500

sp = np.full(n, setpoint_1)
change_index = int(setpoint_change_time / dt)
sp[change_index:] = setpoint_2
t = np.linspace(0, sim_time, n)

# --- 3. Simulation and Analysis Functions ---


def run_simulation(Kp, Ki, Kd):
    """Runs a full PID simulation. Returns the process variable and controller output arrays."""
    pv = np.full(n, initial_temp)
    co = np.zeros(n)
    e = np.zeros(n)
    integral = 0
    prev_error = 0
    fan_input_history = np.full(int(np.ceil(theta / dt)), initial_fan_percent)
    co[0] = initial_fan_percent

    for i in range(1, n):
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

        delayed_fan_input = fan_input_history[0]
        u_fraction = delayed_fan_input / 100.0
        dpv_dt = (1 / tau) * (K * u_fraction - (pv[i - 1] - initial_temp))
        pv[i] = pv[i - 1] + dpv_dt * dt

        fan_input_history = np.roll(fan_input_history, -1)
        fan_input_history[-1] = co[i]

    return pv, co


def calculate_settling_time(pv, sp, t, error_band_percent=2.0):
    """
    Calculates the time it takes for the PV to enter and stay within an error band.
    We only check settling for the first setpoint.
    """
    # Define the error band for the first setpoint
    error_margin = setpoint_1 * (error_band_percent / 100.0)
    upper_bound = setpoint_1 + error_margin
    lower_bound = setpoint_1 - error_margin

    # We only care about settling to the first setpoint
    settle_region_end_index = change_index

    # Iterate backwards from the setpoint change time
    for i in range(settle_region_end_index - 1, 0, -1):
        if not (lower_bound <= pv[i] <= upper_bound):
            # First point found outside the band is the settling time
            return t[i]

    # If it's always in the band (unlikely) or never enters, return max time as a penalty
    return sim_time


# --- 4. Optimizer Objective Function ---
# This function is the same, but it will be called by a different optimizer.
def objective_function(params):
    """
    Function to be minimized. Takes PID params, runs sim, returns settling time.
    """
    Kp, Ki, Kd = params

    # Run simulation with the given parameters
    pv_result, _ = run_simulation(Kp, Ki, Kd)

    # Calculate settling time
    settling_time = calculate_settling_time(pv_result, sp, t)

    # We don't need to print here anymore as differential_evolution has a `disp` option.
    return settling_time


# --- 5. Run the Optimization ---

# The "initial_guess" is still useful for the final comparison plot.
initial_guess = [-10.0, -0.2, -50.0]

# Bounds for the parameters have been significantly increased to give the
# optimizer more freedom to find a better solution.
bounds = [(-50.0, -0.1), (-5.0, -0.001), (-300.0, 0.0)]

print("--- Starting PID Optimization with Differential Evolution ---")
print("This may take longer, as it's performing a more thorough global search.")
start_time = time.time()

# We switched to `differential_evolution`, a global optimizer that is much
# better at exploring the entire search space and not getting stuck.
# `disp=True` will print the progress of the optimization.
result = differential_evolution(objective_function, bounds, disp=True)

end_time = time.time()
print(f"--- Optimization Finished in {end_time - start_time:.2f} seconds ---")

# Extract the optimal parameters
optimal_params = result.x
Kp_opt, Ki_opt, Kd_opt = optimal_params

print("\n--- Optimal PID Parameters ---")
print(f"Kp: {Kp_opt:.4f}")
print(f"Ki: {Ki_opt:.4f}")
print(f"Kd: {Kd_opt:.4f}")
print(f"Optimal Settling Time: {result.fun:.2f}s")
print("----------------------------\n")


# --- 6. Plot Final Comparison ---
print("Running final simulations for plotting...")
# Rerun simulation with initial best and optimal params to get all data
pv_initial, co_initial = run_simulation(
    initial_guess[0], initial_guess[1], initial_guess[2]
)
pv_optimal, co_optimal = run_simulation(Kp_opt, Ki_opt, Kd_opt)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Optimized PID Control Comparison", fontsize=16)

# Plot 1: Temperature Response
ax1.plot(t, sp, label="Setpoint (SP)", linestyle="--", color="k", linewidth=2)
ax1.plot(
    t,
    pv_initial,
    label=f"Initial Best PV (Settle: {calculate_settling_time(pv_initial, sp, t):.1f}s)",
    linestyle=":",
    linewidth=2,
)
ax1.plot(
    t, pv_optimal, label=f"Optimized PV (Settle: {result.fun:.1f}s)", linewidth=2.5
)
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("System Response: Initial vs. Optimized")
ax1.legend()
ax1.grid(True)

# Plot 2: Controller Output
ax2.plot(t, co_initial, label="Initial Best CO", linestyle=":", linewidth=2)
ax2.plot(t, co_optimal, label="Optimized CO", linewidth=2.5)
ax2.set_ylabel("Fan Output (%)")
ax2.set_xlabel("Time (seconds)")
ax2.set_title("Controller Output: Initial vs. Optimized")
ax2.set_ylim(-5, 105)
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
