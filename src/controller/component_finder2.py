import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Standard Component Value Generation ---

def generate_e_series(base_values, decades):
    """Generates a list of standard component values for a given E-series and decades."""
    series_values = []
    for decade in decades:
        for base in base_values:
            series_values.append(base * decade)
    return sorted(list(set(series_values)))

# E-series base values
E6_BASES = [1.0, 1.5, 2.2, 3.3, 4.7, 6.8] # 20% tolerance
E12_BASES = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2] # 10% tolerance
E24_BASES = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
             3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1] # 5% tolerance

# Define practical component ranges
cap_decades = [1e-9, 1e-8, 1e-7, 1e-6] # nF to uF range
res_decades = [1e3, 1e4, 1e5, 1e6, 1e7] # 1 kOhm to 10 MOhm range

# Generate full component value lists
_capacitors_e6_all = generate_e_series(E6_BASES, cap_decades)
_capacitors_e12_all = generate_e_series(E12_BASES, cap_decades)
_resistors_e12_all = generate_e_series(E12_BASES, res_decades)
_resistors_e24_all = generate_e_series(E24_BASES, res_decades)

# *** EDITED: Apply user's max component limits ***
max_capacitance = 4.7e-6
max_resistance = 10e6

capacitors_e6 = [c for c in _capacitors_e6_all if c <= max_capacitance]
capacitors_e12 = [c for c in _capacitors_e12_all if c <= max_capacitance]
resistors_e12 = [r for r in _resistors_e12_all if r <= max_resistance]
resistors_e24 = [r for r in _resistors_e24_all if r <= max_resistance]


def find_closest(value, series):
    """Finds the closest value in a standard series."""
    return series[np.argmin(np.abs(np.array(series) - value))]

# --- 2. Optimization and Calculation ---

# Target PID gains (using absolute values as the op-amp circuit is inverting)
kp_target = 10.0
ki_target = 0.2
kd_target = 50.0

# Define the scenarios to test
scenarios = [
    {'name': 'R: E24, C: E12', 'resistors': resistors_e24, 'capacitors': capacitors_e12},
    {'name': 'R: E12, C: E12', 'resistors': resistors_e12, 'capacitors': capacitors_e12},
    {'name': 'R: E12, C: E6',  'resistors': resistors_e12, 'capacitors': capacitors_e6},
]

overall_best_components = {}
overall_min_error = float('inf')

print(f"--- Searching for Optimal Components (C_max={max_capacitance*1e6}uF, R_max={max_resistance/1e6}MΩ) ---")
start_time = time.time()

for scenario in scenarios:
    min_scenario_error = float('inf')
    best_scenario_components = {}
    
    resistor_series = scenario['resistors']
    capacitor_series = scenario['capacitors']

    # Brute-force search through all standard R1 values for the current scenario
    for r1_actual in resistor_series:
        # From the PID formulas, derive the ideal component values based on a chosen R1
        c2_ideal = 1 / (ki_target * r1_actual)
        c1_ideal = kd_target / r1_actual
        r2_ideal = r1_actual * (kp_target - kd_target * ki_target)

        # Find the closest commercial components for the current scenario
        c2_actual = find_closest(c2_ideal, capacitor_series)
        c1_actual = find_closest(c1_ideal, capacitor_series)
        r2_actual = find_closest(r2_ideal, resistor_series)

        # Calculate the actual PID gains
        if c2_actual * r1_actual == 0: continue
        ki_actual = 1 / (c2_actual * r1_actual)
        kp_actual = (c1_actual / c2_actual) + (r2_actual / r1_actual)
        kd_actual = c1_actual * r1_actual

        # Calculate the total squared percentage error
        error_kp = ((kp_actual - kp_target) / kp_target)**2
        error_ki = ((ki_actual - ki_target) / ki_target)**2
        error_kd = ((kd_actual - kd_target) / kd_target)**2
        total_error = error_kp + error_ki + error_kd

        if total_error < min_scenario_error:
            min_scenario_error = total_error
            best_scenario_components = {
                'R1': r1_actual, 'R2': r2_actual, 'C1': c1_actual, 'C2': c2_actual,
                'kp_actual': kp_actual, 'ki_actual': ki_actual, 'kd_actual': kd_actual
            }

    # Print the best result for the current scenario
    print(f"\n--- Best Result for Scenario: {scenario['name']} ---")
    if not best_scenario_components:
        print("No suitable components found in this scenario.")
        continue
    print(f"R1: {best_scenario_components['R1']/1e3:.2f} kΩ, R2: {best_scenario_components['R2']/1e3:.2f} kΩ, C1: {best_scenario_components['C1']*1e6:.2f} µF, C2: {best_scenario_components['C2']*1e6:.2f} µF")
    print(f"Achieved Gains: Kp={best_scenario_components['kp_actual']:.3f}, Ki={best_scenario_components['ki_actual']:.3f}, Kd={best_scenario_components['kd_actual']:.3f}")
    
    # Check if this scenario produced the overall best result
    if min_scenario_error < overall_min_error:
        overall_min_error = min_scenario_error
        overall_best_components = best_scenario_components
        overall_best_components['scenario_name'] = scenario['name']


end_time = time.time()
print(f"\n--- Search Complete in {end_time - start_time:.2f} seconds ---\n")

# --- 3. Display Overall Best Results ---
print(f"--- Overall Optimal Component Values (C_max={max_capacitance*1e6}uF, R_max={max_resistance/1e6}MΩ) ---")
print(f"Found in Scenario: {overall_best_components['scenario_name']}")
print(f"R1: {overall_best_components['R1']/1e3:.2f} kΩ")
print(f"R2: {overall_best_components['R2']/1e3:.2f} kΩ")
print(f"C1: {overall_best_components['C1']*1e6:.2f} µF")
print(f"C2: {overall_best_components['C2']*1e6:.2f} µF")
print("-" * 35)

print("--- Overall Best PID Gain Comparison ---")
print(f"       |  Target  | Achievable")
print(f"Kp     | {kp_target:8.3f} | {overall_best_components['kp_actual']:8.3f}")
print(f"Ki     | {ki_target:8.3f} | {overall_best_components['ki_actual']:8.3f}")
print(f"Kd     | {kd_target:8.3f} | {overall_best_components['kd_actual']:8.3f}")
print("-" * 35)


# --- 4. Simulation and Plotting ---

# Incubator System Parameters
K_plant = -57.6964
tau_plant = 104.6271
theta_plant = 6.4081
sim_time = 1000
dt = 0.1
n = int(sim_time / dt)
t = np.linspace(0, sim_time, n)
initial_temp = 153.0
setpoint_1 = 120.0
sp = np.full(n, setpoint_1)

def run_simulation(Kp, Ki, Kd):
    """Runs a full PID simulation. Note: Kp, Ki, Kd should be negative for this reverse-acting system."""
    pv = np.full(n, initial_temp)
    co = np.zeros(n)
    e = np.zeros(n)
    integral = 0
    prev_error = 0
    fan_input_history = np.full(int(np.ceil(theta_plant / dt)), 0.0)

    for i in range(1, n):
        e[i] = sp[i] - pv[i-1]
        p_term = Kp * e[i]
        is_saturated = (co[i-1] >= 100 and p_term + Ki*(integral + e[i]*dt) > 100) or \
                       (co[i-1] <= 0 and p_term + Ki*(integral + e[i]*dt) < 0)
        if not is_saturated: integral += e[i] * dt
        i_term = Ki * integral
        derivative = (e[i] - prev_error) / dt
        d_term = Kd * derivative
        raw_co = p_term + i_term + d_term
        prev_error = e[i]
        co[i] = np.clip(raw_co, 0, 100)
        
        delayed_fan_input = fan_input_history[0]
        u_fraction = delayed_fan_input / 100.0
        dpv_dt = (1 / tau_plant) * (K_plant * u_fraction - (pv[i-1] - initial_temp))
        pv[i] = pv[i-1] + dpv_dt * dt
        fan_input_history = np.roll(fan_input_history, -1)
        fan_input_history[-1] = co[i]
    return pv, co

# Run simulations for both scenarios (remembering to use negative gains for the sim)
pv_target, co_target = run_simulation(-kp_target, -ki_target, -kd_target)
pv_actual, co_actual = run_simulation(-overall_best_components['kp_actual'], -overall_best_components['ki_actual'], -overall_best_components['kd_actual'])

# Plotting the comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"System Response: Ideal vs. Best Achievable ({overall_best_components['scenario_name']}, Cmax={max_capacitance*1e6}uF, Rmax={max_resistance/1e6}MΩ)", fontsize=16)

# Plot 1: Temperature Response
ax1.plot(t, sp, label='Setpoint (SP)', linestyle='--', color='k', linewidth=2)
ax1.plot(t, pv_target, label='Ideal Target Gains', linewidth=3, alpha=0.7)
ax1.plot(t, pv_actual, label='Best Achievable Gains', linestyle='--', linewidth=2)
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Temperature Response Comparison')
ax1.legend()
ax1.grid(True)

# Plot 2: Controller Output
ax2.plot(t, co_target, label='Ideal Target CO', linewidth=3, alpha=0.7)
ax2.plot(t, co_actual, label='Best Achievable CO', linestyle='--', linewidth=2)
ax2.set_ylabel('Fan Output (%)')
ax2.set_xlabel('Time (seconds)')
ax2.set_title('Controller Output Comparison')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
