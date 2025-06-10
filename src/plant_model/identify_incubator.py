import pandas as pd
import numpy as np
from scipy.signal import (
    step,
    TransferFunction,
    savgol_filter,
)  # <-- Added savgol_filter
import matplotlib.pyplot as plt


def calculate_nde(y_experimental, y_model):
    """Calculates the Normalized Deviation Error (NDE)."""
    numerator = np.sum((y_experimental - y_model) ** 2)
    denominator = np.sum((y_experimental - np.mean(y_experimental)) ** 2)
    if denominator == 0:
        return np.inf
    return np.sqrt(numerator / denominator)


def identify_and_validate(csv_file, u_initial, u_final):
    """
    Identifies the FOPDT model of a system from step response data using
    Ziegler-Nichols and Sundaresan-Krishnaswamy methods.

    Args:
        csv_file (str): Path to the CSV file with 'Time (s)' and 'Temperature (C)' columns.
        u_initial (float): The initial input value (e.g., 0 for 0% fan).
        u_final (float): The final input value (e.g., 50 for 50% fan).
    """
    # 1. Load and prepare data
    data = pd.read_csv(csv_file)
    time = data["Time (s)"].values
    temp_raw = data["Temperature (C)"].values  # Keep original data for final comparison

    # --- IMPROVEMENT 1: DATA SMOOTHING ---
    # Apply a Savitzky-Golay filter to get a smooth curve for parameter extraction.
    # Adjust window_length and poly_order if needed. Window must be odd.
    window_length = 21  # Should be odd; adjust based on noise level
    poly_order = 2
    temp = savgol_filter(temp_raw, window_length, poly_order)

    # --- Parameter Extraction (using SMOOTHED data) ---
    y0 = temp[0]
    y_inf = np.mean(temp[-20:])
    delta_y = y_inf - y0
    delta_u = u_final - u_initial

    if delta_u == 0:
        raise ValueError("Input step change (delta_u) cannot be zero.")

    K = delta_y / delta_u

    # 2. Method 1: Ziegler-Nichols (now more robust)
    grad = np.gradient(temp, time)
    idx_infl = np.argmax(np.abs(grad))
    max_slope = grad[idx_infl]
    t_infl = time[idx_infl]
    y_infl = temp[idx_infl]

    # Tangent line: y = max_slope * (t - t_infl) + y_infl
    t_intersect_y0 = t_infl - (y_infl - y0) / max_slope
    t_intersect_y_inf = t_infl + (y_inf - y_infl) / max_slope

    L_zn = max(0, t_intersect_y0)
    tau_zn = t_intersect_y_inf - L_zn
    # Create tangent line for plotting
    tangent_time = np.array([L_zn, t_intersect_y_inf])
    tangent_line = np.array([y0, y_inf])

    # 3. Method 2: Sundaresan-Krishnaswamy (now works for cooling)
    y1 = y0 + 0.353 * delta_y
    y2 = y0 + 0.853 * delta_y

    # --- IMPROVEMENT 2: CONDITIONAL INTERPOLATION ---
    # np.interp requires the 'xp' array (temp) to be monotonically increasing.
    # If delta_y is negative, the curve is decreasing, so we reverse the arrays for interpolation.
    if delta_y > 0:  # Heating curve
        t1 = np.interp(y1, temp, time)
        t2 = np.interp(y2, temp, time)
    else:  # Cooling curve (temp is decreasing)
        t1 = np.interp(y1, temp[::-1], time[::-1])
        t2 = np.interp(y2, temp[::-1], time[::-1])
    # --- END IMPROVEMENT 2 ---

    tau_sk = 0.67 * (t2 - t1)
    L_sk = 1.3 * t1 - 0.29 * t2
    L_sk = max(0, L_sk)

    # 4. Build and Simulate Models
    models = {
        "Ziegler-Nichols": {"K": K, "tau": tau_zn, "L": L_zn},
        "Sundaresan-Krishnaswamy": {"K": K, "tau": tau_sk, "L": L_sk},
    }
    results = {}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig1, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot original (raw) and smoothed data to see the filter's effect
    ax1.plot(
        time,
        temp_raw,
        "o",
        color="gray",
        markersize=3,
        alpha=0.6,
        label="Experimental Data (Raw Avg)",
    )
    ax1.plot(time, temp, "k-", lw=1.5, label="Smoothed Data (for Identification)")
    ax1.plot(tangent_time, tangent_line, "r--", lw=1.5, label="Z-N Tangent Line")

    for name, params in models.items():
        p_K, p_tau, p_L = params["K"], params["tau"], params["L"]

        tf_model = TransferFunction([p_K], [p_tau, 1])
        t_sim, y_sim_unit = step(tf_model, T=np.linspace(0, time[-1], 500))

        model_time = t_sim + p_L
        model_response = y0 + y_sim_unit * delta_u

        # Interpolate onto the original experimental time vector for NDE calculation
        y_model_interp = np.interp(
            time, model_time, model_response, left=y0, right=y_inf
        )

        # NDE is calculated against the original RAW data
        nde = calculate_nde(temp_raw, y_model_interp)
        results[name] = {"params": params, "nde": nde, "response": y_model_interp}

        ax1.plot(
            time, y_model_interp, "--", lw=2.5, label=f"{name} Model (NDE={nde:.4f})"
        )
        ax2.plot(time, temp_raw - y_model_interp, label=f"{name} Error")

    # --- Reporting and Plotting ---
    print("--- System Identification Results (Improved) ---")
    print(f"Input Step: {u_initial}% -> {u_final}% Fan Drive\n")
    print(
        f"{'Method':<25} | {'K (Gain)':<10} | {'L (Dead Time)':<15} | {'τ (Time Const)':<15} | {'NDE':<10}"
    )
    print("-" * 80)
    for name, res in results.items():
        p = res["params"]
        print(
            f"{name:<25} | {p['K']:<10.3f} | {p['L']:<15.3f} | {p['tau']:<15.3f} | {res['nde']:<10.4f}"
        )

    best_model_name = min(results, key=lambda k: results[k]["nde"])
    print(f"\n✅ Best Fit: {best_model_name} method")

    ax1.set_title(f"Incubator Step Response: {u_initial}% → {u_final}% Fan Speed")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Model Error (Experimental - Model)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (°C)")
    ax2.legend()
    ax2.grid(True)
    fig1.tight_layout()

    # (Pole-zero and Root Locus plots remain the same as before)
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for name, res in results.items():
        pole = -1 / res["params"]["tau"]
        ax3.plot(
            np.real(pole),
            np.imag(pole),
            "x",
            markersize=10,
            mew=2,
            label=f"Pole ({name})",
        )
        ax4.plot([pole, pole * 10], [0, 0], "->", lw=2, label=f"Locus ({name})")
        ax4.plot(pole, 0, "x", markersize=10, mew=2, color=ax4.lines[-1].get_color())

    ax3.set_title("Pole-Zero Map (Dominant Pole)"), ax3.set_xlabel(
        "Real Axis"
    ), ax3.set_ylabel("Imaginary Axis"), ax3.grid(True), ax3.axhline(
        0, color="k", lw=0.5
    ), ax3.axvline(
        0, color="k", lw=0.5
    ), ax3.legend()
    ax4.set_title("Root Locus Plot"), ax4.set_xlabel("Real Axis"), ax4.set_ylabel(
        "Imaginary Axis"
    ), ax4.grid(True), ax4.axhline(0, color="k", lw=0.5), ax4.axvline(
        0, color="k", lw=0.5
    ), ax4.legend(), ax4.set_xlim(
        ax4.get_xlim()[0], abs(ax4.get_xlim()[0]) * 0.1
    )
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    # --- Configuration ---
    FAN_INITIAL_PERCENT = 0
    FAN_FINAL_PERCENT = 50
    CSV_FILENAME = "averaged_step_response.csv"

    # Create dummy data if not found
    try:
        pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        print(f"'{CSV_FILENAME}' not found. Creating a sample file for demonstration.")
        time_demo = np.arange(0, 301, 1)
        temp_demo = 160 - 25 * (1 - np.exp(-(np.maximum(0, time_demo - 15)) / 60))
        temp_demo += np.random.normal(
            0, 0.4, len(time_demo)
        )  # Added more noise for a better test
        temp_demo[time_demo < 15] = 160 + np.random.normal(
            0, 0.4, len(temp_demo[time_demo < 15])
        )
        pd.DataFrame({"Time (s)": time_demo, "Temperature (C)": temp_demo}).to_csv(
            CSV_FILENAME, index=False
        )

    identify_and_validate(
        csv_file=CSV_FILENAME, u_initial=FAN_INITIAL_PERCENT, u_final=FAN_FINAL_PERCENT
    )
