import pandas as pd
import numpy as np
from scipy.signal import step, TransferFunction, savgol_filter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse


def calculate_nde(y_experimental, y_model):
    """Calculates the Normalized Deviation Error (NDE)."""
    numerator = np.sum((y_experimental - y_model) ** 2)
    denominator = np.sum((y_experimental - np.mean(y_experimental)) ** 2)
    if denominator == 0:
        return np.inf
    return np.sqrt(numerator / denominator)


def simulate_fopdt(params, time_exp, u_initial, u_final, y0):
    """Simulates a FOPDT model for a given set of conditions."""
    K, L, tau = params

    if tau <= 0 or L < 0:
        return np.full_like(time_exp, np.nan)

    tf_model = TransferFunction([K], [tau, 1])
    t_sim, y_sim_unit = step(
        tf_model, T=np.linspace(0, time_exp[-1], len(time_exp) * 2)
    )

    delta_u = u_final - u_initial
    model_time = t_sim + L
    model_response = y0 + y_sim_unit * delta_u

    y_model_interp = np.interp(
        time_exp,
        model_time,
        model_response,
        left=model_response[0],
        right=model_response[-1],
    )

    return y_model_interp


def get_sk_params(time, temp, u_initial, u_final):
    """Calculates FOPDT parameters using Sundaresan-Krishnaswamy method."""
    y0 = temp[0]
    y_inf = np.mean(temp[-20:])
    delta_y = y_inf - y0
    delta_u = u_final - u_initial

    if delta_u == 0:
        return 0, 0, 1

    K = delta_y / delta_u
    y1 = y0 + 0.353 * delta_y
    y2 = y0 + 0.853 * delta_y

    if delta_y > 0:
        t1 = np.interp(y1, temp, time)
        t2 = np.interp(y2, temp, time)
    else:
        t1 = np.interp(y1, temp[::-1], time[::-1])
        t2 = np.interp(y2, temp[::-1], time[::-1])

    tau_sk = 0.67 * (t2 - t1)
    L_sk = 1.3 * t1 - 0.29 * t2
    return K, max(0, L_sk), max(1, tau_sk)  # Ensure tau > 0


def objective_function(params, datasets):
    """Calculates the total NDE across all datasets."""
    total_nde = 0
    for key, data_info in datasets.items():
        y_model = simulate_fopdt(
            params,
            data_info["time"],
            data_info["u_initial"],
            data_info["u_final"],
            data_info["temp"][0],
        )

        if np.isnan(y_model).any():
            return 1e6  # Penalize invalid params

        nde = calculate_nde(data_info["temp"], y_model)
        total_nde += nde**2  # Minimize sum of squares for robustness

    return total_nde


def main():
    # --- Configuration with updated file paths ---
    try:
        datasets_config = {
            "0_to_50": {
                "file": "testes/0-50/averaged_step_response.csv",
                "u_initial": 0,
                "u_final": 50,
            },
            "50_to_100": {
                "file": "testes/50-100/averaged_step_response.csv",
                "u_initial": 50,
                "u_final": 100,
            },
            "100_to_50": {
                "file": "testes/100-50/averaged_step_response.csv",
                "u_initial": 100,
                "u_final": 50,
            },
        }

        # Load and smooth data
        for key, config in datasets_config.items():
            data = pd.read_csv(config["file"])
            config["time"] = data["Time (s)"].values
            raw_temp = data["Temperature (C)"].values
            config["temp"] = savgol_filter(raw_temp, 21, 2)
    except FileNotFoundError as e:
        print(f"Error: Could not find a data file. Please check paths.")
        print(f"(Missing file: {e.filename})")
        return

    # --- Re-run SK method on each dataset to find a better initial guess ---
    print("--- Finding Initial Guess from Individual SK Models ---")
    params_list = []
    print(f"{'Dataset':<12} | {'K':<10} | {'L':<10} | {'Tau':<10}")
    print("-" * 47)
    for key, cfg in datasets_config.items():
        K, L, tau = get_sk_params(
            cfg["time"], cfg["temp"], cfg["u_initial"], cfg["u_final"]
        )
        params_list.append([K, L, tau])
        print(f"{key:<12} | {K:<10.3f} | {L:<10.3f} | {tau:<10.3f}")

    # Use the average of the individual models as the initial guess
    initial_guess = np.mean(params_list, axis=0)
    print(
        f"\nUsing Average SK as Initial Guess: K={initial_guess[0]:.3f}, L={initial_guess[1]:.3f}, Tau={initial_guess[2]:.3f}\n"
    )

    # --- Optimization --- (K, L, tau)
    bounds = ((-2.0, -0.1), (0.1, 60.0), (10.0, 200.0))

    print("--- Starting Global Optimization ---")
    result = minimize(
        objective_function,
        initial_guess,
        args=(datasets_config,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    # --- Results ---
    if result.success:
        optimal_K, optimal_L, optimal_tau = result.x
        print("\n--- Optimization Successful ---")
        print(f"Globally Optimal K    = {optimal_K:.4f}")
        print(f"Globally Optimal L    = {optimal_L:.4f} s")
        print(f"Globally Optimal τ    = {optimal_tau:.4f} s")
        print("\nOptimal Transfer Function:")
        print(
            f" G(s) = ({optimal_K:.3f} * exp(-{optimal_L:.2f}s)) / ({optimal_tau:.2f}s + 1) \n"
        )
    else:
        print("\n--- Optimization Failed ---")
        print(result.message)
        return

    # --- Plotting Validation ---
    print("Plotting results for each dataset...")
    num_plots = len(datasets_config)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True)
    fig.suptitle("Globally Optimized Model vs. All Experimental Data", fontsize=16)

    for i, (key, data_info) in enumerate(datasets_config.items()):
        ax = axes[i]

        y_optimal = simulate_fopdt(
            [optimal_K, optimal_L, optimal_tau],
            data_info["time"],
            data_info["u_initial"],
            data_info["u_final"],
            data_info["temp"][0],
        )

        nde_final = calculate_nde(data_info["temp"], y_optimal)

        title = f"Test: {data_info['u_initial']}% → {data_info['u_final']}% (NDE = {nde_final:.4f})"
        ax.plot(
            data_info["time"],
            data_info["temp"],
            "k-",
            lw=1.5,
            label="Experimental Data (Smoothed)",
        )
        ax.plot(
            data_info["time"],
            y_optimal,
            "r--",
            lw=2.5,
            label="Optimal Model Prediction",
        )
        ax.set_title(title)
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
