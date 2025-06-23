import pandas as pd
import numpy as np
from scipy.signal import step, TransferFunction, savgol_filter
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
    return K, max(0, L_sk), max(1, tau_sk)


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
            return 1e6
        nde = calculate_nde(data_info["temp"], y_model)
        total_nde += nde**2
    return total_nde


def main():
    # --- Datasets: specify percent-based setpoints here ---
    datasets_config = {
        "0_to_50": {
            "file": "testes/0-50/averaged_step_response.csv",
            "u_initial_pct": 0,
            "u_final_pct": 50,
        },
        "50_to_100": {
            "file": "testes/50-100/averaged_step_response.csv",
            "u_initial_pct": 50,
            "u_final_pct": 100,
        },
        "100_to_50": {
            "file": "testes/100-50/averaged_step_response.csv",
            "u_initial_pct": 100,
            "u_final_pct": 50,
        },
    }

    # --- Load + smooth + convert to mV, and convert % → Volts (0–12 V) ---
    try:
        for key, cfg in datasets_config.items():
            df = pd.read_csv(cfg["file"])
            cfg["time"] = df["Time (s)"].values

            # smooth temperature in °C
            raw_temp_C = df["Temperature (C)"].values
            smooth_temp_C = savgol_filter(raw_temp_C, 21, 2)

            # convert °C → sensor mV (linear 10 mV/°C)
            cfg["temp"] = smooth_temp_C * 10.0

            # convert PWM % → drive voltage (0 %→0 V, 100 %→12 V)
            cfg["u_initial"] = cfg["u_initial_pct"] / 100.0 * 12.0
            cfg["u_final"] = cfg["u_final_pct"] / 100.0 * 12.0

    except FileNotFoundError as e:
        print("Error: could not find data file:", e.filename)
        return

    # --- SK initial guesses ---
    print("--- SK Initial Guess (in mV/V and seconds) ---")
    header = f"{'Dataset':<12} | {'K (mV/V)':<10} | {'L (s)':<8} | {'τ (s)':<8}"
    print(header)
    print("-" * len(header))
    sk_params = []
    for key, cfg in datasets_config.items():
        K, L, tau = get_sk_params(
            cfg["time"], cfg["temp"], cfg["u_initial"], cfg["u_final"]
        )
        sk_params.append([K, L, tau])
        print(f"{key:<12} | {K:<10.3f} | {L:<8.3f} | {tau:<8.3f}")

    initial_guess = np.mean(sk_params, axis=0)
    print(
        f"\nUsing average SK as initial guess: K={initial_guess[0]:.3f}, "
        f"L={initial_guess[1]:.3f}, τ={initial_guess[2]:.3f}\n"
    )

    # --- Optimize globally ---
    bounds = ((-500.0, 500.0), (0.0, 600.0), (1.0, 2000.0))
    result = minimize(
        objective_function,
        initial_guess,
        args=(datasets_config,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not result.success:
        print("Optimization failed:", result.message)
        return

    K_opt, L_opt, tau_opt = result.x
    print("--- Optimization Results (mV/V) ---")
    print(f"K  = {K_opt:.4f} mV per V")
    print(f"L  = {L_opt:.4f} s")
    print(f"τ  = {tau_opt:.4f} s\n")
    print(
        f"Transfer function: G(s) = {K_opt:.3f} * exp(-{L_opt:.2f}s) / ({tau_opt:.2f}s + 1)\n"
    )

    # --- Plot validation ---
    print("Plotting experimental vs. model (sensor mV output)...")
    n = len(datasets_config)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), sharex=True)
    fig.suptitle("FOPDT Fit: Drive Voltage → Sensor mV", fontsize=16)

    for ax, (key, cfg) in zip(axes, datasets_config.items()):
        y_fit = simulate_fopdt(
            [K_opt, L_opt, tau_opt],
            cfg["time"],
            cfg["u_initial"],
            cfg["u_final"],
            cfg["temp"][0],
        )
        nde = calculate_nde(cfg["temp"], y_fit)

        ax.plot(cfg["time"], cfg["temp"], "k-", lw=1.5, label="Measured (mV)")
        ax.plot(cfg["time"], y_fit, "r--", lw=2, label=f"Model (NDE={nde:.3f})")
        ax.set_title(f"{cfg['u_initial']:.1f} V → {cfg['u_final']:.1f} V")
        ax.set_ylabel("Sensor Output (mV)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
