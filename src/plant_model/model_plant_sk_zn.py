# identify_plant.py
# Script to load averaged step-response CSV and identify plant transfer function G(s)
# using Sundaresan–Krishnaswamy (SK) and Ziegler–Nichols (ZN) tangent methods.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Helper functions ----------------------------------------------


def compute_process_gain(time, temp, delta_u):
    """
    Compute process gain K = delta T / delta u.
    """
    delta_T = temp[-1] - temp[0]
    return delta_T / delta_u


def sundaresan_krishnaswamy(time, temp):
    """
    Estimate L and tau via tangent at inflection (SK method).
    """
    # approximate first and second derivatives
    dt = np.diff(time)
    dy = np.diff(temp)
    dy_dt = dy / dt
    d2y = np.diff(dy_dt) / dt[:-1]
    idx_inflect = np.argmax(np.abs(d2y)) + 1

    t_i = time[idx_inflect]
    y_i = temp[idx_inflect]
    slope = dy_dt[idx_inflect]

    y0 = temp[0]
    yf = temp[-1]
    L = t_i - (y_i - y0) / slope
    tau = (yf - y0) / slope
    return L, tau


def ziegler_nichols_percent_times(time, temp, levels=(0.283, 0.632)):
    """
    Ziegler–Nichols first-order method. Works with increasing or decreasing responses.
    Returns estimates of L and tau.
    """
    y0 = temp[0]
    yf = temp[-1]
    direction = 1 if yf >= y0 else -1

    t_points = {}
    for frac in levels:
        target = y0 + frac * (yf - y0)
        if direction == 1:
            idx = np.where(temp >= target)[0]
        else:
            idx = np.where(temp <= target)[0]

        if len(idx) < 2:
            raise ValueError("Could not find percent-time targets in data.")
        # linear interpolate around idx[0]
        i = idx[0]
        t1, t2 = time[i - 1], time[i]
        y1, y2 = temp[i - 1], temp[i]
        t_target = t1 + (target - y1) * (t2 - t1) / (y2 - y1)
        t_points[frac] = t_target

    t1 = t_points[levels[0]]
    t2 = t_points[levels[1]]
    L = 1.5 * t1 - 0.5 * t2
    tau = 1.5 * (t2 - t1)
    return L, tau


def simulate_with_delay(K, L, tau, time):
    """
    Simulate first-order-plus-deadtime step response manually:
    y(t) = K * (1 - exp(-(t - L)/tau)) * u(t - L)
    """
    # compute delta response
    resp = K * (1 - np.exp(-np.maximum(time - L, 0) / tau))
    # zero before dead time
    resp[time < L] = 0
    return resp


def compute_nde(y_exp, y_model):
    """
    Normalized Deviation Error (NDE)
    NDE = sqrt(sum((y_exp - y_model)^2) / sum((y_exp - mean(y_exp))^2))
    """
    num = np.sum((y_exp - y_model) ** 2)
    den = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return np.sqrt(num / den)


# --- Main routine ----------------------------------------------


def main(csv_path, delta_u):
    df = pd.read_csv(csv_path)
    time = df["Time (s)"].values
    temp = df["Temperature (C)"].values

    # Use delta temperatures for modeling
    temp_delta = temp - temp[0]

    # Compute process gain
    K = compute_process_gain(time, temp, delta_u)
    print(f"Process gain K = {K:.4f} °C/V")
    if K < 0:
        print(
            "⚠️ Warning: Negative process gain detected. Temperature decreases with fan speed."
        )

    # SK method
    L_sk, tau_sk = sundaresan_krishnaswamy(time, temp)
    print(f"SK method: L = {L_sk:.2f} s, tau = {tau_sk:.2f} s")

    # ZN method
    L_zn, tau_zn = ziegler_nichols_percent_times(time, temp)
    print(f"ZN method: L = {L_zn:.2f} s, tau = {tau_zn:.2f} s")

    # Simulate
    y_sk = simulate_with_delay(K, L_sk, tau_sk, time)
    y_zn = simulate_with_delay(K, L_zn, tau_zn, time)

    # Compute NDE on delta
    nde_sk = compute_nde(temp_delta, y_sk)
    nde_zn = compute_nde(temp_delta, y_zn)
    print(f"NDE SK = {nde_sk:.4f}")
    print(f"NDE ZN = {nde_zn:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(time, temp_delta, "k-", label="Experimental ΔT")
    plt.plot(time, y_sk, "b--", label="SK model ΔT")
    plt.plot(time, y_zn, "r-.", label="ZN model ΔT")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔTemperature (°C)")
    plt.legend()
    plt.title("Step Response ΔT Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identify plant from averaged CSV")
    parser.add_argument("csv_path", help="Path to averaged_step_response.csv")
    parser.add_argument(
        "--delta_u",
        type=float,
        default=6.0,
        help="Step change in input (V), e.g., 6 V for 0-50%",
    )
    args = parser.parse_args()
    main(args.csv_path, args.delta_u)
