import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys


def preprocess_step_response(file_paths, interpolation_dt=1.0, apply_smoothing=True):
    """
    Aligns, interpolates, and averages multiple temperature response curves.

    Args:
        file_paths (list): List of CSV paths.
        interpolation_dt (float): Time resolution in seconds.
        apply_smoothing (bool): Whether to apply Savitzky-Golay smoothing.

    Returns:
        (DataFrame, ndarray): DataFrame with averaged temperature curve, and time array.
    """
    interpolated_curves = []
    min_length = float("inf")

    for file_path in file_paths:
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])

        # Convert to relative time (in seconds)
        df["Time (s)"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

        time = df["Time (s)"].to_numpy()
        temp = df["Temperature (C)"].to_numpy()

        # Interpolation base (start at 0, go to max)
        interp_time = np.arange(0, time[-1], interpolation_dt)
        interp_temp = np.interp(interp_time, time, temp)

        if apply_smoothing and len(interp_temp) >= 51:
            interp_temp = savgol_filter(interp_temp, window_length=51, polyorder=2)

        interpolated_curves.append(interp_temp)
        min_length = min(min_length, len(interp_temp))

    # Trim all curves to shortest one
    aligned_curves = np.array([curve[:min_length] for curve in interpolated_curves])
    avg_curve = np.mean(aligned_curves, axis=0)
    time_vector = np.arange(0, min_length * interpolation_dt, interpolation_dt)

    return (
        pd.DataFrame({"Time (s)": time_vector, "Temperature (C)": avg_curve}),
        aligned_curves,
    )


def plot_all_and_average(file_paths, avg_df, aligned_curves):
    """
    Plots raw curves and the average temperature curve.
    """
    plt.figure(figsize=(12, 7))
    plt.style.use("seaborn-v0_8-darkgrid")

    for i, curve in enumerate(aligned_curves):
        plt.plot(avg_df["Time (s)"], curve, alpha=0.4, label=f"Run {i+1}")

    plt.plot(
        avg_df["Time (s)"],
        avg_df["Temperature (C)"],
        color="black",
        linewidth=2.5,
        label="Average",
        zorder=10,
    )

    plt.title("Aligned and Averaged Temperature Step Response", fontsize=16)
    plt.xlabel("Relative Time (s)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_temperature_steps.py <csv1> <csv2> ...")
        sys.exit(1)

    csv_paths = sys.argv[1:]

    avg_df, aligned_curves = preprocess_step_response(csv_paths)
    plot_all_and_average(csv_paths, avg_df, aligned_curves)

    # Save averaged curve
    output_file = "averaged_step_response.csv"
    avg_df.to_csv(output_file, index=False)
    print(f"Averaged curve saved to: {output_file}")
