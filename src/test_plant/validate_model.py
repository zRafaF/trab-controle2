import pandas as pd
import numpy as np
from scipy.signal import step, TransferFunction
import matplotlib.pyplot as plt
import argparse  # Used to accept command-line arguments


def calculate_nde(y_experimental, y_model):
    """Calculates the Normalized Deviation Error (NDE)."""
    numerator = np.sum((y_experimental - y_model) ** 2)
    denominator = np.sum((y_experimental - np.mean(y_experimental)) ** 2)
    if denominator == 0:
        return np.inf
    return np.sqrt(numerator / denominator)


def validate_model(csv_file, u_initial, u_final, model_K, model_L, model_tau):
    """
    Validates a pre-defined FOPDT model against new experimental step response data.

    Args:
        csv_file (str): Path to the validation CSV data.
        u_initial (float): The initial input value for the validation run.
        u_final (float): The final input value for the validation run.
        model_K (float): Process gain of the identified model.
        model_L (float): Dead time of the identified model.
        model_tau (float): Time constant of the identified model.
    """
    # 1. Load the validation data
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    time_exp = data["Time (s)"].values
    temp_exp = data["Temperature (C)"].values

    # 2. Define and simulate the identified transfer function
    print("--- Validating Model ---")
    print(f"G(s) = ({model_K} * exp(-{model_L}s)) / ({model_tau}s + 1)")
    print(f"Against data from: '{csv_file}'")
    print(f"Input Step: {u_initial}% -> {u_final}% Fan Drive")

    tf_model = TransferFunction([model_K], [model_tau, 1])

    # Simulate step response for the given time duration
    t_sim, y_sim_unit = step(
        tf_model, T=np.linspace(0, time_exp[-1], len(time_exp) * 2)
    )

    # 3. Scale the model's response to the new conditions
    delta_u_validation = u_final - u_initial
    y0_validation = temp_exp[0]  # Use the actual starting temp from the new data

    # Apply dead time and scale the output
    model_time = t_sim + model_L
    model_response = y0_validation + y_sim_unit * delta_u_validation

    # Interpolate model response onto the experimental time grid for comparison
    y_model_interp = np.interp(
        time_exp,
        model_time,
        model_response,
        left=model_response[0],
        right=model_response[-1],
    )

    # 4. Calculate NDE for this validation run
    nde = calculate_nde(temp_exp, y_model_interp)
    print(f"\nValidation NDE: {nde:.4f}")

    # 5. Plot the results for visual comparison
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        time_exp,
        temp_exp,
        "ko",
        markersize=3,
        label=f"Experimental Data ({u_initial}% → {u_final}%)",
    )
    ax1.plot(
        time_exp,
        y_model_interp,
        "r--",
        lw=2.5,
        label=f"Model Prediction (NDE={nde:.4f})",
    )

    ax1.set_title(f"Model Validation: Experimental vs. Predicted Response")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_exp, temp_exp - y_model_interp)
    ax2.set_title("Error (Experimental - Predicted)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (°C)")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Model Parameters (from Sundaresan-Krishnaswamy identification) ---
    # These are the results from your best-fit model.
    # We hard-code them here to test them against new data.
    IDENTIFIED_K = -0.646
    IDENTIFIED_L = 5.126
    IDENTIFIED_TAU = 76.678

    # --- Setup Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Validate an incubator FOPDT model against new experimental data."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to the averaged CSV file for validation."
    )
    parser.add_argument(
        "u_initial",
        type=float,
        help="Initial fan drive percentage for the validation run (e.g., 50).",
    )
    parser.add_argument(
        "u_final",
        type=float,
        help="Final fan drive percentage for the validation run (e.g., 100).",
    )

    args = parser.parse_args()

    # --- Run Validation ---
    validate_model(
        csv_file=args.csv_file,
        u_initial=args.u_initial,
        u_final=args.u_final,
        model_K=IDENTIFIED_K,
        model_L=IDENTIFIED_L,
        model_tau=IDENTIFIED_TAU,
    )
