"""Load and process weight data from Garmin export."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Conversion factor
GRAMS_TO_LBS = 0.00220462


def load_weight_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load weight measurements from Garmin biometrics export.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with columns: date, weight_lbs, days_since_start
    """
    data_dir = Path(data_dir)
    biometrics_path = data_dir / "DI_CONNECT/DI-Connect-Wellness/114762117_userBioMetrics.json"

    with open(biometrics_path) as f:
        data = json.load(f)

    # Extract entries with weight data
    records = []
    for entry in data:
        if "weight" not in entry or not entry["weight"]:
            continue

        weight_info = entry["weight"]
        date_str = entry["metaData"]["calendarDate"][:10]
        weight_lbs = weight_info["weight"] * GRAMS_TO_LBS  # Convert from grams to lbs

        records.append({
            "date": pd.to_datetime(date_str),
            "weight_lbs": weight_lbs,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)

    # Add days since first measurement for modeling
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    return df


def prepare_stan_data(df: pd.DataFrame) -> dict:
    """Prepare data dictionary for Stan model.

    Args:
        df: DataFrame from load_weight_data()

    Returns:
        Dictionary with Stan data fields.
    """
    # Standardize time to help with model convergence
    t = df["days_since_start"].values
    t_scaled = t / t.max()  # Scale to [0, 1]

    # Center weight for better sampling
    y = df["weight_lbs"].values
    y_mean = y.mean()
    y_sd = y.std()
    y_centered = (y - y_mean) / y_sd

    return {
        "N": len(df),
        "t": t_scaled,
        "y": y_centered,
        # Store scaling parameters for back-transformation
        "_y_mean": y_mean,
        "_y_sd": y_sd,
        "_t_max": t.max(),
    }
