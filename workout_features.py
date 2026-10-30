"""Cleaning and row-level feature creation for workout intensity prediction."""

import numpy as np
import pandas as pd


def prepare_features(frame: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
    """Return predictors without labels or identifiers used only for submission.

    Every derived feature uses only values in its own workout. This function can
    therefore transform training, validation, and test rows in the same way.
    """
    data = frame.drop(columns=["workout_id", "intensity"], errors="ignore").copy()
    data["altitude_mean_m"] = data["altitude_mean_m"].mask(
        data["altitude_mean_m"] == -9999
    )
    if not engineered:
        return data

    data["elapsed_speed_kmh"] = 60 * data["distance_km"] / data["duration_min"]
    data["speed_gap_kmh"] = data["max_speed_kmh"] - data["speed_kmh"]
    data["climb_m_per_km"] = data["altitude_gain_m"] / data["distance_km"]
    data["altitude_balance_m"] = data["altitude_gain_m"] - data["altitude_loss_m"]
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)
    data["hour_sin"] = np.sin(2 * np.pi * data["start_hour_utc"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["start_hour_utc"] / 24)
    data["is_weekend"] = (data["weekday"] >= 5).astype(int)
    return data.replace([np.inf, -np.inf], np.nan)
