"""Cleaning and row-level feature creation for workout intensity prediction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


class WorkoutFeatureTransformer(TransformerMixin, BaseEstimator):
    """Build train-only user/sport baselines for the tuned tree models."""

    def __init__(
        self, relative: bool = True, categorical_kind: str = "category",
        rank_features: bool = False,
    ):
        self.relative = relative
        self.categorical_kind = categorical_kind
        self.rank_features = rank_features

    def fit(self, X: pd.DataFrame, y=None):
        data = prepare_features(X, engineered=self.relative)
        self.reference_columns_ = [
            "speed_kmh", "max_speed_kmh", "duration_min", "distance_km",
            "altitude_gain_m", "speed_std_kmh",
        ]
        if self.relative:
            self.user_sport_medians_ = data.groupby(["user_id", "sport"])[
                self.reference_columns_
            ].median()
            self.sport_medians_ = data.groupby("sport")[self.reference_columns_].median()
            self.global_medians_ = data[self.reference_columns_].median()
        if self.rank_features:
            self.group_rank_values_ = {
                key: {column: np.sort(group[column].dropna().to_numpy())
                      for column in self.reference_columns_}
                for key, group in data.groupby(["user_id", "sport"])
            }
            self.sport_rank_values_ = {
                key: {column: np.sort(group[column].dropna().to_numpy())
                      for column in self.reference_columns_}
                for key, group in data.groupby("sport")
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = prepare_features(X, engineered=self.relative)
        if self.relative:
            user_sport = data.join(
                self.user_sport_medians_.add_prefix("baseline_"),
                on=["user_id", "sport"],
            )
            sport = data.join(
                self.sport_medians_.add_prefix("sport_baseline_"), on="sport"
            )
            for column in self.reference_columns_:
                baseline = user_sport[f"baseline_{column}"].fillna(
                    sport[f"sport_baseline_{column}"]
                ).fillna(self.global_medians_[column])
                data[f"relative_{column}"] = (
                    data[column] / baseline.replace(0, np.nan)
                )
        if self.rank_features:
            ranks = {
                column: pd.Series(index=data.index, dtype=float)
                for column in self.reference_columns_
            }
            for key, indices in data.groupby(["user_id", "sport"]).groups.items():
                reference = self.group_rank_values_.get(key)
                if reference is None:
                    reference = self.sport_rank_values_.get(key[1])
                if reference is None:
                    continue
                for column in self.reference_columns_:
                    values = reference[column]
                    if len(values):
                        ranks[column].loc[indices] = np.searchsorted(
                            values, data.loc[indices, column].to_numpy(), side="right"
                        ) / len(values)
            for column in self.reference_columns_:
                data[f"percentile_{column}"] = ranks[column]
        data["user_id"] = data["user_id"].astype(str)
        for column in ["user_id", "sport", "gender"]:
            data[column] = data[column].fillna("missing")
            if self.categorical_kind == "category":
                data[column] = data[column].astype("category")
            elif self.categorical_kind != "object":
                raise ValueError("categorical_kind must be 'category' or 'object'")
        return data.replace([np.inf, -np.inf], np.nan)


class IntensityPredictor:
    """Decode a fitted integer-class classifier back to intensity labels."""

    def __init__(self, model, labels, log_offsets=None):
        self.model = model
        self.labels = np.asarray(labels)
        self.log_offsets = None if log_offsets is None else np.asarray(log_offsets)

    def predict(self, X):
        if self.log_offsets is None:
            indexes = np.asarray(self.model.predict(X), dtype=int).ravel()
        else:
            indexes = np.argmax(self.predict_proba(X), axis=1)
        return self.labels[indexes]

    def predict_proba(self, X):
        probabilities = self.model.predict_proba(X)
        if self.log_offsets is None:
            return probabilities
        adjusted = np.log(np.clip(probabilities, 1e-12, 1)) + self.log_offsets
        adjusted = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
        return adjusted / adjusted.sum(axis=1, keepdims=True)
