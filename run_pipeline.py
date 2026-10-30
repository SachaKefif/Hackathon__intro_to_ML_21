"""Run the workout data preparation, validation, and model selection pipeline.

Run from the repository root with: python run_pipeline.py
The original Data/train.csv is preserved once as Data/train_full.csv.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from workout_features import prepare_features


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
SEED = 42
VALIDATION_SHARE = 0.20
CLASSES = ["easy", "moderate", "hard"]
CATEGORICAL = ["user_id", "sport", "gender"]


def make_pipeline(estimator, engineered: bool) -> Pipeline:
    example = prepare_features(pd.read_csv(DATA / "train_full.csv", nrows=1), engineered)
    numeric = [column for column in example if column not in CATEGORICAL]
    encoder = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline([
        ("features", FunctionTransformer(prepare_features, kw_args={"engineered": engineered})),
        ("preprocess", encoder),
        ("model", estimator),
    ])


def evaluate(name: str, estimator, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, object]:
    estimator.fit(train, train["intensity"])
    predicted = estimator.predict(validation)
    actual = validation["intensity"]
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
    }
    return metrics, estimator


def save_encoded(frame: pd.DataFrame, fitted_pipeline: Pipeline, path: Path) -> int:
    prepared = fitted_pipeline.named_steps["features"].transform(frame)
    transformer = fitted_pipeline.named_steps["preprocess"]
    matrix = transformer.transform(prepared)
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    encoded = pd.DataFrame(
        values.astype(np.float32),
        columns=transformer.get_feature_names_out(),
    )
    encoded.insert(0, "workout_id", frame["workout_id"].to_numpy())
    if "intensity" in frame:
        encoded["intensity"] = frame["intensity"].to_numpy()
    encoded.to_parquet(path, index=False)
    return len(transformer.get_feature_names_out())


def main() -> None:
    DATA.mkdir(exist_ok=True)
    original = DATA / "train_full.csv"
    if not original.exists():
        shutil.copy2(DATA / "train.csv", original)
    full_raw = pd.read_csv(original)
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    assert full_raw["workout_id"].is_unique and test["workout_id"].is_unique
    assert set(full_raw["workout_id"]).isdisjoint(test["workout_id"])
    assert full_raw.columns.drop("intensity").tolist() == test.columns.tolist()
    assert full_raw["intensity"].notna().all()
    assert set(full_raw["intensity"]) == set(CLASSES)
    assert sample["workout_id"].equals(test["workout_id"])
    assert (full_raw["duration_min"] > 0).all()
    assert (full_raw["distance_km"] > 0).all()
    assert (test["duration_min"] > 0).all()
    assert (test["distance_km"] > 0).all()

    audit = {
        "full_training_rows": len(full_raw),
        "test_rows": len(test),
        "duplicate_full_training_rows": int(full_raw.duplicated().sum()),
        "missing_values_in_original_train": int(full_raw.isna().sum().sum()),
        "invalid_altitude_mean_train": int((full_raw["altitude_mean_m"] == -9999).sum()),
        "invalid_altitude_mean_test": int((test["altitude_mean_m"] == -9999).sum()),
        "capped_gps_rows_train": int((full_raw["n_points"] == 500).sum()),
        "shared_train_test_users": len(set(full_raw["user_id"]) & set(test["user_id"])),
    }

    # Save cleaned labeled rows. The untouched source remains in train_full.csv.
    full_clean = full_raw.copy()
    full_clean["altitude_mean_m"] = full_clean["altitude_mean_m"].mask(
        full_clean["altitude_mean_m"] == -9999
    )
    train, validation = train_test_split(
        full_clean, test_size=VALIDATION_SHARE, random_state=SEED,
        stratify=full_clean["intensity"],
    )
    train = train.sort_index().reset_index(drop=True)
    validation = validation.sort_index().reset_index(drop=True)
    train.to_csv(DATA / "train.csv", index=False)
    validation.to_csv(DATA / "validation.csv", index=False)

    dummy = DummyClassifier(strategy="most_frequent")
    candidates = {
        "logistic_engineered": make_pipeline(
            LogisticRegression(max_iter=2000), engineered=True
        ),
        "random_forest_raw": make_pipeline(
            RandomForestClassifier(n_estimators=120, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1), engineered=False
        ),
        "random_forest_engineered": make_pipeline(
            RandomForestClassifier(n_estimators=120, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1), engineered=True
        ),
        "extra_trees_engineered": make_pipeline(
            ExtraTreesClassifier(n_estimators=160, min_samples_leaf=3,
                                 random_state=SEED, n_jobs=-1), engineered=True
        ),
    }
    rows = []
    fitted = {}
    baseline, _ = evaluate("dummy_baseline", dummy, train, validation)
    rows.append(baseline)
    for name, pipeline in candidates.items():
        score, fitted_model = evaluate(name, pipeline, train, validation)
        rows.append(score)
        fitted[name] = fitted_model
        print(f"{name}: balanced accuracy={score['balanced_accuracy']:.4f}, macro F1={score['macro_f1']:.4f}")

    comparison = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1"], ascending=False
    )
    comparison.to_csv(DATA / "model_comparison.csv", index=False)
    best_name = comparison.loc[comparison["model"] != "dummy_baseline", "model"].iloc[0]
    best = fitted[best_name]
    val_predictions = best.predict(validation)
    pd.DataFrame({
        "workout_id": validation["workout_id"],
        "actual_intensity": validation["intensity"],
        "predicted_intensity": val_predictions,
    }).to_csv(DATA / "validation_predictions.csv", index=False)

    encoded_columns = save_encoded(train, best, DATA / "train_encoded.parquet")
    save_encoded(validation, best, DATA / "validation_encoded.parquet")
    save_encoded(test, best, DATA / "test_encoded.parquet")

    detailed = classification_report(
        validation["intensity"], val_predictions, labels=CLASSES,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "seed": SEED,
        "validation_share": VALIDATION_SHARE,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_class_counts": train["intensity"].value_counts().to_dict(),
        "validation_class_counts": validation["intensity"].value_counts().to_dict(),
        "audit": audit,
        "model_comparison": rows,
        "selected_model": best_name,
        "selected_model_report": detailed,
        "confusion_matrix_class_order": CLASSES,
        "confusion_matrix": confusion_matrix(
            validation["intensity"], val_predictions, labels=CLASSES
        ).tolist(),
        "encoded_feature_count": encoded_columns,
    }
    (DATA / "pipeline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Refit the chosen specification on every labeled workout for test prediction.
    final_model = clone(best).fit(full_clean, full_clean["intensity"])
    joblib.dump(final_model, DATA / "final_model.joblib", compress=3)
    submission = pd.DataFrame({
        "workout_id": test["workout_id"],
        "intensity": final_model.predict(test),
    })
    submission.to_csv(DATA / "submission.csv", index=False)

    raw_score = comparison.set_index("model").loc["random_forest_raw", "balanced_accuracy"]
    engineered_score = comparison.set_index("model").loc["random_forest_engineered", "balanced_accuracy"]
    selected_score = comparison.set_index("model").loc[best_name, "balanced_accuracy"]
    report = f"""# Workout intensity pipeline report

The untouched source is `train_full.csv` ({len(full_raw):,} labeled workouts). A stratified 80/20 split with seed {SEED} wrote `train.csv` ({len(train):,}) and `validation.csv` ({len(validation):,}). The only changed source value was the `altitude_mean_m = -9999` sentinel, replaced by a missing value in the split files; the model imputes it from the training fold. `test.csv` remains untouched.

The pipeline drops `workout_id` as a predictor. It one-hot encodes `user_id`, `sport`, and `gender`; numeric fields are median-imputed and scaled. The candidate engineered fields are elapsed speed, speed gap, climb per km, altitude balance, cyclic month/hour components, and weekend. Encoders and imputation statistics are fitted only on `train.csv` for validation. The selected model has {encoded_columns} encoded input columns in the saved split-level Parquet files.

## Validation results

{comparison.to_markdown(index=False, floatfmt='.4f')}

The selected model is **{best_name}**, with balanced accuracy **{selected_score:.4f}**. The random forest changes from **{raw_score:.4f}** on original features to **{engineered_score:.4f}** with the candidate features (difference **{engineered_score - raw_score:+.4f}**). The final saved model was refitted on all {len(full_clean):,} labeled workouts after selection; its `submission.csv` has {len(submission):,} test predictions.

Validation scores describe held-out workouts from users already seen in training. They do not estimate performance on new users. Near-zero individual feature importance in `data_analysis.ipynb` is not proof a correlated column can be removed safely.
"""
    (DATA / "pipeline_report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Selected: {best_name}; encoded columns: {encoded_columns}")


if __name__ == "__main__":
    main()
