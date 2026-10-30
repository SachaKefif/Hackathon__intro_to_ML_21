"""Grid-search several tabular models for workout macro F1.

Run each family with `python tune_models.py --family lightgbm` (or catboost,
xgboost, random_forest), then `python tune_models.py --finalize`.
All hyperparameter selection uses 3-fold cross-validation on Data/train.csv.
Data/validation.csv is held out until the finalist comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from run_pipeline import make_pipeline
from workout_features import IntensityPredictor, WorkoutFeatureTransformer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
SEED = 42
LABELS = ["easy", "moderate", "hard"]
LABEL_TO_INT = {label: index for index, label in enumerate(LABELS)}
FAMILIES = ["lightgbm", "lightgbm_rank", "catboost", "xgboost", "random_forest"]


def gpu_available() -> bool:
    """Use NVIDIA acceleration when the device is visible to this process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    validation = pd.read_csv(DATA / "validation.csv")
    full = pd.read_csv(DATA / "train_full.csv")
    test = pd.read_csv(DATA / "test.csv")
    X_train = train.drop(columns="intensity")
    X_val = validation.drop(columns="intensity")
    y_train = train.intensity.map(LABEL_TO_INT).to_numpy()
    y_val = validation.intensity.map(LABEL_TO_INT).to_numpy()
    return X_train, y_train, X_val, y_val, full, test


def model_and_grid(family: str):
    gpu = gpu_available()
    if family in ["lightgbm", "lightgbm_rank"]:
        model = Pipeline([
            ("features", WorkoutFeatureTransformer(
                categorical_kind="category", rank_features=family == "lightgbm_rank"
            )),
            ("model", LGBMClassifier(
                objective="multiclass", n_estimators=450, learning_rate=0.05,
                min_child_samples=35, random_state=SEED, n_jobs=4,
                verbosity=-1, force_col_wise=True,
            )),
        ])
        if family == "lightgbm":
            grid = {
                "features__relative": [True],
                "model__num_leaves": [127],
                "model__min_child_samples": [15, 35, 50, 75],
                "model__n_estimators": [650, 950, 1250],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            }
        else:
            grid = {
                "model__num_leaves": [127],
                "model__n_estimators": [650, 950, 1250],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            }
    elif family == "catboost":
        model = Pipeline([
            ("features", WorkoutFeatureTransformer(categorical_kind="object")),
            ("model", CatBoostClassifier(
                loss_function="MultiClass", cat_features=("user_id", "sport", "gender"),
                iterations=450, learning_rate=0.05, random_seed=SEED,
                thread_count=4, verbose=False, allow_writing_files=False,
                task_type="GPU" if gpu else "CPU",
                **({"devices": "0"} if gpu else {}),
            )),
        ])
        grid = {
            "features__relative": [False, True],
            "model__depth": [5, 7],
            "model__l2_leaf_reg": [3, 8],
        }
    elif family == "xgboost":
        model = Pipeline([
            ("features", WorkoutFeatureTransformer(categorical_kind="category")),
            ("model", XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                enable_categorical=True, n_estimators=450, learning_rate=0.05,
                random_state=SEED, n_jobs=4, verbosity=0,
                device="cuda" if gpu else "cpu",
            )),
        ])
        grid = {
            "features__relative": [False, True],
            "model__max_depth": [3, 5],
            "model__min_child_weight": [3, 10],
        }
    elif family == "random_forest":
        model = make_pipeline(
            RandomForestClassifier(random_state=SEED, n_jobs=4), engineered=False
        )
        grid = {
            "model__n_estimators": [200, 400],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.5],
        }
    else:
        raise ValueError(f"Unknown family: {family}")
    return model, grid


def search_family(family: str):
    X_train, y_train, *_ = load_data()
    model, grid = model_and_grid(family)
    print(f"Searching {family} on {'GPU' if gpu_available() and family in ['catboost', 'xgboost'] else 'CPU'}", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        model, grid, scoring="f1_macro", cv=cv, refit=True, n_jobs=1,
        verbose=2, error_score="raise", return_train_score=False,
    )
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    results.insert(0, "family", family)
    results.to_csv(DATA / f"tuning_{family}_cv.csv", index=False)
    joblib.dump(search.best_estimator_, DATA / f"tuning_{family}_best.joblib", compress=3)
    print(f"BEST {family}: CV macro F1={search.best_score_:.4f}; {search.best_params_}")


def add_score(rows, name, y_true, predicted, cv_score=None):
    predicted = np.asarray(predicted).ravel()
    rows.append({
        "model": name,
        "cv_macro_f1": cv_score,
        "validation_macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "validation_accuracy": float(accuracy_score(y_true, predicted)),
    })


def finalize():
    X_train, y_train, X_val, y_val, full, test = load_data()
    fitted = {}
    cv_scores = {}
    rows = []
    for family in FAMILIES:
        path = DATA / f"tuning_{family}_best.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Run --family {family} first")
        fitted[family] = joblib.load(path)
        if family != "random_forest":
            # Older search artifacts predate the optional percentile feature.
            fitted[family].named_steps["features"].__dict__.setdefault(
                "rank_features", False
            )
        cv_scores[family] = float(
            pd.read_csv(DATA / f"tuning_{family}_cv.csv").iloc[0]["mean_test_score"]
        )
        add_score(rows, family, y_val, fitted[family].predict(X_val), cv_scores[family])

    # Tune class decision offsets on out-of-fold training predictions only.
    offset_family = max(["lightgbm", "lightgbm_rank"], key=lambda name: cv_scores[name])
    offset_name = f"{offset_family}_offset"
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(
        clone(fitted[offset_family]), X_train, y_train, cv=folds,
        method="predict_proba", n_jobs=1,
    )
    log_oof = np.log(np.clip(oof_proba, 1e-12, 1))
    best_offset_score = -1.0
    best_offsets = np.zeros(3)
    for moderate_offset in np.arange(-0.25, 0.251, 0.05):
        for hard_offset in np.arange(-0.25, 0.251, 0.05):
            offsets = np.array([0.0, moderate_offset, hard_offset])
            predicted = np.argmax(log_oof + offsets, axis=1)
            score = f1_score(y_train, predicted, average="macro")
            if score > best_offset_score:
                best_offset_score = score
                best_offsets = offsets
    val_proba = fitted[offset_family].predict_proba(X_val)
    offset_prediction = np.argmax(
        np.log(np.clip(val_proba, 1e-12, 1)) + best_offsets, axis=1
    )
    add_score(rows, offset_name, y_val, offset_prediction, best_offset_score)

    # The two LightGBM variants are complementary. Select their blend weight
    # and class offsets using aligned out-of-fold training probabilities.
    oof_plain = oof_proba if offset_family == "lightgbm" else cross_val_predict(
        clone(fitted["lightgbm"]), X_train, y_train, cv=folds,
        method="predict_proba", n_jobs=1,
    )
    oof_rank = oof_proba if offset_family == "lightgbm_rank" else cross_val_predict(
        clone(fitted["lightgbm_rank"]), X_train, y_train, cv=folds,
        method="predict_proba", n_jobs=1,
    )
    blend_best_score = -1.0
    blend_best_weight = 0.5
    blend_best_offsets = np.zeros(3)
    for rank_weight in [0.25, 0.5, 0.75]:
        blended = (1 - rank_weight) * oof_plain + rank_weight * oof_rank
        log_blended = np.log(np.clip(blended, 1e-12, 1))
        for moderate_offset in np.arange(-0.15, 0.151, 0.05):
            for hard_offset in np.arange(-0.15, 0.151, 0.05):
                offsets = np.array([0.0, moderate_offset, hard_offset])
                score = f1_score(
                    y_train, np.argmax(log_blended + offsets, axis=1),
                    average="macro",
                )
                if score > blend_best_score:
                    blend_best_score = score
                    blend_best_weight = rank_weight
                    blend_best_offsets = offsets
    blended_val = (
        (1 - blend_best_weight) * fitted["lightgbm"].predict_proba(X_val)
        + blend_best_weight * fitted["lightgbm_rank"].predict_proba(X_val)
    )
    blend_prediction = np.argmax(
        np.log(np.clip(blended_val, 1e-12, 1)) + blend_best_offsets,
        axis=1,
    )
    add_score(rows, "lightgbm_weighted", y_val, blend_prediction, blend_best_score)

    # Test whether a small GPU-trained CatBoost contribution adds useful
    # information to the two LightGBM variants. Select the weight on OOF data.
    oof_catboost = cross_val_predict(
        clone(fitted["catboost"]), X_train, y_train, cv=folds,
        method="predict_proba", n_jobs=1,
    )
    base_oof = (1 - blend_best_weight) * oof_plain + blend_best_weight * oof_rank
    base_val = blended_val
    cat_val = fitted["catboost"].predict_proba(X_val)
    cat_best_score = -1.0
    cat_best_weight = 0.0
    cat_best_offsets = np.zeros(3)
    for cat_weight in [0.05, 0.10, 0.20, 0.30]:
        mixed = (1 - cat_weight) * base_oof + cat_weight * oof_catboost
        log_mixed = np.log(np.clip(mixed, 1e-12, 1))
        for moderate_offset in np.arange(-0.15, 0.151, 0.05):
            for hard_offset in np.arange(-0.15, 0.151, 0.05):
                offsets = np.array([0.0, moderate_offset, hard_offset])
                score = f1_score(
                    y_train, np.argmax(log_mixed + offsets, axis=1),
                    average="macro",
                )
                if score > cat_best_score:
                    cat_best_score = score
                    cat_best_weight = cat_weight
                    cat_best_offsets = offsets
    cat_mixed_val = (1 - cat_best_weight) * base_val + cat_best_weight * cat_val
    cat_blend_prediction = np.argmax(
        np.log(np.clip(cat_mixed_val, 1e-12, 1)) + cat_best_offsets,
        axis=1,
    )
    add_score(rows, "lightgbm_catboost_weighted", y_val, cat_blend_prediction, cat_best_score)

    # Equal-probability ensembles are fixed in advance, without tuning weights
    # on the held-out validation labels.
    ensemble_options = {
        "boosting_ensemble": [offset_family, "catboost", "xgboost"],
        "lightgbm_catboost": [offset_family, "catboost"],
        "lightgbm_variants": ["lightgbm", "lightgbm_rank"],
    }
    ensemble_proba = {}
    for name, members in ensemble_options.items():
        proba = np.mean([fitted[member].predict_proba(X_val) for member in members], axis=0)
        ensemble_proba[name] = proba
        add_score(rows, name, y_val, np.argmax(proba, axis=1))

    comparison = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    comparison.to_csv(DATA / "tuned_model_comparison.csv", index=False)
    winner = comparison.iloc[0]["model"]
    selected_offsets = None
    if winner == "lightgbm_catboost_weighted":
        validation_prediction = cat_blend_prediction
        final_numeric = VotingClassifier(
            estimators=[
                ("lightgbm", clone(fitted["lightgbm"])),
                ("lightgbm_rank", clone(fitted["lightgbm_rank"])),
                ("catboost", clone(fitted["catboost"])),
            ],
            voting="soft",
            weights=[
                (1 - cat_best_weight) * (1 - blend_best_weight),
                (1 - cat_best_weight) * blend_best_weight,
                cat_best_weight,
            ],
            n_jobs=1,
        )
        selected_offsets = cat_best_offsets
    elif winner == "lightgbm_weighted":
        validation_prediction = blend_prediction
        final_numeric = VotingClassifier(
            estimators=[
                ("lightgbm", clone(fitted["lightgbm"])),
                ("lightgbm_rank", clone(fitted["lightgbm_rank"])),
            ],
            voting="soft", weights=[1 - blend_best_weight, blend_best_weight],
            n_jobs=1,
        )
        selected_offsets = blend_best_offsets
    elif winner == offset_name:
        validation_prediction = offset_prediction
        final_numeric = clone(fitted[offset_family])
        selected_offsets = best_offsets
    elif winner in fitted:
        validation_prediction = np.asarray(fitted[winner].predict(X_val)).ravel()
        final_numeric = clone(fitted[winner])
    else:
        validation_prediction = np.argmax(ensemble_proba[winner], axis=1)
        members = ensemble_options[winner]
        final_numeric = VotingClassifier(
            estimators=[(member, clone(fitted[member])) for member in members],
            voting="soft", n_jobs=1,
        )

    y_full = full.intensity.map(LABEL_TO_INT).to_numpy()
    final_numeric.fit(full.drop(columns="intensity"), y_full)
    final_model = IntensityPredictor(final_numeric, LABELS, selected_offsets)
    joblib.dump(final_model, DATA / "final_model.joblib", compress=3)
    submission = pd.DataFrame({
        "workout_id": test.workout_id,
        "intensity": final_model.predict(test),
    })
    submission.to_csv(DATA / "submission.csv", index=False)
    validation_labels = np.asarray(LABELS)[validation_prediction]
    pd.DataFrame({
        "workout_id": X_val.workout_id,
        "actual_intensity": np.asarray(LABELS)[y_val],
        "predicted_intensity": validation_labels,
    }).to_csv(DATA / "validation_predictions.csv", index=False)

    metrics = {
        "selected_model": winner,
        "nvidia_gpu_detected": gpu_available(),
        "gpu_search_families": ["catboost", "xgboost"] if gpu_available() else [],
        "selection_metric": "validation_macro_f1",
        "cross_validation": "3-fold stratified, training split only",
        "validation_rows": len(y_val),
        "comparison": rows,
        "class_order": LABELS,
        "confusion_matrix": confusion_matrix(y_val, validation_prediction, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_val, validation_prediction, labels=[0, 1, 2],
            target_names=LABELS, output_dict=True, zero_division=0,
        ),
        "lightgbm_oof_log_offsets": best_offsets.tolist(),
        "lightgbm_oof_macro_f1_after_offsets": best_offset_score,
        "offset_family": offset_family,
        "lightgbm_blend_rank_weight": blend_best_weight,
        "lightgbm_blend_log_offsets": blend_best_offsets.tolist(),
        "lightgbm_blend_oof_macro_f1": blend_best_score,
        "catboost_blend_weight": cat_best_weight,
        "catboost_blend_oof_macro_f1": cat_best_score,
    }
    (DATA / "tuning_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = f"""# Macro F1 model tuning

The original train/validation split was kept fixed: {len(X_train):,} training and {len(X_val):,} held-out validation workouts. Hyperparameters and the use of user/sport-relative features were selected with 3-fold stratified **grid search using macro F1** on training rows only. LightGBM class decision offsets and blend weights were separately tuned on out-of-fold training predictions. The held-out set was used to compare the tuned model variants and ensembles. It was not used to fit preprocessing, hyperparameters, offsets, or ensemble weights.

## Held-out results

{comparison.to_markdown(index=False, floatfmt='.4f')}

Selected **{winner}** with validation macro F1 **{comparison.iloc[0]['validation_macro_f1']:.4f}**. For reference, the earlier untuned random forest scored 0.5508 on this same validation split. The chosen specification was then refitted on all {len(full):,} labeled workouts and saved as `final_model.joblib`; `submission.csv` contains {len(submission):,} test predictions.

CatBoost and XGBoost searches used the NVIDIA GPU when available. LightGBM used CPU because this installation does not include a GPU tree learner. The held-out estimate applies to workouts from users represented in training. No test labels were available to measure test macro F1. The detailed grid results are in `tuning_<family>_cv.csv`.
"""
    (DATA / "tuning_report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"SELECTED {winner}: validation macro F1={comparison.iloc[0]['validation_macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", choices=FAMILIES)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        search_family(args.family)
