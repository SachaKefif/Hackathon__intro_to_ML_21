"""One-command training pipeline optimized for validation macro F1.

Usage:
    python train_best_model.py

This preserves the original labeled data, regenerates the cleaned split,
grid-searches five model variants with 3-fold CV, evaluates the tuned finalists
on the held-out validation set, and refits the winner on all labeled rows.
CatBoost and XGBoost automatically use an NVIDIA GPU when available.
"""

import argparse

from run_pipeline import DATA, main as prepare_data_and_baselines
from tune_models import FAMILIES, finalize, gpu_available, search_family


def main(reuse_search: bool = False) -> None:
    print("Preparing data and baseline models...", flush=True)
    prepare_data_and_baselines()
    print(
        "NVIDIA GPU detected: " + str(gpu_available())
        + ". CatBoost and XGBoost use it when available; LightGBM uses CPU.",
        flush=True,
    )
    for family in FAMILIES:
        cached = (DATA / f"tuning_{family}_best.joblib").exists() and (
            DATA / f"tuning_{family}_cv.csv"
        ).exists()
        if reuse_search and cached:
            print(f"Reusing completed {family} grid search", flush=True)
        else:
            search_family(family)
    print("Comparing tuned models and refitting the winner...", flush=True)
    finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-search", action="store_true",
        help="Reuse completed grid results; the default reruns every search.",
    )
    main(reuse_search=parser.parse_args().reuse_search)
