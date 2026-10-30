# Workout intensity hackathon

Run the complete data preparation, GPU-enabled model search, and final training
from this directory:

```powershell
pip install -r requirements.txt
python train_best_model.py
```

The first run preserves the original labeled dataset as `Data/train_full.csv`.
It then creates a cleaned, stratified 80/20 split in `Data/train.csv` and
`Data/validation.csv` (seed 42). Re-running the script uses `train_full.csv`, so
the split is reproducible. `Data/test.csv` is never modified.

The preparation stage checks the data, replaces the invalid `-9999`
mean-altitude marker with a missing value in the split, and fits median
imputation and one-hot encoding on training rows only. The tuning stage runs
3-fold grid searches for LightGBM, CatBoost, XGBoost, and random forests using
**macro F1**. CatBoost and XGBoost use an NVIDIA GPU when one is available;
the installed LightGBM package uses CPU. The best validation model is refitted
on all labeled rows. If no NVIDIA GPU is available, the GPU models run on CPU.
Use `python train_best_model.py --reuse-search` to reuse completed grid results
and refit the winner without repeating the searches.

The main results are in `Data/tuning_report.md`,
`Data/tuned_model_comparison.csv`, and `Data/tuning_metrics.json`. Individual
grid scores are saved as `Data/tuning_<family>_cv.csv`. The earlier baseline
comparison remains in `Data/pipeline_report.md`.
The fitted model is `Data/final_model.joblib`; `Data/submission.csv` contains
test predictions in the sample submission format. The split-level encoded
inputs are `Data/train_encoded.parquet`, `Data/validation_encoded.parquet`, and
`Data/test_encoded.parquet` (these belong to the baseline models; the tuned
models transform raw columns internally).
`Data/data_analysis.ipynb` is the exploratory analysis of the original data.

To predict with the saved model, run Python from this directory so
`workout_features.py` is importable:

```python
import joblib
import pandas as pd

model = joblib.load("Data/final_model.joblib")
predictions = model.predict(pd.read_csv("Data/test.csv"))
```
