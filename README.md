# Workout intensity hackathon

Run the complete data and modeling pipeline from this directory:

```powershell
pip install -r requirements.txt
python run_pipeline.py
```

The first run preserves the original labeled dataset as `Data/train_full.csv`.
It then creates a cleaned, stratified 80/20 split in `Data/train.csv` and
`Data/validation.csv` (seed 42). Re-running the script uses `train_full.csv`, so
the split is reproducible. `Data/test.csv` is never modified.

The script checks the data, replaces the invalid `-9999` mean-altitude marker
with a missing value in the split, and fits median imputation and one-hot
encoding on training rows only. It compares a baseline, logistic regression,
random forest with and without engineered features, and Extra Trees. The model
with the highest validation balanced accuracy is refitted on all labeled rows.

Results are in `Data/pipeline_report.md` and `Data/model_comparison.csv`.
`Data/pipeline_metrics.json` includes class metrics and a confusion matrix.
The fitted model is `Data/final_model.joblib`; `Data/submission.csv` contains
test predictions in the sample submission format. The split-level encoded
inputs are `Data/train_encoded.parquet`, `Data/validation_encoded.parquet`, and
`Data/test_encoded.parquet` (Parquet keeps the one-hot data compact).
`Data/data_analysis.ipynb` is the exploratory analysis of the original data.

To predict with the saved model, run Python from this directory so
`workout_features.py` is importable:

```python
import joblib
import pandas as pd

model = joblib.load("Data/final_model.joblib")
predictions = model.predict(pd.read_csv("Data/test.csv"))
```
