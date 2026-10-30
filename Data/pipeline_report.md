# Workout intensity pipeline report

The untouched source is `train_full.csv` (22,728 labeled workouts). A stratified 80/20 split with seed 42 wrote `train.csv` (18,182) and `validation.csv` (4,546). The only changed source value was the `altitude_mean_m = -9999` sentinel, replaced by a missing value in the split files; the model imputes it from the training fold. `test.csv` remains untouched.

The pipeline drops `workout_id` as a predictor. It one-hot encodes `user_id`, `sport`, and `gender`; numeric fields are median-imputed and scaled. The candidate engineered fields are elapsed speed, speed gap, climb per km, altitude balance, cyclic month/hour components, and weekend. Encoders and imputation statistics are fitted only on `train.csv` for validation. The selected model has 206 encoded input columns in the saved split-level Parquet files.

## Validation results

| model                    |   accuracy |   balanced_accuracy |   macro_f1 |
|:-------------------------|-----------:|--------------------:|-----------:|
| random_forest_raw        |     0.5532 |              0.5536 |     0.5508 |
| random_forest_engineered |     0.5466 |              0.5470 |     0.5443 |
| extra_trees_engineered   |     0.5370 |              0.5373 |     0.5328 |
| logistic_engineered      |     0.4461 |              0.4463 |     0.4426 |
| dummy_baseline           |     0.3346 |              0.3333 |     0.1671 |

The selected model is **random_forest_raw**, with balanced accuracy **0.5536**. The random forest changes from **0.5536** on original features to **0.5470** with the candidate features (difference **-0.0066**). The final saved model was refitted on all 22,728 labeled workouts after selection; its `submission.csv` has 7,669 test predictions.

Validation scores describe held-out workouts from users already seen in training. They do not estimate performance on new users. Near-zero individual feature importance in `data_analysis.ipynb` is not proof a correlated column can be removed safely.
