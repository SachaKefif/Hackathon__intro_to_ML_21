# Macro F1 model tuning

The original train/validation split was kept fixed: 18,182 training and 4,546 held-out validation workouts. Hyperparameters and the use of user/sport-relative features were selected with 3-fold stratified **grid search using macro F1** on training rows only. LightGBM class decision offsets and blend weights were separately tuned on out-of-fold training predictions. The held-out set was used to compare the tuned model variants and ensembles. It was not used to fit preprocessing, hyperparameters, offsets, or ensemble weights.

## Held-out results

| model                      |   cv_macro_f1 |   validation_macro_f1 |   validation_accuracy |
|:---------------------------|--------------:|----------------------:|----------------------:|
| lightgbm_weighted          |        0.5898 |                0.6033 |                0.6040 |
| lightgbm_catboost_weighted |        0.5902 |                0.6000 |                0.5992 |
| lightgbm_variants          |      nan      |                0.5989 |                0.6010 |
| lightgbm_rank              |        0.5852 |                0.5979 |                0.5992 |
| lightgbm                   |        0.5863 |                0.5938 |                0.5961 |
| lightgbm_offset            |        0.5873 |                0.5935 |                0.5941 |
| lightgbm_catboost          |      nan      |                0.5922 |                0.5944 |
| boosting_ensemble          |      nan      |                0.5791 |                0.5814 |
| random_forest              |        0.5582 |                0.5683 |                0.5695 |
| catboost                   |        0.5570 |                0.5565 |                0.5576 |
| xgboost                    |        0.5600 |                0.5307 |                0.5337 |

Selected **lightgbm_weighted** with validation macro F1 **0.6033**. For reference, the earlier untuned random forest scored 0.5508 on this same validation split. The chosen specification was then refitted on all 22,728 labeled workouts and saved as `final_model.joblib`; `submission.csv` contains 7,669 test predictions.

CatBoost and XGBoost searches used the NVIDIA GPU when available. LightGBM used CPU because this installation does not include a GPU tree learner. The held-out estimate applies to workouts from users represented in training. No test labels were available to measure test macro F1. The detailed grid results are in `tuning_<family>_cv.csv`.
