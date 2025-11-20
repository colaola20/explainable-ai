from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
import json

# TO RUN:
# python scripts/train_xgb.py


# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]  # repo root
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ================================================
# LOAD DATA
# ================================================
try:
    df = pd.read_csv('../data_preprocessing/data/processed/preprocessed_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Preprocessed CSV not found. Check your path.")


print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.2%}")
print(f"Class distribution:\n{df['default'].value_counts()}")


# prepare X/y
y = df['default'].astype(int)
X = df.drop(columns=['default'])

# save preprocessing metadata and feature list
feature_names = X.columns.to_list()
num_cols = X.select_dtypes(include='number').columns.tolist()
preproc = {
    "feature_names": feature_names,
    "num_cols": num_cols,
    "median": X[feature_names].median().to_dict()
}
joblib.dump(preproc, MODELS / "preproc.joblib")

# ================================================
# SPLIT & IMBALANCE
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# class imbalance baseline
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / max(1, pos)
print(f"neg/pos = {neg}/{pos}, scale_pos_weight={scale_pos_weight:.2f}")

# ================================================
# MODEL DEFINITION + GRID SEARCH
# ================================================
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1,
    early_stopping_rounds=50,
)

# Controlled hyperparameter grid
param_grid = {
    "n_estimators": [500, 1000],
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "gamma": [0, 0.1],
    "lambda": [1, 2],
    "alpha": [0, 1]
}

grid_search = GridSearchCV(
    estimator = xgb_model, 
    param_grid = param_grid, 
    scoring = "roc_auc", 
    cv = 3, verbose = 1, 
    n_jobs=-1
    )

grid_search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
    )

print("\nBest parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)


# ================================================
# EVALUATE BEST MODEL
# ================================================
final_model = xgb.XGBClassifier(
    **grid_search.best_params_
)
final_model.fit(
    X_train, y_train
    # eval_set=[(X_test, y_test)],
    # verbose=50
    )


y_pred = final_model.predict((X_test))
y_proba = final_model.predict_proba(X_test)[:, 1]

optimal_threshold = 0.4
y_pred_new = (y_proba >= optimal_threshold).astype(int)


print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ================================================
# SAVE RESULTS & MODEL
# ================================================
# metrics = {
#     "classification_report": classification_report(y_test, y_pred, output_dict=True),
#     "roc_auc": float(roc_auc_score(y_test, y_proba)),
#     "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
#     "train_size": len(X_train),
#     "test_size": len(X_test),
#     "scale_pos_weight": scale_pos_weight,
# }
# with open(RESULTS / "metrics.json", "w") as f:
#     json.dump(metrics, f, indent=2)

# save model
# joblib.dump(final_model, MODELS / "xgb_baseline.joblib")
# print("Saved sklearn wrapper model to", MODELS / "xgb_baseline.joblib")