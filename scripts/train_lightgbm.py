from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import lightgbm as lgb
import joblib
import json

# to run
# python scripts/train_lightgbm.py

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_lightgbm"
MODELS = ROOT / "models_lightgbm"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ================================================
# LOAD DATA
# ================================================
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.2%}")

y = df['default'].astype(int)
X = df.drop(columns=['default'])

# ================================================
# SPLIT DATA
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"\nTrain: {len(X_train_sub)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ================================================
# LIGHTGBM MODEL
# ================================================
print("\n" + "="*60)
print("Training LightGBM Model")
print("="*60)

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=3,
    is_unbalance=True,  # Handles class imbalance automatically
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Train with early stopping
lgb_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\nBest iteration: {lgb_model.best_iteration_}")
print(f"Best validation score: {lgb_model.best_score_['valid_0']['auc']:.4f}")

# ================================================
# FIND OPTIMAL THRESHOLD
# ================================================
y_proba_val = lgb_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)

if thresholds.size > 0:
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])
else:
    optimal_threshold = 0.5

print(f"\nOptimal threshold (based on F1): {optimal_threshold:.3f}")

# ================================================
# EVALUATE ON TEST SET
# ================================================
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

print("\n--- With Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\n--- With Optimal Threshold ({optimal_threshold:.3f}) ---")
print(classification_report(y_test, y_pred_optimal))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_optimal))

# ================================================
# FEATURE IMPORTANCE
# ================================================
feature_importance = pd.DataFrame({
    'feature': X_train_sub.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 15 Important Features ---")
print(feature_importance.head(15))

# ================================================
# SAVE RESULTS & MODEL
# ================================================
metrics = {
    "model": "LightGBM",
    "optimal_threshold": float(optimal_threshold),
    "best_iteration": int(lgb_model.best_iteration_),
    "best_val_auc": float(lgb_model.best_score_['valid_0']['auc']),
    "default_threshold": {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    },
    "optimal_threshold_metrics": {
        "classification_report": classification_report(y_test, y_pred_optimal, output_dict=True),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_optimal).tolist(),
    },
}

with open(RESULTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

feature_importance.to_csv(RESULTS / "feature_importance.csv", index=False)
joblib.dump(lgb_model, MODELS / "lgb_model.joblib")

print(f"\n{'='*60}")
print("SAVED FILES:")
print(f"{'='*60}")
print(f"✓ Model: {MODELS / 'lgb_model.joblib'}")
print(f"✓ Metrics: {RESULTS / 'metrics.json'}")
print(f"✓ Feature importance: {RESULTS / 'feature_importance.csv'}")
print("\nLightGBM training complete!")