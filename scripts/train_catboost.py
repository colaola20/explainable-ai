from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from catboost import CatBoostClassifier, Pool
import joblib
import json

# to run
# python scripts/train_catboost.py

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_catboost"
MODELS = ROOT / "models_catboost"

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
# CATBOOST MODEL
# ================================================
print("\n" + "="*60)
print("Training CatBoost Model")
print("="*60)

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    l2_leaf_reg=3,
    subsample=0.8,
    colsample_bylevel=0.8,
    min_data_in_leaf=20,
    random_seed=42,
    auto_class_weights='Balanced',  # Handles class imbalance
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=50,
    task_type='CPU',
    thread_count=-1
)

# Create Pool objects (CatBoost's efficient data structure)
train_pool = Pool(X_train_sub, y_train_sub)
val_pool = Pool(X_val, y_val)

# Train
cat_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

print(f"\nBest iteration: {cat_model.best_iteration_}")
print(f"Best validation score: {cat_model.best_score_['validation']['AUC']:.4f}")

# ================================================
# FIND OPTIMAL THRESHOLD
# ================================================
y_proba_val = cat_model.predict_proba(X_val)[:, 1]
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

y_pred = cat_model.predict(X_test)
y_proba = cat_model.predict_proba(X_test)[:, 1]
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
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 15 Important Features ---")
print(feature_importance.head(15))

# ================================================
# SAVE RESULTS & MODEL
# ================================================
metrics = {
    "model": "CatBoost",
    "optimal_threshold": float(optimal_threshold),
    "best_iteration": int(cat_model.best_iteration_),
    "best_val_auc": float(cat_model.best_score_['validation']['AUC']),
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
cat_model.save_model(str(MODELS / "catboost_model.cbm"))

print(f"\n{'='*60}")
print("SAVED FILES:")
print(f"{'='*60}")
print(f"✓ Model: {MODELS / 'catboost_model.cbm'}")
print(f"✓ Metrics: {RESULTS / 'metrics.json'}")
print(f"✓ Feature importance: {RESULTS / 'feature_importance.csv'}")
print("\nCatBoost training complete!")