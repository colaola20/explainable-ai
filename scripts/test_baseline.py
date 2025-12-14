# Save this as: beat_baseline_single.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# Load data
df_original = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data.csv')

df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X = df.drop(columns=['default'])

# Use EXACT same split as original
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

scale_pos_weight = (y_train_sub == 0).sum() / (y_train_sub == 1).sum()

# Your original best params but with MORE trees and patience
best_params = {
    'alpha': 1,
    'colsample_bytree': 0.7,
    'gamma': 0,
    'reg_lambda': 3,
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 2000,  # MORE TREES (was 500)
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
}

print("\n" + "="*80)
print("TRYING TO BEAT BASELINE WITH SINGLE MODEL")
print("="*80)
print(f"\nUsing ALL {X.shape[1]} features")
print(f"Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Max trees: 2000 (vs original 500)")
print(f"Early stopping patience: 100 rounds (vs original 50)")

model = xgb.XGBClassifier(**best_params, early_stopping_rounds=100)
model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nBest iteration: {model.best_iteration}")

# Test
y_pred = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Test AUC: {test_auc:.4f}")
print(f"Original baseline: 0.7843")
print(f"Difference: {(test_auc - 0.7843)*100:+.2f}%")

if test_auc > 0.7843:
    print("\n🎉 SUCCESS! Beat the baseline!")
else:
    print("\n😔 Did not beat baseline. Try Option B (Optuna)")