from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt

# TO RUN:
# pip install imbalanced-learn  (if not installed)
# python scripts/train_xgb_with_smote.py

# SMOTE COUSE OVERFITTING :(       <------------------------

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_smote_comparison"
MODELS = ROOT / "models_smote_comparison"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ================================================
# LOAD DATA
# ================================================
try:
    df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
except FileNotFoundError:
    raise FileNotFoundError("Preprocessed CSV not found. Check your path.")

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.2%}")
print(f"Class distribution:\n{df['default'].value_counts()}")

# Prepare X/y
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

print(f"\nOriginal split:")
print(f"Train: {len(X_train_sub)} (Class 0: {(y_train_sub==0).sum()}, Class 1: {(y_train_sub==1).sum()})")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# Class imbalance
neg = (y_train_sub == 0).sum()
pos = (y_train_sub == 1).sum()
scale_pos_weight = neg / max(1, pos)
print(f"\nOriginal class imbalance: {neg}/{pos} = {scale_pos_weight:.2f}")

# ================================================
# BEST PARAMS
# ================================================
best_params = {
    'alpha': 1,
    'colsample_bytree': 0.7,
    'gamma': 0,
    'reg_lambda': 3,  # Note: 'lambda' → 'reg_lambda' in XGBoost API
    'learning_rate': 0.01,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.8
}

# ================================================
# BASELINE: NO SMOTE
# ================================================
print("\n" + "="*70)
print("BASELINE: Training without SMOTE")
print("="*70)

baseline_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,  # Use scale_pos_weight for imbalance
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbose=0
)

baseline_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_baseline = baseline_model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_pred_baseline)

print(f"Baseline AUC: {baseline_auc:.4f}")
print(f"Training samples: {len(X_train_sub)}")

# ================================================
# SMOTE VARIANTS
# ================================================
results = []

# Add baseline to results
results.append({
    'method': 'Baseline (No SMOTE)',
    'train_samples': len(X_train_sub),
    'class_0': (y_train_sub==0).sum(),
    'class_1': (y_train_sub==1).sum(),
    'val_auc': baseline_model.best_score,
    'test_auc': baseline_auc
})

# ================================================
# 1. STANDARD SMOTE
# ================================================
print("\n" + "="*70)
print("METHOD 1: Standard SMOTE")
print("="*70)

smote = SMOTE(
    random_state=42,
    k_neighbors=5,
    sampling_strategy='auto'  # Balance to 1:1 ratio
)

X_train_smote, y_train_smote = smote.fit_resample(X_train_sub, y_train_sub)

print(f"After SMOTE:")
print(f"  Train samples: {len(X_train_smote)}")
print(f"  Class 0: {(y_train_smote==0).sum()}")
print(f"  Class 1: {(y_train_smote==1).sum()}")
print(f"  Ratio: {(y_train_smote==0).sum() / (y_train_smote==1).sum():.2f}")

# Train with SMOTE data - DON'T use scale_pos_weight (data is already balanced)
model_smote = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=1.0,  # Set to 1.0 since SMOTE already balanced the data
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbose=0
)

model_smote.fit(
    X_train_smote, y_train_smote,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_smote = model_smote.predict_proba(X_test)[:, 1]
smote_auc = roc_auc_score(y_test, y_pred_smote)

print(f"SMOTE AUC: {smote_auc:.4f} ({smote_auc - baseline_auc:+.4f})")

results.append({
    'method': 'SMOTE',
    'train_samples': len(X_train_smote),
    'class_0': (y_train_smote==0).sum(),
    'class_1': (y_train_smote==1).sum(),
    'val_auc': model_smote.best_score,
    'test_auc': smote_auc
})

# ================================================
# 2. BORDERLINE SMOTE
# ================================================
print("\n" + "="*70)
print("METHOD 2: Borderline SMOTE (focuses on decision boundary)")
print("="*70)

borderline_smote = BorderlineSMOTE(
    random_state=42,
    k_neighbors=5,
    sampling_strategy='auto'
)

X_train_border, y_train_border = borderline_smote.fit_resample(X_train_sub, y_train_sub)

print(f"After Borderline SMOTE:")
print(f"  Train samples: {len(X_train_border)}")
print(f"  Class 0: {(y_train_border==0).sum()}")
print(f"  Class 1: {(y_train_border==1).sum()}")

model_border = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbose=0
)

model_border.fit(
    X_train_border, y_train_border,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_border = model_border.predict_proba(X_test)[:, 1]
border_auc = roc_auc_score(y_test, y_pred_border)

print(f"Borderline SMOTE AUC: {border_auc:.4f} ({border_auc - baseline_auc:+.4f})")

results.append({
    'method': 'Borderline SMOTE',
    'train_samples': len(X_train_border),
    'class_0': (y_train_border==0).sum(),
    'class_1': (y_train_border==1).sum(),
    'val_auc': model_border.best_score,
    'test_auc': border_auc
})

# ================================================
# 3. ADASYN (Adaptive Synthetic Sampling)
# ================================================
print("\n" + "="*70)
print("METHOD 3: ADASYN (adaptive density-based)")
print("="*70)

adasyn = ADASYN(
    random_state=42,
    n_neighbors=5,
    sampling_strategy='auto'
)

try:
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_sub, y_train_sub)
    
    print(f"After ADASYN:")
    print(f"  Train samples: {len(X_train_adasyn)}")
    print(f"  Class 0: {(y_train_adasyn==0).sum()}")
    print(f"  Class 1: {(y_train_adasyn==1).sum()}")
    
    model_adasyn = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbose=0
    )
    
    model_adasyn.fit(
        X_train_adasyn, y_train_adasyn,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred_adasyn = model_adasyn.predict_proba(X_test)[:, 1]
    adasyn_auc = roc_auc_score(y_test, y_pred_adasyn)
    
    print(f"ADASYN AUC: {adasyn_auc:.4f} ({adasyn_auc - baseline_auc:+.4f})")
    
    results.append({
        'method': 'ADASYN',
        'train_samples': len(X_train_adasyn),
        'class_0': (y_train_adasyn==0).sum(),
        'class_1': (y_train_adasyn==1).sum(),
        'val_auc': model_adasyn.best_score,
        'test_auc': adasyn_auc
    })
except Exception as e:
    print(f"ADASYN failed: {e}")
    print("Skipping ADASYN method")

# ================================================
# 4. SMOTE + TOMEK LINKS (Hybrid)
# ================================================
print("\n" + "="*70)
print("METHOD 4: SMOTE-Tomek (over-sample + clean boundaries)")
print("="*70)

smotetomek = SMOTETomek(
    random_state=42,
    sampling_strategy='auto'
)

X_train_st, y_train_st = smotetomek.fit_resample(X_train_sub, y_train_sub)

print(f"After SMOTE-Tomek:")
print(f"  Train samples: {len(X_train_st)}")
print(f"  Class 0: {(y_train_st==0).sum()}")
print(f"  Class 1: {(y_train_st==1).sum()}")

model_st = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbose=0
)

model_st.fit(
    X_train_st, y_train_st,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_st = model_st.predict_proba(X_test)[:, 1]
st_auc = roc_auc_score(y_test, y_pred_st)

print(f"SMOTE-Tomek AUC: {st_auc:.4f} ({st_auc - baseline_auc:+.4f})")

results.append({
    'method': 'SMOTE-Tomek',
    'train_samples': len(X_train_st),
    'class_0': (y_train_st==0).sum(),
    'class_1': (y_train_st==1).sum(),
    'val_auc': model_st.best_score,
    'test_auc': st_auc
})

# ================================================
# 5. PARTIAL SMOTE (Balance to 2:1 instead of 1:1)
# ================================================
print("\n" + "="*70)
print("METHOD 5: Partial SMOTE (balance to 2:1 ratio)")
print("="*70)

partial_smote = SMOTE(
    random_state=42,
    k_neighbors=5,
    sampling_strategy=0.5  # Minority will be 50% of majority (2:1 ratio)
)

X_train_partial, y_train_partial = partial_smote.fit_resample(X_train_sub, y_train_sub)

print(f"After Partial SMOTE:")
print(f"  Train samples: {len(X_train_partial)}")
print(f"  Class 0: {(y_train_partial==0).sum()}")
print(f"  Class 1: {(y_train_partial==1).sum()}")
print(f"  Ratio: {(y_train_partial==0).sum() / (y_train_partial==1).sum():.2f}")

# Use moderate scale_pos_weight since we still have some imbalance
ratio_partial = (y_train_partial==0).sum() / (y_train_partial==1).sum()

model_partial = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=ratio_partial,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbose=0
)

model_partial.fit(
    X_train_partial, y_train_partial,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_partial = model_partial.predict_proba(X_test)[:, 1]
partial_auc = roc_auc_score(y_test, y_pred_partial)

print(f"Partial SMOTE AUC: {partial_auc:.4f} ({partial_auc - baseline_auc:+.4f})")

results.append({
    'method': 'Partial SMOTE (2:1)',
    'train_samples': len(X_train_partial),
    'class_0': (y_train_partial==0).sum(),
    'class_1': (y_train_partial==1).sum(),
    'val_auc': model_partial.best_score,
    'test_auc': partial_auc
})

# ================================================
# COMPARE ALL METHODS
# ================================================
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("COMPARISON OF ALL METHODS")
print("="*70)
print(results_df.to_string(index=False))

# Find best method
best_idx = results_df['test_auc'].idxmax()
best_method = results_df.iloc[best_idx]

print("\n" + "="*70)
print("BEST METHOD:")
print("="*70)
print(f"Method: {best_method['method']}")
print(f"Test AUC: {best_method['test_auc']:.4f}")
print(f"Improvement: {best_method['test_auc'] - baseline_auc:+.4f}")
print(f"Training samples: {best_method['train_samples']}")

# Save results
results_df.to_csv(RESULTS / 'smote_comparison.csv', index=False)
print(f"\nSaved comparison to {RESULTS / 'smote_comparison.csv'}")

# ================================================
# DETAILED EVALUATION OF BEST METHOD
# ================================================
print("\n" + "="*70)
print(f"DETAILED EVALUATION: {best_method['method']}")
print("="*70)

# Get the best model
if best_method['method'] == 'Baseline (No SMOTE)':
    best_model = baseline_model
    X_test_eval = X_test
elif best_method['method'] == 'SMOTE':
    best_model = model_smote
    X_test_eval = X_test
elif best_method['method'] == 'Borderline SMOTE':
    best_model = model_border
    X_test_eval = X_test
elif best_method['method'] == 'ADASYN':
    best_model = model_adasyn
    X_test_eval = X_test
elif best_method['method'] == 'SMOTE-Tomek':
    best_model = model_st
    X_test_eval = X_test
else:  # Partial SMOTE
    best_model = model_partial
    X_test_eval = X_test

# Find optimal threshold
y_proba_val_best = best_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val_best)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Optimal threshold: {optimal_threshold:.3f}")

# Predictions
y_pred = best_model.predict(X_test_eval)
y_proba = best_model.predict_proba(X_test_eval)[:, 1]
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

print("\n--- Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\n--- Optimal Threshold ({optimal_threshold:.3f}) ---")
print(classification_report(y_test, y_pred_optimal))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_optimal))

# ================================================
# VISUALIZATIONS
# ================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Test AUC comparison
axes[0, 0].barh(range(len(results_df)), results_df['test_auc'])
axes[0, 0].set_yticks(range(len(results_df)))
axes[0, 0].set_yticklabels(results_df['method'])
axes[0, 0].axvline(x=baseline_auc, color='r', linestyle='--', label='Baseline')
axes[0, 0].set_xlabel('Test AUC')
axes[0, 0].set_title('Test AUC Comparison')
axes[0, 0].legend()
axes[0, 0].set_xlim([0.77, results_df['test_auc'].max() * 1.01])

# Plot 2: AUC improvement
improvement = results_df['test_auc'] - baseline_auc
colors = ['green' if x >= 0 else 'red' for x in improvement]
axes[0, 1].barh(range(len(results_df)), improvement, color=colors)
axes[0, 1].set_yticks(range(len(results_df)))
axes[0, 1].set_yticklabels(results_df['method'])
axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_xlabel('AUC Change from Baseline')
axes[0, 1].set_title('Improvement over Baseline')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Plot 3: Sample size comparison
axes[1, 0].bar(range(len(results_df)), results_df['train_samples'])
axes[1, 0].set_xticks(range(len(results_df)))
axes[1, 0].set_xticklabels(results_df['method'], rotation=45, ha='right')
axes[1, 0].set_ylabel('Training Samples')
axes[1, 0].set_title('Training Set Size After Sampling')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Class balance
x = np.arange(len(results_df))
width = 0.35
axes[1, 1].bar(x - width/2, results_df['class_0'], width, label='Class 0 (No Default)', alpha=0.8)
axes[1, 1].bar(x + width/2, results_df['class_1'], width, label='Class 1 (Default)', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(results_df['method'], rotation=45, ha='right')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Class Distribution After Sampling')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS / 'smote_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS / 'smote_comparison.png'}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)