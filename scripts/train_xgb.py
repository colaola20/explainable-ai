from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import os
import json
import matplotlib.pyplot as plt

# TO RUN:
# python scripts/train_xgb.py


# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]  # repo root
RESULTS = ROOT / "results_for_original_and_engineered_features"
MODELS = ROOT / "models_for_original_and_engineered_features"

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

# Further split training into train/validation
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"\nTrain: {len(X_train_sub)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# class imbalance baseline
neg = (y_train_sub == 0).sum()
pos = (y_train_sub == 1).sum()
scale_pos_weight = neg / max(1, pos)
print(f"neg/pos = {neg}/{pos}, scale_pos_weight={scale_pos_weight:.2f}")

# ================================================
# MODEL DEFINITION + GRID SEARCH
# ================================================
# xgb_model = xgb.XGBClassifier(
#     objective="binary:logistic",
#     eval_metric="auc",
#     scale_pos_weight=scale_pos_weight,
#     random_state=42,
#     use_label_encoder=False,
#     n_jobs=-1,
# )

# Controlled hyperparameter grid
# param_grid = {
#     "n_estimators": [500, 1000, 1500],
#     "max_depth": [4, 6, 8],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "subsample": [0.7, 0.8, 0.9],
#     "colsample_bytree": [0.7, 0.8, 0.9],
#     "gamma": [0, 0.1, 0.5],
#     "lambda": [1, 2, 3],
#     "alpha": [0, 0.5, 1]
# }

# Reduce the grid size

# param_grid = {
#     "n_estimators": [500, 1000],
#     "max_depth": [4, 6],
#     "learning_rate": [0.01, 0.05],
#     "subsample": [0.7, 0.9],
#     "colsample_bytree": [0.7, 0.9],
#     "gamma": [0, 0.5],
#     "lambda": [1, 3],
#     "alpha": [0, 1]
# }

# print("\n" + "="*60)
# print("Starting Grid Search...")
# print("="*60)

# ================================================
# GRID SEARCH
# ================================================


# grid_search = GridSearchCV(
#     estimator = xgb_model, 
#     param_grid = param_grid, 
#     scoring = "roc_auc", 
#     cv = 3, verbose = 3, 
#     n_jobs=-1,
#     return_train_score=False
#     )

# grid_search.fit(X_train_sub, y_train_sub)

# print("\nBest parameters:", grid_search.best_params_)
# print("Best CV ROC AUC:", grid_search.best_score_)

# ================================================
# Random GRID SEARCH
# ================================================

# Randomized Search
# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_grid,  # Same grid
#     n_iter=100,  # Try only 100 random combinations
#     scoring="roc_auc",
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=42
# )

# random_search.fit(X_train_sub, y_train_sub)



# ================================================
# TRAIN MODEL WITH BEST PARAMS (from GridSearch)
# ================================================

# BEST PARAMS FOR preprocessed_data.csv dataset

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

# BEST PARAMS FOR preprocessed_data_top_30.csv dataset

# best_params = {
#     'alpha': 1,
#     'colsample_bytree': 0.9,
#     'gamma': 0.1,
#     'reg_lambda': 3,  # Note: 'lambda' → 'reg_lambda' in XGBoost API
#     'learning_rate': 0.01,
#     'max_depth': 4,
#     'n_estimators': 500,
#     'subsample': 0.7
# }



# SKIP GridSearch for future running on preprocessed_data.csv dataset

print("\n" + "="*60)
print("Training final model with best params + early stopping...")
print("="*60)

baseline_model = xgb.XGBClassifier(
    #**grid_search.best_params_,
    **best_params,

    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,

# ================================================
# MODEL 1: BALANCED CONFIGURATION
# ================================================

    # KEY CHANGES for balanced features:
    # max_depth=3,              # Lower depth = can't rely on one feature
    # min_child_weight=10,      # Need more samples per leaf
    # learning_rate=0.05,       # Faster learning
    # n_estimators=500,         
    # subsample=0.7,            # Use 70% of data per tree
    # colsample_bytree=0.5,     # Use only 50% of features per tree! KEY!
    # colsample_bylevel=0.7,    # And 70% per level
    
    # # Regularization
    # reg_alpha=1,              # L1 regularization
    # reg_lambda=3,             # L2 regularization
    # gamma=0.5,                # Minimum loss reduction
)

# ================================================
# MODEL 2: VERY SHALLOW TREES
# ================================================

# final_model = xgb.XGBClassifier(
#     objective="binary:logistic",
#     eval_metric="auc",
#     scale_pos_weight=scale_pos_weight,
#     random_state=42,
#     use_label_encoder=False,
#     n_jobs=-1,
#     early_stopping_rounds=50,
    
#     max_depth=2,              # VERY shallow
#     min_child_weight=20,      
#     learning_rate=0.1,        
#     n_estimators=1000,        # More trees to compensate
#     subsample=0.8,            
#     colsample_bytree=0.6,     
    
#     reg_alpha=0.5,            
#     reg_lambda=2,             
#     gamma=0.3,                
# )

# ================================================
# MODEL 3: DART (Dropouts)
# ================================================

# final_model = xgb.XGBClassifier(
#     objective="binary:logistic",
#     eval_metric="auc",
#     scale_pos_weight=scale_pos_weight,
#     random_state=42,
#     use_label_encoder=False,
#     n_jobs=-1,
#     early_stopping_rounds=50,
    
#     booster='dart',           # Use DART instead of gbtree
#     rate_drop=0.3,            # Drop 30% of trees each round
#     skip_drop=0.5,            # 50% chance to skip dropout
    
#     max_depth=4,              
#     min_child_weight=5,       
#     learning_rate=0.05,       
#     n_estimators=500,         
#     subsample=0.8,            
#     colsample_bytree=0.7,     
    
#     reg_alpha=1,              
#     reg_lambda=2,             
# )

# ================================================


baseline_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nBest iteration: {baseline_model.best_iteration}")
print(f"Best validation score: {baseline_model.best_score:.4f}")

# Evaluate baseline on test set
y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_proba_baseline)
print(f"Baseline - Test AUC: {baseline_auc:.4f}")

# ================================================
# STEP 2: PERMUTATION IMPORTANCE ANALYSIS
# ================================================

print("\n" + "="*70)
print("STEP 2: Calculating permutation importance...")
print("="*70)

perm_importance = permutation_importance(
    baseline_model, 
    X_val, 
    y_val,
    n_repeats=30,
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1
)

# Create DataFrame
perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std,
    'importance_min': perm_importance.importances.min(axis=1),
    'importance_max': perm_importance.importances.max(axis=1)
}).sort_values('importance_mean', ascending=True)

# Identify noisy features
print("\n" + "=" * 70)
print("POTENTIALLY NOISY FEATURES (negative or near-zero importance):")
print("=" * 70)

noisy_candidates = perm_df[perm_df['importance_mean'] <= 0.0001]
print(f"\nFound {len(noisy_candidates)} features with importance <= 0.0001")
print(noisy_candidates.head(20))

# Save permutation importance
perm_df.to_csv(RESULTS / 'permutation_importance.csv', index=False)
print(f"\nSaved permutation importance to {RESULTS / 'permutation_importance.csv'}")

# ================================================
# STEP 3: TEST FEATURE REMOVAL IN TIERS
# ================================================
print("\n" + "="*70)
print("STEP 3: Testing feature removal in tiers...")
print("="*70)

# Define tiers based on permutation importance
highly_negative = perm_df[perm_df['importance_mean'] < -0.0001]['feature'].tolist()
all_negative = perm_df[perm_df['importance_mean'] < 0]['feature'].tolist()
near_zero = perm_df[(perm_df['importance_mean'] >= 0) & 
                    (perm_df['importance_mean'] < 0.0001)]['feature'].tolist()

print(f"\nHighly negative features (< -0.0001): {len(highly_negative)}")
print(f"All negative features: {len(all_negative)}")
print(f"Near-zero features (0 to 0.0001): {len(near_zero)}")

# Test configurations
test_configs = [
    ("Baseline (all features)", []),
    ("Remove highly negative", highly_negative),
    ("Remove all negative", all_negative),
    ("Remove all negative + near-zero", all_negative + near_zero),
]

results = []

for config_name, features_to_remove in test_configs:
    print(f"\nTesting: {config_name}")
    print(f"  Removing {len(features_to_remove)} features")
    
    # Prepare data
    if len(features_to_remove) > 0:
        X_train_test = X_train_sub.drop(columns=features_to_remove, errors='ignore')
        X_val_test = X_val.drop(columns=features_to_remove, errors='ignore')
        X_test_test = X_test.drop(columns=features_to_remove, errors='ignore')
    else:
        X_train_test = X_train_sub
        X_val_test = X_val
        X_test_test = X_test
    
    # Train model
    model_test = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbose=0
    )
    
    model_test.fit(
        X_train_test, y_train_sub,
        eval_set=[(X_val_test, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model_test.predict_proba(X_test_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred)
    
    result = {
        'config': config_name,
        'features_removed': len(features_to_remove),
        'features_kept': X_train_test.shape[1],
        'test_auc': test_auc,
        'improvement': test_auc - baseline_auc
    }
    results.append(result)
    
    print(f"  Features kept: {X_train_test.shape[1]}")
    print(f"  Test AUC: {test_auc:.4f} ({test_auc - baseline_auc:+.4f})")

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("SUMMARY OF ALL TESTS")
print("="*70)
print(results_df.to_string(index=False))

# Find best configuration
best_idx = results_df['test_auc'].idxmax()
best_config = results_df.iloc[best_idx]

print("\n" + "="*70)
print("BEST CONFIGURATION:")
print("="*70)
print(f"Config: {best_config['config']}")
print(f"Features kept: {best_config['features_kept']}")
print(f"Features removed: {best_config['features_removed']}")
print(f"Test AUC: {best_config['test_auc']:.4f}")
print(f"Improvement: {best_config['improvement']:+.4f}")

# Save results
results_df.to_csv(RESULTS / 'feature_removal_comparison.csv', index=False)

# ================================================
# STEP 4: TRAIN FINAL MODEL WITH BEST CONFIG
# ================================================
print("\n" + "="*70)
print("STEP 4: Training final model with best configuration...")
print("="*70)

# Determine which features to use based on best config
if best_config['improvement'] > 0.0005:  # Meaningful improvement
    print(f"\nUsing configuration: {best_config['config']}")
    features_to_remove_final = test_configs[best_idx][1]
    X_train_final = X_train_sub.drop(columns=features_to_remove_final, errors='ignore')
    X_val_final = X_val.drop(columns=features_to_remove_final, errors='ignore')
    X_test_final = X_test.drop(columns=features_to_remove_final, errors='ignore')
else:
    print("\nNo improvement from feature removal - using all features")
    features_to_remove_final = []
    X_train_final = X_train_sub
    X_val_final = X_val
    X_test_final = X_test

# Train final model
final_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

final_model.fit(
    X_train_final, y_train_sub,
    eval_set=[(X_val_final, y_val)],
    verbose=50
)

print(f"\nFinal model - Best iteration: {final_model.best_iteration}")
print(f"Final model - Best validation score: {final_model.best_score:.4f}")

# ================================================
# FIND OPTIMAL THRESHOLD
# ================================================
y_proba_val = final_model.predict_proba(X_val_final)[:, 1]

# Use precision-recall curve to find best threshold
precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"\nOptimal threshold (based on F1): {optimal_threshold:.3f}")

# ================================================
# EVALUATE ON TEST SET
# ================================================

print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)


y_pred = final_model.predict(X_test_final)
y_proba = final_model.predict_proba(X_test_final)[:, 1]

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
    'feature': X_train_final.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 15 Important Features ---")
print(feature_importance.head(15))



# ================================================
# SAVE RESULTS & MODEL
# ================================================
# metrics = {
#     "best_params": best_params,
#     # "best_params": grid_search.best_params_,
#     # "best_cv_roc_auc": float(grid_search.best_score_),
#     "optimal_threshold": float(optimal_threshold),
#     "best_iteration": int(final_model.best_iteration),
#     "default_threshold": {
#         "classification_report": classification_report(y_test, y_pred, output_dict=True),
#         "roc_auc": float(roc_auc_score(y_test, y_proba)),
#         "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
#     },
#     "optimal_threshold": {
#         "classification_report": classification_report(y_test, y_pred_optimal, output_dict=True),
#         "roc_auc": float(roc_auc_score(y_test, y_proba)),
#         "confusion_matrix": confusion_matrix(y_test, y_pred_optimal).tolist(),
#     },
#     "train_size": len(X_train_sub),
#     "val_size": len(X_val),
#     "test_size": len(X_test),
#     "scale_pos_weight": float(scale_pos_weight),
# }
# with open(RESULTS / "metrics.json", "w") as f:
#     json.dump(metrics, f, indent=2)

# # Save feature importance
# feature_importance.to_csv(RESULTS / "feature_importance.csv", index=False)

# # Save preprocessing metadata
# feature_names = X.columns.to_list()
# num_cols = X.select_dtypes(include='number').columns.tolist()
# preproc = {
#     "feature_names": feature_names,
#     "num_cols": num_cols,
#     "median": X[feature_names].median().to_dict(),
#     "optimal_threshold": float(optimal_threshold),
# }
# joblib.dump(preproc, MODELS / "preproc.joblib")

# # save model
# joblib.dump(final_model, MODELS / "xgb_model.joblib")
# print("Saved sklearn wrapper model to", MODELS / "xgb_model.joblib")

print(f"\n{'='*60}")
print("SAVED FILES:")
print(f"{'='*60}")
print(f"✓ Model: {MODELS / 'xgb_model.joblib'}")
print(f"✓ Preprocessing: {MODELS / 'preproc.joblib'}")
print(f"✓ Metrics: {RESULTS / 'metrics.json'}")
print(f"✓ Feature importance: {RESULTS / 'feature_importance.csv'}")
print("\nTraining complete!")




# ================================================
# VISUALIZATIONS
# ================================================
print("\nGenerating visualizations...")

# Plot 1: Permutation importance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

bottom_20 = perm_df.head(20)
axes[0].barh(range(len(bottom_20)), bottom_20['importance_mean'])
axes[0].set_yticks(range(len(bottom_20)))
axes[0].set_yticklabels(bottom_20['feature'], fontsize=8)
axes[0].axvline(x=0, color='r', linestyle='--', label='Zero importance')
axes[0].set_xlabel('Permutation Importance')
axes[0].set_title('Bottom 20 Features (Potential Noise)')
axes[0].legend()

axes[1].errorbar(
    range(len(perm_df)), 
    perm_df['importance_mean'], 
    yerr=perm_df['importance_std'],
    fmt='o', markersize=3, alpha=0.6
)
axes[1].axhline(y=0, color='r', linestyle='--', label='Zero line')
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Permutation Importance')
axes[1].set_title('All Features - Importance with Uncertainty')
axes[1].legend()

plt.tight_layout()
plt.savefig(RESULTS / 'noise_detection_permutation.png', dpi=300)
print(f"✓ Saved: {RESULTS / 'noise_detection_permutation.png'}")

# Plot 2: Feature removal comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(results_df['features_kept'], results_df['test_auc'], 'o-', 
             linewidth=2, markersize=8)
axes[0].axhline(y=baseline_auc, color='r', linestyle='--', label='Baseline')
axes[0].set_xlabel('Number of Features Kept')
axes[0].set_ylabel('Test ROC AUC')
axes[0].set_title('Model Performance vs Feature Count')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

improvement = results_df['improvement']
colors = ['green' if x >= 0 else 'red' for x in improvement]
axes[1].barh(range(len(results_df)), improvement, color=colors)
axes[1].set_yticks(range(len(results_df)))
axes[1].set_yticklabels(results_df['config'], fontsize=9)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('AUC Change from Baseline')
axes[1].set_title('Performance Change by Configuration')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS / 'feature_removal_comparison.png', dpi=300)
print(f"✓ Saved: {RESULTS / 'feature_removal_comparison.png'}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)