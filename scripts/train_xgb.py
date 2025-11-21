from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
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
RESULTS = ROOT / "results_for_top_30"
MODELS = ROOT / "models_for_top_30"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


# ================================================
# LOAD DATA
# ================================================
try:
    df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_top_30.csv')
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

# # Controlled hyperparameter grid
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

# print("\n" + "="*60)
# print("Starting Grid Search...")
# print("="*60)

# grid_search = GridSearchCV(
#     estimator = xgb_model, 
#     param_grid = param_grid, 
#     scoring = "roc_auc", 
#     cv = 3, verbose = 2, 
#     n_jobs=-1
#     )

# grid_search.fit(X_train_sub, y_train_sub)

# print("\nBest parameters:", grid_search.best_params_)
# print("Best CV ROC AUC:", grid_search.best_score_)




# ================================================
# TRAIN MODEL WITH BEST PARAMS (from GridSearch)
# ================================================

# BEST PARAMS FOR preprocessed_data.csv dataset

# best_params = {
#     'alpha': 1,
#     'colsample_bytree': 0.7,
#     'gamma': 0,
#     'reg_lambda': 3,  # Note: 'lambda' → 'reg_lambda' in XGBoost API
#     'learning_rate': 0.01,
#     'max_depth': 6,
#     'n_estimators': 500,
#     'subsample': 0.8
# }

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

final_model = xgb.XGBClassifier(
    # **grid_search.best_params_,
    #**best_params,

    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1,
    early_stopping_rounds=50,

# ================================================
# MODEL 1: BALANCED CONFIGURATION
# ================================================

    # KEY CHANGES for balanced features:
    max_depth=3,              # Lower depth = can't rely on one feature
    min_child_weight=10,      # Need more samples per leaf
    learning_rate=0.05,       # Faster learning
    n_estimators=500,         
    subsample=0.7,            # Use 70% of data per tree
    colsample_bytree=0.5,     # Use only 50% of features per tree! KEY!
    colsample_bylevel=0.7,    # And 70% per level
    
    # Regularization
    reg_alpha=1,              # L1 regularization
    reg_lambda=3,             # L2 regularization
    gamma=0.5,                # Minimum loss reduction
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


final_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nBest iteration: {final_model.best_iteration}")
print(f"Best validation score: {final_model.best_score:.4f}")

# ================================================
# FIND OPTIMAL THRESHOLD
# ================================================
y_proba_val = final_model.predict_proba(X_val)[:, 1]

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


y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

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
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 15 Important Features ---")
print(feature_importance.head(15))


# ================================================
# SAVE RESULTS & MODEL
# ================================================
# metrics = {
#     #"best_params": best_params,
#     "best_params": grid_search.best_params_,
#     "best_cv_roc_auc": float(grid_search.best_score_),
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