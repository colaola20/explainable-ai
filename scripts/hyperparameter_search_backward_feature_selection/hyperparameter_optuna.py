"""
OPTUNA HYPERPARAMETER OPTIMIZATION
===================================
Run this after Day 1 to fine-tune hyperparameters on selected features.
Uses intelligent Bayesian optimization (much smarter than grid search!)

Usage: python scripts/hyperparameter_search_backward_feature_selection/hyperparameter_optuna.py

Requirements:
    pip install optuna optuna-dashboard lightgbm
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "optuna_results"
MODELS = ROOT / "optuna_models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*80)

# ================================================
# LOAD DATA & SELECTED FEATURES
# ================================================
print("\n[1/5] Loading data and selected features...")

# Load full dataset
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X_full = df.drop(columns=['default'])

# Try to load selected features from Day 1
feature_selection_results = ROOT / "feature_selection_results"
selected_features_file = feature_selection_results / "selected_features.txt"

if selected_features_file.exists():
    print(f"  ✓ Loading selected features from Day 1")
    with open(selected_features_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    X = X_full[selected_features]
    print(f"  Using {len(selected_features)} selected features")
else:
    print(f"  ⚠ Selected features not found. Using all {X_full.shape[1]} features.")
    print(f"  Run day1_pipeline.py first for best results!")
    X = X_full
    selected_features = list(X_full.columns)

print(f"  Dataset shape: {X.shape}")
print(f"  Default rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

scale_pos_weight = (y_train_sub == 0).sum() / (y_train_sub == 1).sum()
print(f"  Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")


# ================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ================================================

class XGBoostOptimizer:
    """Optuna optimizer for XGBoost."""
    
    def __init__(self, X_train, y_train, X_val, y_val, scale_pos_weight, cv_folds=3):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scale_pos_weight = scale_pos_weight
        self.cv_folds = cv_folds
        self.best_model = None
    
    def objective(self, trial):
        """Objective function for Optuna."""
        
        # Suggest hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'verbosity': 0,
            
            # Tree parameters
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 2),
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            
            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            
            # Fixed parameters
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': self.scale_pos_weight,
        }
        
        # DART-specific parameters
        if params['booster'] == 'dart':
            params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
        
        # Use cross-validation for more robust evaluation
        if self.cv_folds > 1:
            kfold = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=42
            )
            
            cv_scores = []
            for train_idx, val_idx in kfold.split(self.X_train, self.y_train):
                X_tr = self.X_train.iloc[train_idx]
                y_tr = self.y_train.iloc[train_idx]
                X_v = self.X_train.iloc[val_idx]
                y_v = self.y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_v, y_v)],
                    verbose=False
                )
                
                pred = model.predict_proba(X_v)[:, 1]
                score = roc_auc_score(y_v, pred)
                cv_scores.append(score)
                
                # Pruning for efficiency
                trial.report(score, len(cv_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        else:
            # Single validation set
            model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            
            pred = model.predict_proba(self.X_val)[:, 1]
            score = roc_auc_score(self.y_val, pred)
            
            return score


class LightGBMOptimizer:
    """Optuna optimizer for LightGBM."""
    
    def __init__(self, X_train, y_train, X_val, y_val, scale_pos_weight, cv_folds=3):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scale_pos_weight = scale_pos_weight
        self.cv_folds = cv_folds
        self.best_model = None
    
    def objective(self, trial):
        """Objective function for Optuna."""
        
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            
            # Tree parameters
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            
            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 1),
            
            # Fixed parameters
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': self.scale_pos_weight,
        }
        
        # DART-specific parameters
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.0, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
        
        # Use cross-validation
        if self.cv_folds > 1:
            kfold = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=42
            )
            
            cv_scores = []
            for train_idx, val_idx in kfold.split(self.X_train, self.y_train):
                X_tr = self.X_train.iloc[train_idx]
                y_tr = self.y_train.iloc[train_idx]
                X_v = self.X_train.iloc[val_idx]
                y_v = self.y_train.iloc[val_idx]
                
                model = LGBMClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_v, y_v)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                
                pred = model.predict_proba(X_v)[:, 1]
                score = roc_auc_score(y_v, pred)
                cv_scores.append(score)
                
                # Pruning
                trial.report(score, len(cv_scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        else:
            # Single validation set
            model = LGBMClassifier(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            pred = model.predict_proba(self.X_val)[:, 1]
            score = roc_auc_score(self.y_val, pred)
            
            return score


# ================================================
# RUN OPTUNA OPTIMIZATION
# ================================================

def optimize_model(model_name, optimizer_class, X_train, y_train, X_val, y_val, 
                   scale_pos_weight, n_trials=50, timeout=3600, cv_folds=3):
    """
    Run Optuna optimization for a given model.
    
    Parameters:
    -----------
    model_name: str, name of the model
    optimizer_class: class, optimizer class
    n_trials: int, maximum number of trials
    timeout: int, maximum time in seconds (3600 = 1 hour)
    cv_folds: int, number of CV folds (1 = single validation)
    """
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZING {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Max trials: {n_trials}")
    print(f"Timeout: {timeout}s ({timeout/3600:.1f} hours)")
    print(f"CV folds: {cv_folds}")
    print(f"Starting optimization...\n")
    
    start_time = time.time()
    
    # Create optimizer
    optimizer = optimizer_class(X_train, y_train, X_val, y_val, scale_pos_weight, cv_folds)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    study.optimize(
        optimizer.objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=1  # Don't parallelize to avoid conflicts
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best CV score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    return study, optimizer


# ================================================
# OPTIMIZE XGBOOST
# ================================================

print("\n" + "="*80)
print("STEP 1: XGBOOST OPTIMIZATION")
print("="*80)

import lightgbm as lgb  # Import here to avoid issues

xgb_study, xgb_optimizer = optimize_model(
    model_name='XGBoost',
    optimizer_class=XGBoostOptimizer,
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val,
    y_val=y_val,
    scale_pos_weight=scale_pos_weight,
    n_trials=50,
    timeout=3600,  # 1 hour
    cv_folds=3
)

# Train final XGBoost model with best params
xgb_best_params = xgb_study.best_params.copy()
xgb_best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'scale_pos_weight': scale_pos_weight
})

# xgb_best_params = {
#     "booster": "gbtree",
#     "max_depth": 3,
#     "min_child_weight": 20,
#     "gamma": 1.1882,
#     "learning_rate": 0.0339,
#     "n_estimators": 500,
#     "subsample": 0.6747,
#     "colsample_bytree": 0.9901,
#     "colsample_bylevel": 0.5021,
#     "reg_alpha": 9.8572,
#     "reg_lambda": 7.6303,
# }

print("\nTraining final XGBoost model...")
xgb_final = xgb.XGBClassifier(**xgb_best_params, early_stopping_rounds=50)
xgb_final.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_test_pred = xgb_final.predict_proba(X_test)[:, 1]
xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)
print(f"XGBoost Test AUC: {xgb_test_auc:.4f}")


# ================================================
# OPTIMIZE LIGHTGBM
# ================================================

print("\n" + "="*80)
print("STEP 2: LIGHTGBM OPTIMIZATION")
print("="*80)

lgb_study, lgb_optimizer = optimize_model(
    model_name='LightGBM',
    optimizer_class=LightGBMOptimizer,
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val,
    y_val=y_val,
    scale_pos_weight=scale_pos_weight,
    n_trials=50,
    timeout=3600,  # 1 hour
    cv_folds=3
)

# Train final LightGBM model with best params
lgb_best_params = lgb_study.best_params.copy()
lgb_best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'scale_pos_weight': scale_pos_weight,
    'verbosity': -1
})

print("\nTraining final LightGBM model...")
lgb_final = LGBMClassifier(**lgb_best_params)
lgb_final.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(0)
    ]
)

lgb_test_pred = lgb_final.predict_proba(X_test)[:, 1]
lgb_test_auc = roc_auc_score(y_test, lgb_test_pred)
print(f"LightGBM Test AUC: {lgb_test_auc:.4f}")


# ================================================
# SAVE RESULTS
# ================================================

print("\n[3/3] Saving results...")

# Save best parameters
results = {
    'xgboost': {
        'best_params': xgb_best_params,
        'best_cv_score': float(xgb_study.best_value),
        'test_auc': float(xgb_test_auc),
        'n_trials': len(xgb_study.trials),
        'best_iteration': int(xgb_final.best_iteration)
    },
    'lightgbm': {
        'best_params': lgb_best_params,
        'best_cv_score': float(lgb_study.best_value),
        'test_auc': float(lgb_test_auc),
        'n_trials': len(lgb_study.trials),
        'best_iteration': int(lgb_final.best_iteration_)
    }
}

# results = {
#     'xgboost': {
#         'best_params': xgb_best_params,
#         'best_cv_score': 0.7869,
#         'test_auc': float(xgb_test_auc),
#         'n_trials': 50,
#         'best_iteration': int(xgb_final.best_iteration)
#     },
#     'lightgbm': {
#         'best_params': lgb_best_params,
#         'best_cv_score': float(lgb_study.best_value),
#         'test_auc': float(lgb_test_auc),
#         'n_trials': len(lgb_study.trials),
#         'best_iteration': int(lgb_final.best_iteration_)
#     }
# }

# Convert numpy types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

results = convert_to_serializable(results)

with open(RESULTS / 'optuna_best_params.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save models
joblib.dump(xgb_final, MODELS / 'xgb_optuna_optimized.joblib')
joblib.dump(lgb_final, MODELS / 'lgb_optuna_optimized.joblib')

# Save studies for later analysis
# joblib.dump(xgb_study, RESULTS / 'xgb_optuna_study.pkl')
joblib.dump(lgb_study, RESULTS / 'lgb_optuna_study.pkl')

# Save parameter importance
print("\nAnalyzing parameter importance...")

try:
    xgb_importance = optuna.importance.get_param_importances(xgb_study)
    lgb_importance = optuna.importance.get_param_importances(lgb_study)
    
    importance_df = pd.DataFrame({
        'xgb_param': list(xgb_importance.keys()),
        'xgb_importance': list(xgb_importance.values()),
    })
    
    lgb_importance_df = pd.DataFrame({
        'lgb_param': list(lgb_importance.keys()),
        'lgb_importance': list(lgb_importance.values()),
    })
    
    importance_df.to_csv(RESULTS / 'xgb_param_importance.csv', index=False)
    lgb_importance_df.to_csv(RESULTS / 'lgb_param_importance.csv', index=False)
except:
    print("  Could not compute parameter importance (need more trials)")


# ================================================
# FINAL SUMMARY
# ================================================

print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)

summary = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Best CV AUC': [xgb_study.best_value, lgb_study.best_value],
    'Test AUC': [xgb_test_auc, lgb_test_auc],
    'N Trials': [len(xgb_study.trials), len(lgb_study.trials)],
    'Best Iteration': [xgb_final.best_iteration, lgb_final.best_iteration_]
})

print("\n" + summary.to_string(index=False))

best_model = 'XGBoost' if xgb_test_auc > lgb_test_auc else 'LightGBM'
best_auc = max(xgb_test_auc, lgb_test_auc)
print(f"\n🏆 Best model: {best_model} with Test AUC: {best_auc:.4f}")

print("\n" + "="*80)
print("SAVED FILES")
print("="*80)
print(f"✓ Best parameters: {RESULTS / 'optuna_best_params.json'}")
print(f"✓ XGBoost model: {MODELS / 'xgb_optuna_optimized.joblib'}")
print(f"✓ LightGBM model: {MODELS / 'lgb_optuna_optimized.joblib'}")
print(f"✓ XGBoost study: {RESULTS / 'xgb_optuna_study.pkl'}")
print(f"✓ LightGBM study: {RESULTS / 'lgb_optuna_study.pkl'}")
print(f"✓ Parameter importance: {RESULTS / '*_param_importance.csv'}")


# ================================================
# VISUALIZATIONS
# ================================================

print("\nGenerating visualizations...")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 12))

# Plot 1: Optimization History - XGBoost
ax1 = plt.subplot(3, 3, 1)
xgb_trials_df = xgb_study.trials_dataframe()
ax1.plot(xgb_trials_df['number'], xgb_trials_df['value'], 'o-', alpha=0.6)
ax1.axhline(y=xgb_study.best_value, color='r', linestyle='--', label=f'Best: {xgb_study.best_value:.4f}')
ax1.set_xlabel('Trial')
ax1.set_ylabel('CV AUC')
ax1.set_title('XGBoost: Optimization History')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Optimization History - LightGBM
ax2 = plt.subplot(3, 3, 2)
lgb_trials_df = lgb_study.trials_dataframe()
ax2.plot(lgb_trials_df['number'], lgb_trials_df['value'], 'o-', alpha=0.6, color='green')
ax2.axhline(y=lgb_study.best_value, color='r', linestyle='--', label=f'Best: {lgb_study.best_value:.4f}')
ax2.set_xlabel('Trial')
ax2.set_ylabel('CV AUC')
ax2.set_title('LightGBM: Optimization History')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Comparison
ax3 = plt.subplot(3, 3, 3)
models = ['XGBoost', 'LightGBM']
cv_scores = [xgb_study.best_value, lgb_study.best_value]
test_scores = [xgb_test_auc, lgb_test_auc]
x = np.arange(len(models))
width = 0.35
ax3.bar(x - width/2, cv_scores, width, label='CV AUC', alpha=0.8)
ax3.bar(x + width/2, test_scores, width, label='Test AUC', alpha=0.8)
ax3.set_ylabel('AUC')
ax3.set_title('Model Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Parameter Importance - XGBoost (top 10)
ax4 = plt.subplot(3, 3, 4)
try:
    xgb_imp_df = pd.DataFrame({
        'param': list(xgb_importance.keys()),
        'importance': list(xgb_importance.values())
    }).sort_values('importance', ascending=False).head(10)
    
    ax4.barh(range(len(xgb_imp_df)), xgb_imp_df['importance'])
    ax4.set_yticks(range(len(xgb_imp_df)))
    ax4.set_yticklabels(xgb_imp_df['param'])
    ax4.set_xlabel('Importance')
    ax4.set_title('XGBoost: Top 10 Important Parameters')
    ax4.invert_yaxis()
except:
    ax4.text(0.5, 0.5, 'Insufficient trials\nfor importance analysis', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('XGBoost: Parameter Importance')

# Plot 5: Parameter Importance - LightGBM (top 10)
ax5 = plt.subplot(3, 3, 5)
try:
    lgb_imp_df = pd.DataFrame({
        'param': list(lgb_importance.keys()),
        'importance': list(lgb_importance.values())
    }).sort_values('importance', ascending=False).head(10)
    
    ax5.barh(range(len(lgb_imp_df)), lgb_imp_df['importance'], color='green')
    ax5.set_yticks(range(len(lgb_imp_df)))
    ax5.set_yticklabels(lgb_imp_df['param'])
    ax5.set_xlabel('Importance')
    ax5.set_title('LightGBM: Top 10 Important Parameters')
    ax5.invert_yaxis()
except:
    ax5.text(0.5, 0.5, 'Insufficient trials\nfor importance analysis', 
             ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('LightGBM: Parameter Importance')

# Plot 6: Summary table visualization
ax6 = plt.subplot(3, 3, 6)
ax6.axis('tight')
ax6.axis('off')
table_data = [
    ['Metric', 'XGBoost', 'LightGBM'],
    ['CV AUC', f'{xgb_study.best_value:.4f}', f'{lgb_study.best_value:.4f}'],
    ['Test AUC', f'{xgb_test_auc:.4f}', f'{lgb_test_auc:.4f}'],
    ['Trials', f'{len(xgb_study.trials)}', f'{len(lgb_study.trials)}'],
    ['Best Iter', f'{xgb_final.best_iteration}', f'{lgb_final.best_iteration_}']
]
table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')
ax6.set_title('Performance Summary', pad=20, fontweight='bold')

# Plot 7: Feature Importance - XGBoost
ax7 = plt.subplot(3, 3, 7)
xgb_feat_imp = pd.DataFrame({
    'feature': selected_features,
    'importance': xgb_final.feature_importances_
}).sort_values('importance', ascending=False).head(15)

ax7.barh(range(len(xgb_feat_imp)), xgb_feat_imp['importance'])
ax7.set_yticks(range(len(xgb_feat_imp)))
ax7.set_yticklabels(xgb_feat_imp['feature'], fontsize=8)
ax7.set_xlabel('Importance')
ax7.set_title('XGBoost: Top 15 Features')
ax7.invert_yaxis()

# Plot 8: Feature Importance - LightGBM
ax8 = plt.subplot(3, 3, 8)
lgb_feat_imp = pd.DataFrame({
    'feature': selected_features,
    'importance': lgb_final.feature_importances_
}).sort_values('importance', ascending=False).head(15)

ax8.barh(range(len(lgb_feat_imp)), lgb_feat_imp['importance'], color='green')
ax8.set_yticks(range(len(lgb_feat_imp)))
ax8.set_yticklabels(lgb_feat_imp['feature'], fontsize=8)
ax8.set_xlabel('Importance')
ax8.set_title('LightGBM: Top 15 Features')
ax8.invert_yaxis()

# Plot 9: Progress comparison
ax9 = plt.subplot(3, 3, 9)
ax9.plot(xgb_trials_df['number'], xgb_trials_df['value'], 
         'o-', alpha=0.5, label='XGBoost', markersize=3)
ax9.plot(lgb_trials_df['number'], lgb_trials_df['value'], 
         'o-', alpha=0.5, label='LightGBM', markersize=3, color='green')
ax9.set_xlabel('Trial Number')
ax9.set_ylabel('CV AUC')
ax9.set_title('Optimization Progress: Both Models')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'optuna_optimization_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: {RESULTS / 'optuna_optimization_summary.png'}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("Optuna optimization complete! Now you can:")
print("1. Review best parameters in 'optuna_best_params.json'")
print("2. Check parameter importance to understand what matters most")
print("3. Run Day 2 pipeline: 10-Fold Ensemble with these optimized params")
print("\nThese optimized parameters will give you better performance in the ensemble!")
print("="*80)

# Optional: Print top parameter insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

try:
    print("\nXGBoost - Most Important Parameters:")
    for param, imp in list(xgb_importance.items())[:5]:
        print(f"  {param}: {imp:.3f}")
    
    print("\nLightGBM - Most Important Parameters:")
    for param, imp in list(lgb_importance.items())[:5]:
        print(f"  {param}: {imp:.3f}")
except:
    print("Run more trials to get parameter importance insights")

print("\n" + "="*80)
print("ALL COMPLETE! 🎉")
print("="*80)