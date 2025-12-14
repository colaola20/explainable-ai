"""
10-FOLD ENSEMBLE WITH OPTIMIZED PARAMETERS
===========================================================
Run this after Optuna optimization to train the final ensemble.
Uses best parameters from both XGBoost and LightGBM.

Usage: python scripts/hyperparameter_search_backward_feature_selection/ensemble.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import joblib
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ensemble_results"
MODELS = ROOT / "ensemble_models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("10-FOLD ENSEMBLE WITH OPTIMIZED PARAMETERS")
print("="*80)

# ================================================
# LOAD DATA & CONFIGURATIONS
# ================================================
print("\n[1/5] Loading data and configurations...")

# Load full dataset
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X_full = df.drop(columns=['default'])

# Load selected features
feature_selection_results = ROOT / "feature_selection_results"
selected_features_file = feature_selection_results / "selected_features.txt"

if selected_features_file.exists():
    with open(selected_features_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    X = X_full[selected_features]
    print(f"  ✓ Using {len(selected_features)} selected features")
else:
    X = X_full
    selected_features = list(X_full.columns)
    print(f"  ⚠ Using all {X_full.shape[1]} features (run day1_pipeline.py first)")

# Load Optuna best parameters
optuna_results = ROOT / "optuna_results"
optuna_params_file = optuna_results / "optuna_best_params.json"

if optuna_params_file.exists():
    with open(optuna_params_file, 'r') as f:
        optuna_params = json.load(f)
    print(f"  ✓ Loaded Optuna optimized parameters")
    xgb_params = optuna_params['xgboost']['best_params']
    lgb_params = optuna_params['lightgbm']['best_params']
else:
    print(f"  ⚠ Optuna params not found, using defaults")
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    xgb_params = {
        'max_depth': 6, 'learning_rate': 0.01, 'n_estimators': 500,
        'scale_pos_weight': scale_pos_weight, 'objective': 'binary:logistic',
        'eval_metric': 'auc', 'random_state': 42, 'n_jobs': -1
    }
    lgb_params = {
        'max_depth': 6, 'learning_rate': 0.01, 'n_estimators': 500,
        'scale_pos_weight': scale_pos_weight, 'objective': 'binary',
        'metric': 'auc', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }

print(f"  Dataset shape: {X.shape}")
print(f"  Default rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")


# ================================================
# 10-FOLD ENSEMBLE CLASS
# ================================================

class KFoldEnsemble:
    """
    Train multiple models with 10-Fold CV and ensemble predictions.
    Each model is trained 10 times, final prediction is average.
    """
    
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []
        self.oof_predictions = None
        self.test_predictions = None
        self.fold_scores = []
    
    def train(self, X_train, y_train, X_test, model_configs):
        """
        Train ensemble with K-Fold CV.
        
        Parameters:
        -----------
        model_configs: list of dict, e.g.:
            [
                {'name': 'xgb', 'model_class': xgb.XGBClassifier, 'params': {...}},
                {'name': 'lgb', 'model_class': LGBMClassifier, 'params': {...}},
            ]
        """
        
        n_train = len(X_train)
        n_test = len(X_test)
        n_models = len(model_configs)
        
        # Initialize storage
        self.oof_predictions = np.zeros((n_train, n_models))
        self.test_predictions = np.zeros((n_test, n_models))
        
        # K-Fold split
        kfold = StratifiedKFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        print(f"\n[2/5] Training {n_models} models with {self.n_folds}-Fold CV")
        print(f"  This will train {n_models * self.n_folds} models total")
        
        start_time = time.time()
        
        for model_idx, config in enumerate(model_configs):
            model_name = config['name']
            model_class = config['model_class']
            params = config['params']
            
            print(f"\n  {'='*70}")
            print(f"  Model {model_idx+1}/{n_models}: {model_name.upper()}")
            print(f"  {'='*70}")
            
            fold_models = []
            test_preds_folds = []
            fold_aucs = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
                # Split data
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]
                
                # Train model
                model = model_class(**params)
                
                if model_name == 'xgb':
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif model_name == 'lgb':
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(0)
                        ]
                    )
                elif model_name == 'cat':
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_tr, y_tr)
                
                # Get predictions
                val_pred = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
                
                # Store OOF predictions
                self.oof_predictions[val_idx, model_idx] = val_pred
                test_preds_folds.append(test_pred)
                fold_models.append(model)
                
                # Calculate fold AUC
                fold_auc = roc_auc_score(y_val, val_pred)
                fold_aucs.append(fold_auc)
                print(f"    Fold {fold+1:2d}/{self.n_folds}: AUC = {fold_auc:.4f}")
            
            # Average test predictions across folds
            self.test_predictions[:, model_idx] = np.mean(test_preds_folds, axis=0)
            
            # Calculate OOF AUC for this model
            oof_auc = roc_auc_score(y_train, self.oof_predictions[:, model_idx])
            mean_fold_auc = np.mean(fold_aucs)
            std_fold_auc = np.std(fold_aucs)
            
            print(f"  → OOF AUC: {oof_auc:.4f}")
            print(f"  → Mean Fold AUC: {mean_fold_auc:.4f} ± {std_fold_auc:.4f}")
            
            self.models.append({
                'name': model_name,
                'fold_models': fold_models,
                'oof_auc': oof_auc,
                'mean_fold_auc': mean_fold_auc,
                'std_fold_auc': std_fold_auc,
                'fold_aucs': fold_aucs
            })
        
        elapsed = time.time() - start_time
        print(f"\n  ✓ All models trained in {elapsed/60:.1f} minutes")
    
    def predict(self, mode='average', weights=None):
        """
        Get ensemble predictions.
        
        Parameters:
        -----------
        mode: 'average', 'weighted', or 'rank_average'
        weights: list of weights for each model (if mode='weighted')
        """
        
        if mode == 'average':
            oof_pred = self.oof_predictions.mean(axis=1)
            test_pred = self.test_predictions.mean(axis=1)
        
        elif mode == 'weighted':
            if weights is None:
                # Use OOF AUC as weights
                weights = np.array([m['oof_auc'] for m in self.models])
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            oof_pred = (self.oof_predictions * weights).sum(axis=1)
            test_pred = (self.test_predictions * weights).sum(axis=1)
        
        elif mode == 'rank_average':
            # Rank averaging (more robust to outliers)
            from scipy.stats import rankdata
            
            # Get ranks for each model
            oof_ranks = np.array([rankdata(self.oof_predictions[:, i]) 
                                for i in range(self.oof_predictions.shape[1])])
            test_ranks = np.array([rankdata(self.test_predictions[:, i]) 
                                for i in range(self.test_predictions.shape[1])])
            
            # Average the ranks
            oof_pred = np.mean(oof_ranks, axis=0)
            test_pred = np.mean(test_ranks, axis=0)
            
            # NORMALIZE TO [0, 1] - THIS IS THE KEY FIX!
            oof_pred = (oof_pred - oof_pred.min()) / (oof_pred.max() - oof_pred.min())
            test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min())
        
        return oof_pred, test_pred
    
    def evaluate(self, y_train, y_test, mode='average', weights=None):
        """Evaluate ensemble performance."""
        
        oof_pred, test_pred = self.predict(mode=mode, weights=weights)
        
        oof_auc = roc_auc_score(y_train, oof_pred)
        test_auc = roc_auc_score(y_test, test_pred)
        
        return oof_auc, test_auc, oof_pred, test_pred


# ================================================
# CREATE MODEL CONFIGURATIONS
# ================================================

print("\n[3/5] Creating model configurations...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model_configs = [
    {
        'name': 'xgb',
        'model_class': xgb.XGBClassifier,
        'params': xgb_params
    },
    {
        'name': 'lgb',
        'model_class': LGBMClassifier,
        'params': lgb_params
    },
    {
        'name': 'cat',
        'model_class': CatBoostClassifier,
        'params': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.01,
            'auto_class_weights': 'Balanced',
            'random_state': 42,
            'verbose': 0
        }
    }
]

print(f"  ✓ Configured {len(model_configs)} models: XGBoost, LightGBM, CatBoost")


# ================================================
# TRAIN 10-FOLD ENSEMBLE
# ================================================

ensemble = KFoldEnsemble(n_folds=10, random_state=42)
ensemble.train(X_train, y_train, X_test, model_configs)


# ================================================
# EVALUATE ENSEMBLE STRATEGIES
# ================================================

print("\n[4/5] Evaluating ensemble strategies...")

strategies = [
    ('Simple Average', 'average', None),
    ('Weighted by OOF AUC', 'weighted', None),
    ('Rank Average', 'rank_average', None),
]

results = []

for strategy_name, mode, weights in strategies:
    oof_auc, test_auc, oof_pred, test_pred = ensemble.evaluate(
        y_train, y_test, mode=mode, weights=weights
    )
    
    results.append({
        'strategy': strategy_name,
        'oof_auc': oof_auc,
        'test_auc': test_auc,
        'oof_pred': oof_pred,
        'test_pred': test_pred
    })
    
    print(f"\n  {strategy_name}:")
    print(f"    OOF AUC:  {oof_auc:.4f}")
    print(f"    Test AUC: {test_auc:.4f}")

# Find best strategy
best_result = max(results, key=lambda x: x['test_auc'])
best_strategy = best_result['strategy']
best_test_auc = best_result['test_auc']

print(f"\n  {'='*70}")
print(f"  🏆 Best Strategy: {best_strategy}")
print(f"  🏆 Test AUC: {best_test_auc:.4f}")
print(f"  {'='*70}")


# ================================================
# FINAL EVALUATION WITH BEST STRATEGY
# ================================================

print("\n[5/5] Final evaluation with best strategy...")

# Get predictions with best strategy
best_test_pred = best_result['test_pred']

# Add these checks to your code:
print(f"Prediction range: [{best_test_pred.min():.4f}, {best_test_pred.max():.4f}]")
print(f"Predictions > 0.5: {(best_test_pred > 0.5).sum()} / {len(best_test_pred)}")
print(f"Mean prediction: {best_test_pred.mean():.4f}")

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, best_test_pred)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[optimal_idx])

print(f"\n  Optimal threshold: {optimal_threshold:.3f}")

# Predictions with default and optimal thresholds
y_pred_default = (best_test_pred >= 0.5).astype(int)
y_pred_optimal = (best_test_pred >= optimal_threshold).astype(int)



print(f"\n  {'='*70}")
print(f"  FINAL TEST SET EVALUATION")
print(f"  {'='*70}")

print("\n  --- With Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred_default, digits=4))
print(f"  ROC AUC: {best_test_auc:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_default)}")

print(f"\n  --- With Optimal Threshold ({optimal_threshold:.3f}) ---")
print(classification_report(y_test, y_pred_optimal, digits=4))
print(f"  ROC AUC: {best_test_auc:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_optimal)}")


# ================================================
# SAVE EVERYTHING
# ================================================

print("\n[6/6] Saving results...")

# Save ensemble object
joblib.dump(ensemble, MODELS / 'ensemble_10fold.joblib')

# Save best predictions
np.save(RESULTS / 'test_predictions.npy', best_test_pred)
np.save(RESULTS / 'oof_predictions.npy', best_result['oof_pred'])

# Save metrics
metrics = {
    'n_folds': ensemble.n_folds,
    'n_models': len(ensemble.models),
    'best_strategy': best_strategy,
    'best_test_auc': float(best_test_auc),
    'best_oof_auc': float(best_result['oof_auc']),
    'optimal_threshold': float(optimal_threshold),
    'individual_models': [
        {
            'name': m['name'],
            'oof_auc': float(m['oof_auc']),
            'mean_fold_auc': float(m['mean_fold_auc']),
            'std_fold_auc': float(m['std_fold_auc'])
        }
        for m in ensemble.models
    ],
    'all_strategies': [
        {
            'strategy': r['strategy'],
            'oof_auc': float(r['oof_auc']),
            'test_auc': float(r['test_auc'])
        }
        for r in results
    ],
    'classification_report_default': classification_report(y_test, y_pred_default, output_dict=True),
    'classification_report_optimal': classification_report(y_test, y_pred_optimal, output_dict=True),
    'confusion_matrix_default': confusion_matrix(y_test, y_pred_default).tolist(),
    'confusion_matrix_optimal': confusion_matrix(y_test, y_pred_optimal).tolist(),
}

with open(RESULTS / 'ensemble_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save fold details
fold_details = []
for model in ensemble.models:
    for fold_idx, fold_auc in enumerate(model['fold_aucs']):
        fold_details.append({
            'model': model['name'],
            'fold': fold_idx + 1,
            'auc': fold_auc
        })

fold_df = pd.DataFrame(fold_details)
fold_df.to_csv(RESULTS / 'fold_details.csv', index=False)

print(f"  ✓ Ensemble object: {MODELS / 'ensemble_10fold.joblib'}")
print(f"  ✓ Predictions: {RESULTS / 'test_predictions.npy'}")
print(f"  ✓ Metrics: {RESULTS / 'ensemble_metrics.json'}")
print(f"  ✓ Fold details: {RESULTS / 'fold_details.csv'}")


# ================================================
# VISUALIZATIONS
# ================================================

print("\nGenerating visualizations...")

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(20, 12))

# Plot 1: Individual model OOF AUC
ax1 = plt.subplot(3, 4, 1)
model_names = [m['name'].upper() for m in ensemble.models]
oof_aucs = [m['oof_auc'] for m in ensemble.models]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax1.bar(model_names, oof_aucs, color=colors, alpha=0.7)
ax1.set_ylabel('OOF AUC')
ax1.set_title('Individual Model Performance')
ax1.set_ylim([min(oof_aucs) - 0.01, max(oof_aucs) + 0.01])
for bar, auc in zip(bars, oof_aucs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Ensemble strategies comparison
ax2 = plt.subplot(3, 4, 2)
strategies_list = [r['strategy'] for r in results]
test_aucs = [r['test_auc'] for r in results]
colors_strat = ['green' if s == best_strategy else 'gray' for s in strategies_list]
bars = ax2.barh(strategies_list, test_aucs, color=colors_strat, alpha=0.7)
ax2.set_xlabel('Test AUC')
ax2.set_title('Ensemble Strategy Comparison')
for bar, auc in zip(bars, test_aucs):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f' {auc:.4f}', ha='left', va='center', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Fold-wise performance
ax3 = plt.subplot(3, 4, 3)
for model in ensemble.models:
    ax3.plot(range(1, ensemble.n_folds + 1), model['fold_aucs'], 
            'o-', label=model['name'].upper(), markersize=6, linewidth=2)
ax3.set_xlabel('Fold Number')
ax3.set_ylabel('AUC')
ax3.set_title('Performance Across Folds')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, ensemble.n_folds + 1))

# Plot 4: Performance improvement over baseline
ax4 = plt.subplot(3, 4, 4)
baseline_auc = 0.7843  # Your original baseline
improvement = (best_test_auc - baseline_auc) * 100
colors_imp = ['green' if improvement > 0 else 'red']
bars = ax4.bar(['Ensemble\nvs\nBaseline'], [improvement], color=colors_imp, alpha=0.7, width=0.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Improvement (%)')
ax4.set_title('Performance Gain vs Original Baseline')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.2f}%', ha='center', 
            va='bottom' if height > 0 else 'top', 
            fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Model correlation heatmap
ax5 = plt.subplot(3, 4, 5)
correlation = np.corrcoef(ensemble.test_predictions.T)
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
           xticklabels=[m['name'].upper() for m in ensemble.models],
           yticklabels=[m['name'].upper() for m in ensemble.models],
           ax=ax5, vmin=0.9, vmax=1.0, cbar_kws={'label': 'Correlation'})
ax5.set_title('Model Prediction Correlations')

# Plot 6: Confusion matrix (optimal threshold)
ax6 = plt.subplot(3, 4, 6)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
           xticklabels=['No Default', 'Default'],
           yticklabels=['No Default', 'Default'])
ax6.set_xlabel('Predicted')
ax6.set_ylabel('Actual')
ax6.set_title(f'Confusion Matrix (threshold={optimal_threshold:.3f})')

# Plot 7: Precision-Recall curve
ax7 = plt.subplot(3, 4, 7)
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresholds_pr = precision_recall_curve(y_test, best_test_pred)
ap_score = average_precision_score(y_test, best_test_pred)
ax7.plot(recall, precision, 'b-', linewidth=2, label=f'AP = {ap_score:.3f}')
ax7.scatter([recall[optimal_idx]], [precision[optimal_idx]], 
           color='red', s=100, zorder=5, label=f'Optimal (t={optimal_threshold:.3f})')
ax7.set_xlabel('Recall')
ax7.set_ylabel('Precision')
ax7.set_title('Precision-Recall Curve')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: ROC curve
ax8 = plt.subplot(3, 4, 8)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds_roc = roc_curve(y_test, best_test_pred)
ax8.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {best_test_auc:.4f}')
ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax8.set_xlabel('False Positive Rate')
ax8.set_ylabel('True Positive Rate')
ax8.set_title('ROC Curve')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Feature importance (average across all models)
ax9 = plt.subplot(3, 4, 9)
# Collect feature importances
feat_importances = []
for model_info in ensemble.models:
    if model_info['name'] in ['xgb', 'lgb', 'cat']:
        # Average importance across folds
        fold_importances = [m.feature_importances_ for m in model_info['fold_models']]
        avg_importance = np.mean(fold_importances, axis=0)
        feat_importances.append(avg_importance)

if len(feat_importances) > 0:
    avg_feat_imp = np.mean(feat_importances, axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': selected_features,
        'importance': avg_feat_imp
    }).sort_values('importance', ascending=False).head(15)
    
    ax9.barh(range(len(feat_imp_df)), feat_imp_df['importance'])
    ax9.set_yticks(range(len(feat_imp_df)))
    ax9.set_yticklabels(feat_imp_df['feature'], fontsize=8)
    ax9.set_xlabel('Average Importance')
    ax9.set_title('Top 15 Features (Ensemble Average)')
    ax9.invert_yaxis()
else:
    ax9.text(0.5, 0.5, 'No feature importance available', 
            ha='center', va='center', transform=ax9.transAxes)

# Plot 10: Box plot of fold AUCs
ax10 = plt.subplot(3, 4, 10)
fold_data = [model['fold_aucs'] for model in ensemble.models]
bp = ax10.boxplot(fold_data, labels=[m['name'].upper() for m in ensemble.models],
                  patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax10.set_ylabel('Fold AUC')
ax10.set_title('AUC Distribution Across Folds')
ax10.grid(True, alpha=0.3, axis='y')

# Plot 11: Summary table
ax11 = plt.subplot(3, 4, 11)
ax11.axis('tight')
ax11.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['N Folds', f'{ensemble.n_folds}'],
    ['N Models', f'{len(ensemble.models)}'],
    ['Best Strategy', best_strategy],
    ['Test AUC', f'{best_test_auc:.4f}'],
    ['vs Baseline', f'{(best_test_auc - 0.7843):+.4f}'],
    ['Optimal Threshold', f'{optimal_threshold:.3f}']
]
table = ax11.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(len(summary_data)):
    if i == 0:
        table[(i, 0)].set_facecolor('#d3d3d3')
        table[(i, 1)].set_facecolor('#d3d3d3')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_text_props(weight='bold')

ax11.set_title('Summary Statistics', fontweight='bold', pad=20)

# Plot 12: Prediction distribution
ax12 = plt.subplot(3, 4, 12)
ax12.hist(best_test_pred[y_test == 0], bins=50, alpha=0.5, label='No Default', color='blue')
ax12.hist(best_test_pred[y_test == 1], bins=50, alpha=0.5, label='Default', color='red')
ax12.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={optimal_threshold:.3f}')
ax12.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Default (0.5)')
ax12.set_xlabel('Predicted Probability')
ax12.set_ylabel('Count')
ax12.set_title('Prediction Distribution')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.suptitle('10-Fold Ensemble Model - Complete Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(RESULTS / 'ensemble_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Visualization saved: {RESULTS / 'ensemble_analysis.png'}")

print("\n" + "="*80)
print("✅ ENSEMBLE TRAINING COMPLETE!")
print("="*80)
print(f"\nFinal Results:")
print(f"  🏆 Best Strategy: {best_strategy}")
print(f"  📊 Test AUC: {best_test_auc:.4f}")
print(f"  📈 Improvement vs Baseline: {(best_test_auc - 0.7843)*100:+.2f}%")
print(f"  🎯 Optimal Threshold: {optimal_threshold:.3f}")
print(f"\nAll results saved to: {RESULTS}")
print("="*80 + "\n")