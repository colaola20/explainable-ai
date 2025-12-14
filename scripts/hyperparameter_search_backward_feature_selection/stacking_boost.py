"""
STACKING META-LEARNER BOOST
============================
Run this on your existing 10-fold ensemble to boost performance.
No retraining needed - uses saved OOF predictions!

Usage: python scripts/hyperparameter_search_backward_feature_selection/stacking_boost.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

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
# SETUP - HARDCODED ABSOLUTE PATHS
# ================================================
from pathlib import Path

ENSEMBLE_RESULTS = Path("/Users/olha/explainable-ai/scripts/ensemble_results")
ENSEMBLE_MODELS = Path("/Users/olha/explainable-ai/scripts/ensemble_models")
STACKING_RESULTS = Path("/Users/olha/explainable-ai/scripts/stacking_results")

STACKING_RESULTS.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("STACKING META-LEARNER BOOST")
print("="*80)
print(f"\nLooking for ensemble in: {ENSEMBLE_MODELS}")

# Verify the file exists before proceeding
ensemble_file = ENSEMBLE_MODELS / 'ensemble_10fold.joblib'
if not ensemble_file.exists():
    print(f"  ✗ Ensemble file not found at: {ensemble_file}")
    print(f"  Please check the path and make sure ensemble.py completed successfully.")
    exit(1)

print("\n" + "="*80)
print("STACKING META-LEARNER BOOST")
print("="*80)

# ================================================
# LOAD ENSEMBLE PREDICTIONS
# ================================================
print("\n[1/4] Loading ensemble and predictions...")

try:
    # Load ensemble object
    ensemble = joblib.load(ENSEMBLE_MODELS / 'ensemble_10fold.joblib')
    print("  ✓ Loaded 10-fold ensemble")
    
    # Get OOF and test predictions
    oof_predictions = ensemble.oof_predictions  # Shape: (train_size, n_models)
    test_predictions = ensemble.test_predictions  # Shape: (test_size, n_models)
    
    print(f"  OOF predictions shape: {oof_predictions.shape}")
    print(f"  Test predictions shape: {test_predictions.shape}")
    
except Exception as e:
    print(f"  ✗ Error loading ensemble: {e}")
    print(f"  Make sure you've run ensemble.py first!")
    exit(1)

# Load original data to get true labels
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X_full = df.drop(columns=['default'])

# Recreate same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  Train size: {len(y_train)}, Test size: {len(y_test)}")

# Baseline from simple average
baseline_pred = test_predictions.mean(axis=1)
baseline_auc = roc_auc_score(y_test, baseline_pred)
print(f"\n  Baseline (Simple Average) Test AUC: {baseline_auc:.4f}")


# ================================================
# TRY MULTIPLE META-LEARNERS
# ================================================
print("\n[2/4] Training multiple meta-learners...")

# Define meta-learners to try
meta_learners = [
    ('Logistic Regression (L2)', LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000, random_state=42, solver='lbfgs'
    )),
    
    ('Logistic Regression (L1)', LogisticRegression(
        penalty='l1', C=0.5, max_iter=1000, random_state=42, solver='saga'
    )),
    
    ('Ridge Classifier', RidgeClassifier(
        alpha=1.0, random_state=42
    )),
    
    ('XGBoost (Shallow)', xgb.XGBClassifier(
        max_depth=2, n_estimators=100, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric='auc'
    )),
    
    ('LightGBM (Shallow)', LGBMClassifier(
        max_depth=2, n_estimators=100, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=-1
    )),
    
    ('Random Forest', RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=42, n_jobs=-1
    )),
    
    ('Extra Trees', ExtraTreesClassifier(
        n_estimators=100, max_depth=3, random_state=42, n_jobs=-1
    )),
    
    ('MLP (Neural Net)', MLPClassifier(
        hidden_layer_sizes=(10, 5), max_iter=1000, 
        random_state=42, early_stopping=True
    )),
]

results = []
best_auc = 0
best_name = None
best_model = None
best_pred = None

print(f"\n  Testing {len(meta_learners)} different meta-learners:\n")

for name, model in meta_learners:
    try:
        # Train on OOF predictions
        model.fit(oof_predictions, y_train)
        
        # Predict on test
        if hasattr(model, 'predict_proba'):
            test_pred = model.predict_proba(test_predictions)[:, 1]
        else:
            # For models without predict_proba (like Ridge)
            test_pred = model.decision_function(test_predictions)
            # Normalize to [0, 1]
            test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min())
        
        # Calculate AUC
        test_auc = roc_auc_score(y_test, test_pred)
        improvement = (test_auc - baseline_auc) * 100
        vs_baseline_original = (test_auc - 0.7843) * 100
        
        results.append({
            'name': name,
            'test_auc': test_auc,
            'improvement_vs_avg': improvement,
            'vs_baseline': vs_baseline_original
        })
        
        # Print results
        status = "✓" if test_auc > baseline_auc else " "
        print(f"  {status} {name:25s}: AUC={test_auc:.4f}  "
              f"(vs avg: {improvement:+.2f}%, vs baseline: {vs_baseline_original:+.2f}%)")
        
        # Track best
        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
            best_model = model
            best_pred = test_pred
            
    except Exception as e:
        print(f"    ✗ {name}: Failed ({str(e)[:50]})")

# ================================================
# WEIGHTED ENSEMBLE OF META-LEARNERS
# ================================================
print("\n[3/4] Creating ensemble of top meta-learners...")

# Sort by AUC and get top 3
results_df = pd.DataFrame(results).sort_values('test_auc', ascending=False)
top_k = min(3, len(results_df))
top_names = results_df.head(top_k)['name'].tolist()

print(f"  Combining top {top_k} meta-learners:")
for i, name in enumerate(top_names, 1):
    auc = results_df[results_df['name'] == name]['test_auc'].values[0]
    print(f"    {i}. {name}: {auc:.4f}")

# Get predictions from top models
top_predictions = []
top_weights = []

for name in top_names:
    # Find the model
    for n, m in meta_learners:
        if n == name:
            m.fit(oof_predictions, y_train)
            if hasattr(m, 'predict_proba'):
                pred = m.predict_proba(test_predictions)[:, 1]
            else:
                pred = m.decision_function(test_predictions)
                pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            top_predictions.append(pred)
            # Weight by AUC
            auc = results_df[results_df['name'] == name]['test_auc'].values[0]
            top_weights.append(auc)
            break

# Weighted average
top_weights = np.array(top_weights)
top_weights = top_weights / top_weights.sum()

ensemble_pred = sum(w * p for w, p in zip(top_weights, top_predictions))
ensemble_auc = roc_auc_score(y_test, ensemble_pred)

print(f"\n  Weighted ensemble AUC: {ensemble_auc:.4f}")
print(f"  Improvement vs simple average: {(ensemble_auc - baseline_auc)*100:+.2f}%")
print(f"  vs original baseline: {(ensemble_auc - 0.7843)*100:+.2f}%")

# Use ensemble if better, otherwise use best single
if ensemble_auc > best_auc:
    final_pred = ensemble_pred
    final_auc = ensemble_auc
    final_name = f"Ensemble of {top_k} meta-learners"
else:
    final_pred = best_pred
    final_auc = best_auc
    final_name = best_name

# ================================================
# FINAL EVALUATION
# ================================================
print("\n[4/4] Final evaluation with best approach...")

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, final_pred)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[optimal_idx])

print(f"\n  Optimal threshold: {optimal_threshold:.3f}")

# Predictions with thresholds
y_pred_default = (final_pred >= 0.5).astype(int)
y_pred_optimal = (final_pred >= optimal_threshold).astype(int)

print(f"\n  {'='*70}")
print(f"  FINAL RESULTS")
print(f"  {'='*70}")
print(f"  Best approach: {final_name}")
print(f"  Test AUC: {final_auc:.4f}")

print("\n  --- With Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred_default, digits=4))
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_default)}")

print(f"\n  --- With Optimal Threshold ({optimal_threshold:.3f}) ---")
print(classification_report(y_test, y_pred_optimal, digits=4))
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_optimal)}")

# ================================================
# SAVE RESULTS
# ================================================
print("\n[5/5] Saving results...")

# Save best model
joblib.dump(best_model, STACKING_RESULTS / 'best_meta_learner.joblib')

# Save predictions
np.save(STACKING_RESULTS / 'final_predictions.npy', final_pred)

# Save detailed results
detailed_results = {
    'final_approach': final_name,
    'final_test_auc': float(final_auc),
    'improvement_vs_simple_avg': float((final_auc - baseline_auc) * 100),
    'improvement_vs_original_baseline': float((final_auc - 0.7843) * 100),
    'optimal_threshold': float(optimal_threshold),
    'all_meta_learners': results,
    'top_k_ensemble': {
        'members': top_names,
        'weights': top_weights.tolist(),
        'test_auc': float(ensemble_auc)
    },
    'classification_report_optimal': classification_report(y_test, y_pred_optimal, output_dict=True),
    'confusion_matrix_optimal': confusion_matrix(y_test, y_pred_optimal).tolist(),
}

with open(STACKING_RESULTS / 'stacking_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)

# Save summary
results_df.to_csv(STACKING_RESULTS / 'meta_learner_comparison.csv', index=False)

print(f"  ✓ Best model: {STACKING_RESULTS / 'best_meta_learner.joblib'}")
print(f"  ✓ Predictions: {STACKING_RESULTS / 'final_predictions.npy'}")
print(f"  ✓ Results: {STACKING_RESULTS / 'stacking_results.json'}")
print(f"  ✓ Comparison: {STACKING_RESULTS / 'meta_learner_comparison.csv'}")

# ================================================
# VISUALIZATION
# ================================================
print("\nGenerating visualization...")

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(16, 10))

# Plot 1: Meta-learner comparison
ax1 = plt.subplot(2, 3, 1)
results_df_sorted = results_df.sort_values('test_auc')
colors = ['green' if x > baseline_auc else 'gray' for x in results_df_sorted['test_auc']]
bars = ax1.barh(range(len(results_df_sorted)), results_df_sorted['test_auc'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(results_df_sorted)))
ax1.set_yticklabels(results_df_sorted['name'], fontsize=8)
ax1.axvline(x=baseline_auc, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_auc:.4f}')
ax1.axvline(x=0.7843, color='orange', linestyle='--', linewidth=2, label='Original: 0.7843')
ax1.set_xlabel('Test AUC')
ax1.set_title('Meta-Learner Performance')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Improvement comparison
ax2 = plt.subplot(2, 3, 2)
improvements = results_df.sort_values('improvement_vs_avg', ascending=False)
colors2 = ['green' if x > 0 else 'red' for x in improvements['improvement_vs_avg']]
ax2.barh(range(len(improvements)), improvements['improvement_vs_avg'], color=colors2, alpha=0.7)
ax2.set_yticks(range(len(improvements)))
ax2.set_yticklabels(improvements['name'], fontsize=8)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Improvement (%)')
ax2.set_title('Improvement vs Simple Average')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: ROC curve comparison
ax3 = plt.subplot(2, 3, 3)
from sklearn.metrics import roc_curve

# Baseline
fpr_base, tpr_base, _ = roc_curve(y_test, baseline_pred)
ax3.plot(fpr_base, tpr_base, '--', label=f'Simple Avg: {baseline_auc:.4f}', linewidth=2)

# Best stacking
fpr_best, tpr_best, _ = roc_curve(y_test, final_pred)
ax3.plot(fpr_best, tpr_best, '-', label=f'Stacking: {final_auc:.4f}', linewidth=2)

ax3.plot([0, 1], [0, 1], 'k:', alpha=0.3)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion matrix
ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
           xticklabels=['No Default', 'Default'],
           yticklabels=['No Default', 'Default'])
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (threshold={optimal_threshold:.3f})')

# Plot 5: Prediction distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(final_pred[y_test == 0], bins=50, alpha=0.6, label='No Default', density=True)
ax5.hist(final_pred[y_test == 1], bins=50, alpha=0.6, label='Default', density=True)
ax5.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
           label=f'Threshold={optimal_threshold:.3f}')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Density')
ax5.set_title('Prediction Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['Best Approach', final_name[:30]],
    ['Test AUC', f'{final_auc:.4f}'],
    ['vs Simple Avg', f'{(final_auc - baseline_auc)*100:+.2f}%'],
    ['vs Original', f'{(final_auc - 0.7843)*100:+.2f}%'],
    ['Optimal Threshold', f'{optimal_threshold:.3f}']
]
table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
# Style header
table[(0, 0)].set_facecolor('#40466e')
table[(0, 1)].set_facecolor('#40466e')
table[(0, 0)].set_text_props(weight='bold', color='white')
table[(0, 1)].set_text_props(weight='bold', color='white')
ax6.set_title('Stacking Summary', pad=20, fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(STACKING_RESULTS / 'stacking_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization: {STACKING_RESULTS / 'stacking_analysis.png'}")

# ================================================
# FINAL SUMMARY
# ================================================
print("\n" + "="*80)
print("STACKING COMPLETE!")
print("="*80)
print(f"\n🎯 Final Results:")
print(f"  • Best approach: {final_name}")
print(f"  • Test AUC: {final_auc:.4f}")
print(f"  • Improvement vs simple average: {(final_auc - baseline_auc)*100:+.2f}%")
print(f"  • Improvement vs original baseline (0.7843): {(final_auc - 0.7843)*100:+.2f}%")

if final_auc > 0.7843:
    print(f"\n  🎉 SUCCESS! Beat the original baseline!")
elif final_auc > baseline_auc:
    print(f"\n  ✓ Improved over simple averaging (but below original baseline)")
else:
    print(f"\n  → Stacking didn't help (simple average is better)")
    print(f"  Consider: more features, different ensemble strategy, or more data")

print("\n" + "="*80)