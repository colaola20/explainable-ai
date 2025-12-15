"""
PIPELINE: FAST HYPERPARAMETER SEARCH + BACKWARD FEATURE SELECTION
========================================================================
Run this script to:
1. Find good hyperparameters quickly (~30 min)
2. Select best 40-50 features (~2-3 hours)
3. Save results for Day 2 ensemble training

Usage: python scripts/hyperparameter_search_backward_feature_selection/feature_selection.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import json
from tqdm import tqdm
import time

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "feature_selection_results"
MODELS = ROOT / "feature_selection_models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("DAY 1 PIPELINE: SUCCESSIVE HALVING + BACKWARD FEATURE SELECTION")
print("="*80)

# ================================================
# LOAD DATA
# ================================================
print("\n[1/4] Loading data...")

df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X = df.drop(columns=['default'])

print(f"  Dataset shape: {X.shape}")
print(f"  Features: {X.shape[1]}")
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
# STEP 1: SUCCESSIVE HALVING HYPERPARAMETER SEARCH
# ================================================

class SuccessiveHalvingSearch:
    """
    Ultra-fast hyperparameter search.
    Starts with many configs, keeps best, increases training budget.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = None
        self.search_history = []
    
    def search(self, n_configs=64, min_estimators=50, max_estimators=1000):
        """Run successive halving search."""
        
        print("\n[2/4] Running Successive Halving Hyperparameter Search...")
        print(f"  Initial configs: {n_configs}")
        print(f"  Budget range: {min_estimators} → {max_estimators} estimators")
        
        start_time = time.time()
        
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        # Generate random configurations
        print("\n  Generating random configurations...")
        configs = []
        np.random.seed(42)
        
        for i in range(n_configs):
            config = {
                'max_depth': int(np.random.randint(3, 9)),
                'min_child_weight': int(np.random.randint(1, 21)),
                'gamma': float(np.random.uniform(0, 1)),
                'learning_rate': float(np.random.uniform(0.001, 0.3)),
                'subsample': float(np.random.uniform(0.5, 1.0)),
                'colsample_bytree': float(np.random.uniform(0.5, 1.0)),
                'colsample_bylevel': float(np.random.uniform(0.5, 1.0)),
                'reg_alpha': float(np.random.uniform(0, 10)),
                'reg_lambda': float(np.random.uniform(0, 10)),
                'scale_pos_weight': scale_pos_weight,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
            }
            configs.append(config)
        
        # Successive halving rounds
        current_configs = configs
        n_estimators = min_estimators
        round_num = 1
        
        while len(current_configs) > 1 and n_estimators <= max_estimators:
            print(f"\n  Round {round_num}:")
            print(f"    Testing {len(current_configs)} configs with {n_estimators} estimators")
            
            scores = []
            for i, config in enumerate(tqdm(current_configs, desc="    Training")):
                model = xgb.XGBClassifier(**config, n_estimators=n_estimators)
                model.fit(self.X_train, self.y_train, verbose=False)
                pred = model.predict_proba(self.X_val)[:, 1]
                score = roc_auc_score(self.y_val, pred)
                scores.append(score)
            
            # Keep top half
            sorted_indices = np.argsort(scores)[::-1]
            top_half = max(1, len(current_configs) // 2)
            current_configs = [current_configs[i] for i in sorted_indices[:top_half]]
            
            best_score = max(scores)
            print(f"    Best AUC: {best_score:.4f}")
            print(f"    Keeping top {top_half} configs")
            
            self.search_history.append({
                'round': round_num,
                'n_configs': len(current_configs) * 2,  # Before pruning
                'n_estimators': n_estimators,
                'best_auc': best_score
            })
            
            # Double budget
            n_estimators = min(n_estimators * 2, max_estimators)
            round_num += 1
        
        # Final training with full budget
        print(f"\n  Final training with {max_estimators} estimators...")
        self.best_params = current_configs[0]
        final_model = xgb.XGBClassifier(**self.best_params, n_estimators=max_estimators)
        final_model.fit(self.X_train, self.y_train, verbose=False)
        final_pred = final_model.predict_proba(self.X_val)[:, 1]
        final_score = roc_auc_score(self.y_val, final_pred)
        
        elapsed = time.time() - start_time
        
        print(f"\n  ✓ Search complete in {elapsed/60:.1f} minutes")
        print(f"  Final validation AUC: {final_score:.4f}")
        print(f"\n  Best hyperparameters:")
        for param, value in self.best_params.items():
            if param not in ['objective', 'eval_metric', 'random_state', 'n_jobs', 'scale_pos_weight']:
                if isinstance(value, float):
                    print(f"    {param}: {value:.4f}")
                else:
                    print(f"    {param}: {value}")
        
        return self.best_params, final_score


print("\n" + "="*80)
print("STEP 1: SUCCESSIVE HALVING")
print("="*80)

halving_search = SuccessiveHalvingSearch(X_train_sub, y_train_sub, X_val, y_val)
best_params, baseline_auc = halving_search.search(
    n_configs=64,
    min_estimators=50,
    max_estimators=1000
)

# Save search history
search_df = pd.DataFrame(halving_search.search_history)
search_df.to_csv(RESULTS / 'halving_search_history.csv', index=False)


# ================================================
# STEP 2: BACKWARD FEATURE SELECTION
# ================================================

class BackwardFeatureSelection:
    """
    Greedy backward feature elimination.
    Removes features one at a time that hurt performance least.
    """
    
    def __init__(self, base_params, X_train, y_train, X_val, y_val):
        self.base_params = base_params
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.selected_features = None
        self.elimination_history = []
    
    def _evaluate_features(self, features):
        """Train model with given features and return validation AUC."""
        X_tr = self.X_train[features]
        X_v = self.X_val[features]
        
        model = xgb.XGBClassifier(
            **self.base_params,
            n_estimators=200,  # Use fewer estimators for speed during selection
        )
        model.fit(X_tr, self.y_train, verbose=False)
        pred = model.predict_proba(X_v)[:, 1]
        return roc_auc_score(self.y_val, pred)
    
    def fit(self, target_features=40, early_stopping_rounds=5):
        """
        Perform backward feature selection.
        
        Parameters:
        -----------
        target_features: Target number of features to keep
        early_stopping_rounds: Stop if no improvement for N rounds
        """
        
        print("\n[3/4] Running Backward Feature Selection...")
        print(f"  Starting features: {self.X_train.shape[1]}")
        print(f"  Target features: {target_features}")
        print(f"  Early stopping: {early_stopping_rounds} rounds")
        
        start_time = time.time()
        
        features = list(self.X_train.columns)
        
        # Baseline score
        print("\n  Calculating baseline score...")
        baseline_score = self._evaluate_features(features)
        print(f"  Baseline validation AUC: {baseline_score:.4f}")
        
        self.elimination_history.append({
            'features_remaining': len(features),
            'removed_feature': 'baseline',
            'validation_auc': baseline_score
        })
        
        best_score = baseline_score
        no_improvement_count = 0
        
        pbar = tqdm(total=len(features) - target_features, desc="  Removing features")
        
        while len(features) > target_features:
            scores = {}
            
            # Try removing each feature
            for feature in features:
                remaining = [f for f in features if f != feature]
                score = self._evaluate_features(remaining)
                scores[feature] = score
            
            # Find feature whose removal hurts least (or helps most)
            feature_to_remove = max(scores, key=scores.get)
            new_score = scores[feature_to_remove]
            
            # Update progress bar
            pbar.set_postfix({
                'features': len(features) - 1,
                'AUC': f'{new_score:.4f}',
                'change': f'{new_score - best_score:+.4f}'
            })
            pbar.update(1)
            
            # Check early stopping
            if new_score < best_score - 0.001:  # Degraded by more than 0.001
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_rounds:
                    pbar.close()
                    print(f"\n  Early stopping: Performance degraded for {early_stopping_rounds} consecutive rounds")
                    break
            else:
                no_improvement_count = 0
                if new_score > best_score:
                    best_score = new_score
            
            # Remove feature
            features.remove(feature_to_remove)
            
            self.elimination_history.append({
                'features_remaining': len(features),
                'removed_feature': feature_to_remove,
                'validation_auc': new_score
            })
        
        pbar.close()
        
        self.selected_features = features
        elapsed = time.time() - start_time
        
        print(f"\n  ✓ Selection complete in {elapsed/60:.1f} minutes")
        print(f"  Selected features: {len(features)}")
        print(f"  Final validation AUC: {best_score:.4f}")
        print(f"  Change from baseline: {best_score - baseline_score:+.4f}")
        
        return features


print("\n" + "="*80)
print("STEP 2: BACKWARD FEATURE SELECTION")
print("="*80)

selector = BackwardFeatureSelection(
    base_params=best_params,
    X_train=X_train_sub,
    y_train=y_train_sub,
    X_val=X_val,
    y_val=y_val
)

selected_features = selector.fit(
    target_features=40,
    early_stopping_rounds=5
)

# Save elimination history
elim_df = pd.DataFrame(selector.elimination_history)
elim_df.to_csv(RESULTS / 'feature_elimination_history.csv', index=False)


# ================================================
# STEP 3: EVALUATE FINAL MODEL
# ================================================

print("\n[4/4] Evaluating final model with selected features...")

# Train final model with selected features and full budget
X_train_selected = X_train_sub[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]

final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=1000,  # Full budget
    early_stopping_rounds=50
)

final_model.fit(
    X_train_selected, y_train_sub,
    eval_set=[(X_val_selected, y_val)],
    verbose=False
)

# Evaluate
y_pred_val = final_model.predict_proba(X_val_selected)[:, 1]
y_pred_test = final_model.predict_proba(X_test_selected)[:, 1]

val_auc = roc_auc_score(y_val, y_pred_val)
test_auc = roc_auc_score(y_test, y_pred_test)

print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Best iteration: {final_model.best_iteration}")


# ================================================
# STEP 4: SAVE EVERYTHING
# ================================================

print("\n[5/5] Saving results...")

# Save selected features
with open(RESULTS / 'selected_features.txt', 'w') as f:
    for feat in selected_features:
        f.write(feat + '\n')

# Save best params
with open(RESULTS / 'best_hyperparameters.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    params_serializable = {}
    for k, v in best_params.items():
        if isinstance(v, (np.integer, np.floating)):
            params_serializable[k] = float(v)
        else:
            params_serializable[k] = v
    json.dump(params_serializable, f, indent=2)

# Save model
joblib.dump(final_model, MODELS / 'feature_selected_model.joblib')

# Save evaluation metrics
metrics = {
    'baseline_auc': float(baseline_auc),
    'n_features_original': len(X.columns),
    'n_features_selected': len(selected_features),
    'validation_auc': float(val_auc),
    'test_auc': float(test_auc),
    'best_iteration': int(final_model.best_iteration),
    'selected_features': selected_features
}

with open(RESULTS / 'day1_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Create summary visualization data
summary = pd.DataFrame({
    'Metric': ['Original Features', 'Selected Features', 'Feature Reduction', 
               'Validation AUC', 'Test AUC', 'vs Baseline'],
    'Value': [
        len(X.columns),
        len(selected_features),
        f"{(1 - len(selected_features)/len(X.columns))*100:.1f}%",
        f"{val_auc:.4f}",
        f"{test_auc:.4f}",
        f"{test_auc - 0.7843:+.4f}"  # Compare to your original baseline
    ]
})

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(summary.to_string(index=False))

print("\n" + "="*80)
print("SAVED FILES")
print("="*80)
print(f"✓ Selected features: {RESULTS / 'selected_features.txt'}")
print(f"✓ Best hyperparameters: {RESULTS / 'best_hyperparameters.json'}")
print(f"✓ Model: {MODELS / 'feature_selected_model.joblib'}")
print(f"✓ Metrics: {RESULTS / 'day1_metrics.json'}")
print(f"✓ Search history: {RESULTS / 'halving_search_history.csv'}")
print(f"✓ Elimination history: {RESULTS / 'feature_elimination_history.csv'}")



# ================================================
# GENERATE VISUALIZATION
# ================================================

import matplotlib.pyplot as plt

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Successive Halving Progress
ax1 = axes[0, 0]
search_df_plot = pd.DataFrame(halving_search.search_history)
ax1.plot(search_df_plot['round'], search_df_plot['best_auc'], 'o-', 
         linewidth=2, markersize=8, color='blue')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Best Validation AUC', fontsize=12)
ax1.set_title('Successive Halving: Performance per Round', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add annotations
for idx, row in search_df_plot.iterrows():
    ax1.annotate(f"{row['n_configs']} configs\n{row['n_estimators']} trees", 
                xy=(row['round'], row['best_auc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.7)

# Plot 2: Feature Elimination Progress
ax2 = axes[0, 1]
elim_df_plot = pd.DataFrame(selector.elimination_history)
ax2.plot(elim_df_plot['features_remaining'], elim_df_plot['validation_auc'], 
         'o-', linewidth=2, markersize=6, color='green')
ax2.axhline(y=baseline_auc, color='r', linestyle='--', label='Baseline', linewidth=2)
ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('Validation AUC', fontsize=12)
ax2.set_title('Backward Selection: AUC vs Features', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Importance (Top 20)
ax3 = axes[1, 0]
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

ax3.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
ax3.set_yticks(range(len(feature_importance)))
ax3.set_yticklabels(feature_importance['feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_title('Top 20 Selected Features by Importance', fontsize=14, fontweight='bold')
ax3.invert_yaxis()

# Plot 4: Summary Comparison
ax4 = axes[1, 1]
comparison_data = {
    'Metric': ['Original\nFeatures', 'Selected\nFeatures', 'Test AUC\n(Original)', 'Test AUC\n(Selected)'],
    'Value': [len(X.columns), len(selected_features), 0.7843, test_auc]
}
colors = ['#ff7f0e', '#2ca02c', '#ff7f0e', '#2ca02c']
bars = ax4.bar(comparison_data['Metric'], comparison_data['Value'], color=colors, alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}' if height > 1 else f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('Value', fontsize=12)
ax4.set_title('Before vs After Feature Selection', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS / 'day1_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved: {RESULTS / 'day1_summary.png'}")

print("\n" + "="*80)
print("ALL COMPLETE! 🎉")
print("="*80)