import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# to run
# python scripts/selective_feature_engineering.py

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ================================================
# LOAD DATA (with all engineered features)
# ================================================
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')

print(f"Dataset shape: {df.shape}")

# Prepare X and y
y = df['default'].astype(int)
X = df.drop(columns=['default', 'id'], errors='ignore')

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================================================
# METHOD 1: XGBoost Feature Importance
# ================================================
print("\n" + "="*60)
print("METHOD 1: XGBoost Feature Importance")
print("="*60)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

# Train a model to get feature importances
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_,
    'importance_type': 'gain'
}).sort_values('importance', ascending=False)

print("\nTop 30 features by XGBoost importance:")
print(importance_df.head(30).to_string(index=False))


# ================================================
# METHOD 2: Mutual Information
# ================================================
print("\n" + "="*60)
print("METHOD 2: Mutual Information Score")
print("="*60)

# Calculate mutual information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 30 features by Mutual Information:")
print(mi_df.head(30).to_string(index=False))

# ================================================
# METHOD 3: Combined Ranking
# ================================================
print("\n" + "="*60)
print("METHOD 3: Combined Ranking (XGBoost + MI)")
print("="*60)

# Normalize scores to 0-1 range
importance_df['importance_norm'] = (
    importance_df['importance'] / importance_df['importance'].max()
)
mi_df['mi_norm'] = mi_df['mi_score'] / mi_df['mi_score'].max()

# Merge and average
combined = importance_df[['feature', 'importance', 'importance_norm']].merge(
    mi_df[['feature', 'mi_score', 'mi_norm']], 
    on='feature'
)
combined['combined_score'] = (combined['importance_norm'] + combined['mi_norm']) / 2
combined = combined.sort_values('combined_score', ascending=False)

print("\nTop 30 features by combined score:")
print(combined.head(30)[['feature', 'combined_score', 'importance', 'mi_score']].to_string(index=False))


# ================================================
# FEATURE SELECTION RECOMMENDATIONS
# ================================================
print("\n" + "="*60)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*60)

# Categorize features
original_features = [
    'limit_bal', 'sex', 'education', 'marriage', 'age',
    'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
    'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
    'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
]

engineered_features = [f for f in X.columns if f not in original_features]

print(f"\nOriginal features: {len(original_features)}")
print(f"Engineered features: {len(engineered_features)}")

# Top engineered features
top_engineered = combined[combined['feature'].isin(engineered_features)].head(20)
print(f"\nTop 20 engineered features:")
print(top_engineered[['feature', 'combined_score']].to_string(index=False))

# ================================================
# SAVE SELECTED FEATURE SETS
# ================================================
# Top 20 features overall
top_20 = combined.head(20)['feature'].tolist()

# Top 30 features overall
top_30 = combined.head(30)['feature'].tolist()

# Top 40 features overall
top_40 = combined.head(40)['feature'].tolist()

# Hybrid: All original + top engineered
top_10_engineered = combined[combined['feature'].isin(engineered_features)].head(10)['feature'].tolist()
hybrid_features = original_features + top_10_engineered

# Save selection configs
feature_sets = {
    'top_20': top_20,
    'top_30': top_30,
    'top_40': top_40,
    'hybrid': hybrid_features,
    'all': X.columns.tolist()
}

import json
with open(RESULTS / 'feature_selections.json', 'w') as f:
    json.dump(feature_sets, f, indent=2)

print(f"\n✓ Saved feature selections to {RESULTS / 'feature_selections.json'}")

# ================================================
# CREATE REDUCED DATASETS
# ================================================
print("\n" + "="*60)
print("CREATING REDUCED DATASETS")
print("="*60)

for name, features in feature_sets.items():
    if name == 'all':
        continue
    
    # Create reduced dataset
    df_reduced = df[features + ['default']].copy()
    if 'id' in df.columns and 'id' not in features:
        df_reduced.insert(0, 'id', df['id'])
    
    output_path = f'./data_preprocessing/data/processed/preprocessed_data_{name}.csv'
    df_reduced.to_csv(output_path, index=False)
    print(f"✓ Saved {name} dataset ({len(features)} features) to {output_path}")

# ================================================
# VISUALIZATION
# ================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Plot top 20 features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost importance
top_20_importance = importance_df.head(20).sort_values('importance')
axes[0].barh(range(len(top_20_importance)), top_20_importance['importance'])
axes[0].set_yticks(range(len(top_20_importance)))
axes[0].set_yticklabels(top_20_importance['feature'])
axes[0].set_xlabel('Importance (Gain)')
axes[0].set_title('Top 20 Features - XGBoost Importance')
axes[0].grid(axis='x', alpha=0.3)

# Combined score
top_20_combined = combined.head(20).sort_values('combined_score')
axes[1].barh(range(len(top_20_combined)), top_20_combined['combined_score'])
axes[1].set_yticks(range(len(top_20_combined)))
axes[1].set_yticklabels(top_20_combined['feature'])
axes[1].set_xlabel('Combined Score (Normalized)')
axes[1].set_title('Top 20 Features - Combined Score')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {RESULTS / 'feature_importance.png'}")

# ================================================
# SUMMARY & RECOMMENDATIONS
# ================================================
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("""
Based on the analysis, here are your options:

1. **TOP_20** (20 features) - MOST AGGRESSIVE
   - Best for generalization
   - Fastest training & inference
   - Easiest SHAP interpretation
   - Risk: Might lose some signal

2. **TOP_30** (30 features) - BALANCED ⭐ RECOMMENDED
   - Good balance of performance and generalization
   - Still interpretable for SHAP
   - Lower overfitting risk than 70+ features

3. **TOP_40** (40 features) - CONSERVATIVE
   - Minimal information loss
   - More complex model
   - Good if test performance is priority

4. **HYBRID** (24 original + 10 best engineered) - SAFE
   - Keeps all original features
   - Adds only most valuable interactions
   - Good starting point if unsure

Try training with different feature sets and compare:
- Cross-validation scores
- Test set performance
- Training vs validation gap (overfitting indicator)

For SHAP analysis, I recommend TOP_30 or HYBRID.
""")

print("\n✓ Feature selection complete!")
print(f"✓ All results saved to {RESULTS}")