from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import time
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# ================================================
# DIR SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_for_original_and_engineered_features" / "shap_xgb_model(after XAI)"
MODELS = ROOT / "models_for_original_and_engineered_features"

RESULTS.mkdir(parents=True, exist_ok=True)

print("="*60)
print("SHAP ANALYSIS")
print("="*60)

# ================================================
# LOAD MODEL & DATA
# ================================================
print("\nLoading model and data...")
# model = CatBoostClassifier()
# model.load_model(str(MODELS / "catboost_model.cbm"))

model = joblib.load(MODELS / "xgb_model.joblib")

df = pd.read_csv(ROOT / 'data_preprocessing/data/processed/preprocessed_data_with_features.csv')

y = df['default'].astype(int)
X = df.drop(columns=['default'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"   Model loaded")
# Test 6000 samples that the model was not trained on 
print(f"   Test set size: {len(X_test)}")

# ================================================
# COMPUTE SHAP VALUES
# ================================================
print("\nComputing SHAP values...")

start_time = time.time()

# Use TreeExplainer for XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

computation_time = time.time() - start_time

print(f"   SHAP computation completed in {computation_time:.2f} seconds")
print(f"   SHAP values shape: {shap_values.shape}")

# ================================================
# SHAP VISUALIZATIONS
# ================================================
print("\nGenerating SHAP visualizations...")

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Summary Plot (Global Feature Importance with Values)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False, max_display=20)
plt.title('SHAP Summary Plot: Feature Importance & Impact', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_summary_plot.png")

# 2. Bar Plot (Mean Absolute SHAP Values)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS / 'shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_feature_importance_bar.png")

# 3. Waterfall Plot (Local Explanation - First Instance)
plt.figure(figsize=(12, 10))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns.tolist()
    ),
    max_display=15,
    show=False
)
plt.title('SHAP Waterfall Plot: Individual Prediction Explanation (Sample 1)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS / 'shap_waterfall_sample_0.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: shap_waterfall_sample_0.png")

# 4. Multiple Waterfall Plots
for idx in [1, 5, 10]:
    plt.figure(figsize=(12, 10))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=15,
        show=False
    )
    plt.tight_layout()
    plt.savefig(RESULTS / f'shap_waterfall_sample_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()



# # 5. Force Plot (save as image)
# plt.figure(figsize=(20, 3))
# shap.force_plot(
#     explainer.expected_value,
#     shap_values[0],
#     X_test.iloc[0],
#     matplotlib=True,
#     show=False
# )
# plt.tight_layout()
# plt.savefig(RESULTS / 'shap_force_plot_sample_0.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("   Saved: shap_force_plot_sample_0.png")

# ================================================
# CALCULATE SHAP METRICS FOR MCDM
# ================================================
print("\nCalculating SHAP metrics for MCDM...")

# Feature importance ranking
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (SHAP):")
print(feature_importance_df.head(10).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv(RESULTS / 'shap_feature_importance.csv', index=False)

# MCDM Criteria Metrics
shap_metrics = {
    'method': 'SHAP',
    
    # 1. Computational Efficiency
    'computation_time_seconds': float(computation_time),
    'avg_time_per_instance': float(computation_time / len(X_test)),
    
    # 2. Consistency (lower variance = more consistent)
    'consistency_variance': float(np.var(np.abs(shap_values))),
    'consistency_std': float(np.std(np.abs(shap_values))),
    
    # 3. Sparsity (how many features have significant impact)
    'avg_nonzero_features': float((np.abs(shap_values) > 0.01).sum(axis=1).mean()),
    'avg_features_above_threshold': float((np.abs(shap_values) > 0.05).sum(axis=1).mean()),
    
    # 4. Completeness
    'features_covered': int(X_test.shape[1]),
    'features_with_nonzero_importance': int((mean_abs_shap > 0).sum()),
    
    # 5. Stability
    'shap_values_std': float(np.std(shap_values)),
    'shap_values_mean': float(np.mean(np.abs(shap_values))),
    'shap_values_max': float(np.max(np.abs(shap_values))),
    'shap_values_min': float(np.min(np.abs(shap_values))),
    
    # 6. Feature importance data
    'top_10_features': feature_importance_df.head(10).to_dict('records'),
    
    # 7. Sample info
    'n_samples_explained': len(X_test)
}

# 1. Add dependence plots for top features (shows feature interactions)
for i, feature in enumerate(feature_importance_df.head(5)['feature']):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, 
        shap_values, 
        X_test,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS / f'shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: shap_dependence_{feature}.png")

# 2. Add decision plot (shows how features accumulate to final prediction)
plt.figure(figsize=(12, 8))
shap.decision_plot(
    explainer.expected_value, 
    shap_values[:100],  # First 100 samples
    X_test.iloc[:100],
    show=False
)
plt.title('SHAP Decision Plot: Feature Contribution Paths', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'shap_decision_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Add beeswarm plot (alternative to summary plot, sometimes clearer)
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test.values,
        feature_names=X_test.columns.tolist()
    ),
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(RESULTS / 'shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
plt.close()


# Add this to your code to analyze feature distribution:
shap_stats = {
    'total_features': 114,
    'features_with_nonzero_importance': (mean_abs_shap > 0).sum(),
    'top_10_cumulative_importance': feature_importance_df.head(10)['importance'].sum(),
    'top_20_cumulative_importance': feature_importance_df.head(20)['importance'].sum(),
    'concentration_ratio': feature_importance_df.head(10)['importance'].sum() / 
                          feature_importance_df['importance'].sum()
}

print(shap_stats)

# Save metrics
with open(RESULTS / 'shap_metrics.json', 'w') as f:
    json.dump(shap_metrics, f, indent=2)

# Save SHAP values for later analysis
np.save(RESULTS / 'shap_values.npy', shap_values)
np.save(RESULTS / 'shap_expected_value.npy', explainer.expected_value)

print(f"\nResults saved:")
print(f"   SHAP metrics: {RESULTS / 'shap_metrics.json'}")
print(f"   SHAP values: {RESULTS / 'shap_values.npy'}")
print(f"   Feature importance: {RESULTS / 'shap_feature_importance.csv'}")

# ================================================
# SUMMARY
# ================================================
print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE!")
print("="*60)
print(f"\nComputation Time: {computation_time:.2f} seconds")
print(f"Samples Explained: {len(X_test)}")
print(f"Features Analyzed: {X_test.shape[1]}")
print(f"\nNEXT STEP:")
print("   Run LIME analysis: python 04_lime_analysis.py")
print("\n" + "="*60)