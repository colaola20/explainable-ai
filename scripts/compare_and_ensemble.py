from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from catboost import CatBoostClassifier

# to run
# python scripts/compare_and_ensemble.py

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_ensemble"
RESULTS.mkdir(parents=True, exist_ok=True)

# ================================================
# LOAD DATA
# ================================================
df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data_with_features.csv')

y = df['default'].astype(int)
X = df.drop(columns=['default'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================================================
# LOAD TRAINED MODELS
# ================================================
print("Loading trained models...")

xgb_model = joblib.load(ROOT / "models_for_original_and_engineered_features" / "xgb_model.joblib")
lgb_model = joblib.load(ROOT / "models_lightgbm" / "lgb_model.joblib")
cat_model = CatBoostClassifier()
cat_model.load_model(str(ROOT / "models_catboost" / "catboost_model.cbm"))

print("✓ All models loaded successfully")

# ================================================
# GET PREDICTIONS FROM EACH MODEL
# ================================================
print("\n" + "="*60)
print("Generating predictions from each model...")
print("="*60)

y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_proba_cat = cat_model.predict_proba(X_test)[:, 1]

# Individual model performance
xgb_auc = roc_auc_score(y_test, y_proba_xgb)
lgb_auc = roc_auc_score(y_test, y_proba_lgb)
cat_auc = roc_auc_score(y_test, y_proba_cat)

print(f"\nXGBoost Test AUC:  {xgb_auc:.4f}")
print(f"LightGBM Test AUC: {lgb_auc:.4f}")
print(f"CatBoost Test AUC: {cat_auc:.4f}")

# ================================================
# ENSEMBLE METHODS
# ================================================
print("\n" + "="*60)
print("Testing Ensemble Methods")
print("="*60)

# Method 1: Simple Average
y_proba_avg = (y_proba_xgb + y_proba_lgb + y_proba_cat) / 3
avg_auc = roc_auc_score(y_test, y_proba_avg)
print(f"\n1. Simple Average:        {avg_auc:.4f}")

# Method 2: Weighted Average (weight by validation performance)
# You can adjust these weights based on validation AUC
weights = np.array([xgb_auc, lgb_auc, cat_auc])
weights = weights / weights.sum()  # Normalize to sum to 1

y_proba_weighted = (
    weights[0] * y_proba_xgb +
    weights[1] * y_proba_lgb +
    weights[2] * y_proba_cat
)
weighted_auc = roc_auc_score(y_test, y_proba_weighted)
print(f"2. Weighted Average:      {weighted_auc:.4f}")
print(f"   Weights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

# Method 3: Rank Average (less sensitive to outliers)
from scipy.stats import rankdata
y_rank_xgb = rankdata(y_proba_xgb)
y_rank_lgb = rankdata(y_proba_lgb)
y_rank_cat = rankdata(y_proba_cat)

y_proba_rank = (y_rank_xgb + y_rank_lgb + y_rank_cat) / 3
rank_auc = roc_auc_score(y_test, y_proba_rank)
print(f"3. Rank Average:          {rank_auc:.4f}")

# Method 4: Best single model only (for comparison)
best_single_auc = max(xgb_auc, lgb_auc, cat_auc)
best_model = ['XGBoost', 'LightGBM', 'CatBoost'][np.argmax([xgb_auc, lgb_auc, cat_auc])]
print(f"4. Best Single Model:     {best_single_auc:.4f} ({best_model})")

# ================================================
# FIND BEST ENSEMBLE
# ================================================
ensemble_results = {
    'Simple Average': {'auc': avg_auc, 'predictions': y_proba_avg},
    'Weighted Average': {'auc': weighted_auc, 'predictions': y_proba_weighted},
    'Rank Average': {'auc': rank_auc, 'predictions': y_proba_rank},
}

best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['auc'])
best_name = best_ensemble[0]
best_auc = best_ensemble[1]['auc']
best_predictions = best_ensemble[1]['predictions']

print(f"\n{'='*60}")
print(f"BEST ENSEMBLE: {best_name}")
print(f"Test AUC: {best_auc:.4f}")
print(f"Improvement over best single model: {best_auc - best_single_auc:+.4f}")
print(f"{'='*60}")

# ================================================
# DETAILED EVALUATION OF BEST ENSEMBLE
# ================================================
print("\n" + "="*60)
print("DETAILED EVALUATION - BEST ENSEMBLE")
print("="*60)

# Find optimal threshold
from sklearn.metrics import precision_recall_curve

# Rank average returns ranks, not probabilities, so normalize to [0,1]
if best_name == 'Rank Average':
    best_predictions = (best_predictions - best_predictions.min()) / (best_predictions.max() - best_predictions.min())

precision, recall, thresholds = precision_recall_curve(y_test, best_predictions)
if thresholds.size > 0:
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])
else:
    optimal_threshold = 0.5

print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Predictions with default and optimal thresholds
y_pred_default = (best_predictions >= 0.5).astype(int)
y_pred_optimal = (best_predictions >= optimal_threshold).astype(int)

print("\n--- With Default Threshold (0.5) ---")
print(classification_report(y_test, y_pred_default))
print(f"ROC AUC: {best_auc:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_default))

print(f"\n--- With Optimal Threshold ({optimal_threshold:.3f}) ---")
print(classification_report(y_test, y_pred_optimal))
print(f"ROC AUC: {best_auc:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_optimal))

# ================================================
# PREDICTION CORRELATION ANALYSIS
# ================================================
print("\n" + "="*60)
print("MODEL PREDICTION CORRELATIONS")
print("="*60)

correlation_df = pd.DataFrame({
    'XGBoost': y_proba_xgb,
    'LightGBM': y_proba_lgb,
    'CatBoost': y_proba_cat
})

print("\nCorrelation matrix:")
print(correlation_df.corr().round(4))

print("\nInterpretation:")
print("- High correlation (>0.95): Models are very similar, ensemble may not help much")
print("- Medium correlation (0.85-0.95): Good diversity, ensemble should help")
print("- Low correlation (<0.85): High diversity, ensemble could help significantly")

# ================================================
# SAVE RESULTS
# ================================================
results_summary = {
    "individual_models": {
        "XGBoost": float(xgb_auc),
        "LightGBM": float(lgb_auc),
        "CatBoost": float(cat_auc),
    },
    "ensemble_methods": {
        "simple_average": float(avg_auc),
        "weighted_average": float(weighted_auc),
        "rank_average": float(rank_auc),
    },
    "best_ensemble": {
        "method": best_name,
        "test_auc": float(best_auc),
        "improvement": float(best_auc - best_single_auc),
        "optimal_threshold": float(optimal_threshold),
    },
    "model_correlations": correlation_df.corr().to_dict(),
}

with open(RESULTS / "ensemble_comparison.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Save comparison table
comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Simple Average', 'Weighted Average', 'Rank Average'],
    'Test_AUC': [xgb_auc, lgb_auc, cat_auc, avg_auc, weighted_auc, rank_auc],
    'Type': ['Single', 'Single', 'Single', 'Ensemble', 'Ensemble', 'Ensemble']
})
comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
comparison_df.to_csv(RESULTS / "model_comparison.csv", index=False)

print(f"\n{'='*60}")
print("SAVED FILES:")
print(f"{'='*60}")
print(f"✓ Comparison: {RESULTS / 'model_comparison.csv'}")
print(f"✓ Detailed results: {RESULTS / 'ensemble_comparison.json'}")
print("\n" + "="*60)
print("Analysis complete!")
print("="*60)