"""
SHAP-MCDM Integration for Credit Default Prediction
====================================================
Combines SHAP explainability with MCDM decision-making
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ================================================
# SETUP
# ================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_for_original_and_engineered_features"
SHAP_RESULTS = RESULTS / "shap_xgb_model(after XAI)"
MCDM_RESULTS = RESULTS / "shap_mcdm_integration"

MCDM_RESULTS.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SHAP-MCDM INTEGRATION")
print("="*80)

# ================================================
# LOAD DATA
# ================================================
print("\n[1/6] Loading SHAP results and data...")

# Load SHAP values
shap_values = np.load(SHAP_RESULTS / 'shap_values.npy')
expected_value = np.load(SHAP_RESULTS / 'shap_expected_value.npy')
feature_importance = pd.read_csv(SHAP_RESULTS / 'shap_feature_importance.csv')

# Load original data
df = pd.read_csv(ROOT / 'data_preprocessing/data/processed/preprocessed_data_with_features.csv')
y = df['default'].astype(int)
X = df.drop(columns=['default'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  ✓ Loaded SHAP values: {shap_values.shape}")
print(f"  ✓ Test samples: {len(X_test)}")
print(f"  ✓ Features: {X_test.shape[1]}")


# ================================================
# STAGE 1: FEATURE GROUPING INTO RISK CRITERIA
# ================================================
print("\n[2/6] Grouping features into risk criteria...")

# Define risk criteria based on feature semantics and SHAP importance
risk_criteria_mapping = {
    'Payment_Behavior_Risk': {
        'keywords': ['pay_', 'payment', 'delay'],
        'description': 'Recent payment history and delay patterns'
    },
    'Financial_Capacity_Risk': {
        'keywords': ['bill_amt', 'limit', 'utilization', 'balance'],
        'description': 'Credit utilization and financial capacity'
    },
    'Severe_Default_Risk': {
        'keywords': ['severe', 'consecutive'],
        'description': 'Patterns of severe delays and consecutive issues'
    },
    'Payment_Amount_Risk': {
        'keywords': ['pay_amt', 'pay_ratio'],
        'description': 'Payment amount patterns and ratios'
    },
    'Interaction_Risk': {
        'keywords': ['_x_', 'interaction'],
        'description': 'Complex interaction patterns between features'
    }
}

# Automatically assign features to criteria
def assign_features_to_criteria(feature_names, criteria_mapping):
    """Assign each feature to a risk criterion based on keywords"""
    criteria_features = {criterion: [] for criterion in criteria_mapping.keys()}
    unassigned = []
    
    for feature in feature_names:
        assigned = False
        for criterion, info in criteria_mapping.items():
            if any(keyword in feature.lower() for keyword in info['keywords']):
                criteria_features[criterion].append(feature)
                assigned = True
                break
        
        if not assigned:
            unassigned.append(feature)
    
    # Assign unassigned features to a catch-all category
    if unassigned:
        criteria_features['Other_Risk'] = unassigned
    
    return criteria_features

criteria_features = assign_features_to_criteria(X_test.columns.tolist(), risk_criteria_mapping)

# Display criteria distribution
print("\n  Risk Criteria Distribution:")
for criterion, features in criteria_features.items():
    if len(features) > 0:
        # Calculate total importance for this criterion
        criterion_importance = feature_importance[
            feature_importance['feature'].isin(features)
        ]['importance'].sum()
        
        print(f"    {criterion:30s}: {len(features):3d} features ({criterion_importance:.4f} importance)")

# Save criteria mapping
with open(MCDM_RESULTS / 'risk_criteria_mapping.json', 'w') as f:
    json.dump(criteria_features, f, indent=2)


# ================================================
# STAGE 2: AGGREGATE SHAP VALUES BY CRITERIA
# ================================================
print("\n[3/6] Aggregating SHAP values by risk criteria...")

def aggregate_shap_by_criteria(shap_values, X_test, criteria_features):
    """
    Aggregate SHAP values for features within each criterion
    Returns DataFrame with criteria scores for each sample
    """
    criteria_shap = {}
    
    for criterion, features in criteria_features.items():
        if len(features) == 0:
            continue
            
        # Get indices of features in this criterion
        feature_indices = [X_test.columns.get_loc(f) for f in features if f in X_test.columns]
        
        if feature_indices:
            # Sum SHAP values for this criterion
            criteria_shap[criterion] = shap_values[:, feature_indices].sum(axis=1)
    
    return pd.DataFrame(criteria_shap)

criteria_df = aggregate_shap_by_criteria(shap_values, X_test, criteria_features)

print(f"  ✓ Created criteria matrix: {criteria_df.shape}")
print(f"\n  Criteria Statistics:")
print(criteria_df.describe())

# Save criteria SHAP values
criteria_df.to_csv(MCDM_RESULTS / 'criteria_shap_values.csv', index=False)


# ================================================
# STAGE 3: MCDM METHODS
# ================================================
print("\n[4/6] Applying MCDM methods...")

class MCDM_Methods:
    """Collection of MCDM methods for risk scoring"""
    
    def __init__(self, criteria_df):
        self.criteria_df = criteria_df
        self.scaler = MinMaxScaler()
        # Normalize to [0, 1] - higher values = higher risk
        self.normalized = pd.DataFrame(
            self.scaler.fit_transform(criteria_df),
            columns=criteria_df.columns,
            index=criteria_df.index
        )
    
    def topsis(self, weights=None):
        """
        TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
        For risk: ideal = low risk (minimize), anti-ideal = high risk (maximize)
        """
        if weights is None:
            weights = np.ones(len(self.normalized.columns)) / len(self.normalized.columns)
        
        # Weighted normalized matrix
        weighted = self.normalized * weights
        
        # For risk assessment: lower is better
        ideal = weighted.min(axis=0)  # Best: lowest risk
        anti_ideal = weighted.max(axis=0)  # Worst: highest risk
        
        # Calculate distances
        dist_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        dist_anti_ideal = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
        
        # TOPSIS score: closeness to anti-ideal = risk score
        # Higher score = higher risk (closer to worst case)
        score = dist_anti_ideal / (dist_ideal + dist_anti_ideal + 1e-10)
        
        return score
    
    def saw(self, weights=None):
        """
        SAW: Simple Additive Weighting
        Weighted sum of normalized criteria
        """
        if weights is None:
            weights = np.ones(len(self.normalized.columns)) / len(self.normalized.columns)
        
        # For risk: higher normalized value = higher risk
        score = (self.normalized * weights).sum(axis=1)
        
        return score
    
    def wpm(self, weights=None):
        """
        WPM: Weighted Product Model
        Product of weighted criteria
        """
        if weights is None:
            weights = np.ones(len(self.normalized.columns)) / len(self.normalized.columns)
        
        # Add small epsilon to avoid zeros
        normalized_safe = self.normalized + 1e-10
        
        # For risk: take product of weighted values
        score = np.prod(normalized_safe ** weights, axis=1)
        
        return score
    
    def vikor(self, weights=None, v=0.5):
        """
        VIKOR: VIseKriterijumska Optimizacija I Kompromisno Resenje
        Multi-criteria optimization and compromise solution
        """
        if weights is None:
            weights = np.ones(len(self.normalized.columns)) / len(self.normalized.columns)
        
        weighted = self.normalized * weights
        
        # Best and worst values
        f_star = weighted.min(axis=0)
        f_minus = weighted.max(axis=0)
        
        # Calculate S (maximum group utility)
        S = ((weighted - f_star) / (f_minus - f_star + 1e-10)).sum(axis=1)
        
        # Calculate R (individual regret)
        R = ((weighted - f_star) / (f_minus - f_star + 1e-10)).max(axis=1)
        
        # VIKOR index Q
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()
        
        Q = v * (S - S_star) / (S_minus - S_star + 1e-10) + \
            (1 - v) * (R - R_star) / (R_minus - R_star + 1e-10)
        
        return Q


# ================================================
# CALCULATE MCDM WEIGHTS (IMPORTANCE-BASED)
# ================================================

def calculate_criteria_weights(criteria_features, feature_importance):
    """
    Calculate weights for each criterion based on SHAP feature importance
    """
    weights = {}
    
    for criterion, features in criteria_features.items():
        if len(features) == 0:
            continue
        
        # Sum importance of features in this criterion
        criterion_importance = feature_importance[
            feature_importance['feature'].isin(features)
        ]['importance'].sum()
        
        weights[criterion] = criterion_importance
    
    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return weights

criteria_weights = calculate_criteria_weights(criteria_features, feature_importance)

print("\n  Criteria Weights (based on SHAP importance):")
for criterion, weight in sorted(criteria_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {criterion:30s}: {weight:.4f} ({weight*100:.2f}%)")

# Convert to array in correct order
weight_array = np.array([criteria_weights.get(col, 0) for col in criteria_df.columns])


# ================================================
# APPLY ALL MCDM METHODS
# ================================================

mcdm = MCDM_Methods(criteria_df)

# Calculate scores with different methods
mcdm_scores = pd.DataFrame({
    'topsis_equal': mcdm.topsis(),
    'topsis_weighted': mcdm.topsis(weight_array),
    'saw_equal': mcdm.saw(),
    'saw_weighted': mcdm.saw(weight_array),
    'wpm_equal': mcdm.wpm(),
    'wpm_weighted': mcdm.wpm(weight_array),
    'vikor_equal': mcdm.vikor(),
    'vikor_weighted': mcdm.vikor(weight_array)
})

print(f"\n  ✓ Calculated {len(mcdm_scores.columns)} MCDM scores")


# ================================================
# STAGE 4: ENSEMBLE MCDM SCORE
# ================================================
print("\n[5/6] Creating ensemble MCDM score...")

# Normalize all MCDM scores to [0, 1]
scaler = MinMaxScaler()
mcdm_scores_normalized = pd.DataFrame(
    scaler.fit_transform(mcdm_scores),
    columns=mcdm_scores.columns,
    index=mcdm_scores.index
)

# Create ensemble score (weighted average of methods)
method_weights = {
    'topsis_weighted': 0.30,
    'saw_weighted': 0.25,
    'vikor_weighted': 0.25,
    'wpm_weighted': 0.20
}

ensemble_score = sum(
    mcdm_scores_normalized[method] * weight 
    for method, weight in method_weights.items()
)

mcdm_scores_normalized['ensemble'] = ensemble_score

print(f"\n  Ensemble Score Statistics:")
print(f"    Mean: {ensemble_score.mean():.4f}")
print(f"    Std:  {ensemble_score.std():.4f}")
print(f"    Min:  {ensemble_score.min():.4f}")
print(f"    Max:  {ensemble_score.max():.4f}")


# ================================================
# STAGE 5: COMBINE WITH MODEL PREDICTIONS
# ================================================
print("\n[6/6] Combining MCDM with model predictions...")

# Load model predictions
from xgboost import XGBClassifier
import joblib

model = joblib.load(ROOT / "models_for_original_and_engineered_features" / "xgb_model.joblib")
model_probs = model.predict_proba(X_test)[:, 1]

# Create final risk scores
# Hybrid approach: combine model prediction with MCDM consensus
alpha = 0.6  # Weight for model
beta = 0.4   # Weight for MCDM

final_risk_score = alpha * model_probs + beta * ensemble_score

# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'sample_id': range(len(X_test)),
    'true_label': y_test.values,
    'model_prob': model_probs,
    'mcdm_ensemble': ensemble_score,
    'final_risk_score': final_risk_score,
    'model_prediction': (model_probs > 0.5).astype(int),
    'mcdm_prediction': (ensemble_score > 0.5).astype(int),
    'final_prediction': (final_risk_score > 0.5).astype(int)
})

# Add MCDM method scores
for col in mcdm_scores_normalized.columns:
    results_df[f'mcdm_{col}'] = mcdm_scores_normalized[col].values

# Add criteria scores
for col in criteria_df.columns:
    results_df[f'criteria_{col}'] = criteria_df[col].values

# Save results
results_df.to_csv(MCDM_RESULTS / 'comprehensive_results.csv', index=False)

print(f"\n  ✓ Created comprehensive results: {results_df.shape}")


# ================================================
# EVALUATE PERFORMANCE
# ================================================
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

print("\n" + "="*80)
print("PERFORMANCE EVALUATION")
print("="*80)

print("\n1. Model Only:")
print(f"   AUC: {roc_auc_score(y_test, model_probs):.4f}")

print("\n2. MCDM Ensemble:")
print(f"   AUC: {roc_auc_score(y_test, ensemble_score):.4f}")

print("\n3. Final Hybrid (Model + MCDM):")
print(f"   AUC: {roc_auc_score(y_test, final_risk_score):.4f}")

print("\n4. Classification Report (Final Hybrid):")
print(classification_report(y_test, results_df['final_prediction'], digits=4))

print("\n5. Confusion Matrix (Final Hybrid):")
print(confusion_matrix(y_test, results_df['final_prediction']))


# ================================================
# VISUALIZATIONS
# ================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# 1. Criteria Weights
ax1 = plt.subplot(3, 4, 1)
weights_df = pd.DataFrame(list(criteria_weights.items()), columns=['Criterion', 'Weight'])
weights_df = weights_df.sort_values('Weight', ascending=True)
ax1.barh(weights_df['Criterion'], weights_df['Weight'])
ax1.set_xlabel('Weight')
ax1.set_title('Criteria Weights (SHAP-based)')
ax1.grid(True, alpha=0.3, axis='x')

# 2. MCDM Method Comparison
ax2 = plt.subplot(3, 4, 2)
method_aucs = {
    method: roc_auc_score(y_test, mcdm_scores_normalized[method])
    for method in mcdm_scores_normalized.columns
}
methods = list(method_aucs.keys())
aucs = list(method_aucs.values())
colors = ['green' if 'weighted' in m else 'gray' for m in methods]
ax2.barh(methods, aucs, color=colors, alpha=0.7)
ax2.set_xlabel('AUC')
ax2.set_title('MCDM Method Performance')
ax2.axvline(roc_auc_score(y_test, model_probs), color='red', linestyle='--', label='Model AUC')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# 3. Score Distribution
ax3 = plt.subplot(3, 4, 3)
ax3.hist(model_probs, bins=50, alpha=0.5, label='Model', density=True)
ax3.hist(ensemble_score, bins=50, alpha=0.5, label='MCDM', density=True)
ax3.hist(final_risk_score, bins=50, alpha=0.5, label='Hybrid', density=True)
ax3.set_xlabel('Risk Score')
ax3.set_ylabel('Density')
ax3.set_title('Risk Score Distributions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Correlation Heatmap
ax4 = plt.subplot(3, 4, 4)
score_corr = results_df[['model_prob', 'mcdm_ensemble', 'final_risk_score']].corr()
sns.heatmap(score_corr, annot=True, fmt='.3f', cmap='coolwarm', ax=ax4, vmin=0.5, vmax=1.0)
ax4.set_title('Score Correlations')

# 5. Criteria Heatmap (sample of 100)
ax5 = plt.subplot(3, 4, 5)
criteria_sample = criteria_df.iloc[:100].T
sns.heatmap(criteria_sample, cmap='RdYlGn_r', ax=ax5, cbar_kws={'label': 'Risk Contribution'})
ax5.set_xlabel('Sample')
ax5.set_ylabel('Risk Criterion')
ax5.set_title('Criteria Risk Profile (First 100 Samples)')

# 6. Model vs MCDM Scatter
ax6 = plt.subplot(3, 4, 6)
scatter = ax6.scatter(model_probs, ensemble_score, c=y_test, cmap='RdYlGn_r', alpha=0.5, s=10)
ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax6.set_xlabel('Model Probability')
ax6.set_ylabel('MCDM Score')
ax6.set_title('Model vs MCDM Agreement')
plt.colorbar(scatter, ax=ax6, label='True Label')
ax6.grid(True, alpha=0.3)

# 7. ROC Curves
from sklearn.metrics import roc_curve
ax7 = plt.subplot(3, 4, 7)
for name, scores in [('Model', model_probs), ('MCDM', ensemble_score), ('Hybrid', final_risk_score)]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    ax7.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax7.set_xlabel('False Positive Rate')
ax7.set_ylabel('True Positive Rate')
ax7.set_title('ROC Curves Comparison')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Criteria Box Plot
ax8 = plt.subplot(3, 4, 8)
criteria_df.boxplot(ax=ax8, rot=45)
ax8.set_ylabel('SHAP Value Contribution')
ax8.set_title('Criteria Distribution')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Method Agreement Matrix
ax9 = plt.subplot(3, 4, 9)
method_preds = mcdm_scores_normalized > 0.5
method_agreement = method_preds.corr()
sns.heatmap(method_agreement, annot=True, fmt='.2f', cmap='Blues', ax=ax9)
ax9.set_title('MCDM Method Agreement')

# 10. Feature Importance vs Criteria Weight
ax10 = plt.subplot(3, 4, 10)
top_features = feature_importance.head(15)
ax10.barh(range(len(top_features)), top_features['importance'])
ax10.set_yticks(range(len(top_features)))
ax10.set_yticklabels(top_features['feature'], fontsize=8)
ax10.set_xlabel('SHAP Importance')
ax10.set_title('Top 15 Features (SHAP)')
ax10.invert_yaxis()
ax10.grid(True, alpha=0.3, axis='x')

# 12. Cumulative Risk Distribution
ax12 = plt.subplot(3, 4, 12)
sorted_scores = np.sort(final_risk_score)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax12.plot(sorted_scores, cumulative, linewidth=2)
ax12.axvline(0.5, color='r', linestyle='--', label='Decision Threshold')
ax12.set_xlabel('Risk Score')
ax12.set_ylabel('Cumulative Probability')
ax12.set_title('Cumulative Risk Distribution')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.suptitle('SHAP-MCDM Integration: Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(MCDM_RESULTS / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: comprehensive_analysis.png")


# ================================================
# SAVE SUMMARY
# ================================================
# ================================================
# SAVE SUMMARY
# ================================================

# Fix: Reset index for y_test to match results_df
y_test_values = y_test.values  # Convert to numpy array

summary = {
    'criteria_weights': criteria_weights,
    'method_weights': method_weights,
    'performance': {
        'model_only_auc': float(roc_auc_score(y_test_values, model_probs)),
        'mcdm_ensemble_auc': float(roc_auc_score(y_test_values, ensemble_score)),
        'hybrid_auc': float(roc_auc_score(y_test_values, final_risk_score)),
        'model_only_accuracy': float((results_df['model_prediction'].values == y_test_values).mean()),
        'mcdm_ensemble_accuracy': float((results_df['mcdm_prediction'].values == y_test_values).mean()),
        'hybrid_accuracy': float((results_df['final_prediction'].values == y_test_values).mean())
    },
    'method_aucs': {k: float(v) for k, v in method_aucs.items()},
    'alpha_beta': {'alpha': alpha, 'beta': beta}
}

# ================================================
# VISUALIZATIONS (continued from plot 11)
# ================================================

# 11. Performance Comparison Table
ax11 = plt.subplot(3, 4, 11)
ax11.axis('off')
perf_data = [
    ['Approach', 'AUC', 'Accuracy'],
    ['Model Only', f"{roc_auc_score(y_test_values, model_probs):.4f}", 
     f"{(results_df['model_prediction'].values == y_test_values).mean():.4f}"],
    ['MCDM Ensemble', f"{roc_auc_score(y_test_values, ensemble_score):.4f}",
     f"{(results_df['mcdm_prediction'].values == y_test_values).mean():.4f}"],
    ['Hybrid', f"{roc_auc_score(y_test_values, final_risk_score):.4f}",
     f"{(results_df['final_prediction'].values == y_test_values).mean():.4f}"]
]
table = ax11.table(cellText=perf_data, cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(len(perf_data)):
    if i == 0:
        table[(i, 0)].set_facecolor('#d3d3d3')
        table[(i, 1)].set_facecolor('#d3d3d3')
        table[(i, 2)].set_facecolor('#d3d3d3')
ax11.set_title('Performance Summary', fontweight='bold', pad=20)

with open(MCDM_RESULTS / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ SHAP-MCDM INTEGRATION COMPLETE!")
print("="*80)
print(f"\nResults saved to: {MCDM_RESULTS}")
print(f"\nKey Files:")
print(f"  - comprehensive_results.csv: All scores and predictions")
print(f"  - risk_criteria_mapping.json: Feature-to-criteria mapping")
print(f"  - criteria_shap_values.csv: Aggregated SHAP by criteria")
print(f"  - comprehensive_analysis.png: Visualizations")
print(f"  - summary.json: Performance metrics")
print("="*80)