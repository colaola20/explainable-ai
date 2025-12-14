import pandas as pd
import numpy as np
from pathlib import Path

# to run
# python scripts/feature_engineering.py

ROOT = Path(__file__).resolve().parents[1]  # repo root

# ================================================
# LOAD DATA
# ================================================
# Adjust path as needed
try:
    df = pd.read_csv('./data_preprocessing/data/processed/preprocessed_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Preprocessed CSV not found. Check your path.")

print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# ================================================
# FEATURE ENGINEERING
# ================================================

# 1. CREDIT UTILIZATION FEATURES
# How much of their credit limit are they using?
for i in range(1, 7):
    df[f'util_rate_{i}'] = df[f'bill_amt{i}'] / (df['limit_bal'] + 1)  # +1 to avoid division by zero

# Average and max utilization
util_cols = [f'util_rate_{i}' for i in range(1, 7)]
df['util_rate_mean'] = df[util_cols].mean(axis=1)
df['util_rate_max'] = df[util_cols].max(axis=1)
df['util_rate_min'] = df[util_cols].min(axis=1)
df['util_rate_std'] = df[util_cols].std(axis=1)

print(f"✓ Added {len(util_cols) + 4} utilization features")

# 2. PAYMENT RATIO FEATURES
# What percentage of their bill did they pay?
for i in range(1, 7):
    df[f'pay_ratio_{i}'] = df[f'pay_amt{i}'] / (df[f'bill_amt{i}'] + 1)

# Average payment behavior
pay_ratio_cols = [f'pay_ratio_{i}' for i in range(1, 7)]
df['pay_ratio_mean'] = df[pay_ratio_cols].mean(axis=1)
df['pay_ratio_max'] = df[pay_ratio_cols].max(axis=1)
df['pay_ratio_min'] = df[pay_ratio_cols].min(axis=1)

# Count how many times they paid >= 100% of bill (good behavior)
df['full_payment_count'] = (df[pay_ratio_cols] >= 1.0).sum(axis=1)
# Count how many times they paid < 10% (bad behavior)
df['minimal_payment_count'] = (df[pay_ratio_cols] < 0.1).sum(axis=1)

print(f"✓ Added {len(pay_ratio_cols) + 5} payment ratio features")

# 3. BILL AMOUNT TRENDS
# Is their debt growing or shrinking?
bill_cols = [f'bill_amt{i}' for i in range(1, 7)]
df['bill_amt_mean'] = df[bill_cols].mean(axis=1)
df['bill_amt_max'] = df[bill_cols].max(axis=1)
df['bill_amt_std'] = df[bill_cols].std(axis=1)

# Trend: is most recent bill higher than 6 months ago?
df['bill_trend'] = df['bill_amt1'] - df['bill_amt6']
df['bill_growth_rate'] = (df['bill_amt1'] - df['bill_amt6']) / (df['bill_amt6'] + 1)

# Recent vs historical average
df['bill_recent_vs_avg'] = df['bill_amt1'] / (df['bill_amt_mean'] + 1)

print(f"✓ Added 6 bill trend features")

# 4. PAYMENT AMOUNT TRENDS
# Are they paying more or less over time?
pay_amt_cols = [f'pay_amt{i}' for i in range(1, 7)]
df['pay_amt_mean'] = df[pay_amt_cols].mean(axis=1)
df['pay_amt_max'] = df[pay_amt_cols].max(axis=1)
df['pay_amt_std'] = df[pay_amt_cols].std(axis=1)

# Payment trend
df['pay_trend'] = df['pay_amt1'] - df['pay_amt6']
df['pay_consistency'] = df['pay_amt_std'] / (df['pay_amt_mean'] + 1)  # Lower = more consistent

print(f"✓ Added 5 payment amount trend features")

# 5. DEBT-TO-LIMIT FEATURES
# This is like "debt-to-income" but using credit limit as proxy
df['debt_to_limit'] = df['bill_amt1'] / (df['limit_bal'] + 1)
df['debt_to_limit_max'] = df['bill_amt_max'] / (df['limit_bal'] + 1)

print(f"✓ Added 2 debt-to-limit features")

# 6. REPAYMENT STATUS FEATURES (pay_0, pay_2-6)
# These indicate months of delay (-1=pay duly, 0=revolving, 1=1 month delay, etc.)
pay_status_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']

# Average delay
df['pay_status_mean'] = df[pay_status_cols].mean(axis=1)
df['pay_status_max'] = df[pay_status_cols].max(axis=1)

# Count severe delays (>= 2 months late)
df['severe_delay_count'] = (df[pay_status_cols] >= 2).sum(axis=1)

print(f"✓ Added 5 repayment status features")

# 7. AGE-RELATED INTERACTIONS
# Younger people with high utilization might be riskier
df['age_utilization'] = df['age'] * df['util_rate_mean']
df['age_limit'] = df['age'] * df['limit_bal'] / 100000  # Scaled

print(f"✓ Added 2 age interaction features")

# 8. BEHAVIORAL CONSISTENCY
# Standard deviation across months (higher = more erratic behavior)
df['behavior_stability'] = (
    df['bill_amt_std'] / (df['bill_amt_mean'] + 1) + 
    df['pay_amt_std'] / (df['pay_amt_mean'] + 1)
)

print(f"✓ Added 1 behavioral consistency feature")

# 9. CRISIS INDICATORS
# Sudden changes that might indicate financial trouble
df['crisis_indicator'] = (
    (df['bill_trend'] > 0) & 
    (df['pay_trend'] < 0) & 
    (df['util_rate_mean'] > 0.7)
).astype(int)

print(f"✓ Added 1 crisis indicator feature")

# 10. Payment trend analysis
df['pay_improvement'] = df['pay_0'] - df['pay_6']  # Getting better or worse?
df['pay_volatility'] = df[['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].std(axis=1)
df['pay_deterioration'] = (df['pay_0'] > df['pay_2']).astype(int)  # Recent worsening

# 11. Consecutive delay patterns
df['consecutive_delays'] = 0
for i in range(len(df)):
    pay_cols = df.loc[i, ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']]
    # Count consecutive months with delay (>0)
    consecutive = 0
    max_consecutive = 0
    for val in pay_cols:
        if val > 0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    df.loc[i, 'consecutive_delays'] = max_consecutive

# 12. Recent vs historical payment behavior
df['recent_pay_mean'] = df[['pay_0', 'pay_2', 'pay_3']].mean(axis=1)  # Last 3 months
df['historical_pay_mean'] = df[['pay_4', 'pay_5', 'pay_6']].mean(axis=1)  # Older months
df['pay_trend_change'] = df['recent_pay_mean'] - df['historical_pay_mean']


# 13. Payment sufficiency
df['payment_coverage_1'] = df['pay_amt1'] / (df['bill_amt1'] + 1)  # Did they pay enough?
df['payment_coverage_2'] = df['pay_amt2'] / (df['bill_amt2'] + 1)
df['payment_coverage_mean'] = df[[f'pay_amt{i}' for i in range(1,7)]].sum(axis=1) / \
                                (df[[f'bill_amt{i}' for i in range(1,7)]].sum(axis=1) + 1)

# 14. Bill growth rate
df['bill_growth'] = (df['bill_amt1'] - df['bill_amt6']) / (df['bill_amt6'] + 1)
df['bill_acceleration'] = (df['bill_amt1'] - df['bill_amt2']) - (df['bill_amt2'] - df['bill_amt3'])

# 15 .Minimum payment patterns
df['pays_minimum_only'] = (df['pay_amt1'] < 0.1 * df['bill_amt1']).astype(int)
df['underpayment_count'] = sum((df[f'pay_amt{i}'] < 0.5 * df[f'bill_amt{i}']).astype(int) 
                                for i in range(1, 7))

# Utilization trends
df['util_increasing'] = (df['util_rate_1'] > df['util_rate_6']).astype(int)
df['util_volatility'] = df[['util_rate_1', 'util_rate_2', 'util_rate_3', 
                              'util_rate_4', 'util_rate_5', 'util_rate_6']].std(axis=1)
df['util_spike'] = df['util_rate_max'] - df['util_rate_mean']  # Maximum spike above average

# Maxed out credit
df['months_maxed_out'] = sum((df[f'util_rate_{i}'] > 0.9).astype(int) for i in range(1, 7))
df['recently_maxed_out'] = (df['util_rate_1'] > 0.9).astype(int)


# Credit limit relative to demographics
df['limit_per_age'] = df['limit_bal'] / (df['age'] + 1)
df['high_limit_young'] = ((df['limit_bal'] > df['limit_bal'].median()) & 
                          (df['age'] < 30)).astype(int)

# Education and payment behavior
df['education_risk'] = df['education'].map({1: 0, 2: 1, 3: 2, 4: 3})  # Higher = more risk
df['edu_payment_interaction'] = df['education_risk'] * df['pay_status_mean']

# Marriage and financial stress
df['married_high_util'] = ((df['marriage'] == 1) & 
                           (df['util_rate_mean'] > 0.7)).astype(int)

# Cash advance usage (negative bill amounts indicate overpayment or cash advance)
df['cash_advance_count'] = sum((df[f'bill_amt{i}'] < 0).astype(int) for i in range(1, 7))

# Payment-to-limit ratio
df['payment_capacity'] = df['pay_amt_mean'] / (df['limit_bal'] + 1)

# Financial strain score (composite)
df['financial_strain'] = (
    (df['util_rate_mean'] > 0.8).astype(int) * 2 +
    (df['pay_status_max'] >= 2).astype(int) * 3 +
    (df['bill_growth'] > 0.5).astype(int) * 1 +
    (df['payment_coverage_mean'] < 0.3).astype(int) * 2
)

# Interaction features for top predictors
df['pay_status_x_util'] = df['pay_status_max'] * df['util_rate_max']
df['recent_pay_x_consecutive'] = df['recent_pay_mean'] * df['consecutive_delays']
df['pay_x_limit'] = df['recent_pay_mean'] / (df['limit_bal'] + 1)

# Payment behavior interactions
df['pay_status_x_util'] = df['pay_status_max'] * df['util_rate_max']
df['severe_x_recent'] = df['severe_delay_count'] * df['recent_pay_mean']
df['consecutive_x_util'] = df['consecutive_delays'] * df['util_rate_max']

# Debt stress indicators
df['debt_x_delay'] = df['debt_to_limit_max'] * df['pay_status_max']
df['maxed_x_delay'] = df['months_maxed_out'] * df['consecutive_delays']

# Credit utilization interactions
df['util_x_limit'] = df['util_rate_max'] * df['limit_bal']
df['util_spike_x_pay'] = (df['util_rate_max'] - df['util_rate_mean']) * df['pay_status_mean']

# Age-based risk
df['age_x_debt'] = df['age'] * df['debt_to_limit_max']
df['young_high_util'] = ((df['age'] < 30) & (df['util_rate_max'] > 0.8)).astype(int)

# Payment coverage interactions
df['payment_ratio_x_util'] = (df['pay_amt_mean'] / (df['bill_amt_mean'] + 1)) * df['util_rate_max']

# Square and cube of most important features
df['recent_pay_mean_sq'] = df['recent_pay_mean'] ** 2
df['recent_pay_mean_cube'] = df['recent_pay_mean'] ** 3
df['pay_status_max_sq'] = df['pay_status_max'] ** 2

# Log transforms for skewed features
df['log_limit_bal'] = np.log1p(df['limit_bal'])
# df['log_bill_amt_mean'] = np.log1p(df['bill_amt_mean'])

# Interaction features between top performers
df['severe_recent_x_pay_max'] = df['severe_x_recent'] * df['pay_status_max']
df['consecutive_x_severe'] = df['consecutive_delays'] * df['severe_delay_count']
df['recent_pay_x_pay_max'] = df['recent_pay_mean'] * df['pay_status_max']

# Risk escalation features
df['pay_deterioration'] = df['pay_0'] - df['recent_pay_mean']  # Getting worse?
df['utilization_trend'] = df['util_rate_max'] - df['util_rate_1']  # Increasing usage?

# Behavioral consistency
df['payment_volatility'] = df['pay_amt_std'] / (df['pay_amt_mean'] + 1)
df['bill_stability'] = df['bill_amt_std'] / (df['bill_amt_mean'] + 1)

# More powerful interactions
df['consecutive_x_utilization'] = df['consecutive_delays'] * df['util_rate_max']
#df['payment_stress'] = df['bill_amt_mean'] / (df['pay_amt_mean'] + 1)




# ================================================
# HANDLE INFINITE/NAN VALUES
# ================================================
# Replace inf with large values and NaN with 0
df = df.replace([np.inf, -np.inf], np.nan)

# For ratio features, NaN usually means denominator was 0
# This often indicates no bill or no payment, which we can set to 0
ratio_features = [col for col in df.columns if 'ratio' in col or 'util' in col or 'growth' in col]
df[ratio_features] = df[ratio_features].fillna(0)

# For other features, fill with 0
df = df.fillna(0)

print(f"\n✓ Handled infinite and NaN values")

# ================================================
# SAVE ENGINEERED DATASET
# ================================================
output_path = './data_preprocessing/data/processed/preprocessed_data_with_features.csv'
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"Original features: {25}")
print(f"New features: {df.shape[1] - 25}")
print(f"Total features: {df.shape[1]}")
print(f"\nSaved to: {output_path}")

# ================================================
# FEATURE SUMMARY
# ================================================
new_features = [col for col in df.columns if col not in [
    'id', 'limit_bal', 'sex', 'education', 'marriage', 'age',
    'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
    'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
    'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
    'default'
]]

print(f"\nNew features created ({len(new_features)}):")
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# Show some statistics on key features
print(f"\n{'='*60}")
print("KEY FEATURE STATISTICS")
print(f"{'='*60}")
print("\nUtilization Rate (mean):")
print(df['util_rate_mean'].describe())

print("\nPayment Ratio (mean):")
print(df['pay_ratio_mean'].describe())

print("\nDebt to Limit:")
print(df['debt_to_limit'].describe())

print("\nSevere Delay Count:")
print(df['severe_delay_count'].value_counts().sort_index())

print("\n✓ Ready to train model with enhanced features!")