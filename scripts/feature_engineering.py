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

# Recent vs historical payment status
df['pay_status_recent'] = df['pay_0']  # Most recent
df['pay_status_worst'] = df[pay_status_cols].max(axis=1)  # Worst ever

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