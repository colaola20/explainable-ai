"""
Data Preprocessing for "Give Me Some Credit" Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# ================================================
# SETUP
# ================================================
# Get the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Define paths based on your actual structure
DATA_RAW = SCRIPT_DIR / "data"  # ← YOUR FILE IS HERE!
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "preprocessing"

# Create output folders
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

print("="*60)
print("DATA PREPROCESSING: Give Me Some Credit Dataset")
print("="*60)

# ================================================
# STEP 1: LOAD RAW DATA
# ================================================
print("\n📥 Step 1: Loading raw data...")

# Your actual file location
csv_file = DATA_RAW / "giveMeSomeCredit.csv"

print(f"   Looking for file at: {csv_file}")

if not csv_file.exists():
    print(f"\n   ❌ ERROR: File not found!")
    print(f"   Expected: {csv_file}")
    print(f"\n   📁 Files in {DATA_RAW}:")
    if DATA_RAW.exists():
        for f in DATA_RAW.iterdir():
            print(f"      • {f.name}")
    exit(1)

# Load the CSV (skip index column)
try:
    df = pd.read_csv(csv_file, index_col=0)
    print(f"   ✅ File found and loaded!")
    print(f"   ✅ Shape: {df.shape}")
    print(f"   ✅ Rows: {df.shape[0]:,}")
    print(f"   ✅ Columns: {df.shape[1]}")
except Exception as e:
    print(f"\n   ❌ ERROR loading CSV:")
    print(f"      {str(e)}")
    exit(1)

# ================================================
# STEP 2: INITIAL EXPLORATION
# ================================================
print("\n🔍 Step 2: Initial data exploration...")

print("\n   Original column names:")
for i, col in enumerate(df.columns, 1):
    print(f"      {i}. {col}")

print("\n   Data types:")
print(df.dtypes)

print("\n   First 3 rows:")
print(df.head(3))

# ================================================
# STEP 3: CLEAN COLUMN NAMES
# ================================================
print("\n🧹 Step 3: Cleaning column names...")

# Standardize column names
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'\s+', '_', regex=True)
              .str.replace('[^a-z0-9_]', '', regex=True))

# Rename target to standard name
if 'seriousdlqin2yrs' in df.columns:
    df.rename(columns={'seriousdlqin2yrs': 'default'}, inplace=True)
    print(f"   ✅ Renamed 'seriousdlqin2yrs' → 'default'")
else:
    print(f"   ⚠️  Warning: Could not find 'seriousdlqin2yrs' column")
    print(f"   Available columns: {df.columns.tolist()}")

print(f"\n   Cleaned column names:")
for i, col in enumerate(df.columns, 1):
    print(f"      {i}. {col}")

# ================================================
# STEP 4: HANDLE MISSING VALUES
# ================================================
print("\n🔧 Step 4: Handling missing values...")

# Check for missing values
missing = df.isnull().sum()
total_missing = missing.sum()

print(f"\n   Total missing values: {total_missing:,}")

if total_missing > 0:
    print(f"\n   Columns with missing values:")
    for col, count in missing[missing > 0].items():
        pct = (count / len(df)) * 100
        print(f"      • {col}: {count:,} ({pct:.2f}%)")

# Fill MonthlyIncome with median
if 'monthlyincome' in df.columns:
    missing_income = df['monthlyincome'].isnull().sum()
    if missing_income > 0:
        median_income = df['monthlyincome'].median()
        df['monthlyincome'].fillna(median_income, inplace=True)
        print(f"\n   ✅ Filled {missing_income:,} missing 'monthlyincome' with median: ${median_income:,.2f}")

# Fill NumberOfDependents with 0
if 'numberofdependents' in df.columns:
    missing_deps = df['numberofdependents'].isnull().sum()
    if missing_deps > 0:
        df['numberofdependents'].fillna(0, inplace=True)
        print(f"   ✅ Filled {missing_deps:,} missing 'numberofdependents' with 0")

print(f"\n   Remaining missing values: {df.isnull().sum().sum()}")

# ================================================
# STEP 5: REMOVE DUPLICATES
# ================================================
print("\n🔄 Step 5: Removing duplicates...")

duplicates = df.duplicated().sum()
print(f"   Duplicates found: {duplicates:,}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"   ✅ Removed {duplicates:,} duplicate rows")

print(f"   Remaining rows: {df.shape[0]:,}")

# ================================================
# STEP 6: FIX DATA TYPES
# ================================================
print("\n🔧 Step 6: Fixing data types...")

# Convert target to integer
if 'default' in df.columns:
    df['default'] = df['default'].astype(int)
    print(f"   ✅ 'default' → int")

# Convert count columns to integers
count_columns = [col for col in df.columns if 'number' in col.lower()]
for col in count_columns:
    if col in df.columns:
        df[col] = df[col].astype(int)
        print(f"   ✅ '{col}' → int")

# ================================================
# STEP 7: HANDLE OUTLIERS
# ================================================
print("\n📊 Step 7: Handling outliers...")

rows_before = len(df)

# Remove invalid ages
if 'age' in df.columns:
    invalid_age = ((df['age'] < 18) | (df['age'] > 100)).sum()
    if invalid_age > 0:
        df = df[(df['age'] >= 18) & (df['age'] <= 100)]
        print(f"   ✅ Removed {invalid_age:,} rows with invalid age (<18 or >100)")

# Remove negative debt ratios
if 'debtratio' in df.columns:
    negative_debt = (df['debtratio'] < 0).sum()
    if negative_debt > 0:
        df = df[df['debtratio'] >= 0]
        print(f"   ✅ Removed {negative_debt:,} rows with negative debt ratio")

# Cap extreme utilization rates
if 'revolvingutilizationofunsecuredlines' in df.columns:
    extreme_util = (df['revolvingutilizationofunsecuredlines'] > 2).sum()
    if extreme_util > 0:
        df['revolvingutilizationofunsecuredlines'] = df['revolvingutilizationofunsecuredlines'].clip(upper=2)
        print(f"   ✅ Capped {extreme_util:,} extreme utilization rates at 200%")

rows_after = len(df)
rows_removed = rows_before - rows_after

print(f"\n   Rows before outlier removal: {rows_before:,}")
print(f"   Rows after outlier removal:  {rows_after:,}")
print(f"   Rows removed: {rows_removed:,}")

# ================================================
# STEP 8: FEATURE ENGINEERING
# ================================================
print("\n🔨 Step 8: Feature engineering...")

features_created = []

# 1. Total late payments
late_payment_cols = [col for col in df.columns if 'pastdue' in col.lower() or '90dayslate' in col.lower()]
if len(late_payment_cols) > 0:
    df['total_late_payments'] = df[late_payment_cols].sum(axis=1)
    features_created.append('total_late_payments')
    print(f"   ✅ Created 'total_late_payments'")

# 2. Has late payments (binary)
if 'total_late_payments' in df.columns:
    df['has_late_payments'] = (df['total_late_payments'] > 0).astype(int)
    features_created.append('has_late_payments')
    print(f"   ✅ Created 'has_late_payments'")

# 3. Income to debt ratio
if 'debtratio' in df.columns:
    df['income_to_debt'] = 1 / (df['debtratio'] + 0.001)
    features_created.append('income_to_debt')
    print(f"   ✅ Created 'income_to_debt'")

# 4. Age groups
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 30, 45, 60, 100], 
                             labels=[0, 1, 2, 3])
    df['age_group'] = df['age_group'].astype(int)
    features_created.append('age_group')
    print(f"   ✅ Created 'age_group' (0=young, 1=middle, 2=senior, 3=elderly)")

print(f"\n   Total new features created: {len(features_created)}")

# ================================================
# STEP 9: CLASS BALANCE CHECK
# ================================================
print("\n⚖️  Step 9: Checking class balance...")

if 'default' not in df.columns:
    print("   ❌ ERROR: 'default' column not found!")
    print(f"   Available columns: {df.columns.tolist()}")
    exit(1)

class_counts = df['default'].value_counts().sort_index()
class_pct = (class_counts / len(df) * 100).round(2)

print(f"\n   Class Distribution:")
print(f"      No Default (0): {class_counts[0]:,} ({class_pct[0]}%)")
print(f"      Default (1):    {class_counts[1]:,} ({class_pct[1]}%)")

imbalance_ratio = class_counts[0] / class_counts[1]
print(f"\n   Imbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"   ⚠️  Dataset is HIGHLY imbalanced!")
    print(f"   💡 Use scale_pos_weight={imbalance_ratio:.2f} in XGBoost")

# Visualize class distribution
plt.figure(figsize=(10, 6))
bars = plt.bar(['No Default', 'Default'], class_counts.values, 
               color=['green', 'red'], alpha=0.7, edgecolor='black')
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Default Status', fontsize=12)

# Add value labels on bars
for i, (bar, count, pct) in enumerate(zip(bars, class_counts.values, class_pct.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{count:,}\n({pct}%)', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(RESULTS / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n   ✅ Class distribution plot saved to: {RESULTS / 'class_distribution.png'}")

# ================================================
# STEP 10: CORRELATION ANALYSIS
# ================================================
print("\n📊 Step 10: Correlation analysis...")

# Calculate correlation with target
correlation_matrix = df.corr()
target_corr = correlation_matrix['default'].sort_values(ascending=False)

print(f"\n   Top 10 features most correlated with default:")
print("   " + "-"*55)
for i, (feature, corr) in enumerate(list(target_corr.items())[:11], 1):
    if feature != 'default':
        direction = "↑ Increases" if corr > 0 else "↓ Decreases"
        print(f"   {i:2}. {feature:35} {direction} risk ({corr:+.4f})")
print("   " + "-"*55)

# Save correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.2f',
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Correlation heatmap saved to: {RESULTS / 'correlation_matrix.png'}")

# ================================================
# STEP 11: SAVE PROCESSED DATA
# ================================================
print("\n💾 Step 11: Saving processed data...")

output_file = DATA_PROCESSED / "preprocessed_data.csv"
df.to_csv(output_file, index=False)
print(f"   ✅ Processed data saved to: {output_file}")
print(f"   📊 Shape: {df.shape}")

# Create preprocessing report
report = {
    'dataset': 'Give Me Some Credit',
    'source_file': str(csv_file),
    'output_file': str(output_file),
    'original_rows': rows_before,
    'final_rows': int(df.shape[0]),
    'rows_removed': rows_removed,
    'original_features': 10,
    'engineered_features': len(features_created),
    'final_features': int(df.shape[1] - 1),
    'total_columns': int(df.shape[1]),
    'target_column': 'default',
    'class_balance': {
        'no_default': int(class_counts[0]),
        'default': int(class_counts[1]),
        'default_rate': float(class_pct[1]),
        'imbalance_ratio': float(imbalance_ratio)
    },
    'top_5_correlations': {k: float(v) for k, v in list(target_corr.items())[1:6]},
    'features_created': features_created
}

report_file = RESULTS / 'preprocessing_report.json'
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)
print(f"   ✅ Report saved to: {report_file}")

# ================================================
# FINAL SUMMARY
# ================================================
print("\n" + "="*60)
print("✅ DATA PREPROCESSING COMPLETE!")
print("="*60)

print(f"\n📊 Dataset Summary:")
print(f"   • Original rows:    {rows_before:,}")
print(f"   • Final rows:       {df.shape[0]:,}")
print(f"   • Rows removed:     {rows_removed:,}")
print(f"   • Original features: 10")
print(f"   • New features:      {len(features_created)}")
print(f"   • Total features:    {df.shape[1] - 1}")
print(f"   • Default rate:      {class_pct[1]}%")
print(f"   • Imbalance ratio:   {imbalance_ratio:.2f}:1")

print(f"\n📁 Output Files:")
print(f"   • {output_file}")
print(f"   • {report_file}")
print(f"   • {RESULTS / 'class_distribution.png'}")
print(f"   • {RESULTS / 'correlation_matrix.png'}")

print(f"\n🎯 Next Steps:")
print(f"   1. Review the visualizations in: {RESULTS}")
print(f"   2. Check preprocessing_report.json for details")
print(f"   3. Update your model training script with:")
print(f"      scale_pos_weight={imbalance_ratio:.2f}")
print(f"   4. Run: python 02_model_training.py")

print("\n" + "="*60)