# Data Cleaning with AI Support

## Student Information
- Name: CAINE IVAN R. BAUTISTA
- Course Year: BSCS 4
- Date: 2025-09-29

## Dataset
- Source: [Adult (UCI Dataset)](https://archive.ics.uci.edu/dataset/2/adult)
- Name: Adult (Census Income) 

## Issues found
- Missing values: 4,262
- Duplicates: 24
- Inconsistencies: 0

## Cleaning steps
1. Missing values: 0
2. Duplicates: 0
3. Inconsistencies: 0
4. Outliers: 14,374

## AI prompts used
- Prompt 1: "Generate a Pandas code that will perform the following steps for cleaning and reporting before and after cleaning on the dataset from UCI - Adult (Census Income):

    Load the raw dataset and perform exploratory checks 
        df.info()
        df.describe() 
        missing values
        duplicates
    Apply cleaning steps: 
        missing value handling
        duplicate removal
        standardize formats
        detect/treat outliers
    Include before and after snapshots of the following:
        shapes
        sample rows
        summary statistics
    Save cleaned dataset as data/cleaned_dataset.csv."
- Generated code: 
``` python
# # UCI Adult Census Income - Data Cleaning Pipeline
# 
# This notebook performs comprehensive data cleaning on the UCI Adult (Census Income) dataset.
# 
# **Steps:**
# 1. Load the raw dataset and perform exploratory checks
# 2. Apply cleaning steps (missing values, duplicates, formatting, outliers)
# 3. Compare before and after snapshots
# 4. Save cleaned dataset

# ## Setup and Imports

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# ## Configuration

# Column names (UCI Adult dataset doesn't include headers)
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# URL for the dataset
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# ---
# # STEP 1: Load Raw Dataset and Exploratory Checks

print("Loading raw dataset...")
df = pd.read_csv(DATA_URL, names=COLUMN_NAMES, skipinitialspace=True, na_values='?')
print(f"✓ Dataset loaded successfully!")

# ### 1.1 Dataset Shape

print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# ### 1.2 Dataset Info

df.info()

# ### 1.3 Statistical Summary

df.describe()

# ### 1.4 Missing Values Analysis

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Percentage': missing_percent
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if not missing_df.empty:
    display(missing_df)
    print(f"\nTotal missing values: {missing_values.sum():,}")
else:
    print("No missing values found.")

# ### 1.5 Duplicate Analysis

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")
if duplicates > 0:
    print(f"Percentage of duplicates: {(duplicates/len(df)*100):.2f}%")

# ### 1.6 Sample Rows (Before Cleaning)

df.head()

# ### 1.7 Save "Before" Statistics

before_stats = {
    'shape': df.shape,
    'missing_total': missing_values.sum(),
    'duplicates': duplicates,
    'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
}

print("Before Cleaning Snapshot:")
print(f"  Shape: {before_stats['shape']}")
print(f"  Missing Values: {before_stats['missing_total']:,}")
print(f"  Duplicates: {before_stats['duplicates']:,}")
print(f"  Memory: {before_stats['memory_usage']:.2f} MB")

# ---
# # STEP 2: Apply Cleaning Steps

# Create a copy for cleaning
df_clean = df.copy()

# ## 2.1 Handle Missing Values

print("Handling Missing Values...")

cols_with_missing = df_clean.columns[df_clean.isnull().any()].tolist()

if cols_with_missing:
    print(f"Columns with missing values: {cols_with_missing}\n")
    
    # Strategy: Mode imputation for categorical columns
    for col in cols_with_missing:
        if df_clean[col].dtype == 'object':
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"✓ Filled '{col}' with mode: '{mode_value}'")
        else:
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            print(f"✓ Filled '{col}' with median: {median_value}")
    
    print(f"\n✓ Missing values remaining: {df_clean.isnull().sum().sum()}")
else:
    print("No missing values to handle.")

# ## 2.2 Remove Duplicate Rows

print("Removing Duplicate Rows...")
before_dup = len(df_clean)
df_clean.drop_duplicates(inplace=True)
after_dup = len(df_clean)
removed_dup = before_dup - after_dup
print(f"✓ Removed {removed_dup:,} duplicate rows")

# ## 2.3 Standardize Formats

print("Standardizing Formats...\n")

# Strip whitespace from all string columns
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].str.strip()
    print(f"✓ Stripped whitespace from '{col}'")

# Standardize income labels
df_clean['income'] = df_clean['income'].str.replace('.', '', regex=False)
print(f"✓ Standardized 'income' labels (removed periods)")

# Convert categorical columns to category dtype
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category')
print(f"✓ Converted {len(categorical_cols)} columns to category dtype")

# ## 2.4 Detect and Treat Outliers

print("Detecting and Treating Outliers...\n")

# Identify numerical columns
numerical_cols = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']

outlier_info = {}
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    outlier_count = outliers_mask.sum()
    outlier_info[col] = outlier_count
    
    print(f"'{col}': {outlier_count:,} outliers detected")
    
    # Cap outliers using IQR method
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print(f"\n✓ Outliers capped using IQR method (1.5 × IQR)")

# ---
# # STEP 3: Before and After Comparison

# ## 3.1 Shapes Comparison

print("SHAPES COMPARISON")
print("-" * 60)
print(f"Before: {before_stats['shape'][0]:,} rows × {before_stats['shape'][1]} columns")
print(f"After:  {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"Rows removed: {before_stats['shape'][0] - df_clean.shape[0]:,}")

# ## 3.2 Sample Rows Comparison

print("\n=== BEFORE (First 3 rows) ===")
display(df.head(3))

print("\n=== AFTER (First 3 rows) ===")
display(df_clean.head(3))

# ## 3.3 Data Quality Metrics

print("DATA QUALITY METRICS")
print("-" * 60)
print(f"Missing Values:")
print(f"  Before: {before_stats['missing_total']:,}")
print(f"  After:  {df_clean.isnull().sum().sum()}")
print()
print(f"Duplicates:")
print(f"  Before: {before_stats['duplicates']:,}")
print(f"  After:  {df_clean.duplicated().sum()}")
print()
print(f"Memory Usage:")
print(f"  Before: {before_stats['memory_usage']:.2f} MB")
print(f"  After:  {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ## 3.4 Statistical Summary (After Cleaning)

df_clean.describe()

# ## 3.5 Categorical Distribution

print("Income Distribution:")
print(df_clean['income'].value_counts())

# Visual distribution
df_clean['income'].value_counts().plot(kind='bar', title='Income Distribution (After Cleaning)')

# ---
# # STEP 4: Save Cleaned Dataset

# Create data directory if it doesn't exist
output_dir = Path('data')
output_dir.mkdir(exist_ok=True)

# Save cleaned dataset
output_path = output_dir / 'cleaned_dataset.csv'
df_clean.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")

# ---
# # Summary Report

print("=" * 60)
print("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"✓ Rows: {before_stats['shape'][0]:,} → {df_clean.shape[0]:,}")
print(f"✓ Missing values: {before_stats['missing_total']:,} → 0")
print(f"✓ Duplicates removed: {before_stats['duplicates']:,}")
print(f"✓ Outliers treated: {sum(outlier_info.values()):,} values capped")
print(f"✓ Output: {output_path}")
print("=" * 60)

# ## Verify Cleaned Dataset

# Quick verification
print("\nFinal Dataset Info:")
df_clean.info()

print("\nCleaning process complete! The dataset is now ready for analysis and modeling.")
```

- Prompt 2: "make the indenfication of numerical columns dynamic"
- Generated Code: 
``` python
# Dynamically identify numerical columns (excluding 'education_num' which is ordinal)
numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude ordinal/categorical numeric columns
exclude_cols = ['education_num']  # Add other columns to exclude if needed
numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

print(f"Numerical columns identified: {numerical_cols}\n")

outlier_info = {}
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    outlier_count = outliers_mask.sum()
    outlier_info[col] = outlier_count
    
    print(f"'{col}': {outlier_count:,} outliers detected")
    
    # Cap outliers using IQR method
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print(f"\n✓ Outliers capped using IQR method (1.5 × IQR)")
```

- Prompt 3: "why does there are still duplicates after cleaning?"
- Generated Code: 
``` python
df_clean = df_clean.drop_duplicates()
df_clean.reset_index(drop=True, inplace=True)
```

## Results
- Rows before: 32,561
- Rows after: 32,508

Video: [Data Cleaning with AI video presentation](https://youtu.be/wuC-eFUbOWU)
