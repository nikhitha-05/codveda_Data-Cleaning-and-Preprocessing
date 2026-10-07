# Import Libraries
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# ============================================================
# STEP 1 : Load Dataset
# ============================================================

df = pd.read_csv("data.csv")     # Replace with your dataset

print("="*60)
print("Original Dataset")
print("="*60)

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nInformation")
print(df.info())

print("\nStatistical Summary")
print(df.describe(include='all'))

# ============================================================
# STEP 2 : Check Missing Values
# ============================================================

print("\nMissing Values")

print(df.isnull().sum())

# ============================================================
# STEP 3 : Handle Missing Values
# ============================================================

# Separate Numerical and Categorical Columns

numerical_columns = df.select_dtypes(include=np.number).columns

categorical_columns = df.select_dtypes(exclude=np.number).columns

# Numerical → Mean Imputation

num_imputer = SimpleImputer(strategy="mean")

df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

# Categorical → Most Frequent

cat_imputer = SimpleImputer(strategy="most_frequent")

df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

print("\nMissing Values After Cleaning")

print(df.isnull().sum())

# ============================================================
# STEP 4 : Remove Duplicate Records
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df = df.drop_duplicates()

print("Shape After Removing Duplicates :", df.shape)

# ============================================================
# STEP 5 : Detect & Remove Outliers (IQR Method)
# ============================================================

print("\nRemoving Outliers...")

Q1 = df[numerical_columns].quantile(0.25)

Q3 = df[numerical_columns].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

df = df[~((df[numerical_columns] < lower) |
          (df[numerical_columns] > upper)).any(axis=1)]

print("Shape After Outlier Removal :", df.shape)

# ============================================================
# STEP 6 : Encode Categorical Variables
# ============================================================

label_encoder = LabelEncoder()

for column in categorical_columns:

    if df[column].nunique() <= 2:

        df[column] = label_encoder.fit_transform(df[column])

# One-Hot Encoding

df = pd.get_dummies(df, drop_first=True)

print("\nDataset After Encoding")

print(df.head())

# ============================================================
# STEP 7 : Feature Scaling
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nStandardized Dataset")

print(df_scaled.head())

# ============================================================
# STEP 8 : Save Cleaned Dataset
# ============================================================

df_scaled.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

# ============================================================
# STEP 9 : Visualization
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(df_scaled.iloc[:,0], bins=30)

plt.title("Distribution of First Feature")

plt.xlabel("Value")

plt.ylabel("Frequency")

plt.show()

# ============================================================
# Final Report
# ============================================================

print("="*60)

print("PREPROCESSING COMPLETED")

print("="*60)

print("✔ Missing Values Handled")
print("✔ Duplicates Removed")
print("✔ Outliers Removed")
print("✔ Categorical Data Encoded")
print("✔ Features Standardized")
print("✔ Dataset Saved")

print("="*60)
