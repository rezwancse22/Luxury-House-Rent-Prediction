import pandas as pd
import numpy as np


# 1. Load Dataset
df = pd.read_csv("Book200.csv")

print("Dataset Shape:", df.shape)


# 2. Check Missing Values

print("\nMissing Values Before Handling:")
print(df.isnull().sum())


# 3. Separate Numeric & Categorical
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# 4. Numeric Missing Value Handling
# Skewness < 0.5 → Mean
# Skewness >= 0.5 → Median

imputation_log = {}

for col in num_cols:
    if df[col].isnull().sum() > 0:
        skewness = df[col].skew()

        if abs(skewness) < 0.5:
            fill_value = df[col].mean()
            method = "Mean"
        else:
            fill_value = df[col].median()
            method = "Median"

        df[col].fillna(fill_value, inplace=True)
        imputation_log[col] = method

# 5. Categorical Missing Value Handling
# Mode (Most Frequent Value)
for col in cat_cols:
    if df[col].isnull().sum() > 0:

        most_frequent = df[col].mode()[0]


        df[col] = df[col].fillna(most_frequent)

        imputation_log[col] = f"Mode ({most_frequent})"

# 6. Show What Method Used Where

print("\nImputation Methods Used:")
for k, v in imputation_log.items():
    print(f"{k} → {v}")

# 7. Final Missing Value Check

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# 8. Save Stage-2 Dataset

df.to_csv("Book410.csv", index=False)

