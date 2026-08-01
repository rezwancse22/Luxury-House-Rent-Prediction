
import pandas as pd
import numpy as np
import re

# 1. Load Dataset
df = pd.read_csv("Sheet8.csv")

print("Original Shape:", df.shape)
print("\nPreview:")
display(df.head())

# 2. Drop Unnecessary Columns
drop_columns = ['url','post title']

df.drop(columns=[c for c in drop_columns if c in df.columns], inplace=True)

print("\nAfter Dropping Columns:", df.shape)

# 3. Standardize Column Names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w]", "", regex=True)
)

print("\nCleaned Column Names:")
print(df.columns.tolist())

# 4. Clean Unnecessary Characters

def clean_text(value):
    if isinstance(value, str):
        value = re.sub(r"[;,]", "", value)
        value = re.sub(r"[৳$]", "", value)
        value = value.strip()
    return value

df = df.applymap(clean_text)

# 5. Remove Extra Whitespaces
for col in df.select_dtypes(exclude=np.number).columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

# 6. Convert Numeric Columns Safely
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")
# 7. Remove Duplicate Rows
dup_count = df.duplicated().sum()
print("\nDuplicate Rows Found:", dup_count)

df.drop_duplicates(inplace=True)

# 8. Final Sanity Check

print("\nFinal Shape:", df.shape)
df.info()

print("\nFinal Preview:")
display(df.head())

# 9. Save Clean Dataset
df.to_csv("Book200.csv", index=False)

# Final Row & Column Count

total_rows, total_columns = df.shape

summary = pd.DataFrame({
    "Metric": ["Total Rows", "Total Columns"],
    "Count": [df.shape[0], df.shape[1]]
})

summary

