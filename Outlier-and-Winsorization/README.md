import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Added import for matplotlib


# 1. Load Dataset
df = pd.read_csv("Book410.csv")

print("Dataset Shape:", df.shape)

# 2. Select Numeric Columns

num_cols = df.select_dtypes(include=np.number).columns
print ("\nNumeric Columns:", num_cols.tolist())


plt.figure(figsize=(12,6))
df[num_cols].boxplot(rot=90) # Changed X_iqr to df and selected numeric columns

plt.title("Before Winsorization - Boxplot") # Updated title
plt.xlabel("Features")
plt.ylabel("Values")

plt.grid()
plt.show()
# 3. Function to Detect Outlier Severity (IQR)

def detect_outlier_level(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((series < lower) | (series > upper)).sum()
    ratio = outliers / len(series)

    return ratio

# 4. Apply SMART Winsorization
winsor_log = {}

for col in num_cols:
    outlier_ratio = detect_outlier_level(df[col])

    # Auto decision
    if outlier_ratio < 0.01:
        lower_p, upper_p = 0.05, 0.95
    elif outlier_ratio < 0.05:
        lower_p, upper_p = 0.025, 0.97
    else:
        lower_p, upper_p = 0.01, 0.99

    lower_cap = df[col].quantile(lower_p)
    upper_cap = df[col].quantile(upper_p)

    df[col] = np.clip(df[col], lower_cap, upper_cap)

    winsor_log[col] = f"{int(lower_p*100)}%–{int(upper_p*100)}%"

# 5. Show Winsorization Summary
print("\nWinsorization Applied:")
for k, v in winsor_log.items():
    print(f"{k} → capped at {v}")


# 6. Final Shape Check

print("\nFinal Shape After Winsorization:", df.shape)

plt.figure(figsize=(12,6))
df[num_cols].boxplot(rot=90) # Changed X_winsor to df and selected numeric columns

plt.title("After IQR and Winsorization - Boxplot")
plt.xlabel("Features")
plt.ylabel("Values")

plt.grid()
plt.show()
# 7. Save Final Dataset

df.to_csv("Book82.csv", index=False)
