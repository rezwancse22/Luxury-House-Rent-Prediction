import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Added import for train_test_split
from xgboost import XGBRegressor # Added import for XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import time

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
df = pd.read_csv("Book82.csv")

target = "price"
X = df.drop(columns=[target])
y = df[target]


# =====================================================
# Categorical columns - Existing column এ encoding (NO new columns)
# =====================================================
cat_cols = ['facing', 'building_regiration_type', 'area', 'rent_floor', 'extra_aminities', 'road_in_front_of_house'] # Corrected typo in 'building_regiration_type'

# OPTION 1: Label Encoding (সবচেয়ে সহজ - Scikit-learn)
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"✅ {col} encoded with {len(le.classes_)} categories")
    else:
        print(f"Skipping column '{col}' as it is not found in X.") # Added for debugging

print("🚀 Running RFE with Cross Validation...")
start = time.time()

# Base estimator (strong + stable)
estimator = RandomForestRegressor(
    n_estimators=800,
    random_state=42,
    n_jobs=-1
)

# Cross-validation strategy
cv_strategy = KFold(
    n_splits=8,
    shuffle=True,
    random_state=42
)

rfecv = RFECV(
    estimator=estimator,
    step=2,
cv=cv_strategy,
    scoring='r2',
    n_jobs=-1
)

rfecv.fit(X, y);

# ───────────────────────
# 5. Results
# ───────────────────────


selected_features = X.columns[rfecv.support_].tolist()
dropped_features = X.columns[~rfecv.support_].tolist()

print("\n" + "="*55)
print("📊 RFE + CROSS VALIDATION REPORT")
print("="*55)
print(f"✅ Optimal Number of Features: {rfecv.n_features_}")
print(f"✅ Selected Features ({len(selected_features)}):")
print(selected_features)

print(f"\n❌ Dropped Features ({len(dropped_features)}):")
print(dropped_features[:10], "...")

print(f"⏱️ Time Taken: {time.time() - start:.2f} seconds")
print("="*55)

final_df = X[selected_features].copy()
target = "price"
final_df[target] = y.values

final_df.to_csv("Book44.csv", index=False)
