# =========================================================
# 1. INSTALL LIBRARIES
# =========================================================
!pip install -q optuna lime shap catboost scikit-optimize

# =========================================================
# 2. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import shap
import lime.lime_tabular

# Helper function for adjusted R2
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1)/(n - p - 1)

# =========================================================
# 3. LOAD DATA
# =========================================================
df = pd.read_csv("Book44.csv")

print("Dataset Shape:", df.shape)


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 7. FEATURE SELECTION (Mutual Information)
# =========================================================
selector = SelectKBest(mutual_info_regression, k="all")
selector.fit(X_train, y_train)
selected_features = X.columns[selector.get_support()]
X_train = pd.DataFrame(selector.transform(X_train), columns=selected_features)
X_test = pd.DataFrame(selector.transform(X_test), columns=selected_features)
# =========================================================
# 8. CROSS VALIDATION
# =========================================================
cv = KFold(n_splits=8, shuffle=True, random_state=42)

# =========================================================
# 9. MODEL LIBRARY
# =========================================================
models = {

"RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),

"ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),

"GradientBoost": GradientBoostingRegressor(n_estimators=300, random_state=42),

"XGBoost": XGBRegressor(n_estimators=300, random_state=42),

"LightGBM": LGBMRegressor(n_estimators=300, random_state=42),

"CatBoost": CatBoostRegressor(verbose=0, random_state=42),

"LinearRegression": LinearRegression(),

"Ridge": Ridge(),

"Lasso": Lasso()

}

# =========================================================
# 10. MODEL COMPARISON
# =========================================================
results=[]

n_samples = X_test.shape[0]
p_features = X_test.shape[1]

for name,model in models.items():

    cv_score=cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    ).mean()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    adj_r2 = adjusted_r2(test_r2, n_samples, p_features)

    results.append({
        "Model":name,
        "CV_R2":cv_score,
        "Test_R2":test_r2,
        "MAE": mae,
        "RMSE": rmse,
        "Adjusted_R2": adj_r2
    })

results_df=pd.DataFrame(results).sort_values("CV_R2",ascending=True)

print(results_df)

# =========================================================
# 11. SELECT TOP 3 MODELS
# =========================================================
top3=results_df.head(3)["Model"].tolist()

top_models=[(m,models[m]) for m in top3]

# =========================================================
# 12. ENSEMBLE MODELS
# =========================================================
# ------------------------------
# Voting Regressor
# ------------------------------
voting_model = VotingRegressor(top_models)
voting_model.fit(X_train, y_train)

# ------------------------------
# Stacking Regressor
# ------------------------------
stacking_model = StackingRegressor(
    estimators=top_models,
    final_estimator=Ridge()
)
stacking_model.fit(X_train, y_train)

# ------------------------------
# Evaluate all models
# ------------------------------
voting_r2 = r2_score(y_test, voting_model.predict(X_test))
stacking_r2 = r2_score(y_test, stacking_model.predict(X_test))

# ------------------------------
# Select best ensemble
# ------------------------------
best_ensemble = max(
    [(voting_model, voting_r2), (stacking_model, stacking_r2)],
    key=lambda x: x[1]
)[0]

print("R2 Scores:")
print(f"Voting: {voting_r2:.4f}, Stacking: {stacking_r2:.4f}")
print("Best Ensemble:", type(best_ensemble).__name__)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import optuna
from skopt import BayesSearchCV
import numpy as np

# ------------------------------
# Cross-validation
# ------------------------------
cv = KFold(n_splits=8, shuffle=True, random_state=42)


# ------------------------------
# 3. OPTUNA
# ------------------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)
    }
    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

optuna_model = XGBRegressor(**study.best_params)
optuna_model.fit(X_train, y_train)
optuna_r2 = r2_score(y_test, optuna_model.predict(X_test))

# ------------------------------
# 4. BAYESIAN OPTIMIZATION
# ------------------------------
bayes_search = BayesSearchCV(
    XGBRegressor(random_state=42),
    {
        "n_estimators": (200, 800),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.1, "uniform")
    },
    n_iter=30,
    cv=cv,
    scoring="r2",
    random_state=42
)
bayes_search.fit(X_train, y_train)
bayes_r2 = r2_score(y_test, bayes_search.predict(X_test))

# ------------------------------
# SELECT BEST MODEL
# ------------------------------
results = {

    "Optuna": (optuna_model, optuna_r2),
    "Bayesian": (bayes_search, bayes_r2)
}

final_model, final_r2 = max(results.values(), key=lambda x: x[1])
best_name = [k for k, v in results.items() if v[0] == final_model][0]

print("R2 Scores:", {k: round(v[1], 4) for k, v in results.items()})
print("Best Hyperparameter Optimized Model:", best_name)


# 16. FINAL METRICS
# =========================================================
y_pred=final_model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

n_samples = X_test.shape[0]
p_features = X_test.shape[1]

r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, n_samples, p_features)
cc = np.corrcoef(y_test, y_pred)[0,1]
print("MAE:",mae)
print("RMSE:",rmse)
print(f"R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")
print(f"Correlation Coefficient (CC): {cc:.4f}")

final_df.to_csv("Book99.csv", index=False)
