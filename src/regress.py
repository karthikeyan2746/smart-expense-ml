# src/regress.py
# ─────────────────────────────────────────────────────────────
# Module: Spending Prediction (Regression)
# Model : Random Forest Regressor
# Task  : Predict the AMOUNT of an expense from its features
# ─────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (mean_absolute_error,
                                      mean_squared_error,
                                      r2_score)


MODEL_PATH = "models/regressor.pkl"


def train_regressor(X: pd.DataFrame, y: pd.Series,
                    save: bool = True) -> RandomForestRegressor:
    """
    Train a Random Forest regressor to predict spending amount.

    Parameters
    ----------
    X    : feature matrix
    y    : amount (continuous target)
    save : if True, persist model to MODEL_PATH

    Returns
    -------
    Trained RandomForestRegressor
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    reg.fit(X_train, y_train)

    # ── Evaluation ──────────────────────────────────────────────
    y_pred = reg.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print("\n" + "="*55)
    print("  📈 REGRESSOR — Random Forest")
    print("="*55)
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  MAE           : ₹{mae:.2f}")
    print(f"  RMSE          : ₹{rmse:.2f}")
    print(f"  R² Score      : {r2:.4f}")

    # Show a few actual vs predicted comparisons
    print("\n  Sample Actual vs Predicted (₹):")
    print("  {:<12} {:<12}".format("Actual", "Predicted"))
    print("  " + "-"*24)
    for a, p in zip(y_test[:8], y_pred[:8]):
        print(f"  ₹{a:<11.2f} ₹{p:<11.2f}")
    print("="*55)

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(reg, MODEL_PATH)
        print(f"  💾 Model saved → {MODEL_PATH}")

    return reg


def load_regressor() -> RandomForestRegressor:
    """Load a previously saved regressor."""
    return joblib.load(MODEL_PATH)


def predict_amount(reg: RandomForestRegressor,
                   features: dict) -> float:
    """
    Predict the spending amount for a single new expense.

    Parameters
    ----------
    reg      : trained regressor
    features : dict with keys matching feature columns

    Returns
    -------
    Predicted amount (float, in ₹)
    """
    df_input      = pd.DataFrame([features])
    predicted_amt = reg.predict(df_input)[0]
    print(f"\n  💰 Predicted Amount : ₹{predicted_amt:.2f}")
    return predicted_amt
