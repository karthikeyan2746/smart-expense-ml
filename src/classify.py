# src/classify.py
# ─────────────────────────────────────────────────────────────
# Module: Expense Categorization (Classification)
# Model : Random Forest Classifier
# Task  : Predict the CATEGORY of an expense from its features
# ─────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                        confusion_matrix,
                                        accuracy_score)


MODEL_PATH = "models/classifier.pkl"


def train_classifier(X: pd.DataFrame, y: pd.Series,
                     save: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X    : feature matrix
    y    : encoded category labels
    save : if True, persist model to MODEL_PATH

    Returns
    -------
    Trained RandomForestClassifier
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=None,     # let trees grow fully
        random_state=42
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ──────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*55)
    print("  🌲 CLASSIFIER — Random Forest")
    print("="*55)
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Accuracy      : {acc:.2%}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5-fold cross-validation for robustness check
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print("="*55)

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        print(f"  💾 Model saved → {MODEL_PATH}")

    return clf


def load_classifier() -> RandomForestClassifier:
    """Load a previously saved classifier."""
    return joblib.load(MODEL_PATH)


def predict_category(clf: RandomForestClassifier,
                     label_encoder,
                     features: dict) -> str:
    """
    Predict expense category for a single new expense.

    Parameters
    ----------
    clf           : trained classifier
    label_encoder : the LabelEncoder used during training
    features      : dict with keys matching feature columns

    Returns
    -------
    Human-readable category string
    """
    df_input = pd.DataFrame([features])
    encoded  = clf.predict(df_input)[0]
    category = label_encoder.inverse_transform([encoded])[0]
    proba    = clf.predict_proba(df_input)[0]
    conf     = proba.max()

    print(f"\n  🔮 Predicted Category : {category}  (confidence: {conf:.1%})")
    return category
