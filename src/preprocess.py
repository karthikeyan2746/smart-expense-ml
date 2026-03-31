# src/preprocess.py
# ─────────────────────────────────────────────────────────────
# Module: Data Loading & Preprocessing
# Purpose: Load raw CSV, clean it, and engineer features for ML
#
# KEY UPGRADE: keyword_score features
#   Instead of only using word COUNT (desc_len), we now scan
#   the description for domain-specific keywords and produce
#   one score per category.  This lets the model understand
#   WHAT the user typed, not just HOW MANY words they typed.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Keyword dictionaries — one list per category ──────────────
# Any word in the description that appears here adds +1 to that
# category's score.  Words are matched as substrings (lowercase).
CATEGORY_KEYWORDS = {
    "food": [
        "food", "grocery", "groceries", "restaurant", "cafe", "canteen",
        "lunch", "dinner", "breakfast", "snack", "pizza", "biryani",
        "swiggy", "zomato", "dominos", "burger", "vegetable", "vegetables",
        "fruit", "fruits", "milk", "dairy", "rice", "pulses", "bakery",
        "hotel", "tiffin", "meal", "eat", "eating", "dmart", "bigbazaar",
        "supermarket", "market", "shop", "cooking", "kitchen", "order",
        "takeaway", "dine", "dining"
    ],
    "transport": [
        "transport", "uber", "ola", "rapido", "cab", "taxi", "auto",
        "rickshaw", "bus", "train", "metro", "ticket", "fare", "petrol",
        "diesel", "fuel", "bike", "ride", "travel", "trip", "station",
        "airport", "railway", "ksrtc", "indriver", "pass", "recharge",
        "route", "commute", "vehicle", "car", "driving"
    ],
    "healthcare": [
        "health", "healthcare", "hospital", "doctor", "medical", "medicine",
        "medicines", "pharmacy", "clinic", "checkup", "consultation",
        "appointment", "lab", "test", "blood", "urine", "scan", "xray",
        "x-ray", "diagnosis", "treatment", "surgery", "dental", "dentist",
        "eye", "vision", "glasses", "spectacles", "pathology", "vitamin",
        "supplement", "prescription", "drug", "tablet", "capsule", "syrup",
        "injection", "vaccine", "physiotherapy", "therapy", "apollo",
        "fortis", "manipal", "aiims", "specialist", "physician", "nurse",
        "bp", "blood pressure", "sugar", "diabetes", "fever", "cold",
        "cough", "emergency", "icu", "ward", "operation", "nursing"
    ],
    "education": [
        "education", "college", "school", "university", "course", "class",
        "tuition", "fee", "fees", "book", "books", "textbook", "stationery",
        "exam", "examination", "registration", "certification", "certificate",
        "degree", "study", "learning", "tutorial", "coaching", "training",
        "workshop", "seminar", "udemy", "coursera", "nptel", "skillshare",
        "bootcamp", "library", "notebook", "pen", "pencil", "assignment",
        "project", "internship", "scholarship", "hostel", "admission"
    ],
    "entertainment": [
        "entertainment", "movie", "film", "cinema", "pvr", "inox",
        "netflix", "amazon prime", "hotstar", "spotify", "youtube",
        "music", "game", "gaming", "concert", "show", "event", "party",
        "outing", "trip", "vacation", "holiday", "subscription", "zee5",
        "disney", "streaming", "ticket", "fun", "leisure", "hobby",
        "sport", "sports", "cricket", "football", "gym", "fitness"
    ],
    "utilities": [
        "utility", "utilities", "electricity", "electric", "bill",
        "water", "gas", "lpg", "cylinder", "internet", "wifi", "broadband",
        "mobile", "phone", "recharge", "jio", "airtel", "bsnl",
        "vi", "vodafone", "dth", "tata sky", "dish", "cable", "maintenance",
        "repair", "service", "rent", "maintenance", "insurance", "tax",
        "emi", "loan", "deposit", "charge", "fee"
    ]
}

CATEGORIES = list(CATEGORY_KEYWORDS.keys())


def keyword_scores(description: str) -> dict:
    """
    Scan a description string and return a score per category.
    Score = number of matching keywords found (case-insensitive).

    Example:
      "Apollo hospital doctor consultation blood pressure medication"
      → {"food":0, "transport":0, "healthcare":5,
         "education":0, "entertainment":0, "utilities":0}
    """
    desc_lower = description.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in desc_lower)
    return scores


def load_data(filepath: str) -> pd.DataFrame:
    """Load the expense CSV and parse dates."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"✅ Loaded {len(df)} records from '{filepath}'")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("\n📊 Dataset Info:")
    print(f"   Shape      : {df.shape}")
    print(f"   Columns    : {list(df.columns)}")
    print(f"   Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Categories : {df['category'].unique()}")
    print(f"   Nulls      :\n{df.isnull().sum()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fill/drop nulls, ensure correct types."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["amount", "category", "description"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numerical features from raw columns.

    Features produced:
      Date-based  → day_of_week, month, day_of_month, is_weekend
      Text-based  → desc_len  (word count)
                  → kw_food, kw_transport, kw_healthcare,
                    kw_education, kw_entertainment, kw_utilities
                    (keyword hit-count per category)
      Payment     → payment_enc  (label-encoded)
    """
    df = df.copy()

    # ── Date features ──────────────────────────────────────────
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # ── Text: word count ───────────────────────────────────────
    df["desc_len"] = df["description"].str.split().str.len()

    # ── Text: keyword scores per category ─────────────────────
    for cat in CATEGORIES:
        col = f"kw_{cat}"
        df[col] = df["description"].apply(
            lambda desc: sum(1 for kw in CATEGORY_KEYWORDS[cat]
                             if kw in str(desc).lower())
        )

    # ── Encode payment method ───────────────────────────────────
    le = LabelEncoder()
    df["payment_enc"] = le.fit_transform(df["payment_method"])

    return df


def encode_target(df: pd.DataFrame) -> tuple:
    """Label-encode the target column 'category'."""
    le = LabelEncoder()
    df = df.copy()
    df["category_enc"] = le.fit_transform(df["category"])
    print(f"\n🏷  Category mapping : "
          f"{dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, le


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Return:
      X     — feature DataFrame (date + keyword + payment features)
      y_cat — encoded category labels  (classification target)
      y_amt — amount                   (regression target)
    """
    feature_cols = [
        "day_of_week", "month", "day_of_month", "is_weekend",
        "desc_len", "payment_enc",
        "kw_food", "kw_transport", "kw_healthcare",
        "kw_education", "kw_entertainment", "kw_utilities"
    ]
    X     = df[feature_cols]
    y_cat = df["category_enc"]
    y_amt = df["amount"]
    return X, y_cat, y_amt