# main.py
# ═══════════════════════════════════════════════════════════════
#  Smart Expense Categorization & Spending Prediction
#  Author  : [Your Name]
#  Subject : Machine Learning Techniques (Mini Project)
# ═══════════════════════════════════════════════════════════════
#
#  HOW TO RUN:
#      python main.py   (in Spyder: press F5)
#
#  FLOW:
#  1. Load & clean data  →  train both models
#  2. Ask user for expense details (description, date, payment)
#  3. Convert user input into ML features automatically
#     — date parts  (day, month, weekday, weekend flag)
#     — keyword scores (how many healthcare/food/etc. words found)
#     — payment encoding
#  4. Predict CATEGORY  +  estimate AMOUNT
#  5. Ask if user wants to predict another expense
# ═══════════════════════════════════════════════════════════════

import sys
import os
from datetime import datetime

# ── make src/ importable regardless of Spyder working directory ─
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocess import (load_data, clean_data, engineer_features,
                        encode_target, get_feature_matrix,
                        keyword_scores, CATEGORY_KEYWORDS)
from classify   import train_classifier
from regress    import train_regressor
from visualize  import generate_all_plots
import pandas as pd


# ── Payment options shown to the user ─────────────────────────
PAYMENT_MAP = {
    "1": ("Cash",        0),
    "2": ("Credit Card", 1),
    "3": ("Net Banking", 2),
    "4": ("UPI",         3),
}

FEATURE_COLS = [
    "day_of_week", "month", "day_of_month", "is_weekend",
    "desc_len", "payment_enc",
    "kw_food", "kw_transport", "kw_healthcare",
    "kw_education", "kw_entertainment", "kw_utilities"
]


# ─────────────────────────────────────────────────────────────
# HELPER: build ML feature dict from user's raw input
# ─────────────────────────────────────────────────────────────
def build_features(date_obj, description: str, payment_enc: int) -> dict:
    """
    Convert raw user inputs into numerical features the model needs.

    Steps:
      1. Extract date parts from the date object
      2. Count words in description  (desc_len)
      3. Scan description for category keywords  (kw_* scores)
      4. Add the encoded payment integer
    """
    dow      = date_obj.weekday()               # 0=Mon … 6=Sun
    kw       = keyword_scores(description)      # dict: cat → score

    return {
        "day_of_week"      : dow,
        "month"            : date_obj.month,
        "day_of_month"     : date_obj.day,
        "is_weekend"       : 1 if dow >= 5 else 0,
        "desc_len"         : len(description.split()),
        "payment_enc"      : payment_enc,
        "kw_food"          : kw["food"],
        "kw_transport"     : kw["transport"],
        "kw_healthcare"    : kw["healthcare"],
        "kw_education"     : kw["education"],
        "kw_entertainment" : kw["entertainment"],
        "kw_utilities"     : kw["utilities"],
    }


# ─────────────────────────────────────────────────────────────
# INPUT: collect one expense from the user interactively
# ─────────────────────────────────────────────────────────────
def get_user_input():
    """
    Ask the user for description, date, and payment method.
    Returns (date_obj, description, payment_label, features_dict)
    or None if the user types 'q' to quit.
    """
    print("\n" + "─" * 58)
    print("  Enter your expense details below")
    print("  (type  q  at any prompt to quit)")
    print("─" * 58)

    # ── Description ────────────────────────────────────────────
    print("\n  📝 Expense description")
    print("     Be descriptive — mention what the expense is for.")
    print("     Example: 'Apollo hospital doctor consultation and")
    print("               blood pressure medication purchase'")
    desc = input("  > ").strip()
    if desc.lower() == "q":
        return None
    if not desc:
        print("  ⚠  Description cannot be empty.")
        return get_user_input()

    # ── Show keyword detection live ────────────────────────────
    kw = keyword_scores(desc)
    hits = {cat: score for cat, score in kw.items() if score > 0}
    if hits:
        hit_str = "  ,  ".join(f"{cat}({score})" for cat, score in
                                sorted(hits.items(), key=lambda x: -x[1]))
        print(f"\n  🔍 Keywords detected → {hit_str}")
    else:
        print("\n  🔍 No strong keywords found — model will use date/payment signals")

    # ── Date ───────────────────────────────────────────────────
    print("\n  📅 Date of expense")
    print("     Format : DD-MM-YYYY   (press Enter to use today's date)")
    date_str = input("  > ").strip()
    if date_str.lower() == "q":
        return None
    if date_str == "":
        date_obj = datetime.today()
        print(f"     Using today: {date_obj.strftime('%d-%m-%Y')}")
    else:
        try:
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            print("  ⚠  Invalid format. Use DD-MM-YYYY  e.g. 22-03-2024")
            return get_user_input()

    # ── Payment Method ─────────────────────────────────────────
    print("\n  💳 Payment method")
    print("     1. Cash          2. Credit Card")
    print("     3. Net Banking   4. UPI")
    pay = input("  Enter number (1-4): ").strip()
    if pay.lower() == "q":
        return None
    if pay not in PAYMENT_MAP:
        print("  ⚠  Please enter a number between 1 and 4.")
        return get_user_input()

    payment_label, payment_enc = PAYMENT_MAP[pay]
    features = build_features(date_obj, desc, payment_enc)
    return date_obj, desc, payment_label, features


# ─────────────────────────────────────────────────────────────
# OUTPUT: run both models and display a rich result
# ─────────────────────────────────────────────────────────────
def show_prediction(clf, reg, label_encoder,
                    date_obj, desc, payment_label, features):
    """Run classifier + regressor and print a detailed result."""

    feat_df = pd.DataFrame([features])[FEATURE_COLS]

    # ── Classification ─────────────────────────────────────────
    cat_enc    = clf.predict(feat_df)[0]
    category   = label_encoder.inverse_transform([cat_enc])[0]
    all_probas = clf.predict_proba(feat_df)[0]
    top_conf   = all_probas.max()
    classes    = label_encoder.classes_

    # ── Regression ─────────────────────────────────────────────
    pred_amt = reg.predict(feat_df)[0]

    # ── Keyword signal summary ─────────────────────────────────
    kw_summary = []
    for cat in ["food", "transport", "healthcare",
                "education", "entertainment", "utilities"]:
        score = features[f"kw_{cat}"]
        if score > 0:
            kw_summary.append(f"{cat}:{score}")

    # ── Display ────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  PREDICTION RESULTS")
    print("═" * 58)
    print(f"  Description  : {desc}")
    print(f"  Date         : {date_obj.strftime('%d %B %Y')}  "
          f"({'Weekend' if features['is_weekend'] else 'Weekday'})")
    print(f"  Payment      : {payment_label}")
    print(f"  Keywords hit : {', '.join(kw_summary) if kw_summary else 'none'}")
    print("─" * 58)
    print(f"  Predicted Category  :  {category}")
    print(f"  Confidence          :  {top_conf:.0%}")
    print(f"  Estimated Amount    :  Rs. {pred_amt:,.2f}")
    print("─" * 58)

    # Show top-3 category probabilities
    print("  Category probability breakdown:")
    top3_idx = all_probas.argsort()[::-1][:3]
    for idx in top3_idx:
        bar_len = int(all_probas[idx] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {classes[idx]:<16} {bar}  {all_probas[idx]:.0%}")

    print("═" * 58)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓" * 60)
    print("  Smart Expense Categorization & Spending Prediction")
    print("▓" * 60)

    # ── Step 1 : Load & prepare training data ──────────────────
    print("\n[1/4] Loading dataset ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, "data", "expenses.csv")
    df = load_data(data_path)
    df = clean_data(df)

    # ── Step 2 : Feature engineering ───────────────────────────
    print("[2/4] Engineering features (with keyword detection) ...")
    df = engineer_features(df)
    df, label_encoder = encode_target(df)
    X, y_cat, y_amt = get_feature_matrix(df)
    print(f"      Features used : {list(X.columns)}")

    # ── Step 3 : Train models ───────────────────────────────────
    print("[3/4] Training models ...")
    clf = train_classifier(X, y_cat, save=True)
    reg = train_regressor(X, y_amt,  save=True)

    # ── Step 4 : Generate visualizations ───────────────────────
    print("[4/4] Generating visualizations ...")
    generate_all_plots(df, clf=clf, feature_names=list(X.columns))

    # ── Step 5 : Interactive prediction loop ───────────────────
    print("\n" + "▓" * 60)
    print("  Models ready!  Now enter YOUR expense to predict.")
    print("▓" * 60)

    while True:
        result = get_user_input()

        if result is None:
            print("\n  Goodbye!\n")
            break

        date_obj, desc, payment_label, features = result
        show_prediction(clf, reg, label_encoder,
                        date_obj, desc, payment_label, features)

        print("\n  Predict another expense?")
        again = input("  Enter 'y' to continue, any other key to exit: ").strip().lower()
        if again != "y":
            print("\n  Thank you! Charts → outputs/plots/   Models → models/\n")
            break


if __name__ == "__main__":
    main()