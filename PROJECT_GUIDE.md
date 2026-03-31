# COMPLETE PROJECT GUIDE
## Smart Expense Categorization & Spending Prediction
### Machine Learning Techniques — Mini Project

---

## 📌 1. PROBLEM STATEMENT

### Problem Definition
In today's fast-paced world, people make dozens of small transactions daily — groceries,
transport, food orders, subscriptions — and lose track of where their money is going.
Manually tagging each expense is tedious and error-prone.

**This project solves two problems using Machine Learning:**

1. **Automatic Categorization** — Given a new expense entry (date, description, payment
   method), automatically classify it into one of 6 categories:
   Food | Transport | Entertainment | Healthcare | Education | Utilities

2. **Spending Prediction** — Predict the likely spending amount for a new expense based
   on its contextual features (day, month, payment method, etc.)

### Real-World Relevance
- Apps like **CRED, Walnut, Money View** use similar ML pipelines for auto-categorization
- Banks use spending prediction to offer personalized financial advice
- Students and young earners benefit most from automated expense tracking
- Foundation for building full personal finance assistants

### Project Objectives
- Build a multi-class classification model for expense categorization
- Build a regression model for spending amount prediction
- Visualize spending patterns with charts
- Create a reusable, modular ML pipeline

---

## 🧠 2. PROJECT OVERVIEW

### Simple Terms
The system takes an expense record (when you spent, what you bought, how you paid) and:
- **Tells you WHAT category it belongs to** (like a smart tag)
- **Estimates HOW MUCH you're likely to spend** on similar future expenses

### Technical Terms

| Component | Role | Algorithm |
|-----------|------|-----------|
| Preprocessing | Clean data, extract features | pandas, LabelEncoder |
| Classifier | Multi-class prediction | Random Forest Classifier |
| Regressor | Continuous value prediction | Random Forest Regressor |
| Evaluator | Measure model quality | accuracy, MAE, R² |
| Visualizer | Generate insights charts | matplotlib |

### Why Random Forest?
- Works well with small datasets (even 80–200 rows)
- Handles mixed feature types (numbers, categories)
- Resistant to overfitting (uses ensemble of trees)
- No need for feature scaling
- Provides feature importance scores

---

## 📂 3. PROJECT STRUCTURE

```
smart-expense-ml/
│
├── data/
│   └── expenses.csv          ← 80-row realistic Indian expense dataset
│
├── src/
│   ├── preprocess.py         ← Step 1: Load, clean, engineer features
│   ├── classify.py           ← Step 2: Train & use classifier
│   ├── regress.py            ← Step 3: Train & use regressor
│   └── visualize.py          ← Step 4: Generate all 4 charts
│
├── notebooks/
│   └── expense_ml_notebook.ipynb  ← Jupyter walkthrough (same steps)
│
├── models/                   ← Auto-created when you run main.py
│   ├── classifier.pkl
│   └── regressor.pkl
│
├── outputs/plots/            ← Auto-created charts
│   ├── category_distribution.png
│   ├── spending_trend.png
│   ├── monthly_spending.png
│   └── feature_importance.png
│
├── main.py                   ← ENTRY POINT — run this file
├── requirements.txt
├── .gitignore
└── README.md
```

**Each file explained:**
- `expenses.csv` — Your training data. More rows = better accuracy
- `preprocess.py` — The data pipeline: raw CSV → clean features
- `classify.py` — All classification logic in one place
- `regress.py` — All regression logic in one place
- `visualize.py` — All chart generation; saves PNG files
- `main.py` — Orchestrates everything in order
- `notebooks/` — Same project in interactive Jupyter format for demo

---

## 📊 4. DATASET

### Column Descriptions

| Column | Type | Example | Purpose |
|--------|------|---------|---------|
| `date` | Date | 2024-01-02 | When the expense happened |
| `description` | Text | "Grocery shopping at DMart" | What was bought |
| `amount` | Float | 1850.00 | How much was spent (₹) |
| `payment_method` | Category | UPI / Cash / Credit Card | How payment was made |
| `category` | Category | Food / Transport / etc. | **Target label** |

### Sample Rows
```
date,description,amount,payment_method,category
2024-01-02,Grocery shopping at DMart,1850.00,UPI,Food
2024-01-03,Uber ride to college,220.50,UPI,Transport
2024-01-05,Netflix subscription,649.00,Credit Card,Entertainment
2024-01-07,Medical checkup at clinic,500.00,Cash,Healthcare
2024-01-08,Electricity bill payment,1200.00,Net Banking,Utilities
2024-01-12,New textbooks for semester,1450.00,Credit Card,Education
```

### Category Distribution (80 rows)
- Food          : ~25 rows
- Transport     : ~18 rows
- Entertainment : ~10 rows
- Healthcare    : ~10 rows
- Utilities     : ~10 rows
- Education     : ~7 rows

**Note to students:** The more data you add, the higher your model accuracy.
Real-world apps train on thousands of transactions.

---

## ⚙️ 5. STEP-BY-STEP IMPLEMENTATION

### Step 1 — Data Preprocessing
```
Raw CSV → Load with pandas → Drop duplicates → Remove nulls
       → Convert amount to float → Remove negative values
```
Key function: `clean_data()` in `src/preprocess.py`

### Step 2 — Feature Engineering
Raw data has text and dates — ML needs numbers. We convert:

| Raw Feature | → | Engineered Feature | How |
|-------------|---|-------------------|-----|
| date | → | day_of_week (0–6) | `.dt.dayofweek` |
| date | → | month (1–12) | `.dt.month` |
| date | → | day_of_month (1–31) | `.dt.day` |
| date | → | is_weekend (0/1) | `dayofweek >= 5` |
| description | → | desc_len (word count) | `.str.split().str.len()` |
| payment_method | → | payment_enc (0/1/2/3) | `LabelEncoder` |

Key function: `engineer_features()` in `src/preprocess.py`

### Step 3a — Classification Model
```python
# Split → Train → Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```
**Metric:** Accuracy (% of correct category predictions)

### Step 3b — Regression Model
```python
# Same split strategy
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
```
**Metrics:** MAE (average ₹ error), R² score (0–1, higher is better)

### Step 4 — Model Evaluation
- Classifier: `accuracy_score`, `classification_report`, 5-fold CV
- Regressor: `mean_absolute_error`, `r2_score`

### Step 5 — Saving Models
Both models are saved as `.pkl` files using `joblib.dump()`.
You can reload them later with `joblib.load()` without retraining.

---

## 📦 9. requirements.txt

```
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
numpy>=1.23.0
joblib>=1.2.0
```

Install with: `pip install -r requirements.txt`

---

## 🌐 12. GITHUB UPLOAD STEPS

Run these commands one by one in your terminal:

```bash
# Step 1: Go to project folder
cd smart-expense-ml

# Step 2: Initialize git
git init

# Step 3: Add all files
git add .

# Step 4: First commit
git commit -m "Initial commit: Smart Expense ML Project"

# Step 5: Create repo on GitHub (go to github.com → New Repository)
# Name it: smart-expense-ml
# Do NOT initialize with README (we already have one)

# Step 6: Link your local repo to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/smart-expense-ml.git

# Step 7: Rename branch to main
git branch -M main

# Step 8: Push to GitHub
git push -u origin main
```

**Then go to github.com/YOUR_USERNAME/smart-expense-ml to see your project live!**

---

## 🎤 13. VIVA QUESTIONS & ANSWERS

**Q1. What is the difference between classification and regression?**
> Classification predicts a **discrete label/category** (e.g., Food, Transport).
> Regression predicts a **continuous numerical value** (e.g., ₹850).
> In this project, we use classification to predict expense category and regression to predict amount.

**Q2. Why did you choose Random Forest over a simple Decision Tree?**
> A single Decision Tree overfits the training data.
> Random Forest builds 100 trees, each trained on a random subset of data and features,
> then takes a majority vote (classification) or average (regression).
> This reduces variance and gives more stable predictions.

**Q3. What is feature engineering? Give an example from your project.**
> Feature engineering is creating new useful numerical features from raw data.
> Example: From the `date` column, we extracted `day_of_week`, `month`, `is_weekend`.
> These help the model understand temporal spending patterns.

**Q4. What does accuracy mean in your classifier?**
> Accuracy = (Correct predictions) / (Total predictions) × 100%.
> If 17 out of 20 test expenses are categorized correctly, accuracy = 85%.
> Note: Accuracy alone can be misleading for imbalanced datasets — we also use precision, recall, and F1-score.

**Q5. What is MAE in your regression model?**
> MAE = Mean Absolute Error = average of |actual - predicted| values.
> If MAE = ₹150, it means on average our prediction is off by ₹150.
> Lower MAE = better model.

**Q6. What is R² score?**
> R² (R-squared) measures how well the regression model explains the variance in data.
> R² = 1.0 → perfect fit. R² = 0 → model is as good as predicting the mean.
> Negative R² means the model is performing worse than a simple average baseline.

**Q7. What is overfitting? How does Random Forest avoid it?**
> Overfitting: model memorizes training data but fails on new data.
> Random Forest uses two techniques: (1) **Bagging** — each tree trains on a random sample,
> (2) **Feature randomness** — each split considers a random subset of features.
> This diversity prevents any single tree from memorizing patterns.

**Q8. What is Label Encoding? Why do you need it?**
> ML algorithms work with numbers, not text.
> Label Encoding converts text categories to integers:
> Cash → 0, Credit Card → 1, Net Banking → 2, UPI → 3.
> We use `sklearn.preprocessing.LabelEncoder` for this.

**Q9. What is cross-validation and why did you use it?**
> With a small dataset (80 rows), a single train-test split can give misleading results.
> 5-fold CV splits data into 5 parts, trains on 4, tests on 1, repeats 5 times, and averages.
> This gives a more reliable accuracy estimate.

**Q10. What are the limitations of your project?**
> (1) Small dataset — accuracy improves significantly with 1000+ records.
> (2) Description text is only used as word count, not semantically analyzed.
> (3) No user interface — currently command-line only.
> (4) Predictions are based on structural patterns, not the meaning of words.

**Q11. How would you improve accuracy with better text features?**
> Instead of just word count, use TF-IDF or word embeddings (Word2Vec/BERT) to convert
> the description into a dense vector that captures meaning.
> E.g., "grocery", "supermarket", "vegetables" would all map close to "Food" category.

**Q12. What is joblib.dump() used for?**
> After training, we save the model to a .pkl (pickle) file using joblib.dump().
> This lets us reload the trained model later without retraining from scratch.
> In production systems, models are trained once and served thousands of predictions.

---

## 🚀 14. FUTURE ENHANCEMENTS

1. **NLP on Descriptions** — Use TF-IDF or sentence embeddings for better classification
2. **Streamlit Web App** — Build a simple UI where users enter expenses and see predictions
3. **Budget Alert System** — Send alert when category spending crosses monthly budget
4. **LSTM Time-Series** — Forecast next month's total spending using sequence models
5. **Receipt OCR** — Automatically extract expense data from uploaded bill photos
6. **REST API** — Expose models as API endpoints so any app can use predictions
7. **User-Specific Models** — Train personalized models per user for higher accuracy
8. **Anomaly Detection** — Flag unusually high expenses automatically
9. **Dashboard** — Interactive charts using Plotly/Dash instead of static matplotlib
10. **Export Report** — Generate monthly PDF spending report automatically

---

## 🧪 8. SAMPLE INPUT & OUTPUT

### Input (new expense to predict)
```python
{
    "day_of_week"  : 1,     # Tuesday
    "month"        : 3,     # March
    "day_of_month" : 5,
    "is_weekend"   : 0,
    "desc_len"     : 6,     # "Weekly grocery shopping at Big Bazaar"
    "payment_enc"  : 3      # UPI
}
```

### Output
```
🔮 Predicted Category : Food  (confidence: 91.0%)
💰 Predicted Amount   : ₹986.38
```

### Another Example
```python
# Expense: Ola cab to railway station, UPI, Saturday
{
    "day_of_week": 5, "month": 3, "day_of_month": 9,
    "is_weekend": 1, "desc_len": 5, "payment_enc": 3
}
# Output:
🔮 Predicted Category : Transport  (confidence: 65.0%)
💰 Predicted Amount   : ₹787.28
```
