# 💰 Smart Expense Categorization & Spending Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-1.5%2B-lightblue?logo=pandas)
![License](https://img.shields.io/badge/license-MIT-green)

> **Mini Project — Machine Learning Techniques**  
> A complete end-to-end ML system that automatically categorizes expenses and predicts spending amounts using Random Forest models.

---

## 📌 Problem Statement

Managing personal finances is challenging. People often struggle to:
- Track which category their money goes to (food, transport, etc.)
- Anticipate future spending based on past patterns

This project builds an ML pipeline that:
1. **Classifies** an expense into a category (Food, Transport, Entertainment, Healthcare, Education, Utilities)
2. **Predicts** the likely spending amount for a new expense

---

## 🧠 How It Works

```
Raw Expense Data  →  Preprocessing  →  Feature Engineering
        ↓
  ┌─────────────┐      ┌──────────────┐
  │ Classifier  │      │  Regressor   │
  │ (Category)  │      │  (Amount ₹)  │
  └─────────────┘      └──────────────┘
        ↓                     ↓
   "Food" / "Transport"   "₹ 850.00"
```

| Model | Algorithm | Task | Input | Output |
|-------|-----------|------|-------|--------|
| Classifier | Random Forest | Multi-class Classification | Features | Category label |
| Regressor | Random Forest | Regression | Features | Amount (₹) |

---

## 📂 Project Structure

```
smart-expense-ml/
│
├── data/
│   └── expenses.csv          ← Raw dataset (40 rows)
│
├── src/
│   ├── preprocess.py         ← Data loading, cleaning, feature engineering
│   ├── classify.py           ← Train & predict with classifier
│   ├── regress.py            ← Train & predict with regressor
│   └── visualize.py          ← All chart generation
│
├── notebooks/
│   └── expense_ml_notebook.ipynb  ← Interactive Jupyter walkthrough
│
├── models/
│   ├── classifier.pkl        ← Saved classifier (auto-created)
│   └── regressor.pkl         ← Saved regressor  (auto-created)
│
├── outputs/
│   └── plots/                ← Generated charts (auto-created)
│       ├── category_distribution.png
│       ├── spending_trend.png
│       ├── monthly_spending.png
│       └── feature_importance.png
│
├── main.py                   ← Entry point — run this!
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| `day_of_week` | Day of week (0=Mon … 6=Sun) |
| `month` | Month number (1–12) |
| `day_of_month` | Day of the month (1–31) |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |
| `desc_len` | Word count in expense description |
| `payment_enc` | Label-encoded payment method |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-expense-ml.git
cd smart-expense-ml
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python main.py
```

### 5. (Optional) Open Jupyter Notebook
```bash
pip install jupyter
jupyter notebook notebooks/expense_ml_notebook.ipynb
```

---

## 📈 Sample Results

```
🌲 CLASSIFIER — Random Forest
  Accuracy      : ~87%
  5-Fold CV     : ~85% ± 4%

📈 REGRESSOR — Random Forest
  MAE           : ₹142.50
  R² Score      : 0.72
```

---

## 🖼 Visualizations

The project generates 4 charts automatically:

- **Pie Chart** — Spending distribution across categories
- **Line Chart** — Daily spending trend over time
- **Bar Chart** — Monthly total spending comparison
- **Feature Importance** — Which features matter most to the classifier

---

## 🔮 Sample Prediction Output

```
── Expense #1 ──────────────────────────────────────
  Description : Weekly grocery shopping at Big Bazaar
  Payment     : UPI
  Date        : 2024-03-05

  🔮 Predicted Category : Food  (confidence: 91.0%)
  💰 Predicted Amount   : ₹1642.33
```

---

## 🚀 Future Enhancements

- [ ] Add NLP (TF-IDF / BERT) on expense descriptions for better classification
- [ ] Flask/Streamlit web interface for live predictions
- [ ] Budget alert system (notify when category spend exceeds limit)
- [ ] Time-series forecasting with LSTM for monthly spending
- [ ] Export PDF report of monthly expense summary
- [ ] Mobile app integration via REST API

---

## 📚 Tech Stack

- **Language:** Python 3.8+
- **ML Library:** scikit-learn
- **Data:** pandas, numpy
- **Visualization:** matplotlib
- **Persistence:** joblib

---

## 👨‍💻 Author

Karthikeyan C
(Cyber Security Engineering Student)  
Subject: Machine Learning Techniques  

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
