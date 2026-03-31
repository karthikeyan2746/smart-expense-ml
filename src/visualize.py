# src/visualize.py
# ─────────────────────────────────────────────────────────────
# Module: Visualization
# Purpose: Generate and save charts for analysis & reporting
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Color palette (consistent across all charts) ──────────────
COLORS = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7",
          "#3a0ca3", "#480ca8", "#560bad", "#b5179e"]


def pie_chart_categories(df: pd.DataFrame) -> None:
    """
    Pie chart: distribution of total spending across categories.
    Saved to outputs/plots/category_distribution.png
    """
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        cat_totals,
        labels=cat_totals.index,
        autopct="%1.1f%%",
        colors=COLORS[:len(cat_totals)],
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)

    ax.set_title("💰 Expense Distribution by Category",
                 fontsize=14, fontweight="bold", pad=20)

    path = os.path.join(PLOT_DIR, "category_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved → {path}")


def line_chart_spending_trend(df: pd.DataFrame) -> None:
    """
    Line chart: total spending trend over time (by date).
    Saved to outputs/plots/spending_trend.png
    """
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily = daily.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily["amount"],
            color="#4361ee", linewidth=2.2, marker="o",
            markersize=5, markerfacecolor="#f72585", zorder=3)
    ax.fill_between(daily["date"], daily["amount"],
                    alpha=0.12, color="#4361ee")

    ax.set_title("📈 Daily Spending Trend", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Amount (₹)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=35, ha="right")

    path = os.path.join(PLOT_DIR, "spending_trend.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved → {path}")


def bar_chart_monthly(df: pd.DataFrame) -> None:
    """
    Bar chart: total monthly spending.
    Saved to outputs/plots/monthly_spending.png
    """
    df = df.copy()
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    monthly = df.groupby("month_label")["amount"].sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(monthly.index, monthly.values,
                  color=COLORS[:len(monthly)],
                  edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"₹{bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title("🗓  Monthly Total Spending", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_ylabel("Amount (₹)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"
    ))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    path = os.path.join(PLOT_DIR, "monthly_spending.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved → {path}")


def feature_importance_chart(clf, feature_names: list) -> None:
    """
    Horizontal bar chart of classifier feature importances.
    Saved to outputs/plots/feature_importance.png
    """
    importances = clf.feature_importances_
    indices     = importances.argsort()          # ascending

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices],
        [importances[i]   for i in indices],
        color="#4361ee", edgecolor="white"
    )
    ax.set_title("🔍 Feature Importance (Classifier)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved → {path}")


def generate_all_plots(df: pd.DataFrame, clf=None,
                       feature_names: list = None) -> None:
    """Run all visualizations at once."""
    print("\n🎨 Generating visualizations...")
    pie_chart_categories(df)
    line_chart_spending_trend(df)
    bar_chart_monthly(df)
    if clf is not None and feature_names is not None:
        feature_importance_chart(clf, feature_names)
    print("✅ All plots saved to outputs/plots/")
