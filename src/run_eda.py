from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FEATURE_DATA_PATH, REPORTS_DIR, FIGURES_DIR
from src.prepare_data import main as prepare_main
from src.data_access import load_raw_dataset

sns.set_theme(style="whitegrid", context="notebook")


def save_plot(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Class balance
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_labels = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().reindex(["Stayed", "Churned"])
    ax.bar(churn_labels.index.astype(str), churn_labels.values)
    ax.set(title="Customer churn distribution", xlabel="Customer outcome", ylabel="Customers")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    save_plot(fig, "01_churn_distribution.png")

    # 2. Churn rate by contract
    fig, ax = plt.subplots(figsize=(8, 4.5))
    work = df.assign(ChurnRate=df["Churn"])
    order = work.groupby("Contract", observed=False)["Churn"].mean().sort_values(ascending=False).index
    contract_means = work.groupby("Contract", observed=False)["Churn"].mean().reindex(order)
    ax.bar(contract_means.index.astype(str), contract_means.values)
    ax.set(title="Observed churn rate by contract type", xlabel="Contract", ylabel="Churn rate")
    ax.set_ylim(0, min(1, max(0.1, work["Churn"].mean() * 2.2)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_plot(fig, "02_churn_by_contract.png")

    # 3. Churn by tenure bucket
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = df.groupby("tenure_group", observed=False)["Churn"].mean().dropna()
    ax.bar(grouped.index.astype(str), grouped.values)
    ax.set(title="Observed churn rate by tenure group", xlabel="Tenure group", ylabel="Churn rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_plot(fig, "03_churn_by_tenure_group.png")

    # 4. Churn by internet service
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = df.groupby("InternetService", dropna=False)["Churn"].mean().sort_values(ascending=False)
    ax.bar(grouped.index.astype(str), grouped.values)
    ax.set(title="Observed churn rate by internet service", xlabel="Internet service", ylabel="Churn rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    save_plot(fig, "04_churn_by_internet_service.png")

    # 5. Monthly charges by churn label
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=df, x=df["Churn"].map({0: "Stayed", 1: "Churned"}), y="MonthlyCharges", ax=ax)
    ax.set(title="Monthly charges by observed churn outcome", xlabel="Customer outcome", ylabel="Monthly charges")
    save_plot(fig, "05_monthly_charges_by_churn.png")

    # 6. Tenure distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df, x="tenure", hue=df["Churn"].map({0: "Stayed", 1: "Churned"}), bins=24, multiple="layer", ax=ax)
    ax.set(title="Tenure distribution by observed churn outcome", xlabel="Tenure (months)", ylabel="Customers")
    save_plot(fig, "06_tenure_distribution.png")

    # 7. Average monthly charge vs tenure, descriptive only
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample = df.sample(min(len(df), 2500), random_state=42)
    sns.scatterplot(data=sample, x="tenure", y="MonthlyCharges", hue=sample["Churn"].map({0: "Stayed", 1: "Churned"}), alpha=.55, s=24, ax=ax)
    ax.set(title="Tenure and monthly charges (sampled for readability)", xlabel="Tenure (months)", ylabel="Monthly charges")
    save_plot(fig, "07_tenure_vs_monthly_charges.png")


def write_eda_summary(df: pd.DataFrame) -> Path:
    churn_rate = float(df["Churn"].mean())
    churn_contract = df.groupby("Contract", observed=False)["Churn"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    churn_tenure = df.groupby("tenure_group", observed=False)["Churn"].agg(["count", "mean"]).dropna().sort_index()
    churn_payment = df.groupby("PaymentMethod", observed=False)["Churn"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    top_contract = churn_contract.index[0]
    top_tenure = churn_tenure["mean"].idxmax()
    top_payment = churn_payment.index[0]
    text = f"""# Part 1 — Exploratory Data Analysis Summary

## Dataset overview

- Records: **{len(df):,}**
- Fields in feature dataset: **{df.shape[1]}**
- Churned customers: **{int(df['Churn'].sum()):,}** ({churn_rate:.1%})
- Customers retained: **{int((df['Churn'] == 0).sum()):,}** ({1-churn_rate:.1%})
- Duplicate customer IDs after cleaning: **{int(df['customerID'].duplicated().sum())}**
- Missing cells in prepared features: **{int(df.isna().sum().sum()):,}** (includes expected undefined average charge for tenure=0)

## Initial observations to investigate

- Highest observed churn-rate contract group: **{top_contract}**.
- Highest observed churn-rate tenure bucket: **{top_tenure}**.
- Highest observed churn-rate payment method group: **{top_payment}**.
- These are descriptive associations in a cross-sectional sample; they do not establish that a contract, payment method, or offer causes churn.

## Churn rate by contract

{churn_contract.rename(columns={'count':'customers','mean':'churn_rate'}).to_markdown(floatfmt='.3f')}

## Churn rate by tenure group

{churn_tenure.rename(columns={'count':'customers','mean':'churn_rate'}).to_markdown(floatfmt='.3f')}

## Churn rate by payment method

{churn_payment.rename(columns={'count':'customers','mean':'churn_rate'}).to_markdown(floatfmt='.3f')}

## Data limitations

1. This dataset is a **single customer snapshot**, not monthly transaction history. It can show differences across tenure and customer attributes, but cannot directly calculate month-by-month usage trends or a true forward-looking time-to-churn sequence.
2. It contains no explicit complaint/ticket count. `TechSupport` is a subscribed service flag, **not** a complaint variable. Do not rename it as a complaint feature.
3. `TotalCharges` is cumulative historical billing, not profit or customer margin. Later CLV work must state its assumptions and avoid presenting this field as true lifetime profit.
4. `customerID` is an identifier and should not be used as a predictive input.
5. The churn label records observed churn in the dataset; it does not demonstrate that a retention offer would change that outcome.

## Generated figures

See `reports/figures/` for the churn distribution, contract, tenure, internet service, monthly charge, tenure distribution, and tenure-versus-monthly-charge charts.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "eda_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    if not FEATURE_DATA_PATH.exists():
        prepare_main()
    df = pd.read_csv(FEATURE_DATA_PATH)
    # Preserve the target as integer when read back from CSV.
    df["Churn"] = pd.to_numeric(df["Churn"], errors="raise").astype(int)
    plot_eda(df)
    summary = write_eda_summary(df)
    print(f"EDA charts saved under: {FIGURES_DIR.resolve()}")
    print(f"EDA summary saved: {summary.resolve()}")
    print(f"Observed churn rate: {df['Churn'].mean():.2%}")

if __name__ == "__main__":
    main()
