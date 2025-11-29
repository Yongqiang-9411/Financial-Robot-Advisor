# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:15:06 2025

@author: Micha
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ETF Selector (Final Version)", layout="wide")

# ------------------------------------------------------------
# Load and clean data
# ------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "ETF List.xlsx"
    xls = pd.ExcelFile(file_path)
    all_dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df["SourceSheet"] = sheet
        df = df.dropna(how="all")
        df.replace(["-", "N/A", "None", "na"], np.nan, inplace=True)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined, xls.sheet_names


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df, sheet_names = load_data()
st.title("ETF Selector (Dynamic + Portfolio Builder)")
st.caption("Dynamic filtering with multi-sheet portfolio builder and adjustable weights.")

# ------------------------------------------------------------
# Step 1: Select ETF Group (sheet)
# ------------------------------------------------------------
st.sidebar.header("① Select ETF Group (sheet)")
selected_sheet = st.sidebar.selectbox("Choose an ETF category:", sheet_names)

df_selected = df[df["SourceSheet"] == selected_sheet].copy()
st.write(f"**Loaded {len(df_selected)} ETFs from '{selected_sheet}'**")


# ------------------------------------------------------------
# Step 2: Dynamic filter logic (UNCHANGED)
# ------------------------------------------------------------
filter_logic = {
    "EM bonds": {
        "category": ["Category", "Subcategory"],
        "numeric": []
    },
    "sector ETFs": {
        "category": [],
        "numeric": ["1D Volume", "30D Avg Volume", "Bid Ask Spread", "Open Interest"]
    },
    "high yield corporate": {
        "category": [],
        "numeric": ["1D Volume", "Bid Ask Spread", "Short Interest%", "Open Interest"]
    },
    "Canadian": {
        "category": [],
        "numeric": ["1D Volume", "Bid Ask Spread", "Implied Liquidity"]
    },
    "bond etfs": {
        "category": [],
        "numeric": ["1D Volume", "Bid Ask Spread", "Agg Traded Val (M USD)"]
    },
    "thematic_strategy": {
        "category": [],
        "numeric": ["Fund Assets (AUM) (M USD)", "Expense Ratio", "YTD Return", "12M Yield", "YTD Flow", "Holdings"]
    }
}


# ------------------------------------------------------------
# Step 3: Display appropriate filters (UNCHANGED)
# ------------------------------------------------------------
st.sidebar.header("② Filter Options")

logic = filter_logic.get(selected_sheet, {"category": [], "numeric": []})

# Category filters
for col in logic["category"]:
    if col in df_selected.columns:
        opts = df_selected[col].dropna().unique().tolist()
        if len(opts) > 0:
            chosen = st.sidebar.multiselect(f"{col}:", options=opts, default=opts)
            df_selected = df_selected[df_selected[col].isin(chosen)]

# Numeric sliders
for col in logic["numeric"]:
    if col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")
        valid = df_selected[col].dropna()
        if len(valid) > 0:
            low, high = float(valid.min()), float(valid.max())
            low_val, high_val = st.sidebar.slider(f"{col} range", low, high, (low, high))
            df_selected = df_selected[(df_selected[col] >= low_val) & (df_selected[col] <= high_val)]


# ------------------------------------------------------------
# Step 4: Display filtered results (UNCHANGED)
# ------------------------------------------------------------
st.subheader(f"Filtered ETFs in {selected_sheet}: {len(df_selected)}")
st.dataframe(df_selected)


# ------------------------------------------------------------
# NEW PART ★ Step 5 – Portfolio Optimization (MVO + Efficient Frontier)
# ---------------------------------------------------------------

st.subheader("📊 Automatic Portfolio Optimization (Mean-Variance Model + Efficient Frontier)")

st.write("Select ETFs from ANY sheet. This tool will automatically compute optimal weights, plot the efficient frontier, and find the maximum Sharpe ratio portfolio.")

# Standardize names
df["Name_Std"] = (
    df["Name"]
    .fillna(df.get("Security"))
    .fillna(df.get("Description"))
    .fillna("Unknown")
)

df["Ticker_Std"] = (
    df["Ticker"]
    .fillna(df.get("Security"))
    .fillna(df.get("Description"))
    .fillna("Unknown")
)

# Select ETFs
all_indices = df.index.tolist()
selected_rows = st.multiselect(
    "Select ETFs to include in your portfolio:",
    options=all_indices,
)

if selected_rows:

    portfolio = df.loc[selected_rows, ["Name_Std", "Ticker_Std", "SourceSheet"]].copy()
    portfolio.reset_index(drop=True, inplace=True)

    st.markdown("### Step 1: Choose Risk Preference")
    risk_choice = st.radio("Risk Level:", ["Low", "Medium", "High"])

    # ---------------------------------------------------
    # Step 2: Download Close prices
    # ---------------------------------------------------
    import yfinance as yf
    import numpy as np
    import pandas as pd

    tickers = portfolio["Ticker_Std"].tolist()

    st.write("Fetching 1-year daily close prices...")

    try:
        price_data = yf.download(tickers, period="1y")["Close"]
    except:
        st.error("Error fetching Close prices from yfinance.")
        st.stop()

    returns = price_data.pct_change().dropna()

    mu = returns.mean() * 252
    cov = returns.cov() * 252
    rf_rate = 0.02    # risk-free rate (2%)

    # ---------------------------------------------------
    # Step 3: Mean-Variance Optimization
    # ---------------------------------------------------
    import cvxpy as cp

    n = len(tickers)
    w = cp.Variable(n)

    if risk_choice == "Low":
        lam = 10.0
    elif risk_choice == "Medium":
        lam = 3.0
    else:
        lam = 0.5

    objective = cp.Maximize(mu.values @ w - lam * cp.quad_form(w, cov.values))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    mvo_weights = np.round(w.value, 4)

    portfolio["MVO_Weight"] = mvo_weights

    # ---------------------------------------------------
    # Step 4: Efficient Frontier
    # ---------------------------------------------------
    num_points = 50
    frontier_returns = []
    frontier_risk = []
    frontier_weights = []

    for t in np.linspace(0, 1, num_points):
        w_front = cp.Variable(n)
        objective_front = cp.Minimize(cp.quad_form(w_front, cov.values))
        constraints_front = [
            cp.sum(w_front) == 1,
            w_front >= 0,
            mu.values @ w_front == mu.min() * (1 - t) + mu.max() * t
        ]
        problem_front = cp.Problem(objective_front, constraints_front)
        try:
            problem_front.solve()
            w_val = np.array(w_front.value).flatten()
            frontier_weights.append(w_val)
            frontier_returns.append(mu.values @ w_val)
            frontier_risk.append(np.sqrt(w_val.T @ cov.values @ w_val))
        except:
            pass

    # ---------------------------------------------------
    # Step 5: Maximum Sharpe Ratio Portfolio
    # ---------------------------------------------------
    w_sharpe = cp.Variable(n)
    objective_sharpe = cp.Maximize((mu.values @ w_sharpe - rf_rate) /
                                   cp.sqrt(cp.quad_form(w_sharpe, cov.values)))

    constraints_sharpe = [
        cp.sum(w_sharpe) == 1,
        w_sharpe >= 0
    ]

    problem_sharpe = cp.Problem(objective_sharpe, constraints_sharpe)
    problem_sharpe.solve()

    sharpe_weights = np.round(w_sharpe.value, 4)

    portfolio["MaxSharpe_Weight"] = sharpe_weights

    # ---------------------------------------------------
    # Step 6: Final Portfolio Performance Metrics
    # ---------------------------------------------------
    def portfolio_stats(weights):
        w = np.array(weights)
        ret = mu.values @ w
        risk = np.sqrt(w.T @ cov.values @ w)
        sharpe = (ret - rf_rate) / risk
        return ret, risk, sharpe

    mvo_ret, mvo_risk, mvo_sharpe = portfolio_stats(mvo_weights)
    ms_ret, ms_risk, ms_sharpe = portfolio_stats(sharpe_weights)

    # ---------------------------------------------------
    # Step 7: Visualization
    # ---------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frontier_risk, frontier_returns, s=10, label="Efficient Frontier")
    ax.scatter(mvo_risk, mvo_ret, color="red", label="MVO Portfolio", s=50)
    ax.scatter(ms_risk, ms_ret, color="green", label="Max Sharpe Portfolio", s=50)

    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------------------
    # Step 8: Display Results
    # ---------------------------------------------------
    st.markdown("### ✅ Optimized Portfolio Results")

    st.write("**MVO Portfolio Performance:**")
    st.write(f"- Annualized Return: **{mvo_ret:.4f}**")
    st.write(f"- Annualized Volatility: **{mvo_risk:.4f}**")
    st.write(f"- Sharpe Ratio: **{mvo_sharpe:.4f}**")

    st.write("---")

    st.write("**Maximum Sharpe Portfolio Performance:**")
    st.write(f"- Annualized Return: **{ms_ret:.4f}**")
    st.write(f"- Annualized Volatility: **{ms_risk:.4f}**")
    st.write(f"- Sharpe Ratio: **{ms_sharpe:.4f}**")

    st.write("---")

    st.dataframe(portfolio)

    # Download final table
    csv = portfolio.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Portfolio as CSV",
        csv,
        "optimized_portfolio.csv",
        "text/csv"
    )

else:
    st.info("Select ETFs to build your optimized portfolio.")

