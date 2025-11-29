# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:15:06 2025

@author: Micha
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="ETF Robo-Advisor – MVO Optimizer",
    layout="wide"
)

st.title("ETF Robo-Advisor – Portfolio Optimization (MVO + Efficient Frontier)")
st.caption(
    "This app uses a local ETF database (metadata, prices, returns) to filter ETFs, "
    "construct a portfolio universe, and run Mean–Variance Optimization with an "
    "efficient frontier and maximum Sharpe portfolio."
)

# ============================================================
# Load local database
# ============================================================

@st.cache_data
def load_data():
    # Read local Excel databases
    meta = pd.read_excel("etf_metadata.xlsx")
    prices = pd.read_excel("prices_1y.xlsx", index_col=0)
    returns = pd.read_excel("returns_1y.xlsx", index_col=0)

    # Ensure datetime index
    prices.index = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)

    # Check ticker column
    if "YF_Ticker" not in meta.columns:
        raise ValueError("Column 'YF_Ticker' is missing in etf_metadata.xlsx.")

    # Keep only tickers that exist in all datasets
    tickers_meta = set(meta["YF_Ticker"].dropna().astype(str))
    tickers_prices = set(prices.columns.astype(str))
    tickers_returns = set(returns.columns.astype(str))

    valid_tickers = sorted(list(tickers_meta & tickers_prices & tickers_returns))

    # Subset to valid tickers
    prices = prices[valid_tickers]
    returns = returns[valid_tickers]
    meta_use = meta[meta["YF_Ticker"].isin(valid_tickers)].reset_index(drop=True)

    return meta_use, prices, returns


meta, prices, returns = load_data()

# ============================================================
# Step 1 – Filter ETF universe using metadata
# ============================================================

st.header("Step 1 – Filter ETF Universe")

col1, col2, col3, col4 = st.columns(4)

# Filter by SourceSheet (asset group)
with col1:
    if "SourceSheet" in meta.columns:
        groups = sorted(meta["SourceSheet"].unique().tolist())
    else:
        groups = ["All"]
    selected_groups = st.multiselect(
        "Asset group (sheet)",
        options=groups,
        default=groups
    )

# Numeric filters (only applied if the column exists)
with col2:
    if "1D Volume" in meta.columns:
        vol_min = st.number_input(
            "Min 1D Volume",
            min_value=0,
            value=int(meta["1D Volume"].quantile(0.25))
        )
    else:
        vol_min = None

with col3:
    if "Open Interest" in meta.columns:
        oi_min = st.number_input(
            "Min Open Interest",
            min_value=0,
            value=int(meta["Open Interest"].quantile(0.25))
        )
    else:
        oi_min = None

with col4:
    if "Bid Ask Spread" in meta.columns:
        bid_max = st.number_input(
            "Max Bid-Ask Spread",
            min_value=0.0,
            value=float(meta["Bid Ask Spread"].quantile(0.75))
        )
    else:
        bid_max = None


# Helper: safe numeric conversion
def safe_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Apply filters
universe = meta.copy()

if "SourceSheet" in universe.columns and selected_groups:
    universe = universe[universe["SourceSheet"].isin(selected_groups)]

universe = safe_numeric(universe, "1D Volume")
universe = safe_numeric(universe, "Open Interest")
universe = safe_numeric(universe, "Bid Ask Spread")

if vol_min is not None and "1D Volume" in universe.columns:
    universe = universe[universe["1D Volume"] >= vol_min]

if oi_min is not None and "Open Interest" in universe.columns:
    universe = universe[universe["Open Interest"] >= oi_min]

if bid_max is not None and "Bid Ask Spread" in universe.columns:
    universe = universe[universe["Bid Ask Spread"] <= bid_max]

universe = universe.reset_index(drop=True)

st.write(f"Number of ETFs after filtering: **{len(universe)}**")

if len(universe) == 0:
    st.error("No ETFs passed the filters. Please loosen the filters above.")
    st.stop()

st.dataframe(
    universe[
        [
            col for col in [
                "Name", "Ticker", "YF_Ticker", "SourceSheet",
                "1D Volume", "30D Avg Volume",
                "Open Interest", "Bid Ask Spread"
            ]
            if col in universe.columns
        ]
    ],
    use_container_width=True
)

# ============================================================
# Step 2 – Select ETFs for optimization
# ============================================================

st.header("Step 2 – Select ETFs for Optimization")

options = universe.index.tolist()
labels = [
    f"{universe.loc[i, 'YF_Ticker']} – {universe.loc[i, 'Name']}"
    for i in options
]

selected_indices = st.multiselect(
    "Choose ETFs to include in the optimized portfolio:",
    options=options,
    format_func=lambda i: labels[i]
)

if len(selected_indices) < 2:
    st.info("Please select at least **2** ETFs to run optimization.")
    st.stop()

selected_universe = universe.loc[selected_indices].reset_index(drop=True)
tickers_sel = selected_universe["YF_Ticker"].tolist()

st.write(f"Selected ETFs ({len(tickers_sel)}): {tickers_sel}")

# ============================================================
# Step 3 – Return data for selected ETFs
# ============================================================

st.header("Step 3 – Return Data for Selected ETFs")

# Subset returns matrix
ret = returns[tickers_sel].copy()

# Drop rows with all NaNs and then drop columns that become all NaN
ret = ret.dropna(how="all")
ret = ret.dropna(axis=1, how="all")

if ret.shape[1] < 2:
    st.error(
        "Selected ETFs do not have enough valid return data after cleaning. "
        "Please choose a different set of ETFs."
    )
    st.stop()

st.write(f"Return matrix shape: {ret.shape[0]} days × {ret.shape[1]} ETFs")
st.line_chart(ret)

# ============================================================
# Step 4 – Run Mean–Variance Optimization
# ============================================================

st.header("Step 4 – Mean–Variance Optimization (GMV & Max Sharpe)")

run_opt = st.button("🚀 Run Optimization")

if run_opt:

    # Annualized mean and covariance
    mu = ret.mean() * 252
    cov = ret.cov() * 252

    # Regularize covariance to avoid numerical issues
    eps = 1e-6
    Sigma = cov.values
    Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])
    Sigma_inv = np.linalg.pinv(Sigma_reg)

    n = len(mu)
    ones = np.ones(n)
    rf = 0.02  # risk-free rate

    # ---------- GMV portfolio (closed form) ----------
    # w_gmv = Σ^{-1} 1 / (1' Σ^{-1} 1)
    num_gmv = Sigma_inv @ ones
    den_gmv = ones.T @ Sigma_inv @ ones
    w_gmv = num_gmv / den_gmv

    # ---------- Max Sharpe (tangency) portfolio ----------
    # w_ms ∝ Σ^{-1} (μ - rf 1), then normalized to sum 1
    excess = mu.values - rf * ones
    raw_ms = Sigma_inv @ excess
    w_ms = raw_ms / np.sum(raw_ms)

    def portfolio_stats(w: np.ndarray):
        """Compute annualized return, volatility and Sharpe."""
        w = np.array(w)
        p_ret = float(mu.values @ w)
        p_vol = float(np.sqrt(w.T @ Sigma_reg @ w))
        if p_vol > 0:
            p_sharpe = (p_ret - rf) / p_vol
        else:
            p_sharpe = np.nan
        return p_ret, p_vol, p_sharpe

    g_ret, g_vol, g_sharpe = portfolio_stats(w_gmv)
    ms_ret, ms_vol, ms_sharpe = portfolio_stats(w_ms)

    # ---------- Efficient frontier (simple convex combination) ----------
    ef_rets = []
    ef_vols = []

    for alpha in np.linspace(0.0, 1.0, 30):
        w_ef = alpha * w_gmv + (1 - alpha) * w_ms
        r, v, _ = portfolio_stats(w_ef)
        ef_rets.append(r)
        ef_vols.append(v)

    # ---------- Plot efficient frontier ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ef_vols, ef_rets, "o-", markersize=3, label="Efficient frontier")
    ax.scatter(g_vol, g_ret, color="orange", s=80, label="GMV portfolio")
    ax.scatter(ms_vol, ms_ret, color="green", s=80, label="Max Sharpe portfolio")
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected annual return (μ)")
    ax.legend()
    st.pyplot(fig)

    # ============================================================
    # Step 5 – Output final portfolio summary
    # ============================================================

    st.header("Step 5 – Portfolio Output Summary")

    # Weights table
    weights_df = pd.DataFrame({
        "ETF": tickers_sel,
        "GMV Weight": w_gmv,
        "Max Sharpe Weight": w_ms
    })

    # Normalize again to avoid tiny numerical drift
    for col in ["GMV Weight", "Max Sharpe Weight"]:
        total = weights_df[col].sum()
        if total != 0:
            weights_df[col] = weights_df[col] / total

    st.subheader("Final Portfolio Weights")
    st.dataframe(weights_df, use_container_width=True)

    # Performance summary
    summary_df = pd.DataFrame({
        "Portfolio": ["GMV", "Max Sharpe"],
        "Annual Return": [g_ret, ms_ret],
        "Annual Volatility": [g_vol, ms_vol],
        "Sharpe Ratio": [g_sharpe, ms_sharpe]
    })

    st.subheader("Performance Summary")
    st.dataframe(summary_df, use_container_width=True)

    # Download weights as Excel
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        weights_df.to_excel(writer, sheet_name="weights", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    st.download_button(
        label="📥 Download Optimized Portfolio (Excel)",
        data=buffer.getvalue(),
        file_name="optimized_portfolio.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


