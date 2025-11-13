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
# NEW PART ⭐ Step 5 — Portfolio Builder (CROSS-SHEET + WEIGHTS)
# ------------------------------------------------------------
st.subheader("③ Build Your Portfolio")

st.write("You can select **any ETFs from ANY sheet**, regardless of current filters.")

# Standardize names to avoid None values
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

# Allow user to select ANY ETF across all sheets
all_indices = df.index.tolist()
selected_portfolio_rows = st.multiselect(
    "Select ETFs to include in your portfolio (cross-sheet supported):",
    options=all_indices,
)

if selected_portfolio_rows:
    portfolio = df.loc[selected_portfolio_rows, ["Name_Std", "Ticker_Std", "SourceSheet"]].copy()
    portfolio.reset_index(drop=True, inplace=True)

    st.markdown("### 🔧 Adjust ETF Weights")

    total_weight = 0
    weight_list = []

    # Weight input for each ETF
    for i in range(len(portfolio)):
        default_w = round(1 / len(portfolio), 2)
        w = st.number_input(
            f"Weight for {portfolio.loc[i, 'Name_Std']} (0-1):",
            min_value=0.0,
            max_value=1.0,
            value=float(default_w),
            step=0.05,
            key=f"weight_{i}"
        )
        weight_list.append(w)
        total_weight += w

    # Normalize weights so total = 1
    portfolio["Weight"] = [w / total_weight for w in weight_list] if total_weight > 0 else 0

    st.markdown("### 📊 Your Final Portfolio")
    st.dataframe(portfolio)

    # Download
    csv = portfolio.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Portfolio as CSV", csv, "portfolio.csv", "text/csv")

else:
    st.info("Select ETFs to build your portfolio.")

