# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 00:44:24 2025

@author: Micha
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ETF Selector (Dynamic)", layout="wide")

# ------------------------------------------------------------
# Function to load and clean data
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
        df.replace(["-", "N/A", "na", "None"], np.nan, inplace=True)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined, xls.sheet_names

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df, sheet_names = load_data()
st.title("ETF Selector (Dynamic Filtering Demo)")
st.caption("The filtering options below automatically adjust based on the selected ETF group (sheet).")

# ------------------------------------------------------------
# Step 1: User selects ETF Group (sheet)
# ------------------------------------------------------------
st.sidebar.header("1️⃣ Select ETF Group (sheet)")
selected_sheet = st.sidebar.selectbox("Choose an ETF category:", sheet_names)

# Filter the data for the selected sheet
df_selected = df[df["SourceSheet"] == selected_sheet].copy()
st.write(f"**Loaded {len(df_selected)} ETFs from '{selected_sheet}'**")

# ------------------------------------------------------------
# Step 2: Dynamic filter logic mapping
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
# Step 3: Display appropriate filters
# ------------------------------------------------------------
st.sidebar.header("2️⃣ Filter Options")

logic = filter_logic.get(selected_sheet, {"category": [], "numeric": []})

# 1) Category filters
for col in logic["category"]:
    if col in df_selected.columns:
        opts = df_selected[col].dropna().unique().tolist()
        if len(opts) > 0:
            selected_opts = st.sidebar.multiselect(f"{col}:", options=opts, default=opts)
            df_selected = df_selected[df_selected[col].isin(selected_opts)]

# 2) Numeric filters
for col in logic["numeric"]:
    if col in df_selected.columns:
        # convert to numeric
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")
        valid = df_selected[col].dropna()
        if len(valid) > 0:
            low, high = float(valid.min()), float(valid.max())
            # slider range
            low_val, high_val = st.sidebar.slider(f"{col} range", low, high, (low, high))
            df_selected = df_selected[(df_selected[col] >= low_val) & (df_selected[col] <= high_val)]

# ------------------------------------------------------------
# Step 4: Display results
# ------------------------------------------------------------
st.subheader(f"Filtered ETFs in {selected_sheet}: {len(df_selected)}")
if len(df_selected) == 0:
    st.warning("No ETFs match the selected filters. Try broadening your range or removing some filters.")
else:
    st.dataframe(df_selected)

# ------------------------------------------------------------
# Step 5: Simple portfolio builder
# ------------------------------------------------------------
st.subheader("3️⃣ Build Your Portfolio")

if len(df_selected) > 0:
    selected_rows = st.multiselect("Select ETFs by index:", options=df_selected.index.tolist())
    if selected_rows:
        portfolio = df_selected.loc[selected_rows].copy()
        portfolio["Weight"] = 1 / len(portfolio)
        st.write("Your Portfolio (Equal-Weighted):")
        cols_to_show = [c for c in ["Name", "Ticker", "SourceSheet", "Weight"] if c in portfolio.columns]
        st.dataframe(portfolio[cols_to_show])
    else:
        st.info("Select some ETFs from the table to form a portfolio.")
else:
    st.info("Please select a valid ETF group to build a portfolio.")
