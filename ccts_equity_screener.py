"""
NIFTY 50 Carbon Credit Trading Scheme (CCTS) Equity Valuation Screener
======================================================================
Institutional quantitative screening tool evaluating the EPS erosion and
constant-multiple equity price drawdown of Indian heavy-industry constituents
under the Bureau of Energy Efficiency (BEE) Carbon Credit Trading Scheme (CCTS).

Usage:
    streamlit run ccts_equity_screener.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================

INDIA_CORPORATE_TAX_RATE: float = 0.2517  # Section 115BAA (22% base + 10% surcharge + 4% cess)

NIFTY50_HEAVY_INDUSTRY_TICKERS: List[str] = [
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ULTRACEMCO.NS",
    "HINDALCO.NS",
    "NTPC.NS",
]

# Resilient institutional fallback snapshot in case Yahoo Finance rate-limits
# or returns partial metadata for NSE (.NS) tickers during market off-hours.
FALLBACK_EQUITY_SNAPSHOT: Dict[str, Dict[str, float]] = {
    "TATASTEEL.NS": {
        "Live_Price_INR": 152.40,
        "Shares_Outstanding": 12_483_520_000,
        "PE_Ratio": 28.50,
    },
    "JSWSTEEL.NS": {
        "Live_Price_INR": 965.00,
        "Shares_Outstanding": 2_445_450_000,
        "PE_Ratio": 26.80,
    },
    "ULTRACEMCO.NS": {
        "Live_Price_INR": 11_250.00,
        "Shares_Outstanding": 288_690_000,
        "PE_Ratio": 42.10,
    },
    "HINDALCO.NS": {
        "Live_Price_INR": 685.50,
        "Shares_Outstanding": 2_220_500_000,
        "PE_Ratio": 14.20,
    },
    "NTPC.NS": {
        "Live_Price_INR": 412.00,
        "Shares_Outstanding": 9_696_670_000,
        "PE_Ratio": 19.40,
    },
}


# ==============================================================================
# 2. MODULE 1: LIVE EQUITY INTEGRATION (yfinance)
# ==============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_equity_data(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pulls live stock prices, shares outstanding, and trailing/forward P/E ratios
    from Yahoo Finance (`yfinance`) for the specified NSE tickers.

    Derives Baseline TTM EPS and Baseline PAT strictly from market price and P/E
    (EPS = Price / PE; PAT = EPS * Shares Outstanding) to maintain internal
    consistency across constant-multiple valuation calculations.
    """
    records = []
    fallback_warnings = []

    for ticker_symbol in tickers:
        fallback = FALLBACK_EQUITY_SNAPSHOT.get(
            ticker_symbol,
            {"Live_Price_INR": 500.0, "Shares_Outstanding": 1_000_000_000, "PE_Ratio": 20.0},
        )
        used_fallback = False

        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info or {}

            # 1. Extract Live Price
            live_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )

            # 2. Extract Shares Outstanding
            shares_out = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")

            # 3. Extract Current P/E Ratio (prefer trailingPE, fallback to forwardPE)
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")

            if not live_price or live_price <= 0:
                live_price = fallback["Live_Price_INR"]
                used_fallback = True
            if not shares_out or shares_out <= 0:
                shares_out = fallback["Shares_Outstanding"]
                used_fallback = True
            if not pe_ratio or pe_ratio <= 0:
                pe_ratio = fallback["PE_Ratio"]
                used_fallback = True

        except Exception:
            live_price = fallback["Live_Price_INR"]
            shares_out = fallback["Shares_Outstanding"]
            pe_ratio = fallback["PE_Ratio"]
            used_fallback = True

        if used_fallback:
            fallback_warnings.append(ticker_symbol)

        # Derive Baseline EPS and Baseline PAT from Price and P/E for strict multiple consistency
        baseline_eps = float(live_price) / float(pe_ratio)
        baseline_pat_inr = baseline_eps * float(shares_out)
        market_cap_inr = float(live_price) * float(shares_out)

        records.append(
            {
                "Ticker": ticker_symbol,
                "Live_Price_INR": float(live_price),
                "Shares_Outstanding": float(shares_out),
                "Market_Cap_Cr": market_cap_inr / 1e7,
                "PE_Ratio": float(pe_ratio),
                "Baseline_EPS_INR": baseline_eps,
                "Baseline_PAT_INR": baseline_pat_inr,
                "Data_Source": "Fallback Snapshot" if used_fallback else "Live yfinance",
            }
        )

    return pd.DataFrame(records), fallback_warnings


# ==============================================================================
# 3. MODULE 2: SYNTHETIC ESG / BRSR DATABASE
# ==============================================================================

def get_synthetic_brsr_data() -> pd.DataFrame:
    """
    Constructs a synthetic Business Responsibility and Sustainability Report (BRSR)
    dataset containing BEE-aligned Baseline Carbon Intensity targets, Actual Carbon
    Intensity figures (tCO2e per unit of output), and Annual Production Volumes.
    """
    brsr_data = [
        {
            "Ticker": "TATASTEEL.NS",
            "Company": "Tata Steel Ltd.",
            "Sector": "Iron & Steel",
            "Output_Unit": "Tonnes Crude Steel",
            "Baseline_Carbon_Intensity": 2.15,  # BEE Target tCO2e / tcs
            "Actual_Carbon_Intensity": 2.48,    # Reported BRSR Scope 1+2 intensity
            "Annual_Production_Volume": 20_800_000,  # ~20.8 MTPA India operations
        },
        {
            "Ticker": "JSWSTEEL.NS",
            "Company": "JSW Steel Ltd.",
            "Sector": "Iron & Steel",
            "Output_Unit": "Tonnes Crude Steel",
            "Baseline_Carbon_Intensity": 2.12,  # BEE Target tCO2e / tcs
            "Actual_Carbon_Intensity": 2.49,    # Higher BF-BOF share intensity
            "Annual_Production_Volume": 24_500_000,  # ~24.5 MTPA India operations
        },
        {
            "Ticker": "ULTRACEMCO.NS",
            "Company": "UltraTech Cement Ltd.",
            "Sector": "Cement",
            "Output_Unit": "Tonnes Cementitious",
            "Baseline_Carbon_Intensity": 0.54,  # BEE Target tCO2e / t cement
            "Actual_Carbon_Intensity": 0.61,    # Clinker factor intensity gap
            "Annual_Production_Volume": 115_000_000,  # ~115 MTPA consolidated domestic
        },
        {
            "Ticker": "HINDALCO.NS",
            "Company": "Hindalco Industries Ltd.",
            "Sector": "Aluminium & Metals",
            "Output_Unit": "Tonnes Aluminium",
            "Baseline_Carbon_Intensity": 8.60,  # BEE Target tCO2e / t primary aluminium
            "Actual_Carbon_Intensity": 9.85,    # Captive coal-fired smelter intensity
            "Annual_Production_Volume": 1_350_000,  # ~1.35 MTPA domestic smelting
        },
        {
            "Ticker": "NTPC.NS",
            "Company": "NTPC Ltd.",
            "Sector": "Thermal Power / Utilities",
            "Output_Unit": "MWh Net Generation",
            "Baseline_Carbon_Intensity": 0.78,  # Threshold tCO2e / MWh
            "Actual_Carbon_Intensity": 0.85,    # Coal fleet average emission factor
            "Annual_Production_Volume": 340_000_000,  # ~340 Million MWh (340 BU)
        },
    ]
    return pd.DataFrame(brsr_data)


# ==============================================================================
# 4. MODULE 3: CCTS VALUATION & EPS IMPACT ENGINE
# ==============================================================================

@dataclass
class ValuationParameters:
    carbon_price_inr: float
    tax_rate: float = INDIA_CORPORATE_TAX_RATE


def run_ccts_valuation_engine(
    equity_df: pd.DataFrame,
    brsr_df: pd.DataFrame,
    params: ValuationParameters,
) -> pd.DataFrame:
    """
    Executes the CCTS liability and constant-P/E valuation drawdown model.

    Formulas:
    1. Intensity_Shortfall (tCO2e/unit) = max(0, Actual_Carbon_Intensity - Baseline_Carbon_Intensity)
    2. Unabated_Emissions_Shortfall (tCO2e) = Intensity_Shortfall * Annual_Production_Volume
    3. Gross_CCTS_Liability (INR) = Unabated_Emissions_Shortfall * Carbon_Price
    4. Net_CCTS_Liability (INR) = Gross_CCTS_Liability * (1 - Corporate_Tax_Rate) [25.17% Tax Shield]
    5. Carbon_Adjusted_PAT (INR) = Baseline_PAT - Net_CCTS_Liability
    6. Carbon_Adjusted_EPS (INR) = Carbon_Adjusted_PAT / Shares_Outstanding
    7. EPS_Erosion_Pct (%) = ((Baseline_EPS - Carbon_Adjusted_EPS) / Baseline_EPS) * 100
    8. Implied_Share_Price (INR) = max(0, Carbon_Adjusted_EPS * PE_Ratio)
    9. Carbon_Discount_Drawdown_Pct (%) = ((Implied_Share_Price - Live_Price) / Live_Price) * 100
    """
    df = pd.merge(brsr_df, equity_df, on="Ticker", how="inner")

    # 1. Carbon Intensity Shortfall (strictly positive liability shortfall)
    df["Intensity_Shortfall"] = np.maximum(
        0.0, df["Actual_Carbon_Intensity"] - df["Baseline_Carbon_Intensity"]
    )

    # 2. Total Compliance Shortfall Volume (tCO2e)
    df["Shortfall_Tonnes_CO2e"] = (
        df["Intensity_Shortfall"] * df["Annual_Production_Volume"]
    )

    # 3. Gross Pre-Tax CCTS Liability (INR)
    df["Gross_CCTS_Liability_INR"] = (
        df["Shortfall_Tonnes_CO2e"] * params.carbon_price_inr
    )

    # 4. Tax Shield (25.17%) & Net Post-Tax CCTS Liability (INR)
    df["Tax_Shield_INR"] = df["Gross_CCTS_Liability_INR"] * params.tax_rate
    df["Net_CCTS_Liability_INR"] = df["Gross_CCTS_Liability_INR"] * (
        1.0 - params.tax_rate
    )

    # Convert INR liabilities to ₹ Crores (1 Crore = 10^7 INR) for institutional readability
    df["Gross_Liability_Cr"] = df["Gross_CCTS_Liability_INR"] / 1e7
    df["Net_Liability_Cr"] = df["Net_CCTS_Liability_INR"] / 1e7
    df["Baseline_PAT_Cr"] = df["Baseline_PAT_INR"] / 1e7

    # 5. Carbon-Adjusted PAT (INR & ₹ Crores)
    df["Carbon_Adjusted_PAT_INR"] = (
        df["Baseline_PAT_INR"] - df["Net_CCTS_Liability_INR"]
    )
    df["Carbon_Adjusted_PAT_Cr"] = df["Carbon_Adjusted_PAT_INR"] / 1e7

    # 6. Carbon-Adjusted EPS & EPS Erosion (%)
    df["Carbon_Adjusted_EPS_INR"] = (
        df["Carbon_Adjusted_PAT_INR"] / df["Shares_Outstanding"]
    )
    df["EPS_Erosion_Pct"] = (
        (df["Baseline_EPS_INR"] - df["Carbon_Adjusted_EPS_INR"])
        / df["Baseline_EPS_INR"]
    ) * 100.0

    # 7. Implied Stock Price at Constant P/E & "Carbon Discount" Drawdown (%)
    # Floor implied equity price at ₹0.00 in extreme stress scenarios where PAT turns negative
    df["Implied_Price_INR"] = np.maximum(
        0.0, df["Carbon_Adjusted_EPS_INR"] * df["PE_Ratio"]
    )
    df["Price_Drawdown_Pct"] = (
        (df["Implied_Price_INR"] - df["Live_Price_INR"]) / df["Live_Price_INR"]
    ) * 100.0

    # Carbon Discount Risk Score (absolute percentage drawdown for descending risk ranking)
    df["Carbon_Discount_Risk_Pct"] = -df["Price_Drawdown_Pct"]

    # Sort descending by Carbon Discount Risk (highest EPS/price erosion at top)
    df = df.sort_values(by="Carbon_Discount_Risk_Pct", ascending=False).reset_index(
        drop=True
    )
    df.index = df.index + 1  # 1-based institutional ranking index
    df.index.name = "Risk Rank"

    return df


# ==============================================================================
# 5. MODULE 4: INSTITUTIONAL STREAMLIT DASHBOARD
# ==============================================================================

def render_dashboard() -> None:
    st.set_page_config(
        page_title="NIFTY 50 CCTS Valuation Screener",
        page_icon="📉",
        layout="wide",
    )

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ CCTS Policy & Pricing Controls")
    st.sidebar.markdown(
        "Configure the Bureau of Energy Efficiency (BEE) **Carbon Credit Certificate (CCC)** "
        "clearing price to stress-test NIFTY 50 heavy-industry earnings."
    )

    carbon_price_inr = st.sidebar.slider(
        label="Carbon Price (₹ / tCO₂e)",
        min_value=500,
        max_value=3000,
        value=1500,
        step=50,
        help="Simulated clearing price per tonne of CO₂ equivalent on the Indian Energy Exchange (IEX).",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🏛️ Fiscal & Valuation Assumptions")
    st.sidebar.metric(
        label="Corporate Tax Shield (Sec 115BAA)",
        value=f"{INDIA_CORPORATE_TAX_RATE * 100:.2f}%",
        help="22% base corporate tax + 10% surcharge + 4% health & education cess = 25.17%. "
        "CCTS compliance costs are treated as deductible operating expenses.",
    )
    st.sidebar.caption(
        "**Valuation Methodology:** Constant P/E multiple framework. "
        "Post-tax CCTS compliance liabilities directly reduce trailing PAT and EPS, "
        "driving a proportional implied equity price drawdown (`ΔP/P = ΔEPS/EPS`)."
    )

    if st.sidebar.button("🔄 Refresh Live Market Data", use_container_width=True):
        st.cache_data.clear()

    # --- Main Header ---
    st.title("🏛️ NIFTY 50 Carbon Credit Trading Scheme (CCTS) Valuation Screener")
    st.markdown(
        "**Institutional Quantitative Desk** | Evaluating the post-tax EPS erosion and "
        "constant-multiple equity valuation discount across Indian heavy-industry leaders."
    )

    # --- Execute Data & Valuation Pipeline ---
    with st.spinner("Fetching live NSE equity data via yfinance and running CCTS valuation engine..."):
        equity_df, fallback_tickers = fetch_live_equity_data(NIFTY50_HEAVY_INDUSTRY_TICKERS)
        brsr_df = get_synthetic_brsr_data()
        valuation_df = run_ccts_valuation_engine(
            equity_df=equity_df,
            brsr_df=brsr_df,
            params=ValuationParameters(carbon_price_inr=float(carbon_price_inr)),
        )

    if fallback_tickers:
        st.info(
            f"**Data Feed Notice:** Using institutional reference snapshot for "
            f"`{', '.join(fallback_tickers)}` due to Yahoo Finance rate-limiting or off-market feed gaps."
        )

    # --- Top-Level Portfolio KPI Strip ---
    total_shortfall_mt = valuation_df["Shortfall_Tonnes_CO2e"].sum() / 1e6
    total_gross_liab_cr = valuation_df["Gross_Liability_Cr"].sum()
    total_net_liab_cr = valuation_df["Net_Liability_Cr"].sum()
    avg_eps_erosion = valuation_df["EPS_Erosion_Pct"].mean()
    worst_row = valuation_df.iloc[0]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        label="Aggregate Shortfall Volume",
        value=f"{total_shortfall_mt:.2f} MtCO₂e",
        delta=f"@ ₹{carbon_price_inr:,}/tCO₂e",
        delta_color="off",
    )
    kpi2.metric(
        label="Post-Tax CCTS Liability (Basket)",
        value=f"₹{total_net_liab_cr:,.0f} Cr",
        delta=f"Gross: ₹{total_gross_liab_cr:,.0f} Cr (25.17% Shield)",
        delta_color="inverse",
    )
    kpi3.metric(
        label="Mean Basket EPS Erosion",
        value=f"-{avg_eps_erosion:.2f}%",
        delta="Constant P/E Implied Drawdown",
        delta_color="inverse",
    )
    kpi4.metric(
        label="Highest Carbon Discount Risk",
        value=f"{worst_row['Ticker'].replace('.NS', '')}",
        delta=f"{worst_row['Price_Drawdown_Pct']:.2f}% Implied Price Impact",
        delta_color="inverse",
    )

    st.divider()

    # --- Main Institutional Ranking Table ---
    st.subheader(
        f"📊 Cross-Sectional 'Carbon Discount' Risk Ranking (Carbon Price: ₹{carbon_price_inr:,}/tCO₂e)"
    )

    display_cols = [
        "Ticker",
        "Company",
        "Sector",
        "Live_Price_INR",
        "PE_Ratio",
        "Baseline_Carbon_Intensity",
        "Actual_Carbon_Intensity",
        "Intensity_Shortfall",
        "Shortfall_Tonnes_CO2e",
        "Gross_Liability_Cr",
        "Net_Liability_Cr",
        "Baseline_PAT_Cr",
        "Carbon_Adjusted_PAT_Cr",
        "Baseline_EPS_INR",
        "Carbon_Adjusted_EPS_INR",
        "EPS_Erosion_Pct",
        "Implied_Price_INR",
        "Price_Drawdown_Pct",
    ]

    styled_table = (
        valuation_df[display_cols]
        .rename(
            columns={
                "Live_Price_INR": "Live Price (₹)",
                "PE_Ratio": "P/E (x)",
                "Baseline_Carbon_Intensity": "Baseline Int.",
                "Actual_Carbon_Intensity": "Actual Int.",
                "Intensity_Shortfall": "Shortfall / Unit",
                "Shortfall_Tonnes_CO2e": "Total Shortfall (tCO₂e)",
                "Gross_Liability_Cr": "Gross Liab. (₹ Cr)",
                "Net_Liability_Cr": "Post-Tax Liab. (₹ Cr)",
                "Baseline_PAT_Cr": "Base PAT (₹ Cr)",
                "Carbon_Adjusted_PAT_Cr": "Adj. PAT (₹ Cr)",
                "Baseline_EPS_INR": "Base EPS (₹)",
                "Carbon_Adjusted_EPS_INR": "Adj. EPS (₹)",
                "EPS_Erosion_Pct": "EPS Erosion (%)",
                "Implied_Price_INR": "Implied Price (₹)",
                "Price_Drawdown_Pct": "Carbon Discount (%)",
            }
        )
        .style.format(
            {
                "Live Price (₹)": "₹{:,.2f}",
                "P/E (x)": "{:.1f}x",
                "Baseline Int.": "{:.2f}",
                "Actual Int.": "{:.2f}",
                "Shortfall / Unit": "{:.2f}",
                "Total Shortfall (tCO₂e)": "{:,.0f}",
                "Gross Liab. (₹ Cr)": "₹{:,.1f}",
                "Net Liab. (₹ Cr)": "₹{:,.1f}",
                "Post-Tax Liab. (₹ Cr)": "₹{:,.1f}",
                "Base PAT (₹ Cr)": "₹{:,.0f}",
                "Adj. PAT (₹ Cr)": "₹{:,.0f}",
                "Base EPS (₹)": "₹{:,.2f}",
                "Adj. EPS (₹)": "₹{:,.2f}",
                "EPS Erosion (%)": "-{:.2f}%",
                "Implied Price (₹)": "₹{:,.2f}",
                "Carbon Discount (%)": "{:.2f}%",
            }
        )
        .background_gradient(
            subset=["EPS Erosion (%)"], cmap="Reds"
        )
        .background_gradient(
            subset=["Carbon Discount (%)"], cmap="Reds_r"
        )
    )

    st.dataframe(styled_table, use_container_width=True, height=260)

    # --- Secondary Analytical Views (Tabs) ---
    tab1, tab2 = st.tabs(
        ["📈 Carbon Price Sensitivity Matrix", "📋 Synthetic BRSR & Production Audit"]
    )

    with tab1:
        st.markdown(
            "##### Implied Stock Price Drawdown (%) Across Carbon Price Scenarios (₹500 – ₹3,000/tCO₂e)"
        )
        price_grid = list(range(500, 3500, 500))
        sensitivity_records = []
        for cp in price_grid:
            sim_df = run_ccts_valuation_engine(
                equity_df=equity_df,
                brsr_df=brsr_df,
                params=ValuationParameters(carbon_price_inr=float(cp)),
            )
            row_dict = {"Carbon Price (₹/tCO₂e)": f"₹{cp:,}"}
            for _, r in sim_df.iterrows():
                row_dict[r["Ticker"]] = f"{r['Price_Drawdown_Pct']:.2f}%"
            sensitivity_records.append(row_dict)

        st.table(pd.DataFrame(sensitivity_records).set_index("Carbon Price (₹/tCO₂e)"))

    with tab2:
        st.markdown(
            "##### Underlying Synthetic BRSR Intensity & Live Equity Metadata"
        )
        audit_cols = [
            "Ticker",
            "Company",
            "Sector",
            "Output_Unit",
            "Annual_Production_Volume",
            "Baseline_Carbon_Intensity",
            "Actual_Carbon_Intensity",
            "Shares_Outstanding",
            "Market_Cap_Cr",
            "Data_Source",
        ]
        st.dataframe(
            valuation_df[audit_cols].style.format(
                {
                    "Annual_Production_Volume": "{:,.0f}",
                    "Baseline_Carbon_Intensity": "{:.2f}",
                    "Actual_Carbon_Intensity": "{:.2f}",
                    "Shares_Outstanding": "{:,.0f}",
                    "Market_Cap_Cr": "₹{:,.0f} Cr",
                }
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    render_dashboard()