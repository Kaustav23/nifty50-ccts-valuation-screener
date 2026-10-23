# NIFTY 50 CCTS Valuation Screener

Institutional quantitative screening tool evaluating the EPS erosion and constant-multiple equity price drawdown of Indian heavy-industry constituents under the Bureau of Energy Efficiency (BEE) Carbon Credit Trading Scheme (CCTS).

## Executive Summary & Investment Thesis
This quantitative engine models the equity pricing impact of India's formal carbon market on heavy-industry stocks. By bridging synthetic SEBI BRSR sustainability disclosures with live equity data, this tool isolates the specific corporate carbon liabilities that traditional P/E ratios often ignore, providing a pure financial view of climate transition risk.

## Key Financial Mechanics
* **Live Equity Integration:** Pulls real-time NSE stock prices, shares outstanding, and P/E ratios via `yfinance` to maintain accurate, live baseline valuations.
* **Valuation Engine:** Calculates the statutory CCTS liability based on carbon intensity shortfalls, applies India's 25.17% corporate tax shield (Section 115BAA), and computes the Carbon-Adjusted Profit After Tax (PAT).
* **Constant-Multiple Drawdown:** Translates the post-tax liability directly into Earnings Per Share (EPS) erosion and projects the implied stock price drawdown assuming P/E multiples remain constant.

## Installation & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
