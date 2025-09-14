# 📊 Marketing Dashboard Project

This project is a **Marketing Intelligence Dashboard** built with **Streamlit**, **Pandas**, and **Plotly**.
It helps marketers and business teams understand how their ad spend connects to revenue, customers, and efficiency metrics.

---

## 🚀 What does it do?

- Cleans raw marketing + business CSV files (Facebook, Google, TikTok, Business data).
- Aggregates them into daily totals, channel summaries, campaign summaries, and state-level performance.
- Computes key metrics like **ROAS, CAC, AOV, CTR, CPC, CPM, Margin ROAS**.
- Provides an **interactive dashboard** with filters (date, channel, state, campaign search).
- Shows easy-to-understand insights with explanations (not just numbers).
- Exports results and charts as CSV/PNG for reports.

---

## 📂 Project Structure

```text
data/
├── raw/             <- put your raw CSV files here (Facebook.csv, Google.csv, TikTok.csv, Business.csv)
├── cleaned/         <- processed & aggregated data lives here (created by scripts)
└── figures/         <- exported charts and anomaly reports (optional)

notebook/
└── 04_EDA.ipynb     <- for data exploration (Jupyter Notebook)

scripts/
├── 01_clean.py      <- cleans raw files (removes duplicates, fixes dates, normalizes columns)
├── 02_aggregate.py  <- aggregates cleaned data into daily totals, channel summaries, etc.
└── 03_metrics.py    <- computes derived metrics (ROAS, CAC, AOV, etc.)

src/app/
└── dashboard.py     <- the main Streamlit dashboard

venv/                <- your Python virtual environment (ignored in Git)
requirements.txt     <- Python dependencies
