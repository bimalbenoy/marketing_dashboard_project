# src/app/dashboard.py
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from typing import Optional

st.set_page_config(layout="wide", page_title="Marketing Intelligence")

# ===== Glossary & helpers =====
GLOSSARY = {
    "Blended ROAS": {
        "formula": "total_revenue ÷ spend",
        "meaning": "How much revenue you generate per $1 of total marketing spend across all channels.",
        "good_when": "Higher is better; > 3.0 is strong for many DTC businesses."
    },
    "ROAS (Attributed)": {
        "formula": "attributed_revenue ÷ spend",
        "meaning": "Channel-attributed revenue per $1 of spend (model-based).",
        "good_when": "Higher is better; compare with blended ROAS to spot attribution gaps."
    },
    "CAC": {
        "formula": "spend ÷ new_customers",
        "meaning": "Average cost to acquire one new customer.",
        "good_when": "Lower is better; compare to first-order margin or LTV."
    },
    "AOV": {
        "formula": "total_revenue ÷ orders",
        "meaning": "Average revenue per order.",
        "good_when": "Higher is better; bundling/upsells increase this."
    },
    "Margin ROAS": {
        "formula": "gross_profit ÷ spend",
        "meaning": "Profit (after COGS) per $1 of spend.",
        "good_when": "Higher is better; captures unit economics."
    },
    "CTR": {
        "formula": "clicks ÷ impressions",
        "meaning": "Share of people who clicked after seeing an ad.",
        "good_when": "Higher is better; creative/targeting lever."
    },
    "CPC": {
        "formula": "spend ÷ clicks",
        "meaning": "Average cost per click.",
        "good_when": "Lower is better; bids/quality score lever."
    },
    "CPM": {
        "formula": "(spend × 1000) ÷ impressions",
        "meaning": "Cost to reach 1,000 impressions.",
        "good_when": "Lower is generally better; depends on audience quality."
    },
}

def fmt_money(x): 
    return "N/A" if pd.isna(x) else f"${x:,.0f}"

def fmt_money2(x): 
    return "N/A" if pd.isna(x) else f"${x:,.2f}"

def fmt_ratio(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def micro_insight(metric: str, value: float, context: dict) -> str:
    """Tiny one-liner inside the popover to guide non-technical users."""
    if metric == "Blended ROAS":
        if pd.isna(value): return "Needs spend & revenue to compute."
        if value >= 3.0:   return "Efficient growth: every $1 drives $3+ revenue."
        if value >= 1.5:   return "Workable—optimize channels and funnel to lift further."
        return "Inefficient: shift budget or improve conversion."
    if metric == "CAC":
        if pd.isna(value): return "Needs spend & new customers."
        goal = context.get("bench_cac")
        if goal is not None:
            return "Above goal (high)" if value > goal else "On/under goal (good)"
        return "Lower is better; compare to LTV/margin."
    if metric == "AOV":
        if pd.isna(value): return "Needs revenue & orders."
        return "Upsells/bundles and free-shipping thresholds can lift AOV."
    if metric == "Margin ROAS":
        if pd.isna(value): return "Needs gross profit & spend."
        if value >= 2.0:   return "Strong unit economics."
        if value >= 1.0:   return "Thin margins—watch discounts & CAC."
        return "Unprofitable after COGS—revisit pricing or targeting."
    return ""

# ===== Data access =====
BASE = Path(__file__).resolve().parents[2]  # project root
CLEAN_DIR = BASE / "data" / "cleaned"

def read_pref(name: str) -> Optional[pd.DataFrame]:
    """Read parquet if available, else csv. Returns DataFrame or None."""
    p_parquet = CLEAN_DIR / f"{name}.parquet"
    p_csv = CLEAN_DIR / f"{name}.csv"
    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            pass
    if p_csv.exists():
        try:
            return pd.read_csv(p_csv, parse_dates=["date"])
        except Exception:
            return pd.read_csv(p_csv)
    return None

@st.cache_data
def load_data():
    daily_total = read_pref("metrics_daily_total")
    daily_channel = read_pref("metrics_daily_channel")
    campaign_metrics = read_pref("campaign_metrics")
    state_metrics = read_pref("state_metrics")
    top_spend = read_pref("top_campaign_by_spend")
    top_roas = read_pref("top_campaign_by_roas")
    return {
        "daily_total": daily_total,
        "daily_channel": daily_channel,
        "campaign_metrics": campaign_metrics,
        "state_metrics": state_metrics,
        "top_spend": top_spend,
        "top_roas": top_roas
    }

data = load_data()
daily_total = data["daily_total"]
daily_channel = data["daily_channel"]
campaign_metrics = data["campaign_metrics"]
state_metrics = data["state_metrics"]
top_spend = data["top_spend"]
top_roas = data["top_roas"]

# -------------------------  Basic checks
if daily_total is None or daily_channel is None:
    st.error("Required artifacts not found in data/cleaned (metrics_daily_total & metrics_daily_channel). Run scripts 02/03 first.")
    st.stop()

# normalize date types
daily_total["date"] = pd.to_datetime(daily_total["date"]).dt.date
daily_channel["date"] = pd.to_datetime(daily_channel["date"]).dt.date

# -------------------------  Sidebar - filters
st.sidebar.header("Filters")
min_date = daily_total["date"].min()
max_date = daily_total["date"].max()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

channels = ["All"] + sorted(daily_channel["channel"].astype(str).unique().tolist()) if "channel" in daily_channel.columns else ["All"]
selected_channel = st.sidebar.selectbox("Channel", channels)

states = ["All"]
if state_metrics is not None and "state" in state_metrics.columns:
    states += sorted(state_metrics["state"].astype(str).unique().tolist())
selected_state = st.sidebar.selectbox("State", states)

campaign_query = st.sidebar.text_input("Campaign search (substring)")

st.sidebar.markdown("---")
download_enabled = st.sidebar.checkbox("Enable CSV export", value=True)

# -------------------------  Apply filters
def mask_df_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df is None or "date" not in df.columns:
        return df
    return df[(df["date"] >= start) & (df["date"] <= end)]

df_total = mask_df_by_date(daily_total.copy(), start_date, end_date)
df_channel = mask_df_by_date(daily_channel.copy(), start_date, end_date)

if selected_channel != "All":
    df_channel = df_channel[df_channel["channel"] == selected_channel]

if campaign_metrics is not None:
    df_campaign = campaign_metrics.copy()
    if campaign_query and "campaign" in df_campaign.columns:
        df_campaign = df_campaign[df_campaign["campaign"].str.contains(campaign_query, case=False, na=False)]
    if selected_channel != "All" and "channel" in df_campaign.columns:
        df_campaign = df_campaign[df_campaign["channel"] == selected_channel]
else:
    df_campaign = pd.DataFrame()

if state_metrics is not None and selected_state != "All":
    state_metrics = state_metrics[state_metrics["state"] == selected_state]

# -------------------------  KPI cards (popover details)
st.title("Marketing Intelligence — 120 days")
st.markdown("Interactive dashboard connecting marketing spend to business outcomes. Use filters on the left.")

df_total = df_total.sort_values("date")

total_spend = float(df_total["spend"].sum())
total_rev   = float(df_total.get("total_revenue", df_total.get("attributed_revenue", 0)).sum())
orders      = float(df_total.get("orders", 0).sum())
new_cust    = float(df_total.get("new_customers", 0).sum())
gross_prof  = float(df_total.get("gross_profit", 0).sum())

blended_roas = (total_rev / total_spend) if total_spend > 0 else np.nan
cac          = (total_spend / new_cust)  if new_cust  > 0 else np.nan
aov          = (total_rev / orders)      if orders    > 0 else np.nan
margin_roas  = (gross_prof / total_spend) if total_spend > 0 and gross_prof != 0 else np.nan

k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.metric("Total Spend", fmt_money(total_spend))
    with st.popover("ℹ️ Details"):
        st.write("Total media spend in the selected period.")

with k2:
    st.metric("Total Revenue", fmt_money(total_rev))
    with st.popover("ℹ️ Details"):
        st.write("Business revenue in the selected period (orders × AOV; includes non-ad factors).")

with k3:
    st.metric("Blended ROAS", fmt_ratio(blended_roas))
    with st.popover("ℹ️ Details"):
        st.markdown(f"**Meaning:** {GLOSSARY['Blended ROAS']['meaning']}")
        st.code(GLOSSARY["Blended ROAS"]["formula"])
        st.caption(GLOSSARY["Blended ROAS"]["good_when"])
        st.write(micro_insight("Blended ROAS", blended_roas, {}))

with k4:
    st.metric("CAC", fmt_money2(cac))
    with st.popover("ℹ️ Details"):
        st.markdown(f"**Meaning:** {GLOSSARY['CAC']['meaning']}")
        st.code(GLOSSARY["CAC"]["formula"])
        st.caption(GLOSSARY["CAC"]["good_when"])
        st.write(micro_insight("CAC", cac, {"bench_cac": None}))  # set your CAC goal to personalize

with k5:
    st.metric("AOV", fmt_money2(aov))
    with st.popover("ℹ️ Details"):
        st.markdown(f"**Meaning:** {GLOSSARY['AOV']['meaning']}")
        st.code(GLOSSARY["AOV"]["formula"])
        st.caption(GLOSSARY["AOV"]["good_when"])
        st.write(micro_insight("AOV", aov, {}))

# Optional sixth card:
# k6, = st.columns(1)
# with k6:
#     st.metric("Margin ROAS", fmt_ratio(margin_roas))
#     with st.popover("ℹ️ Details"):
#         st.markdown(f"**Meaning:** {GLOSSARY['Margin ROAS']['meaning']}")
#         st.code(GLOSSARY["Margin ROAS"]["formula"])
#         st.caption(GLOSSARY["Margin ROAS"]["good_when"])
#         st.write(micro_insight("Margin ROAS", margin_roas, {}))

st.markdown("---")

# -------------------------  Time-series: Spend vs Revenue
st.subheader("Spend vs Revenue (daily)")
df_total = df_total.sort_values("date")
df_total["spend_7d"] = df_total["spend"].rolling(7, min_periods=1).mean()
if "total_revenue" in df_total.columns:
    df_total["revenue_7d"] = df_total["total_revenue"].rolling(7, min_periods=1).mean()
else:
    df_total["revenue_7d"] = df_total["attributed_revenue"].rolling(7, min_periods=1).mean()

fig_ts = px.line(
    df_total, x="date", y=["spend_7d", "revenue_7d"],
    labels={"value": "USD", "variable": "Metric"},
    title="7-day MA: Spend vs Revenue"
)
fig_ts.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------  Channel breakdown (with insights)
st.subheader("Channel performance (period)")

ch_agg = df_channel.groupby("channel", as_index=False).agg({
    "impression":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
})
ch_agg["ctr"]  = ch_agg["clicks"] / ch_agg["impression"].replace(0, np.nan)
ch_agg["cpc"]  = ch_agg["spend"]  / ch_agg["clicks"].replace(0, np.nan)
ch_agg["roas"] = ch_agg["attributed_revenue"] / ch_agg["spend"].replace(0, np.nan)

# heuristic flags (tune to your business)
ch_agg["flags"] = ""
ch_agg.loc[ch_agg["ctr"]  < 0.01, "flags"] += "Low CTR (creative/targeting) • "
ch_agg.loc[ch_agg["cpc"]  > 2.00, "flags"] += "High CPC (bids/quality score) • "
ch_agg.loc[ch_agg["roas"] < 2.00, "flags"] += "Low ROAS (landing/conversion) • "
ch_agg["flags"] = ch_agg["flags"].str.rstrip(" • ")

fig_ch = px.bar(
    ch_agg.sort_values("spend", ascending=False),
    x="channel", y=["spend","attributed_revenue"], barmode="group",
    title="Spend vs Attributed Revenue by Channel",
    hover_data={"ctr":":.2%", "cpc":":.2f", "roas":":.2f", "flags":True}
)
st.plotly_chart(fig_ch, use_container_width=True)

display_cols = ["channel","spend","attributed_revenue","roas","ctr","cpc","flags"]
pretty = ch_agg[display_cols].copy()
pretty["spend"]              = pretty["spend"].map(fmt_money)
pretty["attributed_revenue"] = pretty["attributed_revenue"].map(fmt_money)
pretty["roas"]               = pretty["roas"].map(fmt_ratio)
pretty["ctr"]                = pretty["ctr"].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.2%}")
pretty["cpc"]                = pretty["cpc"].map(fmt_money2)
st.dataframe(pretty.reset_index(drop=True))
st.caption("Flags hint at levers: creative/targeting → CTR, bids/quality score → CPC, landing/conversion → ROAS.")

if download_enabled:
    st.download_button("Download channel breakdown CSV", ch_agg.to_csv(index=False).encode("utf-8"),
                       file_name="channel_breakdown.csv")

st.markdown("---")

# -------------------------  Campaign table
st.subheader("Campaign performance (searchable & sortable)")
if df_campaign.empty:
    st.info("No campaign data available for current filters.")
else:
    for c in ["roas","ctr","cpc"]:
        if c not in df_campaign.columns:
            df_campaign[c] = np.nan
    st.dataframe(df_campaign.sort_values("spend", ascending=False).reset_index(drop=True))

    if download_enabled:
        st.download_button("Download campaign metrics", df_campaign.to_csv(index=False).encode("utf-8"),
                           file_name="campaign_metrics.csv")

st.markdown("---")

# -------------------------  State view
st.subheader("State-level performance")
if state_metrics is None or "state" not in state_metrics.columns:
    st.info("State-level metrics not available.")
else:
    sm = state_metrics.copy()
    sm["roas"] = sm.get("attributed_revenue", 0) / sm["spend"].replace(0, np.nan)

    if sm["state"].str.len().max() <= 3:
        fig_state = px.choropleth(
            sm, locations="state", locationmode="USA-states", color="roas",
            hover_name="state", hover_data=["spend","attributed_revenue"],
            color_continuous_scale="Viridis", title="State-level ROAS"
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        fig_state = px.bar(
            sm.sort_values("spend", ascending=False).head(20),
            x="state", y=["spend","attributed_revenue"], barmode="group"
        )
        st.plotly_chart(fig_state, use_container_width=True)

    st.dataframe(sm.sort_values("spend", ascending=False).reset_index(drop=True))

# -------------------------  Top campaigns quick view
st.subheader("Top campaigns")
c1, c2 = st.columns(2)
if top_spend is not None:
    c1.markdown("**Top by Spend**")
    c1.table(top_spend.head(10)[["channel","campaign","spend","attributed_revenue"]])
if top_roas is not None:
    c2.markdown("**Top by ROAS**")
    c2.table(top_roas.head(10)[["channel","campaign","roas","spend","attributed_revenue"]])

# allow CSV export of top lists
if download_enabled:
    if top_spend is not None:
        st.download_button("Download top_by_spend", top_spend.to_csv(index=False).encode("utf-8"),
                           file_name="top_by_spend.csv")
    if top_roas is not None:
        st.download_button("Download top_by_roas", top_roas.to_csv(index=False).encode("utf-8"),
                           file_name="top_by_roas.csv")

st.markdown("---")
st.markdown("Notes: attribution-based ROAS uses `attributed_revenue`. Blended ROAS uses `total_revenue`. Use CTR/CPC/CAC for efficiency evaluation.")

# =========================  Metrics Glossary (on demand)
with st.expander("📘 Metrics Glossary (open for full explanations)"):
    for name, info in GLOSSARY.items():
        st.markdown(f"### {name}")
        st.write(info["meaning"])
        st.code(info["formula"])
        st.caption(info["good_when"])
