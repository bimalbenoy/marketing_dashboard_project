
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
from typing import Optional
import io

st.set_page_config(layout="wide", page_title="Marketing Intelligence")


BASE = Path(__file__).resolve().parents[2]  # project root
CLEAN_DIR = BASE / "data" / "cleaned"

def read_pref(name: str) -> Optional[pd.DataFrame]:
    """
    Read parquet if available, else csv. Returns DataFrame or None.
    """
    p_parquet = CLEAN_DIR / f"{name}.parquet"
    p_csv = CLEAN_DIR / f"{name}.csv"
    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            # fallback to csv if parquet read fails
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

# -------------------------
# Basic checks
# -------------------------
if daily_total is None or daily_channel is None:
    st.error("Required artifacts not found in data/cleaned (metrics_daily_total & metrics_daily_channel). Run scripts 02/03 first.")
    st.stop()

# normalize date types
daily_total["date"] = pd.to_datetime(daily_total["date"]).dt.date
daily_channel["date"] = pd.to_datetime(daily_channel["date"]).dt.date

# -------------------------
# Sidebar - filters
# -------------------------
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

# -------------------------
# Apply filters
# -------------------------
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
    if campaign_query:
        # defensive: ensure campaign column exists
        if "campaign" in df_campaign.columns:
            df_campaign = df_campaign[df_campaign["campaign"].str.contains(campaign_query, case=False, na=False)]
    if selected_channel != "All" and "channel" in df_campaign.columns:
        df_campaign = df_campaign[df_campaign["channel"] == selected_channel]
else:
    df_campaign = pd.DataFrame()

if state_metrics is not None and selected_state != "All":
    state_metrics = state_metrics[state_metrics["state"] == selected_state]

# -------------------------
# KPI cards
# -------------------------
st.title("Marketing Intelligence — 120 days")
st.markdown("Interactive dashboard connecting marketing spend to business outcomes. Filter on the left.")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_spend = float(df_total["spend"].sum())
total_rev = float(df_total["total_revenue"].sum()) if "total_revenue" in df_total.columns else float(df_total.get("attributed_revenue", 0).sum())
blended_roas = (total_rev / total_spend) if total_spend > 0 else np.nan
new_customers = int(df_total.get("new_customers", pd.Series([0])).sum()) if "new_customers" in df_total.columns else int(df_total.get("new_customers", pd.Series([0])).sum())
orders = int(df_total.get("orders", pd.Series([0])).sum()) if "orders" in df_total.columns else int(df_total.get("orders", pd.Series([0])).sum())
cac = (total_spend / new_customers) if new_customers > 0 else np.nan
aov = (total_rev / orders) if orders > 0 else np.nan
margin_roas = (df_total["gross_profit"].sum() / total_spend) if ("gross_profit" in df_total.columns and total_spend > 0) else np.nan

kpi1.metric("Total Spend", f"${total_spend:,.0f}")
kpi2.metric("Total Revenue", f"${total_rev:,.0f}")
kpi3.metric("Blended ROAS", f"{blended_roas:.2f}" if not np.isnan(blended_roas) else "N/A")
kpi4.metric("CAC", f"${cac:,.2f}" if not np.isnan(cac) else "N/A")
kpi5.metric("AOV", f"${aov:,.2f}" if not np.isnan(aov) else "N/A")

st.markdown("---")

# -------------------------
# Time-series: Spend vs Revenue
# -------------------------
st.subheader("Spend vs Revenue (daily)")

df_total = df_total.sort_values("date")
df_total["spend_7d"] = df_total["spend"].rolling(7, min_periods=1).mean()
if "total_revenue" in df_total.columns:
    df_total["revenue_7d"] = df_total["total_revenue"].rolling(7, min_periods=1).mean()
else:
    df_total["revenue_7d"] = df_total["attributed_revenue"].rolling(7, min_periods=1).mean()

fig_ts = px.line(df_total, x="date", y=["spend_7d", "revenue_7d"], labels={"value":"USD","variable":"Metric"},
                 title="7-day MA: Spend vs Revenue")
fig_ts.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------
# Channel breakdown
# -------------------------
st.subheader("Channel performance (period)")
ch_agg = df_channel.groupby("channel", as_index=False).agg({
    "spend":"sum","attributed_revenue":"sum","impression":"sum","clicks":"sum"
})
ch_agg["roas"] = ch_agg.apply(lambda r: (r["attributed_revenue"]/r["spend"]) if r["spend"]>0 else np.nan, axis=1)
ch_agg["ctr"] = ch_agg["clicks"] / ch_agg["impression"].replace(0, np.nan)
ch_agg["cpc"] = ch_agg["spend"] / ch_agg["clicks"].replace(0, np.nan)

fig_ch = px.bar(ch_agg.sort_values("spend", ascending=False), x="channel", y=["spend","attributed_revenue"], barmode="group",
                title="Spend vs Attributed Revenue by Channel")
st.plotly_chart(fig_ch, use_container_width=True)

st.dataframe(ch_agg.sort_values("roas", ascending=False).reset_index(drop=True))

if download_enabled:
    csv = ch_agg.to_csv(index=False).encode("utf-8")
    st.download_button("Download channel breakdown CSV", csv, file_name="channel_breakdown.csv")

st.markdown("---")

# -------------------------
# Campaign table
# -------------------------
st.subheader("Campaign performance (searchable & sortable)")
if df_campaign.empty:
    st.info("No campaign data available for current filters.")
else:
    # show key metrics and allow sort
    display_cols = ["channel","campaign","impression","clicks","spend","attributed_revenue","roas","ctr","cpc"]
    # ensure columns exist
    for c in ["roas","ctr","cpc"]:
        if c not in df_campaign.columns:
            df_campaign[c] = np.nan
    st.dataframe(df_campaign.sort_values("spend", ascending=False).reset_index(drop=True))

    if download_enabled:
        csv = df_campaign.to_csv(index=False).encode("utf-8")
        st.download_button("Download campaign metrics", csv, file_name="campaign_metrics.csv")

st.markdown("---")

# -------------------------
# State view
# -------------------------
st.subheader("State-level performance")
if state_metrics is None or "state" not in state_metrics.columns:
    st.info("State-level metrics not available.")
else:
    sm = state_metrics.copy()
    # ensure roas exists
    sm["roas"] = sm.get("attributed_revenue", 0) / sm["spend"].replace(0, np.nan)
    # If state codes are 2-letter, show choropleth
    if sm["state"].str.len().max() <= 3:
        fig_state = px.choropleth(sm, locations="state", locationmode="USA-states", color="roas",
                                 hover_name="state", hover_data=["spend","attributed_revenue"],
                                 color_continuous_scale="Viridis", title="State-level ROAS")
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        fig_state = px.bar(sm.sort_values("spend", ascending=False).head(20), x="state", y=["spend","attributed_revenue"], barmode="group")
        st.plotly_chart(fig_state, use_container_width=True)
    st.dataframe(sm.sort_values("spend", ascending=False).reset_index(drop=True))

# -------------------------
# Top campaigns quick view
# -------------------------
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
        st.download_button("Download top_by_spend", top_spend.to_csv(index=False).encode("utf-8"), file_name="top_by_spend.csv")
    if top_roas is not None:
        st.download_button("Download top_by_roas", top_roas.to_csv(index=False).encode("utf-8"), file_name="top_by_roas.csv")

st.markdown("---")
st.markdown("Notes: attribution-based ROAS uses `attributed_revenue`. Blended ROAS uses `total_revenue`. Use CTR/CPC/CAC for efficiency evaluation.")
