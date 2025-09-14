#!/usr/bin/env python3
"""
scripts/03_metrics.py

Compute derived marketing & business metrics from aggregated artifacts.

Inputs (reads from agg_dir):
 - daily_channel.csv / .parquet  (date x channel)
 - daily_total.csv / .parquet    (date totals merged with business)
 - campaign_agg.csv / .parquet   (campaign-level aggregated)
 - state_agg.csv / .parquet      (state-level aggregated)

Outputs (written to out_dir):
 - metrics_daily_channel.csv / .parquet  (with CTR/CPC/CPM/ROAS and rolling averages)
 - metrics_daily_total.csv / .parquet    (with blended ROAS/CAC/AOV/margin ROAS)
 - campaign_metrics.csv / .parquet       (campaign-level metrics & ROAS)
 - state_metrics.csv / .parquet          (state-level metrics & ROAS)
 - metrics_manifest.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("03_metrics")

# ---------------------------
# Helpers
# ---------------------------
def read_preferred(basedir: Path, name: str) -> Optional[pd.DataFrame]:
    p_parquet = basedir / f"{name}.parquet"
    p_csv = basedir / f"{name}.csv"
    if p_parquet.exists():
        log.info("Reading parquet %s", p_parquet)
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        log.info("Reading csv %s", p_csv)
        return pd.read_csv(p_csv, parse_dates=["date"])
    log.warning("Missing aggregated file for %s in %s", name, basedir)
    return None

def safe_div(numer, denom):
    """Elementwise safe division, returns np.nan where denom == 0."""
    numer = np.array(numer, dtype=float)
    denom = np.array(denom, dtype=float)
    out = np.full_like(numer, np.nan, dtype=float)
    mask = denom != 0
    out[mask] = numer[mask] / denom[mask]
    return out

def ensure_numeric(df, cols_defaults):
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
        # coerce numeric
        if isinstance(default, int):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(default).astype(int)
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(default).astype(float)
    return df

# ---------------------------
# Metrics computations
# ---------------------------
def compute_channel_daily_metrics(daily_channel: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Adds CTR, CPC, CPM, ROAS to daily_channel and a rolling window for spend & revenue.
    """
    df = daily_channel.copy()
    # ensure columns exist
    df = ensure_numeric(df, {'impression': 0, 'clicks': 0, 'spend': 0.0, 'attributed_revenue': 0.0})

    # basic metrics
    df['ctr'] = safe_div(df['clicks'], df['impression'])  # clicks / impressions
    df['cpc'] = safe_div(df['spend'], df['clicks'])       # spend / clicks
    df['cpm'] = safe_div(df['spend'] * 1000.0, df['impression'])  # cost per 1000 impressions
    df['roas'] = safe_div(df['attributed_revenue'], df['spend'])  # revenue / spend

    # rolling aggregates by channel
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['channel', 'date'])
    # compute rolling metrics per channel
    roll_df = []
    for ch, g in df.groupby('channel'):
        g2 = g.copy().set_index(pd.to_datetime(g['date']))
        # rolling sums for spend & attributed_revenue (window_days)
        g2[f'spend_{window_days}d'] = g2['spend'].rolling(window=window_days, min_periods=1).mean()
        g2[f'revenue_{window_days}d'] = g2['attributed_revenue'].rolling(window=window_days, min_periods=1).mean()
        g2 = g2.reset_index(drop=True)
        roll_df.append(g2)
    df = pd.concat(roll_df, ignore_index=True, sort=False)

    # keep order
    cols_order = ['date', 'channel', 'impression', 'clicks', 'spend', 'attributed_revenue',
                  'ctr', 'cpc', 'cpm', 'roas']
    other = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other]
    return df

def compute_daily_total_metrics(daily_total: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Adds blended ROAS, CAC, AOV, margin ROAS and rolling averages on spend & revenue.
    """
    df = daily_total.copy()
    # ensure numeric columns exist
    biz_defaults = {'impression': 0, 'clicks': 0, 'spend': 0.0, 'attributed_revenue': 0.0,
                    'orders': 0, 'new_orders': 0, 'new_customers': 0,
                    'total_revenue': 0.0, 'gross_profit': 0.0, 'cogs': 0.0}
    df = ensure_numeric(df, biz_defaults)

    # derived
    df['ctr'] = safe_div(df['clicks'], df['impression'])
    df['cpc'] = safe_div(df['spend'], df['clicks'])
    df['cpm'] = safe_div(df['spend'] * 1000.0, df['impression'])
    # attribution-based ROAS: attributed_revenue / spend
    df['roas_attributed'] = safe_div(df['attributed_revenue'], df['spend'])
    # blended ROAS: total_revenue / spend (business-level)
    df['roas_blended'] = safe_div(df['total_revenue'], df['spend'])
    # margin ROAS: gross_profit / spend
    df['roas_margin'] = safe_div(df['gross_profit'], df['spend'])
    # CAC: spend / new_customers (if new_customers == 0 -> NaN)
    df['cac'] = safe_div(df['spend'], df['new_customers'])
    # AOV: total_revenue / orders
    df['aov'] = safe_div(df['total_revenue'], df['orders'])

    # rolling window smoothing on spend/revenue/roas
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    df['spend_{}_d_ma'.format(window_days)] = df['spend'].rolling(window=window_days, min_periods=1).mean()
    df['total_revenue_{}_d_ma'.format(window_days)] = df['total_revenue'].rolling(window=window_days, min_periods=1).mean()
    df['roas_blended_{}_d_ma'.format(window_days)] = df['roas_blended'].rolling(window=window_days, min_periods=1).mean()

    # keep order
    cols_order = ['date', 'impression', 'clicks', 'spend', 'attributed_revenue',
                  'orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs',
                  'ctr', 'cpc', 'cpm', 'roas_attributed', 'roas_blended', 'roas_margin', 'cac', 'aov']
    other = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other]
    return df

def compute_campaign_metrics(campaign_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute campaign-level performance metrics for the full period.
    """
    df = campaign_agg.copy()
    df = ensure_numeric(df, {'impression': 0, 'clicks': 0, 'spend': 0.0, 'attributed_revenue': 0.0})
    df['ctr'] = safe_div(df['clicks'], df['impression'])
    df['cpc'] = safe_div(df['spend'], df['clicks'])
    df['cpm'] = safe_div(df['spend'] * 1000.0, df['impression'])
    df['roas'] = safe_div(df['attributed_revenue'], df['spend'])
    # also compute contribution share
    total_spend = df['spend'].sum() if df['spend'].sum() != 0 else 1.0
    total_rev = df['attributed_revenue'].sum() if df['attributed_revenue'].sum() != 0 else 1.0
    df['spend_share'] = df['spend'] / total_spend
    df['rev_share'] = df['attributed_revenue'] / total_rev
    # order sensible columns
    cols_order = ['channel', 'campaign', 'impression', 'clicks', 'spend', 'attributed_revenue',
                  'ctr', 'cpc', 'cpm', 'roas', 'spend_share', 'rev_share']
    other = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other]
    return df.sort_values('spend', ascending=False)

def compute_state_metrics(state_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state-level metrics across the period.
    """
    df = state_agg.copy()
    df = ensure_numeric(df, {'impression': 0, 'clicks': 0, 'spend': 0.0, 'attributed_revenue': 0.0})
    df['ctr'] = safe_div(df['clicks'], df['impression'])
    df['cpc'] = safe_div(df['spend'], df['clicks'])
    df['cpm'] = safe_div(df['spend'] * 1000.0, df['impression'])
    df['roas'] = safe_div(df['attributed_revenue'], df['spend'])
    cols_order = ['state', 'impression', 'clicks', 'spend', 'attributed_revenue', 'ctr', 'cpc', 'cpm', 'roas']
    other = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other]
    return df.sort_values('spend', ascending=False)

# ---------------------------
# I/O & main
# ---------------------------
def write_df(out_dir: Path, name: str, df: pd.DataFrame):
    csv_path = out_dir / f"{name}.csv"
    pq_path = out_dir / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        log.warning("Could not write parquet for %s (%s). CSV written.", name, e)
    log.info("Wrote %s rows=%d -> %s", name, len(df), csv_path)

def main(agg_dir: Path, out_dir: Path, window_days: int = 7):
    agg_dir = Path(agg_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_channel = read_preferred(agg_dir, "daily_channel")
    daily_total = read_preferred(agg_dir, "daily_total")
    campaign_agg = read_preferred(agg_dir, "campaign_agg")
    state_agg = read_preferred(agg_dir, "state_agg")

    if daily_channel is None or daily_total is None:
        log.error("Required aggregated artifacts missing (daily_channel/daily_total). Ensure Step 2 ran successfully.")
        return

    # compute
    metrics_daily_channel = compute_channel_daily_metrics(daily_channel, window_days=window_days)
    metrics_daily_total = compute_daily_total_metrics(daily_total, window_days=window_days)

    campaign_metrics = compute_campaign_metrics(campaign_agg) if campaign_agg is not None else pd.DataFrame()
    state_metrics = compute_state_metrics(state_agg) if state_agg is not None else pd.DataFrame()

    # Top-N lists for easy use in dashboard (top 10 campaigns by roas and by spend)
    top_campaign_by_spend = campaign_metrics.sort_values('spend', ascending=False).head(10)
    top_campaign_by_roas = campaign_metrics.sort_values('roas', ascending=False).head(10)

    # write outputs
    write_df(out_dir, "metrics_daily_channel", metrics_daily_channel)
    write_df(out_dir, "metrics_daily_total", metrics_daily_total)
    write_df(out_dir, "campaign_metrics", campaign_metrics)
    write_df(out_dir, "state_metrics", state_metrics)
    write_df(out_dir, "top_campaign_by_spend", top_campaign_by_spend)
    write_df(out_dir, "top_campaign_by_roas", top_campaign_by_roas)

    manifest = {
        "metrics_daily_channel": {"rows": int(len(metrics_daily_channel))},
        "metrics_daily_total": {"rows": int(len(metrics_daily_total))},
        "campaign_metrics": {"rows": int(len(campaign_metrics))},
        "state_metrics": {"rows": int(len(state_metrics))}
    }
    manifest_path = out_dir / "metrics_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Metrics computation finished. Manifest: %s", manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute marketing metrics")
    parser.add_argument("--agg-dir", type=str, default="data/cleaned", help="Path to aggregated artifacts")
    parser.add_argument("--out-dir", type=str, default="data/cleaned", help="Path to write metrics outputs")
    parser.add_argument("--window-days", type=int, default=7, help="Rolling window days for smoothing")
    args = parser.parse_args()
    main(Path(args.agg_dir), Path(args.out_dir), window_days=args.window_days)
