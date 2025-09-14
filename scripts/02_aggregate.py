#!/usr/bin/env python3
"""
scripts/02_aggregate.py

Aggregates cleaned marketing & business data into:
 - daily_channel (date x channel) totals
 - daily_total (date totals across channels) merged with business
 - campaign_agg (campaign-level totals across the period)
 - state_agg (state-level totals across period)

Usage:
    python scripts/02_aggregate.py --cleaned-dir data/cleaned --out-dir data/cleaned

Outputs (written to out-dir):
    - daily_channel.csv / .parquet
    - daily_total.csv / .parquet
    - campaign_agg.csv / .parquet
    - state_agg.csv / .parquet
    - aggregate_manifest.json
"""

from pathlib import Path
import argparse
import logging
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("02_aggregate")


def read_preferred(path_base: Path, name: str):
    """Read parquet if exists, else csv. Returns DataFrame or None."""
    p_parquet = path_base / f"{name}.parquet"
    p_csv = path_base / f"{name}.csv"
    if p_parquet.exists():
        log.info("Reading %s (parquet): %s", name, p_parquet)
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        log.info("Reading %s (csv): %s", name, p_csv)
        return pd.read_csv(p_csv, parse_dates=["date"])
    log.warning("Neither parquet nor csv found for %s in %s", name, path_base)
    return None


def ensure_columns(df: pd.DataFrame, cols_defaults: dict):
    """Ensure df has columns listed in cols_defaults; if missing, create with default value."""
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df


def agg_daily_channel(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate marketing daily by channel."""
    # ensure columns exist
    defaults = {"impression": 0, "clicks": 0, "spend": 0.0, "attributed_revenue": 0.0}
    marketing_df = ensure_columns(marketing_df, defaults)

    # coerce types
    marketing_df['impression'] = pd.to_numeric(marketing_df['impression'], errors='coerce').fillna(0).astype(int)
    marketing_df['clicks'] = pd.to_numeric(marketing_df['clicks'], errors='coerce').fillna(0).astype(int)
    marketing_df['spend'] = pd.to_numeric(marketing_df['spend'], errors='coerce').fillna(0.0)
    marketing_df['attributed_revenue'] = pd.to_numeric(marketing_df['attributed_revenue'], errors='coerce').fillna(0.0)

    group_cols = ['date', 'channel']
    agg = marketing_df.groupby(group_cols, as_index=False).agg({
        'impression': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'attributed_revenue': 'sum'
    }).sort_values(['date', 'channel'])

    return agg


def agg_daily_total(daily_channel_df: pd.DataFrame, business_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across channels to daily totals and merge business metrics."""
    defaults = {"impression": 0, "clicks": 0, "spend": 0.0, "attributed_revenue": 0.0}
    daily_channel_df = ensure_columns(daily_channel_df, defaults)

    daily_total = (daily_channel_df.groupby('date', as_index=False)
                   .agg({'impression': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
                   .sort_values('date'))

    # ensure business_df has date and expected columns, else create zeros
    if business_df is None:
        log.warning("Business dataframe is None. Creating empty business skeleton.")
        business_df = pd.DataFrame({'date': daily_total['date']})
    business_df = business_df.copy()
    # ensure date dtype
    if 'date' in business_df.columns:
        business_df['date'] = pd.to_datetime(business_df['date']).dt.date
    # expected business columns
    biz_defaults = {'orders': 0, 'new_orders': 0, 'new_customers': 0, 'total_revenue': 0.0, 'gross_profit': 0.0, 'cogs': 0.0}
    business_df = ensure_columns(business_df, biz_defaults)

    # coerce types on business fields
    for intcol in ['orders', 'new_orders', 'new_customers']:
        business_df[intcol] = pd.to_numeric(business_df[intcol], errors='coerce').fillna(0).astype(int)
    for fcol in ['total_revenue', 'gross_profit', 'cogs']:
        business_df[fcol] = pd.to_numeric(business_df[fcol], errors='coerce').fillna(0.0)

    # merge daily totals with business on date (left join on daily_total)
    # ensure both date types are date (no time)
    daily_total['date'] = pd.to_datetime(daily_total['date']).dt.date
    business_df['date'] = pd.to_datetime(business_df['date']).dt.date

    merged = pd.merge(daily_total, business_df, on='date', how='left')

    # fill missing business with zeros (if any)
    merged.fillna({'orders': 0, 'new_orders': 0, 'new_customers': 0, 'total_revenue': 0.0, 'gross_profit': 0.0, 'cogs': 0.0}, inplace=True)

    # keep order
    cols_desired = ['date', 'impression', 'clicks', 'spend', 'attributed_revenue',
                    'orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']
    other_cols = [c for c in merged.columns if c not in cols_desired]
    merged = merged[cols_desired + other_cols]

    return merged


def agg_campaign(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate campaign-level totals across the full period."""
    defaults = {"impression": 0, "clicks": 0, "spend": 0.0, "attributed_revenue": 0.0}
    marketing_df = ensure_columns(marketing_df, defaults)

    # Normalize campaign column if missing
    if 'campaign' not in marketing_df.columns:
        marketing_df['campaign'] = 'unknown_campaign'

    group_cols = ['channel', 'campaign']
    agg = marketing_df.groupby(group_cols, as_index=False).agg({
        'impression': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'attributed_revenue': 'sum'
    }).sort_values(['spend'], ascending=False)

    return agg


def agg_state(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by state across the full period."""
    defaults = {"impression": 0, "clicks": 0, "spend": 0.0, "attributed_revenue": 0.0}
    marketing_df = ensure_columns(marketing_df, defaults)

    if 'state' not in marketing_df.columns:
        marketing_df['state'] = 'UNKNOWN'

    agg = marketing_df.groupby(['state'], as_index=False).agg({
        'impression': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'attributed_revenue': 'sum'
    }).sort_values('spend', ascending=False)

    return agg


def write_df(out_dir: Path, name: str, df: pd.DataFrame):
    csv_path = out_dir / f"{name}.csv"
    pq_path = out_dir / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        log.warning("Could not write parquet for %s (%s). CSV written.", name, e)
    log.info("Wrote %s: rows=%d -> %s", name, len(df), csv_path)


def main(cleaned_dir: Path, out_dir: Path):
    cleaned_dir = Path(cleaned_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read cleaned inputs
    fb = read_preferred(cleaned_dir, "facebook_cleaned")
    google = read_preferred(cleaned_dir, "google_cleaned")
    tiktok = read_preferred(cleaned_dir, "tiktok_cleaned")
    biz = read_preferred(cleaned_dir, "business_cleaned")

    # Concatenate marketing channels
    marketing_frames = []
    for df, name in [(fb, 'facebook'), (google, 'google'), (tiktok, 'tiktok')]:
        if df is None:
            log.warning("Missing cleaned file for %s", name)
            continue
        # ensure date is date dtype
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        marketing_frames.append(df)

    if not marketing_frames:
        log.error("No marketing data available. Exiting.")
        return

    marketing = pd.concat(marketing_frames, ignore_index=True, sort=False)

    # Ensure channel column exists
    if 'channel' not in marketing.columns:
        marketing['channel'] = 'unknown'

    # Aggregate outputs
    daily_channel = agg_daily_channel(marketing)
    daily_total = agg_daily_total(daily_channel, biz)
    campaign_agg = agg_campaign(marketing)
    state_agg = agg_state(marketing)

    # Save artifacts
    write_df(out_dir, "daily_channel", daily_channel)
    write_df(out_dir, "daily_total", daily_total)
    write_df(out_dir, "campaign_agg", campaign_agg)
    write_df(out_dir, "state_agg", state_agg)

    # manifest
    manifest = {
        "daily_channel": {"rows": int(len(daily_channel))},
        "daily_total": {"rows": int(len(daily_total))},
        "campaign_agg": {"rows": int(len(campaign_agg))},
        "state_agg": {"rows": int(len(state_agg))}
    }
    manifest_path = out_dir / "aggregate_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Aggregation finished. Manifest: %s", manifest_path)

    # Print quick summary to stdout
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregation step for marketing dashboard")
    parser.add_argument("--cleaned-dir", type=str, default="data/cleaned", help="Path to cleaned data")
    parser.add_argument("--out-dir", type=str, default="data/cleaned", help="Path to write aggregated outputs")
    args = parser.parse_args()
    main(Path(args.cleaned_dir), Path(args.out_dir))
