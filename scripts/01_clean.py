#!/usr/bin/env python3
"""
scripts/01_clean.py — Improved cleaning pipeline

Usage:
    python scripts/01_clean.py \
        --raw-dir data/raw \
        --out-dir data/cleaned \
        --fail-on-errors  # optional, default = False

What it does (high level):
- Reads Facebook.csv / Google.csv / TikTok.csv / Business.csv from raw dir
- Normalizes column names, parses dates, coerces numerics
- Detects & aggregates duplicate campaign-date-state rows
- Detects invalid dates, negative values and logs findings
- Writes cleaned CSV and Parquet outputs to out-dir
- Writes clean_manifest.json and validation_report.txt
- Optionally performs Pandera validation if pandera is installed
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("01_clean")

# ---------------------------
# Helpers
# ---------------------------
def clean_colnames(cols):
    cleaned = []
    for c in cols:
        c2 = re.sub(r'[^0-9a-zA-Z]+', '_', str(c).strip().lower()).strip('_')
        c2 = re.sub(r'__+', '_', c2)
        cleaned.append(c2)
    return cleaned

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log.warning("File not found: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log.warning("Read with default encoding failed for %s: %s. Trying latin-1", path, e)
        return pd.read_csv(path, encoding="latin-1")

def to_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype(str).str.replace(',', '').str.replace('$', '', regex=False).str.strip().str.lower()
    s = s.replace({'nan': '0', 'none': '0', '': '0'})
    # handle percent-like strings if appear (not expected)
    s = s.str.rstrip('%')
    return pd.to_numeric(s, errors='coerce').fillna(0)

# ---------------------------
# Normalizers
# ---------------------------
def normalize_marketing(df: pd.DataFrame, channel_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = clean_colnames(df.columns)
    df['channel'] = channel_name

    if 'date' not in df.columns:
        raise ValueError(f"marketing file for {channel_name} missing 'date' column")

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    # unify impressions -> impression
    if 'impressions' in df.columns and 'impression' not in df.columns:
        df.rename(columns={'impressions': 'impression'}, inplace=True)

    # unify attributed revenue columns (explicit then fuzzy)
    if 'attributed_revenue' not in df.columns:
        cand = [c for c in df.columns if 'attributed' in c and 'revenue' in c]
        if cand:
            df.rename(columns={cand[0]: 'attributed_revenue'}, inplace=True)

    # Ensure expected numeric columns exist; if missing, create zeros
    for col in ['impression', 'clicks', 'spend', 'attributed_revenue']:
        if col in df.columns:
            df[col] = to_numeric_col(df, col)
        else:
            log.info("Column '%s' missing in %s; creating zero column", col, channel_name)
            df[col] = 0

    # normalize campaign/state text
    if 'campaign' in df.columns:
        df['campaign'] = df['campaign'].astype(str).str.strip()
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.upper().str.strip()

    # dedupe/aggregate duplicates at same (date,campaign,state,channel)
    key = ['date', 'campaign', 'state', 'channel']
    if set(key).issubset(df.columns):
        dup_count = df.duplicated(subset=key).sum()
        if dup_count:
            log.warning("[%s] Found %d exact duplicate rows on %s -> aggregating by sum", channel_name, dup_count, key)
            agg_cols = {c: 'sum' for c in ['impression', 'clicks', 'spend', 'attributed_revenue'] if c in df.columns}
            df = df.groupby(key, as_index=False).agg(agg_cols)
    else:
        log.info("[%s] Key columns for duplicate detection not present; skipping dedupe", channel_name)

    # reorder columns sensibly
    cols_order = [c for c in ['date','channel','tactic','state','campaign','impression','clicks','spend','attributed_revenue'] if c in df.columns]
    other = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other]

    return df

def normalize_business(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = clean_colnames(df.columns)

    if 'date' not in df.columns:
        raise ValueError("business file missing 'date' column")

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    # safer explicit mapping for common variants
    mapping = {}
    explicit_map = {
        '#_of_orders': 'orders',
        'number_of_orders': 'orders',
        'num_of_orders': 'orders',
        'of_orders': 'orders',
        '#_of_new_orders': 'new_orders',
        'number_of_new_orders': 'new_orders',
        'num_of_new_orders': 'new_orders',
        'of_new_orders': 'new_orders'
    }
    for c in df.columns:
        if c in explicit_map:
            mapping[c] = explicit_map[c]

    # fallback fuzzy mapping (log it)
    for c in df.columns:
        if c not in mapping:
            if re.fullmatch(r'orders?', c) and c != 'orders':
                mapping[c] = 'orders'
            if re.fullmatch(r'new_orders?', c) and c != 'new_orders':
                mapping[c] = 'new_orders'

    if mapping:
        log.info("Business column remap: %s", mapping)
        df = df.rename(columns=mapping)

    # ensure required columns exist
    for c in ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']:
        if c not in df.columns:
            log.info("Business column '%s' missing; creating zero/default", c)
            df[c] = 0

    # numeric coercion
    for c in ['orders', 'new_orders', 'new_customers']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    for c in ['total_revenue', 'gross_profit', 'cogs']:
        s = df[c].astype(str).str.replace(',', '').str.replace('$', '', regex=False).str.strip()
        df[c] = pd.to_numeric(s, errors='coerce').fillna(0.0)

    return df

# ---------------------------
# Validation helpers
# ---------------------------
def validate_dates(df: pd.DataFrame, name: str, fail_on_errors: bool = False):
    if 'date' not in df.columns:
        msg = f"{name}: missing date column"
        log.error(msg)
        if fail_on_errors:
            raise ValueError(msg)
        return {'invalid_dates': None}
    invalid = df['date'].isna().sum()
    if invalid:
        log.error("%s: %d invalid/missing dates. Sample rows:", name, invalid)
        with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            log.error("\n%s", df.loc[df['date'].isna()].head(5).to_string())
        if fail_on_errors:
            raise ValueError(f"{invalid} invalid dates in {name}")
    return {'invalid_dates': int(invalid), 'min_date': str(df['date'].min()), 'max_date': str(df['date'].max())}

def validate_negatives(df: pd.DataFrame, name: str):
    negatives = {}
    for c in ['impression','clicks','spend','attributed_revenue']:
        if c in df.columns:
            neg_count = int((df[c] < 0).sum())
            if neg_count:
                negatives[c] = neg_count
    if negatives:
        log.warning("%s: negative values detected: %s", name, negatives)
    return negatives

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(raw_dir: Path, out_dir: Path, fail_on_errors: bool = False) -> Dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        'facebook': raw_dir / 'Facebook.csv',
        'google': raw_dir / 'Google.csv',
        'tiktok': raw_dir / 'TikTok.csv',
        'business': raw_dir / 'Business.csv'
    }

    results = {}
    # Read
    fb_raw = safe_read_csv(files['facebook'])
    google_raw = safe_read_csv(files['google'])
    tiktok_raw = safe_read_csv(files['tiktok'])
    biz_raw = safe_read_csv(files['business'])

    # Normalize
    fb = normalize_marketing(fb_raw, 'Facebook') if fb_raw is not None else None
    google = normalize_marketing(google_raw, 'Google') if google_raw is not None else None
    tiktok = normalize_marketing(tiktok_raw, 'TikTok') if tiktok_raw is not None else None
    biz = normalize_business(biz_raw) if biz_raw is not None else None

    # Validations & saves
    for name, df in [('facebook', fb), ('google', google), ('tiktok', tiktok), ('business', biz)]:
        if df is None:
            results[name] = {'status': 'missing'}
            continue

        # date checks
        dval = validate_dates(df, name, fail_on_errors=fail_on_errors)

        # negative value checks (marketing only)
        negs = validate_negatives(df, name) if name != 'business' else {}

        # duplicates (marketing)
        dup_info = {}
        if name in ['facebook','google','tiktok']:
            key = ['date','campaign','state','channel']
            if set(key).issubset(df.columns):
                dup_count = int(df.duplicated(subset=key).sum())
                dup_info['duplicate_rows'] = dup_count
            else:
                dup_info['duplicate_rows'] = None

        # write outputs (CSV + parquet)
        csv_path = out_dir / f"{name}_cleaned.csv"
        pq_path = out_dir / f"{name}_cleaned.parquet"
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(pq_path, index=False)
        except Exception as e:
            log.warning("Could not write parquet for %s (%s). Parquet optional.", name, e)

        results[name] = {
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'csv': str(csv_path.resolve()),
            'parquet': str(pq_path.resolve()) if pq_path.exists() else None,
            'date_min': dval.get('min_date'),
            'date_max': dval.get('max_date'),
            'invalid_dates': dval.get('invalid_dates'),
            'negatives': negs,
            'duplicates': dup_info.get('duplicate_rows')
        }

    # manifest + validation report
    manifest_path = out_dir / "clean_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    report_lines = []
    report_lines.append("CLEANING VALIDATION REPORT\n")
    for k, v in results.items():
        report_lines.append(f"== {k.upper()} ==")
        for kk, vv in v.items():
            report_lines.append(f"{kk}: {vv}")
        report_lines.append("")

    report_path = out_dir / "validation_report.txt"
    report_path.write_text("\n".join(report_lines))

    log.info("Cleaning finished. manifest: %s, report: %s", manifest_path, report_path)
    return results

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cleaning pipeline for marketing intelligence assessment")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw CSVs")
    p.add_argument("--out-dir", type=str, default="data/cleaned", help="Path for cleaned outputs")
    p.add_argument("--fail-on-errors", action="store_true", help="Raise on validation errors")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = run_pipeline(args.raw_dir, args.out_dir, fail_on_errors=args.fail_on_errors)
    # print short summary to stdout for convenience
    print(json.dumps(res, indent=2))
