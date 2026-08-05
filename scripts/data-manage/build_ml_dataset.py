#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ML Dataset for TAT Outlier Detection.

Reads integrated export files (parquet or csv) and constructs the target machine learning dataset
matching the specification in `data/example/機械学習用データセット目標.xlsx` and `data/dictionary/high_deviation_dataset_config.json`.
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Add data/example directory to path to import recalc_tat1_custom_points
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "data" / "example"))
import recalc_tat1_custom_points as recalc


def load_dataset_inputs(input_dir: Path):
    """
    Load measurement and profile data from input_dir.
    Supports both .parquet and .csv formats.
    """
    input_dir = Path(input_dir)
    
    # Try loading parquet first, fall back to csv via recalc.load_inputs
    meas_parquet = input_dir / "measurement.parquet"
    prof_parquet = input_dir / "profile.parquet"
    meta_json = input_dir / "metadata.json"
    
    if meas_parquet.exists() and prof_parquet.exists():
        meas_df = pd.read_parquet(meas_parquet)
        profile_df = pd.read_parquet(prof_parquet)
        metadata = {}
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        # Clean text & coerce numeric types
        for col in ["source_index", recalc.TARGET_ITEM]:
            if col in meas_df.columns:
                meas_df[col] = pd.to_numeric(meas_df[col], errors="coerce")
        for col in ["source_index", "項目No.", "測光ﾎﾟｰﾄ", "処理値", "時間", "吸光度"]:
            if col in profile_df.columns:
                profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce")
                
        if "source_index" in meas_df.columns:
            meas_df["source_index"] = meas_df["source_index"].astype("Int64")
        if "source_index" in profile_df.columns:
            profile_df["source_index"] = profile_df["source_index"].astype("Int64")
            
        if {"source_index", "依頼No."}.issubset(meas_df.columns):
            meas_df["global_request_id"] = (
                meas_df["source_index"].astype(str).str.replace("<NA>", "", regex=False)
                + "_"
                + meas_df["依頼No."].astype(str)
            )
        if {"source_index", "依頼No."}.issubset(profile_df.columns):
            profile_df["global_request_id"] = (
                profile_df["source_index"].astype(str).str.replace("<NA>", "", regex=False)
                + "_"
                + profile_df["依頼No."].astype(str)
            )
        return meas_df, profile_df, metadata
    else:
        return recalc.load_inputs(input_dir)


def calculate_features(output_row: dict, pt_names: list[str]) -> dict:
    """
    Calculate summary statistics and relative ratios (e.g. 4-10/2-8, 14-20/2-8) for a row.
    """
    pt_values = [output_row[name] for name in pt_names]

    # Check if all values are NaN
    if pd.isna(pt_values).all() or len([v for v in pt_values if pd.notna(v)]) == 0:
        output_row["平均"] = np.nan
        output_row["最大"] = np.nan
        output_row["最小"] = np.nan
        output_row["レンジ"] = np.nan
        output_row["最大上昇率"] = np.nan
        output_row["最大上昇速度"] = np.nan
        output_row["最大下落率"] = np.nan
        output_row["最大下落速度"] = np.nan
        for p in pt_names[1:]:
            output_row[f"{p}/2-8"] = np.nan
        return output_row

    valid_pts = [v for v in pt_values if pd.notna(v)]

    mean_val = np.mean(valid_pts)
    max_val = np.max(valid_pts)
    min_val = np.min(valid_pts)
    range_val = max_val - min_val

    # Rise / Fall Rate relative to min/max
    max_rise_rate = (max_val / min_val) - 1.0 if min_val != 0 else np.nan
    max_fall_rate = (min_val / max_val) - 1.0 if max_val != 0 else np.nan

    # Speed: adjacent point change rate (P_i / P_{i-1}) - 1
    speeds = []
    for i in range(1, len(pt_values)):
        prev = pt_values[i - 1]
        curr = pt_values[i]
        if pd.notna(prev) and pd.notna(curr) and prev != 0:
            speeds.append((curr / prev) - 1.0)

    max_rise_speed = np.max(speeds) if speeds else np.nan
    max_fall_speed = np.min(speeds) if speeds else np.nan

    output_row["平均"] = mean_val
    output_row["最大"] = max_val
    output_row["最小"] = min_val
    output_row["レンジ"] = range_val
    output_row["最大上昇率"] = max_rise_rate
    output_row["最大上昇速度"] = max_rise_speed
    output_row["最大下落率"] = max_fall_rate
    output_row["最大下落速度"] = max_fall_speed

    pt_2_8 = output_row.get("2-8", np.nan)
    for p in pt_names[1:]:
        val = output_row.get(p, np.nan)
        if pd.notna(pt_2_8) and pt_2_8 != 0 and pd.notna(val):
            output_row[f"{p}/2-8"] = val / pt_2_8
        else:
            output_row[f"{p}/2-8"] = np.nan

    return output_row


def build_dataset(input_dir: Path, output_excel: Path) -> pd.DataFrame:
    """
    Build machine learning dataset from integrated export directory.
    """
    print(f"Loading data from {input_dir}")
    meas_df, profile_df, metadata = load_dataset_inputs(input_dir)
    meas_df["SampleType"] = meas_df.get("属性", "").apply(recalc.classify_sample_type)

    rates_by_pattern = recalc.build_rates(profile_df)
    curves, _ = recalc.build_calibration_curves(meas_df, rates_by_pattern)

    # Pre-index rates_by_pattern for fast lookup
    rates_lookup = {}
    for pattern_name, rate_df in rates_by_pattern.items():
        sub = rate_df[rate_df["項目名"].astype(str) == recalc.TARGET_ITEM]
        for _, r in sub.iterrows():
            gid = str(r["global_request_id"])
            rates_lookup[(pattern_name, gid)] = pd.to_numeric(r["Rate_mAbs_min"], errors="coerce")

    base_cols = ["source_index", "source_file", "global_request_id", "依頼No.", "SID", "属性", "SampleType"]
    pt_names = [p[0] for p in recalc.PT_PATTERNS]
    rows = []

    unique_samples = meas_df[base_cols].drop_duplicates("global_request_id")

    for _, sample in unique_samples.iterrows():
        global_id = str(sample["global_request_id"])
        source_index = sample.get("source_index", "GLOBAL")
        measurement_row = meas_df[meas_df["global_request_id"].astype(str) == global_id].iloc[0]

        output_row = {col: sample[col] for col in base_cols}
        output_row[f"{recalc.TARGET_ITEM}装置生データ"] = pd.to_numeric(measurement_row.get(recalc.TARGET_ITEM, np.nan), errors="coerce")

        for pattern_name in pt_names:
            rate_value = rates_lookup.get((pattern_name, global_id), np.nan)
            cal_rates, cal_concs, _ = recalc.find_curve(curves, pattern_name, source_index)
            val = recalc.interpolate_or_extrapolate(rate_value, cal_rates, cal_concs)
            # Use pattern_name directly (e.g., '2-8', '4-10') to match target Excel columns
            output_row[pattern_name] = val

        # Calculate summary features and relative ratio columns
        output_row = calculate_features(output_row, pt_names)
        rows.append(output_row)

    df = pd.DataFrame(rows)

    # Define exact target column order matching '機械学習用データセット目標.xlsx'
    target_columns = [
        "source_index", "source_file", "global_request_id", "依頼No.", "SID", "属性", "SampleType",
        "TAT1装置生データ", "2-8", "4-10", "6-12", "8-14", "10-16", "12-18", "14-20",
        "平均", "最大", "最小", "レンジ", "最大上昇率", "最大上昇速度", "最大下落率", "最大下落速度",
        "4-10/2-8", "6-12/2-8", "8-14/2-8", "10-16/2-8", "12-18/2-8", "14-20/2-8"
    ]
    
    # Reorder columns present in target_columns, fallback for any extra columns
    cols_to_use = [c for c in target_columns if c in df.columns] + [c for c in df.columns if c not in target_columns]
    df = df[cols_to_use]

    print(f"Saving dataset to {output_excel}")
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Build ML dataset directly from integrated parquet/CSV files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing measurement and profile data.")
    parser.add_argument("--output-excel", required=True, help="Output Excel file path for the ML dataset.")
    args = parser.parse_args()

    build_dataset(Path(args.input_dir), Path(args.output_excel))
    print("Done.")


if __name__ == "__main__":
    main()
