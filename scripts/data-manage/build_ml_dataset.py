#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
# Make it possible to import recalc_tat1_custom_points.py
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "data" / "example"))
import recalc_tat1_custom_points as recalc

def calculate_features(row):
    # Base columns are already in row
    # The calculated points are 2-8, 4-10, 6-12, 8-14, 10-16, 12-18, 14-20
    pt_names = [p[0] for p in recalc.PT_PATTERNS]
    pt_values = [row[f"{recalc.TARGET_ITEM}_{name}"] for name in pt_names]

    # Check if all values are NaN
    if pd.isna(pt_values).all():
        row["平均"] = np.nan
        row["最大"] = np.nan
        row["最小"] = np.nan
        row["レンジ"] = np.nan
        row["最大上昇率"] = np.nan
        row["最大上昇速度"] = np.nan
        row["最大下落率"] = np.nan
        row["最大下落速度"] = np.nan
        for p in pt_names[1:]:
            row[f"{p}/2-8"] = np.nan
        return row

    # Ignore NaNs for stats if possible, or maybe all should be calculated exactly.
    # Looking at the target excel, it seems when data is missing it's not well defined, but here we assume complete data or handle NaNs
    valid_pts = [v for v in pt_values if not pd.isna(v)]
    if not valid_pts:
        return row

    mean_val = np.mean(valid_pts)
    max_val = np.max(valid_pts)
    min_val = np.min(valid_pts)
    range_val = max_val - min_val

    # Rise / Fall Rate
    # Rate relative to min/max
    max_rise_rate = (max_val / min_val) - 1 if min_val != 0 else np.nan
    max_fall_rate = (min_val / max_val) - 1 if max_val != 0 else np.nan

    # Speed
    # (P_i / P_{i-1}) - 1
    speeds = []
    for i in range(1, len(pt_values)):
        prev = pt_values[i-1]
        curr = pt_values[i]
        if pd.notna(prev) and pd.notna(curr) and prev != 0:
            speeds.append((curr / prev) - 1)

    max_rise_speed = np.max(speeds) if speeds else np.nan
    max_fall_speed = np.min(speeds) if speeds else np.nan

    row["平均"] = mean_val
    row["最大"] = max_val
    row["最小"] = min_val
    row["レンジ"] = range_val
    row["最大上昇率"] = max_rise_rate
    row["最大上昇速度"] = max_rise_speed
    row["最大下落率"] = max_fall_rate
    row["最大下落速度"] = max_fall_speed

    pt_2_8 = row[f"{recalc.TARGET_ITEM}_2-8"]
    for p in pt_names[1:]:
        if pd.notna(pt_2_8) and pt_2_8 != 0 and pd.notna(row[f"{recalc.TARGET_ITEM}_{p}"]):
            row[f"{p}/2-8"] = row[f"{recalc.TARGET_ITEM}_{p}"] / pt_2_8
        else:
            row[f"{p}/2-8"] = np.nan

    return row

def build_dataset(input_dir: Path, output_excel: Path):
    print(f"Loading data from {input_dir}")
    meas_df, profile_df, metadata = recalc.load_inputs(input_dir)
    meas_df["SampleType"] = meas_df.get("属性", "").apply(recalc.classify_sample_type)

    rates_by_pattern = recalc.build_rates(profile_df)
    curves, _ = recalc.build_calibration_curves(meas_df, rates_by_pattern)

    base_cols = ["source_index", "source_file", "global_request_id", "依頼No.", "SID", "属性", "SampleType"]
    rows = []

    # Recalculate
    for _, sample in meas_df[base_cols].drop_duplicates("global_request_id").iterrows():
        global_id = str(sample["global_request_id"])
        source_index = sample.get("source_index", "GLOBAL")
        measurement_row = meas_df[meas_df["global_request_id"].astype(str) == global_id].iloc[0]

        output_row = {col: sample[col] for col in base_cols}
        output_row["TAT1装置生データ"] = pd.to_numeric(measurement_row.get(recalc.TARGET_ITEM, np.nan), errors="coerce")

        pts = []
        for pattern_name, _, _ in recalc.PT_PATTERNS:
            pattern_rates = rates_by_pattern[pattern_name]
            matched_rate = pattern_rates[
                (pattern_rates["global_request_id"].astype(str) == global_id)
                & (pattern_rates["項目名"].astype(str) == recalc.TARGET_ITEM)
            ]
            rate_value = pd.to_numeric(matched_rate.iloc[0]["Rate_mAbs_min"], errors="coerce") if not matched_rate.empty else np.nan

            cal_rates, cal_concs, curve_source = recalc.find_curve(curves, pattern_name, source_index)
            val = recalc.interpolate_or_extrapolate(rate_value, cal_rates, cal_concs)
            output_row[f"{recalc.TARGET_ITEM}_{pattern_name}"] = val

        # Calculate features
        output_row = calculate_features(output_row)

        rows.append(output_row)

    df = pd.DataFrame(rows)

    # Save directly to Excel
    print(f"Saving dataset to {output_excel}")
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)

    return df

def main():
    parser = argparse.ArgumentParser(description="Build ML dataset directly from integrated parquet files without intermediate CSVs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing measurement.parquet, profile.parquet, etc.")
    parser.add_argument("--output-excel", required=True, help="Output Excel file path for the ML dataset.")
    args = parser.parse_args()

    build_dataset(Path(args.input_dir), Path(args.output_excel))
    print("Done.")

if __name__ == "__main__":
    main()
