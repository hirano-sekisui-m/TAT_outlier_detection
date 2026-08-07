"""
Build ML Dataset dynamically based on ML_param and CP_param.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Use generalized custom points calculator
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "scripts"))
# noqa: E402
import recalc_custom_points as recalc  # noqa: E402

# ==========================================================
# 設定エリア (パスを変更したい場合はここを書き換えてください)
# ==========================================================
def parse_path(path_str: str) -> Path:
    s = path_str.strip('\'"').replace("\\", "/").replace("¥", "/")
    p = Path(s)
    if not p.is_absolute():
        return BASE_DIR / p
    return p

DEFAULT_INPUT_DIR = parse_path(r'data/260806_検証用/260806_integrated_export')
DEFAULT_OUTPUT_EXCEL = parse_path(r'data/260806_検証用/260806_ML_dataset.xlsx')
# ==========================================================


def load_dataset_inputs(input_dir: Path, cp_param: dict):
    input_dir = Path(input_dir)
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
        
        target_item_col = recalc.find_target_item_column(meas_df.columns, cp_param)
        if target_item_col is None:
            raise ValueError(f"Target item matching {cp_param.get('項目名')} not found in measurement.parquet")

        for col in ["source_index", target_item_col]:
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
        return meas_df, profile_df, metadata, target_item_col
    else:
        return recalc.load_inputs(input_dir, cp_param)


def calculate_features(output_row: dict, pt_names: list[str]) -> dict:
    pt_values = [output_row[name] for name in pt_names if name in output_row]

    if pd.isna(pt_values).all() or len([v for v in pt_values if pd.notna(v)]) == 0:
        output_row["平均"] = np.nan
        output_row["最大"] = np.nan
        output_row["最小"] = np.nan
        output_row["レンジ"] = np.nan
        output_row["最大上昇率"] = np.nan
        output_row["最大上昇速度"] = np.nan
        output_row["最大下落率"] = np.nan
        output_row["最大下落速度"] = np.nan
        if len(pt_names) > 0:
            base_pt = pt_names[0]
            for p in pt_names[1:]:
                output_row[f"{p}/{base_pt}"] = np.nan
        return output_row

    valid_pts = [v for v in pt_values if pd.notna(v)]
    mean_val = np.mean(valid_pts)
    max_val = np.max(valid_pts)
    min_val = np.min(valid_pts)
    range_val = max_val - min_val

    max_rise_rate = (max_val / min_val) - 1.0 if min_val != 0 else np.nan
    max_fall_rate = (min_val / max_val) - 1.0 if max_val != 0 else np.nan

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

    if len(pt_names) > 0:
        base_pt = pt_names[0]
        base_val = output_row.get(base_pt, np.nan)
        for p in pt_names[1:]:
            val = output_row.get(p, np.nan)
            if pd.notna(base_val) and base_val != 0 and pd.notna(val):
                output_row[f"{p}/{base_pt}"] = val / base_val
            else:
                output_row[f"{p}/{base_pt}"] = np.nan

    return output_row


def build_dataset(input_dir: Path, output_excel: Path, ml_param: dict, cp_param: dict) -> pd.DataFrame:
    print(f"Loading data from {input_dir}")
    meas_df, profile_df, _metadata, target_item_col = load_dataset_inputs(input_dir, cp_param)
    meas_df["SampleType"] = meas_df.get("属性", "").apply(recalc.classify_sample_type)

    pt_names = ml_param.get("ml_photometric_points", [])
    pt_patterns = [recalc.parse_pt_pattern(p) for p in pt_names]

    calc_method = cp_param.get("測定方法", "rate")
    cal_shape = cp_param.get("Cal形状", "piecewise_linear")

    rates_by_pattern = recalc.build_rates(profile_df, target_item_col, pt_patterns, calc_method)
    curves, _ = recalc.build_calibration_curves(meas_df, rates_by_pattern, target_item_col, cp_param)

    rates_lookup = {}
    for pattern_name, rate_df in rates_by_pattern.items():
        sub = rate_df[rate_df["項目名"].astype(str) == target_item_col]
        for _, r in sub.iterrows():
            gid = str(r["global_request_id"])
            rates_lookup[(pattern_name, gid)] = pd.to_numeric(r["Rate_mAbs_min"], errors="coerce")

    base_cols = ["source_index", "source_file", "global_request_id", "依頼No.", "SID", "属性", "SampleType"]
    rows = []

    unique_samples = meas_df[base_cols].drop_duplicates("global_request_id")

    for _, sample in unique_samples.iterrows():
        global_id = str(sample["global_request_id"])
        source_index = sample.get("source_index", "GLOBAL")
        measurement_row = meas_df[meas_df["global_request_id"].astype(str) == global_id].iloc[0]

        output_row = {col: sample[col] for col in base_cols}
        output_row["装置生データ"] = pd.to_numeric(measurement_row.get(target_item_col, np.nan), errors="coerce")

        for pattern_name in pt_names:
            rate_value = rates_lookup.get((pattern_name, global_id), np.nan)
            cal_rates, cal_concs, _ = recalc.find_curve(curves, pattern_name, source_index)
            val = recalc.interpolate_concentration(rate_value, cal_rates, cal_concs, cal_shape)
            output_row[pattern_name] = val

        output_row = calculate_features(output_row, pt_names)
        rows.append(output_row)

    df = pd.DataFrame(rows)

    target_columns = ml_param.get("target_columns", [])
    if not target_columns:
        target_columns = list(df.columns)

    cols_to_use = [c for c in target_columns if c in df.columns] + [c for c in df.columns if c not in target_columns]
    df = df[cols_to_use]

    print(f"Saving dataset to {output_excel}")
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Build ML dataset directly from integrated parquet/CSV files using dynamic parameters.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help=f"Directory containing measurement and profile data. (Default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-excel", default=str(DEFAULT_OUTPUT_EXCEL), help=f"Output Excel file path for the ML dataset. (Default: {DEFAULT_OUTPUT_EXCEL})")
    parser.add_argument("--model", default="TAT_outlier_classification", help="Name of the ML model config to use (without .json extension)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_excel)

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    ml_param_path = BASE_DIR / "param" / "ML_param" / f"{args.model}.json"
    if not ml_param_path.exists():
        print(f"Error: ML config '{ml_param_path}' does not exist.")
        sys.exit(1)

    with open(ml_param_path, "r", encoding="utf-8") as f:
        ml_param = json.load(f)

    target_item = ml_param.get("target_item")
    if not target_item:
        print("Error: 'target_item' not found in ML config.")
        sys.exit(1)

    cp_param_path = BASE_DIR / "param" / "CP_param" / f"{target_item}.json"
    if not cp_param_path.exists():
        print(f"Error: CP config '{cp_param_path}' does not exist.")
        sys.exit(1)

    with open(cp_param_path, "r", encoding="utf-8") as f:
        cp_param = json.load(f)

    build_dataset(input_path, output_path, ml_param, cp_param)
    print("Done.")


if __name__ == "__main__":
    main()
