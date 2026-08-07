#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom photometric-point concentration recalculation.

Reads CP_param and ML_param to dynamically compute concentrations
for specified photometric point ranges based on piecewise_linear
calibration curves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PT_TIMES = {
    1: 0.0, 2: 9.2, 3: 18.0, 4: 27.2, 5: 36.0, 6: 45.2, 7: 54.0,
    8: 63.2, 9: 72.0, 10: 81.2, 11: 90.0, 12: 99.2, 13: 108.0,
    14: 117.2, 15: 126.0, 16: 135.2, 17: 144.0, 18: 153.2,
    19: 162.0, 20: 171.2, 21: 180.0,
}


def parse_pt_pattern(pattern_str: str) -> tuple[str, int, int]:
    parts = pattern_str.split("-")
    if len(parts) == 2:
        return pattern_str, int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid pattern string: {pattern_str}")

def clean_text(value):
    if pd.isna(value):
        return value
    text = str(value).replace("\x00", "").strip()
    for _ in range(2):
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
    return text.replace('""', '"')

def find_target_item_column(columns: pd.Index, cp_param: dict) -> str | None:
    # 1. 検索用項目名による完全一致
    search_names = cp_param.get("検索用項目名", [])
    for col in columns:
        if col in search_names:
            return col

    # 2. 対象カラム正規表現によるマッチ
    regex_pattern = cp_param.get("対象カラム正規表現")
    if regex_pattern:
        for col in columns:
            if re.match(regex_pattern, str(col)):
                return col

    # 3. 項目名(デフォルト)のチェック
    default_name = cp_param.get("項目名")
    if default_name and default_name in columns:
        return default_name

    return None

def load_inputs(input_dir: Path, cp_param: dict):
    measurement_path = input_dir / "measurement.csv"
    profile_path = input_dir / "profile.csv"
    metadata_path = input_dir / "metadata.json"

    if not measurement_path.exists():
        raise FileNotFoundError(f"measurement.csv not found: {measurement_path}")
    if not profile_path.exists():
        raise FileNotFoundError(f"profile.csv not found: {profile_path}")

    meas_df = pd.read_csv(measurement_path, dtype=str, encoding="utf-8-sig", engine="python")
    profile_df = pd.read_csv(profile_path, dtype=str, encoding="utf-8-sig", engine="python")

    for df in (meas_df, profile_df):
        for col in df.columns:
            df[col] = df[col].map(clean_text)

    target_item_col = find_target_item_column(meas_df.columns, cp_param)
    if target_item_col is None:
        raise ValueError(f"Target item matching {cp_param.get('項目名')} not found in measurement.csv")

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

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return meas_df, profile_df, metadata, target_item_col

def classify_sample_type(value) -> str:
    text = str(value).strip()
    upper = text.upper()
    if text in ["", "None", "nan", "NaN"]:
        return "OTHER"
    if "CAL" in upper or "キャリブ" in text or "SALINE" in upper or "生食" in text:
        return "CAL"
    if "CTRL" in upper or "CONTROL" in upper or "CONT" in upper or "コントロール" in text or "パネル" in text:
        return "QC"
    return "SAMPLE"

def calc_rate_mabs_min(time_values, absorbance_values, pt_start: int, pt_end: int) -> float:
    if pt_start not in PT_TIMES or pt_end not in PT_TIMES:
        return np.nan

    times = np.asarray(time_values, dtype=float)
    absorbance = np.asarray(absorbance_values, dtype=float)
    if len(times) == 0 or len(times) != len(absorbance):
        return np.nan

    t_start = PT_TIMES[pt_start]
    t_end = PT_TIMES[pt_end]
    idx_start = int(np.argmin(np.abs(times - t_start)))
    idx_end = int(np.argmin(np.abs(times - t_end)))

    delta_t = times[idx_end] - times[idx_start]
    if delta_t <= 0:
        return np.nan

    return float(((absorbance[idx_end] - absorbance[idx_start]) * 0.1) / (delta_t / 60.0))

def calc_end_absorbance(time_values, absorbance_values, pt_start: int, pt_end: int) -> float:
    times = np.asarray(time_values, dtype=float)
    absorbance = np.asarray(absorbance_values, dtype=float)
    if len(times) == 0 or len(times) != len(absorbance):
        return np.nan

    t_end = PT_TIMES.get(pt_end, None)
    if t_end is None:
        return np.nan

    idx_end = int(np.argmin(np.abs(times - t_end)))
    return float(absorbance[idx_end])

def representative(values, target_n: int, calc_method: str = "median") -> float:
    arr = [float(v) for v in values if not pd.isna(v)]
    n_actual = len(arr)
    if n_actual == 0:
        return np.nan

    if n_actual >= target_n and target_n >= 3:
        if calc_method.lower() == "median":
            return float(np.median(arr[:target_n]))
        else:
            return float(np.mean(arr[:target_n]))
    elif n_actual == 2 or target_n == 2:
        return float(np.mean(arr[:2]))
    else:
        return float(arr[0])

def interpolate_piecewise_linear(rate, cal_rates, cal_concs) -> float:
    if pd.isna(rate) or cal_rates is None or len(cal_rates) < 2:
        return np.nan

    r = np.asarray(cal_rates, dtype=float)
    c = np.asarray(cal_concs, dtype=float)
    order = np.argsort(r)
    r = r[order]
    c = c[order]
    r, unique_idx = np.unique(r, return_index=True)
    c = c[unique_idx]

    if len(r) < 2:
        return np.nan

    if r[0] <= rate <= r[-1]:
        return float(np.interp(rate, r, c))

    if rate < r[0]:
        denom = r[1] - r[0]
        if denom == 0:
            return float(c[0])
        return float(c[0] + (c[1] - c[0]) / denom * (rate - r[0]))

    denom = r[-1] - r[-2]
    if denom == 0:
        return float(c[-1])
    return float(c[-1] + (c[-1] - c[-2]) / denom * (rate - r[-1]))

def interpolate_linear(rate, cal_rates, cal_concs) -> float:
    # 最小二乗法で直線近似
    if pd.isna(rate) or cal_rates is None or len(cal_rates) < 2:
        return np.nan

    r = np.asarray(cal_rates, dtype=float)
    c = np.asarray(cal_concs, dtype=float)

    A = np.vstack([r, np.ones(len(r))]).T
    m, c_val = np.linalg.lstsq(A, c, rcond=None)[0]

    return float(m * rate + c_val)

def interpolate_spline(rate, cal_rates, cal_concs) -> float:
    from scipy.interpolate import UnivariateSpline
    if pd.isna(rate) or cal_rates is None or len(cal_rates) < 4:
        return interpolate_piecewise_linear(rate, cal_rates, cal_concs)

    r = np.asarray(cal_rates, dtype=float)
    c = np.asarray(cal_concs, dtype=float)
    order = np.argsort(r)
    r = r[order]
    c = c[order]
    r, unique_idx = np.unique(r, return_index=True)
    c = c[unique_idx]

    if len(r) < 4:
        return interpolate_piecewise_linear(rate, cal_rates, cal_concs)

    spline = UnivariateSpline(r, c, k=3, s=0) # cubic spline interpolation
    return float(spline(rate))


def interpolate_concentration(rate, cal_rates, cal_concs, cal_shape: str) -> float:
    shape_lower = cal_shape.lower() if cal_shape else ""
    if shape_lower == "spline":
        return interpolate_spline(rate, cal_rates, cal_concs)
    elif shape_lower == "linear":
        return interpolate_linear(rate, cal_rates, cal_concs)
    else: # default to piecewise_linear (折れ線)
        return interpolate_piecewise_linear(rate, cal_rates, cal_concs)

def cal_level_from_attr(attr):
    text = str(attr).strip()
    upper = text.upper()
    if "SALINE" in upper or "生食" in text:
        return 0
    match = re.search(r"CAL\s*[-_ ]?\s*(\d+)", upper)
    if match:
        level = int(match.group(1))
        return level if 0 <= level <= 5 else None
    return None

def build_rates(profile_df: pd.DataFrame, target_item_col: str, pt_patterns: list[tuple[str, int, int]], method: str) -> dict[str, pd.DataFrame]:
    profile = profile_df[profile_df["項目名"].astype(str) == target_item_col].copy()
    if profile.empty:
        pass

    profile["時間"] = pd.to_numeric(profile["時間"], errors="coerce")
    profile["吸光度"] = pd.to_numeric(profile["吸光度"], errors="coerce")
    profile = profile.dropna(subset=["時間", "吸光度"])

    group_cols = ["source_index", "source_file", "global_request_id", "依頼No.", "項目名"]
    grouped_curves = []
    for keys, group in profile.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        sorted_group = group.sort_values("時間")
        grouped_curves.append((
            base,
            sorted_group["時間"].to_numpy(float),
            sorted_group["吸光度"].to_numpy(float),
        ))

    rates_by_pattern = {}
    for pattern_name, pt_start, pt_end in pt_patterns:
        rows = []
        for base, times, absorbance in grouped_curves:
            row = base.copy()
            row["測光Ptパターン"] = pattern_name
            if method == "end":
                row["Rate_mAbs_min"] = calc_end_absorbance(times, absorbance, pt_start, pt_end)
            else:
                row["Rate_mAbs_min"] = calc_rate_mabs_min(times, absorbance, pt_start, pt_end)
            rows.append(row)
        rates_by_pattern[pattern_name] = pd.DataFrame(rows)

    return rates_by_pattern

def build_calibration_curves(meas_df: pd.DataFrame, rates_by_pattern: dict[str, pd.DataFrame], target_item_col: str, cp_param: dict):
    target_n = int(cp_param.get("Cal測定n数", 3))
    calc_method = cp_param.get("Cal代表値算出方法", "median")

    cal_info = cp_param.get("キャリブレーター濃度情報", {})
    std_concs = cal_info.get("濃度リスト", [])
    if not std_concs:
        raise ValueError("キャリブレーター濃度情報（濃度リスト）が CP_param に定義されていません。")

    cal_rows = []
    for _, row in meas_df.iterrows():
        request_no = str(row.get("依頼No.", "")).strip()
        c_match = re.match(r"^C(\d+)$", request_no)
        attr_level = cal_level_from_attr(row.get("属性", ""))

        if not c_match and attr_level is None:
            continue

        raw_value = pd.to_numeric(row.get(target_item_col, np.nan), errors="coerce")
        if pd.isna(raw_value):
            continue

        cal_rows.append({
            "source_index": row.get("source_index", "GLOBAL"),
            "source_file": row.get("source_file", ""),
            "global_request_id": row.get("global_request_id", ""),
            "依頼No.": request_no,
            "属性": row.get("属性", ""),
            "項目名": target_item_col,
            "装置生データ": raw_value,
            "cal_source_type": "C_ID" if c_match else "ATTR",
            "c_num": int(c_match.group(1)) if c_match else np.nan,
            "cal_level_attr": attr_level if attr_level is not None else np.nan,
        })

    cal_df = pd.DataFrame(cal_rows)
    curves_by_pattern_source = {}
    summary_rows = []

    if cal_df.empty:
        return curves_by_pattern_source, pd.DataFrame()

    for pattern_name, rate_df in rates_by_pattern.items():
        curves_by_pattern_source[pattern_name] = {}
        joined = cal_df.merge(
            rate_df[["source_index", "global_request_id", "項目名", "Rate_mAbs_min"]],
            on=["source_index", "global_request_id", "項目名"],
            how="left",
        )
        joined["cal_rate"] = pd.to_numeric(joined["Rate_mAbs_min"], errors="coerce").fillna(
            pd.to_numeric(joined["装置生データ"], errors="coerce")
        )

        for source in list(joined["source_index"].dropna().unique()) + ["GLOBAL"]:
            source_df = joined if source == "GLOBAL" else joined[joined["source_index"].astype(str) == str(source)]
            if source_df.empty:
                continue

            buckets = {i: [] for i in range(len(std_concs))}

            c_sub = source_df[source_df["cal_source_type"] == "C_ID"].dropna(subset=["c_num", "cal_rate"]).sort_values("c_num")
            if len(c_sub) > 0:
                values = c_sub["cal_rate"].astype(float).tolist()
                idx = 0
                level = 0
                while idx < len(values) and level < len(std_concs):
                    buckets[level].append(representative(values[idx:idx + target_n], target_n, calc_method))
                    idx += target_n
                    level += 1

            attr_sub = source_df[source_df["cal_level_attr"].notna()].dropna(subset=["cal_rate"])
            for level, group in attr_sub.groupby("cal_level_attr"):
                level = int(level)
                if 0 <= level < len(std_concs):
                    buckets[level].append(representative(group["cal_rate"].tolist(), target_n, calc_method))

            cal_rates = []
            cal_concs = []
            for level, conc in enumerate(std_concs):
                rate_value = representative(buckets[level], target_n, calc_method)
                if not pd.isna(rate_value):
                    cal_rates.append(rate_value)
                    cal_concs.append(conc)

            if len(cal_rates) >= 2:
                cal_rates = np.array(cal_rates, dtype=float)
                cal_concs = np.array(cal_concs, dtype=float)
                curves_by_pattern_source[pattern_name][source] = (cal_rates, cal_concs)

                record = {
                    "測光Ptパターン": pattern_name,
                    "source_index": source,
                    "項目名": target_item_col,
                    "有効Cal点数": len(cal_rates),
                }
                for i, (rv, cv) in enumerate(zip(cal_rates, cal_concs)):
                    record[f"CalPoint{i}_濃度"] = cv
                    record[f"CalPoint{i}_Rate_mAbs_min"] = rv
                summary_rows.append(record)

    return curves_by_pattern_source, pd.DataFrame(summary_rows)

def find_curve(curves, pattern_name, source_index):
    if pattern_name in curves:
        if source_index in curves[pattern_name]:
            cal_rates, cal_concs = curves[pattern_name][source_index]
            return cal_rates, cal_concs, f"source_index={source_index}"
        if "GLOBAL" in curves[pattern_name]:
            cal_rates, cal_concs = curves[pattern_name]["GLOBAL"]
            return cal_rates, cal_concs, "GLOBAL"
    return None, None, "NO_CURVE"

