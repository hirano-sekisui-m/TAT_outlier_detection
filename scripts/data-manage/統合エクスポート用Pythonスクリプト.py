#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F07 CSV integrated exporter

複数の F07 生CSVを個別にパースして統合し、以下をエクスポートします。

- measurement.parquet
- profile.parquet
- metadata.json
- measurement.csv       # 任意、既定で出力
- profile.csv           # 任意、既定で出力
- export_summary.json

特徴
----
- 各CSVを個別にパースしてから結合するため、CSV間で測定項目列が違っても統合可能
- source_file / source_index / global_request_id を追加し、由来追跡と依頼No.重複を回避
- cp932 / utf-8-sig / utf-8 を順に試行
- 元CSVは既定では移動しない

実行例
------
python export_f07_integrated.py \
  "240624 F07---TAT高値検体-スクリ－ニング (1～40).csv" \
  "240625 F07 TAT高値-スクリ－ニング (No.41～150) CSV_data.csv" \
  "CSV_data_F07_20240604074000.csv" \
  --output-dir ./f07_integrated_export

python export_f07_integrated.py --input-dir ./raw-data --output-dir ./f07_integrated_export
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ==========================================================
# Constants
# ==========================================================

DEFAULT_ATTACHED_FILES = [
    "240624 F07---TAT高値検体-スクリ－ニング (1～40).csv",
    "240625 F07 TAT高値-スクリ－ニング (No.41～150) CSV_data.csv",
    "CSV_data_F07_20240604074000.csv",
]

MISSING_STRINGS = {"", "None", "nan", "NaN", "<NA>"}


# ==========================================================
# Public API
# ==========================================================

def process_csv(csv_path: Path | str, source_index: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    単一の F07 生CSVを読み込み、measurement/profile/metadata に変換する。

    Parameters
    ----------
    csv_path : Path or str
        入力CSVパス
    source_index : int
        統合時のファイル番号。global_request_id に使用する。

    Returns
    -------
    measurement_df, profile_df, metadata
    """
    csv_path = Path(csv_path)
    lines, encoding = read_csv_lines(csv_path)

    measurement_df, metadata = parse_measurement_table(lines)
    profile_df, item_mapping = parse_profile_table(lines)

    metadata["item_mapping"] = item_mapping
    metadata["source_csv"] = csv_path.name
    metadata["source_encoding"] = encoding

    measurement_df = add_source_columns(measurement_df, csv_path.name, source_index)
    profile_df = add_source_columns(profile_df, csv_path.name, source_index)

    return measurement_df, profile_df, metadata


def process_multiple_csvs(
    csv_paths: Sequence[Path | str],
    output_dir: Path | str,
    export_csv: bool = True,
    move_processed: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Path]:
    """
    複数CSVを個別にパースして統合し、エクスポートする。

    Parameters
    ----------
    csv_paths : sequence of Path or str
        統合対象CSV
    output_dir : Path or str
        出力先ディレクトリ
    export_csv : bool
        True の場合、確認用CSVも出力する
    move_processed : bool
        True の場合、成功後に入力CSVを processed/ に移動する

    Returns
    -------
    measurement_df_all, profile_df_all, metadata_all, output_dir
    """
    paths = [Path(p) for p in csv_paths]
    if not paths:
        raise ValueError("csv_paths が空です。")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("入力CSVが見つかりません: " + ", ".join(missing))

    measurement_list: List[pd.DataFrame] = []
    profile_list: List[pd.DataFrame] = []
    metadata_list: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, path in enumerate(paths, start=1):
        try:
            measurement_df, profile_df, metadata = process_csv(path, source_index=idx)
            measurement_list.append(measurement_df)
            profile_list.append(profile_df)
            metadata_list.append(metadata)
        except Exception as exc:
            errors.append({"source_csv": path.name, "error": repr(exc)})

    if not measurement_list and not profile_list:
        raise RuntimeError(f"すべてのCSV処理に失敗しました: {errors}")

    measurement_all = concat_dataframes(measurement_list)
    profile_all = concat_dataframes(profile_list)
    metadata_all = merge_metadata(metadata_list, errors=errors)

    metadata_all["row_counts"] = {
        "measurement": int(len(measurement_all)),
        "profile": int(len(profile_all)),
    }

    output_path = export_integrated_data(
        measurement_df=measurement_all,
        profile_df=profile_all,
        metadata=metadata_all,
        output_dir=output_dir,
        export_csv=export_csv,
    )

    if move_processed and not errors:
        for path in paths:
            move_original_csv(path)

    return measurement_all, profile_all, metadata_all, output_path


def load_parsed_data(parsed_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """エクスポート済みディレクトリから parquet/json を再読込する。"""
    parsed_dir = Path(parsed_dir)

    measurement_df = pd.read_parquet(parsed_dir / "measurement.parquet")
    profile_df = pd.read_parquet(parsed_dir / "profile.parquet")

    with open(parsed_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return measurement_df, profile_df, metadata


# ==========================================================
# Read helpers
# ==========================================================

def read_csv_lines(csv_path: Path) -> Tuple[List[str], str]:
    """CSVを複数エンコーディング候補で読み込む。"""
    encodings = ["cp932", "utf-8-sig", "utf-8"]
    last_exc: Optional[Exception] = None

    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, errors="strict", newline="") as f:
                lines = [line.replace("\x00", "") for line in f.readlines()]
                return lines, enc
        except Exception as exc:
            last_exc = exc

    # 最終フォールバック。壊れた文字は置換して処理を継続する。
    try:
        with open(csv_path, "r", encoding="cp932", errors="replace", newline="") as f:
            lines = [line.replace("\x00", "") for line in f.readlines()]
            return lines, "cp932(errors=replace)"
    except Exception as exc:
        raise RuntimeError(f"CSVを読み込めません: {csv_path}, last_error={last_exc!r}") from exc


def is_blank_csv_line(line: str) -> bool:
    """空行またはカンマだけの行を True とする。"""
    stripped = line.strip()
    if stripped == "":
        return True
    cells = [c.strip().replace('"', "") for c in stripped.split(",")]
    return all(c == "" for c in cells)


def normalize_label(value: Any) -> str:
    """列名・項目名の表記ゆれを最低限正規化する。"""
    if value is None:
        return ""
    s = str(value).strip().replace('"', "")
    if s.startswith("Unnamed:"):
        return ""
    return s


def find_measurement_header_idx(lines: Sequence[str]) -> int:
    for i, line in enumerate(lines):
        if "依頼No." in line and (
            "SID" in line or "PID" in line or "検体" in line or "測定日" in line or "属性" in line
        ):
            return i
    raise ValueError("Measurement header not found.")


def find_profile_header_idx(lines: Sequence[str], start: int = 0) -> Optional[int]:
    for i in range(start, len(lines)):
        line = lines[i]
        if "依頼No." in line and ("項目名" in line or "測光" in line or "ﾌﾟﾛﾌｧｲﾙ" in line or "処理値" in line):
            return i
    return None


# ==========================================================
# Measurement parser
# ==========================================================

def parse_measurement_table(lines: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """測定値テーブルを DataFrame 化する。"""
    header_idx = find_measurement_header_idx(lines)
    profile_header_idx = find_profile_header_idx(lines, start=header_idx + 1)

    end_idx = profile_header_idx if profile_header_idx is not None else len(lines)
    while end_idx > header_idx and is_blank_csv_line(lines[end_idx - 1]):
        end_idx -= 1

    measurement_lines = list(lines[header_idx:end_idx])
    if len(measurement_lines) <= 1:
        raise ValueError("Measurement table has no data rows.")

    csv_text = "".join(measurement_lines)
    try:
        df_raw = pd.read_csv(StringIO(csv_text), dtype=str, engine="python")
    except Exception:
        # 一部装置CSVでは引用符が不完全な行が混ざることがあるため、
        # quote 処理を無効化して再読込する。
        df_raw = pd.read_csv(
            StringIO(csv_text),
            dtype=str,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="warn",
        )
    df_raw = df_raw.dropna(how="all")

    original_columns = list(df_raw.columns)
    fixed_count = infer_fixed_column_count(original_columns)
    fixed_cols = original_columns[:fixed_count]

    out = pd.DataFrame(index=df_raw.index)
    for col in fixed_cols:
        clean_col = normalize_label(col)
        if clean_col:
            out[clean_col] = df_raw[col].astype(str).str.strip().replace("nan", "")

    measurement_items: List[str] = []
    measurement_units: Dict[str, str] = {}

    idx = fixed_count
    while idx < len(original_columns):
        value_col = original_columns[idx]
        flag_col = original_columns[idx + 1] if idx + 1 < len(original_columns) else None

        raw_name = normalize_label(value_col)
        item_name, unit = split_item_and_unit(raw_name)

        pair_is_empty = column_is_empty(df_raw, value_col) and (flag_col is None or column_is_empty(df_raw, flag_col))

        # 末尾の空列群は無視する。
        if item_name == "" and pair_is_empty:
            idx += 2
            continue

        if item_name == "":
            item_name = f"unnamed_item_{idx}"

        if item_name not in measurement_items:
            measurement_items.append(item_name)

        if unit:
            measurement_units[item_name] = unit

        out[item_name] = pd.to_numeric(clean_series(df_raw[value_col]), errors="coerce")

        if flag_col is not None:
            out[f"{item_name}_FLAG"] = clean_series(df_raw[flag_col])

        idx += 2

    metadata = {
        "measurement_items": measurement_items,
        "measurement_units": measurement_units,
    }

    return out.reset_index(drop=True), metadata


def infer_fixed_column_count(columns: Sequence[Any]) -> int:
    """
    固定列数を推定する。
    通常は 依頼No., SID, 検体ﾊﾞｰｺｰﾄﾞ, 測定日, 属性 の5列。
    """
    cleaned = [normalize_label(c) for c in columns]
    if "属性" in cleaned:
        return cleaned.index("属性") + 1
    if "測定日" in cleaned:
        return cleaned.index("測定日") + 1
    return min(5, len(columns))


def split_item_and_unit(raw_name: str) -> Tuple[str, str]:
    match = re.match(r"^(.*?)\((.*?)\)$", raw_name)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return raw_name.strip(), ""


def column_is_empty(df: pd.DataFrame, col: Any) -> bool:
    if col not in df.columns:
        return True
    s = clean_series(df[col])
    return s.isin(list(MISSING_STRINGS)).all()


def clean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace('"', "", regex=False).str.strip().replace("nan", "")


# ==========================================================
# Profile parser
# ==========================================================

def parse_profile_table(lines: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """プロファイルテーブルをロング形式 DataFrame に変換する。"""
    profile_header_idx = find_profile_header_idx(lines, start=0)
    if profile_header_idx is None:
        # プロファイルがないCSVも許容する。
        return pd.DataFrame(columns=["依頼No.", "項目名", "項目No.", "測光ﾎﾟｰﾄ", "処理値", "時間", "吸光度"]), {}

    records: List[Dict[str, Any]] = []
    item_mapping: Dict[str, str] = {}

    i = profile_header_idx + 1
    while i < len(lines):
        row1_line = lines[i]
        if is_blank_csv_line(row1_line):
            i += 1
            continue

        token1 = parse_csv_line(row1_line)
        if not looks_like_profile_row1(token1):
            i += 1
            continue

        row2_idx = find_next_absorbance_row(lines, i + 1)
        if row2_idx is None:
            i += 1
            continue

        token2 = parse_csv_line(lines[row2_idx])

        request_no = get_token(token1, 0)
        item_name = get_token(token1, 1)
        item_no = to_number(get_token(token1, 2))
        photometric_port = to_number(get_token(token1, 3))
        processed_value = to_number(get_token(token1, 4))

        if pd.notna(item_no):
            item_mapping[str(int(item_no))] = item_name

        time_values = parse_numeric_tokens(token1[6:])
        absorb_values = parse_numeric_tokens(token2[6:])
        n = min(len(time_values), len(absorb_values))

        for j in range(n):
            records.append({
                "依頼No.": request_no,
                "項目名": item_name,
                "項目No.": item_no,
                "測光ﾎﾟｰﾄ": photometric_port,
                "処理値": processed_value,
                "時間": time_values[j],
                "吸光度": absorb_values[j],
            })

        i = row2_idx + 1

    profile_df = pd.DataFrame(records)
    expected_cols = ["依頼No.", "項目名", "項目No.", "測光ﾎﾟｰﾄ", "処理値", "時間", "吸光度"]
    for col in expected_cols:
        if col not in profile_df.columns:
            profile_df[col] = pd.Series(dtype="float64" if col not in ["依頼No.", "項目名"] else "object")

    return profile_df[expected_cols].reset_index(drop=True), item_mapping


def parse_csv_line(line: str) -> List[str]:
    try:
        return [cell.replace('"', "").strip() for cell in next(csv.reader([line]))]
    except Exception:
        return [cell.replace('"', "").strip() for cell in line.strip().split(",")]


def get_token(tokens: Sequence[str], index: int, default: str = "") -> str:
    if index >= len(tokens):
        return default
    return str(tokens[index]).strip()


def looks_like_profile_row1(tokens: Sequence[str]) -> bool:
    if len(tokens) < 6:
        return False
    request_no = get_token(tokens, 0)
    item_name = get_token(tokens, 1)
    item_no = get_token(tokens, 2)
    port = get_token(tokens, 3)
    return request_no != "" and item_name != "" and is_number_like(item_no) and is_number_like(port)


def find_next_absorbance_row(lines: Sequence[str], start: int) -> Optional[int]:
    for j in range(start, min(start + 5, len(lines))):
        if is_blank_csv_line(lines[j]):
            continue
        tokens = parse_csv_line(lines[j])
        # 吸光度行は先頭側が空で、6列目以降に数値が並ぶ形式が基本。
        if len(tokens) >= 7 and get_token(tokens, 0) == "" and len(parse_numeric_tokens(tokens[6:])) > 0:
            return j
    return None


def is_number_like(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def to_number(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def parse_numeric_tokens(tokens: Iterable[Any]) -> List[float]:
    values: List[float] = []
    for token in tokens:
        s = str(token).strip()
        if s == "":
            continue
        try:
            values.append(float(s))
        except Exception:
            continue
    return values


# ==========================================================
# Merge / Export
# ==========================================================

def add_source_columns(df: pd.DataFrame, source_file: str, source_index: int) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "source_index", source_index)
    out.insert(1, "source_file", source_file)

    if "依頼No." in out.columns:
        req = out["依頼No."].astype(str).str.strip()
    else:
        req = pd.Series(range(1, len(out) + 1), index=out.index).astype(str)

    out.insert(2, "global_request_id", str(source_index) + "_" + req)
    return out


def concat_dataframes(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)


def merge_metadata(metadata_list: Sequence[Dict[str, Any]], errors: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    source_files: List[str] = []
    encodings: Dict[str, str] = {}
    measurement_items: List[str] = []
    measurement_units: Dict[str, str] = {}
    item_mapping: Dict[str, str] = {}

    for meta in metadata_list:
        source = meta.get("source_csv", "")
        if source:
            source_files.append(source)
            encodings[source] = meta.get("source_encoding", "")

        for item in meta.get("measurement_items", []):
            if item not in measurement_items:
                measurement_items.append(item)

        for key, value in meta.get("measurement_units", {}).items():
            measurement_units.setdefault(key, value)

        for key, value in meta.get("item_mapping", {}).items():
            # 既存値と矛盾しない限り保持。矛盾する場合は既存優先。
            item_mapping.setdefault(str(key), value)

    return {
        "source_csv_files": source_files,
        "source_encodings": encodings,
        "measurement_items": measurement_items,
        "measurement_units": measurement_units,
        "item_mapping": item_mapping,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "errors": errors or [],
    }


def export_integrated_data(
    measurement_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Path | str,
    export_csv: bool = True,
) -> Path:
    """統合データを parquet/json/csv に保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_path = output_dir / "measurement.parquet"
    profile_path = output_dir / "profile.parquet"

    measurement_df.to_parquet(measurement_path, index=False)
    profile_df.to_parquet(profile_path, index=False)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if export_csv:
        # Excelで開きやすいように utf-8-sig で出力。
        measurement_df.to_csv(output_dir / "measurement.csv", index=False, encoding="utf-8-sig")
        profile_df.to_csv(output_dir / "profile.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "files": {
            "measurement_parquet": "measurement.parquet",
            "profile_parquet": "profile.parquet",
            "metadata_json": "metadata.json",
            "measurement_csv": "measurement.csv" if export_csv else None,
            "profile_csv": "profile.csv" if export_csv else None,
        },
        "row_counts": metadata.get("row_counts", {}),
        "source_csv_files": metadata.get("source_csv_files", []),
        "errors": metadata.get("errors", []),
    }

    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_dir


# ==========================================================
# Optional processed move
# ==========================================================

def move_original_csv(csv_path: Path | str) -> Path:
    """処理済みCSVを processed/ に移動する。"""
    csv_path = Path(csv_path)
    processed_dir = csv_path.parent / "processed"
    processed_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = processed_dir / f"{timestamp}_{csv_path.name}"
    shutil.move(str(csv_path), str(dest))
    return dest


# ==========================================================
# Analysis helpers preserved from original style
# ==========================================================

def try_parse_json_obj(value: Any) -> Any:
    """文字列化されたJSONや辞書オブジェクトから値を取り出すヘルパー。"""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                return None
        return s
    return None


def extract_sample_id(measurement_df: pd.DataFrame, preferred_keys: Optional[List[str]] = None) -> pd.Series:
    """measurement_df から検体ID候補を抽出する。"""
    if preferred_keys is None:
        preferred_keys = ["SID", "検体ID", "PID", "SampleID", "ID", "検体ﾊﾞｰｺｰﾄﾞ", "属性", "依頼No."]

    for key in preferred_keys:
        if key not in measurement_df.columns:
            continue

        series = measurement_df[key]
        if key == "属性":
            parsed = series.map(try_parse_json_obj)
            if parsed.dropna().apply(lambda x: isinstance(x, dict)).any():
                for inner_key in ["SID", "検体ID", "PID", "SampleID", "ID"]:
                    extracted = parsed.map(lambda d: d.get(inner_key) if isinstance(d, dict) else None)
                    s = extracted.astype(str).str.strip()
                    if not s.isin(list(MISSING_STRINGS)).all():
                        return s

        s = series.astype(str).str.strip()
        if not s.isin(list(MISSING_STRINGS)).all():
            return s

    first_col = measurement_df.columns[0]
    return measurement_df[first_col].astype(str).str.strip()


def detect_prescription_columns(measurement_df: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
    """metadata の measurement_items と measurement_df 列名の交差を返す。"""
    items = metadata.get("measurement_items", []) if metadata else []
    common = [c for c in measurement_df.columns if c in items]
    if common:
        return common
    return [c for c in measurement_df.columns if str(c).startswith("処方") or "処方" in str(c)]


def load_parsed_for_analysis(parsed_dir: Path | str, sample_id_col_name: str = "SID"):
    """統合済みデータを読み込み、sample id と測定値列候補を付与して返す。"""
    measurement_df, profile_df, metadata = load_parsed_data(parsed_dir)
    sample_series = extract_sample_id(measurement_df)

    measurement_df = measurement_df.copy()
    measurement_df[sample_id_col_name] = sample_series.values

    prescription_columns = detect_prescription_columns(measurement_df, metadata)
    return measurement_df, profile_df, metadata, prescription_columns


# ==========================================================
# CLI
# ==========================================================

def discover_input_files(args: argparse.Namespace) -> List[Path]:
    if args.csv_files:
        return [Path(p) for p in args.csv_files]

    if args.input_dir:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"CSVが見つかりません: {input_dir}")
        return files

    # 引数なしで実行した場合は、スクリプトと同じディレクトリにある添付3ファイル名を探す。
    base_dir = Path(__file__).resolve().parent
    files = [base_dir / name for name in DEFAULT_ATTACHED_FILES]
    existing = [p for p in files if p.exists()]
    if existing:
        return existing

    raise FileNotFoundError("入力CSVが指定されていません。csv_files または --input-dir を指定してください。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F07 CSV 3ファイル統合エクスポートスクリプト")
    parser.add_argument("csv_files", nargs="*", help="統合対象CSVファイル")
    parser.add_argument("--input-dir", default=None, help="CSVをまとめて読み込むディレクトリ")
    parser.add_argument("--output-dir", default="./f07_integrated_export", help="出力先ディレクトリ")
    parser.add_argument("--no-csv", action="store_true", help="確認用CSVを出力しない")
    parser.add_argument("--move-processed", action="store_true", help="成功後、入力CSVを processed/ に移動する")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    csv_paths = discover_input_files(args)

    print("Input CSV files:")
    for p in csv_paths:
        print(f"  - {p}")

    measurement_df, profile_df, metadata, output_dir = process_multiple_csvs(
        csv_paths=csv_paths,
        output_dir=args.output_dir,
        export_csv=not args.no_csv,
        move_processed=args.move_processed,
    )

    print()
    print("Export done")
    print(f"Output dir  : {output_dir}")
    print(f"Measurement : {measurement_df.shape}")
    print(f"Profile     : {profile_df.shape}")
    print(f"Items       : {metadata.get('measurement_items', [])}")
    if metadata.get("errors"):
        print("Warnings/errors:")
        for err in metadata["errors"]:
            print(f"  - {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
