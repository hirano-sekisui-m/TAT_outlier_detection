#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest 高値乖離単独判別モデルの検証・推論スクリプト

使い方:
  python validate_high_deviation_rf.py --input 乖離特徴分析2_測光6ポイント_TAT1のみ.xlsx \
      --model high_deviation_rf_classifier.joblib \
      --output high_deviation_rf_predictions_from_script.csv

仕様:
- 学習済みモデルは joblib 形式で読み込みます。
- G列「乖離区分」は推論・学習には使いません。存在する場合も評価表示にのみ使います。
- H列「205/206比」が存在する場合だけ、正解 y = 205/206比 > 1.15 を作成して検証指標を出します。
- H列がない新規データの場合は、推論結果CSVだけを出します。
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(y_true, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average='binary', zero_division=0
    )
    result = {
        'threshold': float(threshold),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'accuracy': float((tp + tn) / len(y_true)),
    }
    # 片側クラスしかない場合はAUCが計算できないためスキップ
    if len(np.unique(y_true)) == 2:
        result['auc'] = float(roc_auc_score(y_true, score))
        result['average_precision'] = float(average_precision_score(y_true, score))
    return result


def main():
    parser = argparse.ArgumentParser(description='Validate or run high-deviation Random Forest classifier.')
    parser.add_argument('--input', required=True, help='Input .xlsx or .csv file')
    parser.add_argument('--model', required=True, help='Model .joblib file')
    parser.add_argument('--output', default='high_deviation_rf_predictions_from_script.csv', help='Output CSV path')
    parser.add_argument('--threshold', type=float, default=None, help='Probability threshold; default is model metadata value')
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    package = joblib.load(model_path)
    model = package['model']
    feature_cols = package['feature_cols']
    threshold = package.get('probability_threshold', 0.686) if args.threshold is None else args.threshold
    target_col = '205/206比'
    positive_ratio_threshold = package.get('positive_threshold_ratio', 1.15)

    if input_path.suffix.lower() in ['.xlsx', '.xlsm', '.xls']:
        df = pd.read_excel(input_path, engine='openpyxl')
    elif input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError('Unsupported input format. Use .xlsx or .csv')

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    valid_mask = X.notna().all(axis=1)
    if not valid_mask.all():
        print(f'Warning: {int((~valid_mask).sum())} rows have missing/non-numeric feature values and will be excluded.')

    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)
    X_valid = X.loc[valid_mask].reset_index(drop=True)

    score = model.predict_proba(X_valid)[:, 1]
    pred = score >= threshold

    out = pd.DataFrame({
        'row_number_1based_after_header': np.where(valid_mask)[0] + 2,
    })
    for col in ['依頼No.', 'SID', '乖離区分', target_col]:
        if col in df_valid.columns:
            out[col] = df_valid[col].values
    out['probability_high_deviation'] = score
    out['threshold'] = threshold
    out['pred_high_deviation'] = pred
    out.to_csv(output_path, index=False, encoding='utf-8-sig')

    report = {
        'model': str(model_path),
        'input': str(input_path),
        'output': str(output_path),
        'n_rows_input': int(len(df)),
        'n_rows_scored': int(len(df_valid)),
        'threshold': float(threshold),
        'target_definition': package.get('target_definition', f'{target_col} > {positive_ratio_threshold}'),
        'features_used': feature_cols,
        'note': 'G列 乖離区分 は推論には使用していません。',
    }

    if target_col in df_valid.columns:
        y_ratio = pd.to_numeric(df_valid[target_col], errors='coerce')
        eval_mask = y_ratio.notna()
        y_true = (y_ratio.loc[eval_mask] > positive_ratio_threshold).to_numpy()
        report['evaluation'] = compute_metrics(y_true, score[eval_mask.to_numpy()], threshold)

        if '乖離区分' in df_valid.columns:
            report['crosstab_label_eval_only'] = pd.crosstab(
                df_valid.loc[eval_mask, '乖離区分'],
                pd.Series(pred[eval_mask.to_numpy()], name='pred_high_deviation')
            ).to_dict()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
