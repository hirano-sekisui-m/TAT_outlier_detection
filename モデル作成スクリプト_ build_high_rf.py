import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

INPUT = Path('/mnt/data/乖離特徴分析2_測光6ポイント_TAT1のみ.xlsx')
MODEL_OUT = Path('/mnt/data/high_deviation_rf_classifier.joblib')
META_OUT = Path('/mnt/data/high_deviation_rf_metadata.json')
PRED_OUT = Path('/mnt/data/high_deviation_rf_cv_predictions.csv')

RANDOM_STATE = 42
POSITIVE_THRESHOLD_RATIO = 1.15
PROBABILITY_THRESHOLD = 0.686  # CVでF1最大付近だった閾値
FEATURE_COLS = [
    '装置生データ', '2-8', '4-10', '6-12', '8-14', '10-16', '12-18', '14-20',
    '平均', '最大', '最小', 'レンジ', '最大上昇率', '最大上昇速度', '最大下落率', '最大下落速度',
    '4-10/2-8', '6-12/2-8', '8-14/2-8', '10-16/2-8', '12-18/2-8', '終点/始点'
]
TARGET_COL = '205/206比'


def make_model():
    return RandomForestClassifier(
        n_estimators=1000,
        random_state=RANDOM_STATE,
        min_samples_leaf=3,
        class_weight='balanced',
        n_jobs=-1,
    )


def classification_metrics(y_true, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average='binary', zero_division=0
    )
    return {
        'threshold': float(threshold),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'accuracy': float((tp + tn) / len(y_true)),
        'auc': float(roc_auc_score(y_true, score)),
        'average_precision': float(average_precision_score(y_true, score)),
    }


def main():
    df = pd.read_excel(INPUT, engine='openpyxl')
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    y_ratio = pd.to_numeric(df[TARGET_COL], errors='coerce')
    mask = X.notna().all(axis=1) & y_ratio.notna()
    X = X.loc[mask].reset_index(drop=True)
    y_ratio = y_ratio.loc[mask].reset_index(drop=True)
    y_high = y_ratio > POSITIVE_THRESHOLD_RATIO
    df_valid = df.loc[mask].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_model = make_model()
    cv_score = cross_val_predict(base_model, X, y_high, cv=cv, method='predict_proba')[:, 1]
    cv_metrics_default = classification_metrics(y_high, cv_score, PROBABILITY_THRESHOLD)

    # 全データで最終学習。検証値は上のCV結果を見ること。
    final_model = make_model()
    final_model.fit(X, y_high)

    package = {
        'model': final_model,
        'feature_cols': FEATURE_COLS,
        'target_definition': f'{TARGET_COL} > {POSITIVE_THRESHOLD_RATIO}',
        'positive_threshold_ratio': POSITIVE_THRESHOLD_RATIO,
        'probability_threshold': PROBABILITY_THRESHOLD,
        'random_state': RANDOM_STATE,
        'model_type': 'RandomForestClassifier',
        'model_params': final_model.get_params(),
        'note': 'G列の乖離区分は学習に使用していません。教師信号はH列 205/206比 > 1.15 から作成しています。',
        'sklearn_version': sklearn.__version__,
        'python_version': platform.python_version(),
    }
    joblib.dump(package, MODEL_OUT, compress=3)

    pred_df = pd.DataFrame({
        'row_number_1based_after_header': np.arange(2, len(df_valid) + 2),
        '依頼No.': df_valid['依頼No.'] if '依頼No.' in df_valid.columns else np.nan,
        'SID': df_valid['SID'] if 'SID' in df_valid.columns else np.nan,
        '乖離区分_eval_only': df_valid['乖離区分'] if '乖離区分' in df_valid.columns else np.nan,
        TARGET_COL: y_ratio,
        'true_high_by_H_gt_1_15': y_high,
        'cv_probability_high': cv_score,
        f'cv_pred_high_prob_ge_{PROBABILITY_THRESHOLD}': cv_score >= PROBABILITY_THRESHOLD,
    })
    pred_df.to_csv(PRED_OUT, index=False, encoding='utf-8-sig')

    metadata = {
        'input_file': str(INPUT),
        'n_rows_used': int(len(X)),
        'n_positive_H_gt_1_15': int(y_high.sum()),
        'n_negative': int((~y_high).sum()),
        'feature_cols': FEATURE_COLS,
        'target_definition': f'{TARGET_COL} > {POSITIVE_THRESHOLD_RATIO}',
        'probability_threshold': PROBABILITY_THRESHOLD,
        'cv': {'type': 'StratifiedKFold', 'n_splits': 5, 'shuffle': True, 'random_state': RANDOM_STATE},
        'cv_metrics_at_threshold': cv_metrics_default,
        'model_file': str(MODEL_OUT),
        'cv_predictions_file': str(PRED_OUT),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    META_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
