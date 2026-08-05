# 高値乖離単独判別モデル

## 目的
I列～AD列の特徴量から、高値乖離検体を判別する。

## 正解定義
H列「205/206比」 > 1.15 を高値乖離と定義する。
G列「乖離区分」は学習には使用しない。

## 使用モデル
RandomForestClassifier

## 使用特徴量
装置生データ, 2-8, 4-10, ..., 終点/始点

## 推奨判定閾値
高値乖離確率 >= 0.686

## 実行例
python scripts/validate/validate_high_deviation_rf.py \
  --input data/raw/乖離特徴分析2_測光6ポイント_TAT1のみ.xlsx \
  --model models/production/high_deviation_rf_classifier.joblib \
  --output outputs/predictions/predictions.csv
