# TAT 異常検知・高値乖離分析パイプライン

本リポジトリは、自動分析装置CP3kシリーズの生データCSVを統合、キャリブレーション再計算＆特徴量データセット構築、機械学習（分類・回帰）モデルの自動学習・評価、および未知データの推論・予測までを一貫して行う自動パイプラインです。

---

## 全体ドキュメント一覧 (`docs/`)

システム全体の仕様およびスクリプトごとの取扱説明書は以下をご参照ください。

- **[全体ワークフロー概要 (workflow_overview.md)](ROOT_DIR/docs/workflow_overview.md)**: パイプライン全体のフロー図、構造、および概要
- **[データセット構築の取説 (dataset_building_guide.md)](ROOT_DIR/docs/dataset_building_guide.md)**: 生データ統合および特徴量Excel構築スクリプトの取説
- **[モデル学習＆評価の取説 (ml_training_guide.md)](ROOT_DIR/docs/ml_training_guide.md)**: 分類・回帰モデル構築＆多モデル評価サマリー出力スクリプトの取説
- **[未知データ予測の取説 (ml_prediction_guide.md)](ROOT_DIR/docs/ml_prediction_guide.md)**: 保存モデルによる未知データ推論スクリプトの取説

---

## クイック実行

```bash
# 1-1. 生CSVデータのパースと統合
python scripts/01-1_data-merger.py

# 1-2. 機械学習用特徴量データセットの生成
python scripts/01-2_build_ml_dataset.py

# 3. 分類・回帰モデルの学習、評価、サマリー出力
python scripts/02_train_ml_models.py --task-type classification
python scripts/02_train_ml_models.py --task-type regression

# 4. 未知データに対する予測実行
python scripts/03_predict_ml_models.py
```
