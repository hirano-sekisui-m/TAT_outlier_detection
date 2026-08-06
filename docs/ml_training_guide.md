# 機械学習モデル構築＆評価サマリー生成スクリプト 取扱説明書

本ドキュメントでは、作成された特徴量データセット（`機械学習用データセット.xlsx` 等）を入力とし、動的に定義した正解ラベル（目的変数）に対して分類・回帰モデルを交差検証し、包括的な評価サマリーレポートを出力する `scripts/train/train_ml_models.py` の使用方法および仕様について解説します。

---

## 1. 全体概要

本スクリプトは、データ分析・モデル構築作業を効率化するための汎用機械学習パイプラインです。

- **分類タスク (Classification)**: 高値乖離判定、異常検知フラグ、カテゴリ予測など
- **回帰タスク (Regression)**: 装置生データ値や連続数値の予測など

スクリプト上部のコードを数行書き換えるだけで、**タスク種別の切り替え**、**正解ラベル（目的変数）の動的定義**、**モデルの比較学習・レポート出力**を一括して行うことができます。

---

## 2. 入力・出力ファイルの仕様

### 2.1 入力ファイルの要件
- **ファイル形式**: `.xlsx` (Excel) または `.csv`
- **推奨入力**: `build_ml_dataset.py` で出力された `機械学習用データセット.xlsx`
- **使用される特徴量 (22列)**:
  - 換算濃度: `2-8`, `4-10`, `6-12`, `8-14`, `10-16`, `12-18`, `14-20`
  - 装置値: `TAT1装置生データ`
  - 統計量: `平均`, `最大`, `最小`, `レンジ`, `最大上昇率`, `最大上昇速度`, `最大下落率`, `最大下落速度`
  - 相対比率: `4-10/2-8`, `6-12/2-8`, `8-14/2-8`, `10-16/2-8`, `12-18/2-8`, `14-20/2-8`

### 2.2 出力ファイルのパス
デフォルトの出力ディレクトリ:
- **レポート出力先**: `reports/train/`
- **学習済みモデル保存先**: `models/trained/`

#### 生成される成果物:
1. **Excel サマリーレポート (`model_summary_report_classification.xlsx` / `model_summary_report_regression.xlsx`)**
   - **Model_Comparison**: 全モデルの性能比較表（ヘッダー装飾・書式指定済み）
   - **Feature_Importances**: 特徴量重要度 / 係数絶対値のランキング表
   - **CV_Predictions**: 各サンプルの交差検証（CV）による予測値・予測確率データ
   - **Run_Summary**: 実行日時・正解ラベル定義 (`target_definition`)・データ件数・使用モデル等のメタデータ
2. **JSON サマリーレポート (`summary_report_classification.json` / `summary_report_regression.json`)**
   - `target_definition`: 正解ラベル（目的変数）の定義内容（例: `"14-20/2-8 > 1.15"` または `"TAT1装置生データ"`）を含む構造化データ
3. **最優秀モデルファイル (`best_model_*.joblib`)**
   - 5-Fold CVで最も優れたスコア（分類: F1/AUC, 回帰: R²/RMSE）を獲得した学習済みモデルオブジェクト（ターゲット定義メタデータ内包）

---

## 3. スクリプトの書き換え・カスタマイズ方法（設定エリア）

スクリプト `scripts/train/train_ml_models.py` を開いて **25〜60行目付近** にある設定エリアを書き換えます。

```python
# ==========================================================
# 設定エリア (パスや学習条件を変更したい場合はここを修正してください)
# ==========================================================

# 1. 入力・出力パスの設定
DEFAULT_INPUT_DATASET = PROJECT_ROOT / "data" / "example" / "機械学習用データセット.xlsx"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "train"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "trained"

# 2. タスク種別の選択: "classification" (分類) または "regression" (回帰)
TASK_TYPE = "classification"

# 3. 正解ラベル (Target) の動的定義関数
def define_target_label(df: pd.DataFrame, task_type: str = TASK_TYPE) -> pd.Series:
    if task_type == "classification":
        # 【分類の書き換え例】
        # 205/206比 が 1.15 を超える場合を 1 (高値乖離)、それ以外を 0 (非乖離)と定義
        val = pd.to_numeric(df["205/206比"], errors="coerce")
        return (val > 1.15).astype(int)
    else:
        # 【回帰の書き換え例】
        # 「205/206比」をそのまま連続数値ターゲットとして予測対象にする
        return pd.to_numeric(df["205/206比"], errors="coerce")

# 4. パラメータ
RANDOM_STATE = 42
N_SPLITS = 5
# ==========================================================
```

---

## 4. 評価対象モデル

スクリプトは以下の主要アルゴリズムを一度に評価します。

### 分類モデル (Classification)
- **RandomForestClassifier**: 決定木のアンサンブル
- **GradientBoostingClassifier**: 勾配ブースティング
- **ExtraTreesClassifier**: 極度ランダム木
- **LogisticRegression**: 標準化 (`StandardScaler`) パイプライン付きロジスティック回帰
- **SVC**: 標準化 (`StandardScaler`) パイプライン付きサポートベクター分類器

### 回帰モデル (Regression)
- **RandomForestRegressor**: ランダムフォレスト回帰
- **GradientBoostingRegressor**: 勾配ブースティング回帰
- **ExtraTreesRegressor**: エクストラツリー回帰
- **Ridge**: 標準化付き L2 正則化線形回帰
- **Lasso**: 標準化付き L1 正則化線形回帰
- **SVR**: 標準化付き サポートベクター回帰

---

## 5. 実行方法

### 基本実行（スクリプト内設定に従う）
```bash
# 分類タスクを実行
python scripts/train/train_ml_models.py --task-type classification

# 回帰タスクを実行
python scripts/train/train_ml_models.py --task-type regression
```

### オプション指定実行
```bash
python scripts/train/train_ml_models.py \
  --input-dataset "data/example/機械学習用データセット.xlsx" \
  --report-dir "reports/custom_experiment" \
  --model-dir "models/custom_experiment" \
  --task-type classification
```

---

## 6. 保存されたモデルの再利用方法 (Pythonコード例)

保存された `.joblib` モデルは以下のように読み込んで新規データの予測に使用できます。

```python
import joblib
import pandas as pd

# 保存されたモデルパッケージを読み込み
pkg = joblib.load("models/trained/best_model_classification_GradientBoosting.joblib")

model = pkg["model"]
feature_cols = pkg["feature_cols"]

# 未知データの読み込み＆特徴量抽出
df_new = pd.read_excel("data/example/機械学習用データセット.xlsx")
X_new = df_new[feature_cols]

# 予測の実行
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

print("予測結果:", predictions[:5])
print("予測確率:", probabilities[:5])
```
