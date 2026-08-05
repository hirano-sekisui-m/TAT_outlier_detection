「学習データ、モデル、検証スクリプト、メタデータJSON、予測結果」を継続管理するため、データ・コード・モデル・評価結果を明確に分ける構造にします。今回作成した high_deviation_rf_classifier.joblib、validate_high_deviation_rf.py、high_deviation_rf_metadata.json なども、そのまま配置しやすい形で提案します。

ディレクトリ構造
TAT_outlier_detection/
├── README.md
├── requirements.txt
├── config/
│   └── model_config.json
│
├── data/
│   ├── raw/
│   │   └── 乖離特徴分析2_測光6ポイント_TAT1のみ.xlsx
│   ├── processed/
│   │   └── training_dataset_v001.csv
│   └── external/
│
├── models/
│   ├── production/
│   │   ├── high_deviation_rf_classifier.joblib
│   │   └── high_deviation_rf_metadata.json
│   └── archive/
│       └── v001_20260805/
│           ├── high_deviation_rf_classifier.joblib
│           └── high_deviation_rf_metadata.json
│
├── scripts/
│   ├── train/
│   │   └── build_high_rf.py
│   ├── validate/
│   │   └── validate_high_deviation_rf.py
│   └── predict/
│       └── predict_high_deviation.py
│
├── reports/
│   ├── validation/
│   │   ├── high_deviation_rf_validation_report.json
│   │   └── high_deviation_rf_cv_predictions.csv
│   └── figures/
│
├── outputs/
│   └── predictions/
│       └── high_deviation_rf_predictions_from_script.csv
│
└── docs/
    ├── feature_definition.md
    ├── model_specification.md
    └── operation_rule.md

各フォルダの役割
data/raw/

元のExcelやCSVをそのまま保存します。

data/raw/
└── 乖離特徴分析2_測光6ポイント_TAT1のみ.xlsx


ここは手で編集しない場所にするのがおすすめです。
 後から「どの元データで学習したか」を追跡できます。

data/processed/

学習に使いやすい形へ整形したデータを置きます。

例:

training_dataset_v001.csv


作成する場合は、以下のような列を含めると管理しやすいです。

装置生データ, 2-8, 4-10, ..., 終点/始点, target_high_deviation


今回のモデルでは、教師ラベルはG列ではなく、H列から作った

205/206比 > 1.15


です。モデルメタデータにもこの定義を入れています。3.csv

models/production/

現在使う本番モデルを置きます。

models/production/
├── high_deviation_rf_classifier.joblib
└── high_deviation_rf_metadata.json


今回出力済みのモデルファイルとJSONはここに置くのが良いです。1.py

models/archive/

過去モデルを保存します。

models/archive/
└── v001_20260805/
    ├── high_deviation_rf_classifier.joblib
    └── high_deviation_rf_metadata.json


モデルを更新したら、古いモデルは削除せずに archive/ に移します。
 これにより、「以前の判定結果を再現したい」という時に対応できます。

scripts/train/

学習用スクリプトを置きます。

scripts/train/
└── build_high_rf.py


今回作成した build_high_rf.py はここに置くのが自然です。4.json

scripts/validate/

検証用スクリプトを置きます。

scripts/validate/
└── validate_high_deviation_rf.py


今回作成した検証スクリプトはここに配置します。

reports/validation/

検証結果、交差検証の予測結果、評価レポートを置きます。

reports/validation/
├── high_deviation_rf_validation_report.json
└── high_deviation_rf_cv_predictions.csv


交差検証結果はモデル性能の根拠として残しておくと良いです。5.py

outputs/predictions/

実際に検体を判定した出力CSVを置きます。

outputs/predictions/
└── high_deviation_rf_predictions_from_script.csv


新しい検証データや実運用データを流した結果は、ここに日付付きで保存すると良いです。7.csv

ファイル名のおすすめルール

モデルや出力は、日付とバージョンを入れると管理しやすくなります。

high_deviation_rf_classifier_v001_20260805.joblib
high_deviation_rf_metadata_v001_20260805.json
validation_report_v001_20260805.json
predictions_20260805_batch001.csv


本番用だけは短い名前にしておきます。

models/production/high_deviation_rf_classifier.joblib
models/production/high_deviation_rf_metadata.json


過去版は archive/ に保存します。

README.md に書いておくべき内容

最低限、以下は書いておくと後から困りません。

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

requirements.txt 例
pandas
numpy
scikit-learn
openpyxl
joblib


より厳密に再現したい場合は、バージョン固定がおすすめです。今回の出力環境では以下でした。3.csv

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
openpyxl
joblib

私のおすすめ
最初はこの形が一番扱いやすいです。

high_deviation_model/
├── data/
├── models/
├── scripts/
├── reports/
├── outputs/
├── docs/
├── README.md
└── requirements.txt


特に重要なのは、rawデータ、model、script、reportを混ぜないことです。
 この構造にしておくと、モデル更新、検証、過去結果の再現がかなり楽になります。