scripts/下のpythonスクリプトを整理したいです。
現在、ディレクトリがdata-manage、train、validate、oldと別れていますが、oldだけ残して、他のディレクトリ下にあるpythonスクリプトは、全てscripts/下で一括管理したいです。
その上で、ファイルをリネームします。

(変更前)
scripts\data-manage\統合エクスポート用Pythonスクリプト.py
scripts\data-manage\build_ml_dataset.py
scripts\train\train_ml_models.py
scripts\validate\predict_ml_models.py

(変更後)
scripts\01-1_data-merger.py
scripts\01-2_build_ml_dataset.py
scripts\02_train_ml_models.py
scripts\03_predict_ml_models.py

上記のように変更をお願いします。
また、ファイル名変更に伴い、ファイル内でパスの依存関係などがあれば調査して、リネーム後にエラーが生じないように事前対策をお願いします。

さらに、各pythonスクリプト内でのパス入力簡便化を行いたいです。
具体的には、以下のような記述です。

scripts\data-manage\統合エクスポート用Pythonスクリプト.py
```python
# 統合データ (measurement.parquet, profile.parquet等) の出力先フォルダパス
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "260806_検証用" / "260806_integrated_export"
```
現在、上記のフォルダ指定をする場合、スラッシュで区切り、さらにダブルクォーテーションで囲ってディレクトリ指定をする必要があります。
これでは入力の際に面倒なので、フォルダを作成して「相対パスのコピー&ペースト」で、フォルダパスを指定できるように変更してください。
フォルダパスの指定は、scripts\01-1_data-merger.py、scripts\01-2_build_ml_dataset.py、scripts\02_train_ml_models.py、scripts\03_predict_ml_models.pyの内部で入力データや出力データ、参照モデルのパスなど、全てで簡便なパス指定が可能になるように変更をお願いします。
Windowsでは、パスをコピーした際に、スラッシュがバックスラッシュになったり¥(円マーク)になることがありましたっけ？その場合にも対応できるようにロバストな設計にしてください。

変更内容について、不明点があれば事前に私に確認をしてから実装をしてください。
修正内容が明確であれば、早速実装に取り掛かってください。