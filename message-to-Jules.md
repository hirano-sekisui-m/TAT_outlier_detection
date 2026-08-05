scripts\data-manage\統合エクスポート用Pythonスクリプト.pyの機能拡張をお願いします。

現在、統合エクスポート用Pythonスクリプト.pyは、複数の「*.csv」を、measurement.parquet、profile.parquet、metadata.jsonにデータを分割してまとめる機能を有しています。これを、csvファイルが一つであってもmeasurement.parquet、profile.parquet、metadata.jsonにデータを分割してまとめるようにしてください。出力形式は、入力csvが一つでも複数でも同一にします。

想定入力データの例：