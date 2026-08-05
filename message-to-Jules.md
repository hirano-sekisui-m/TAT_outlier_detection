scripts\data-manage\統合エクスポート用Pythonスクリプト.pyの機能拡張をお願いします。

現在、統合エクスポート用Pythonスクリプト.pyは、複数の「*.csv」を、measurement.parquet、profile.parquet、metadata.jsonにデータを分割してまとめる機能を有しています。これを、csvファイルが一つであってもmeasurement.parquet、profile.parquet、metadata.jsonにデータを分割してまとめるようにしてください。出力形式は、入力csvが一つでも複数でも同一にします。

想定入力データの例：data\example

データを同一形式でまとめた後、今度は機械学習へかけるデータセットの構築を行います。
今度は入力データが、先ほどの出力データ(measurement.parquet、profile.parquet、metadata.json)になります。
これらを基に、データセットの最終形態は「data\example\機械学習用データセット目標.xlsx」を目指します。

まずはデータに含める列は、Aから順に、source_index	source_file	global_request_id	依頼No.	SID	属性	SampleType　とします。
その右側の列は、機械学習に入れるデータで、data\dictionary下のjsonを参照してデータカラムを設計します。(data\dictionary\high_deviation_dataset_config.json←穴あきデータですが、例を示します。)

2-8や4-10などのデータについては、「data\example\recalc_tat1_custom_points.py」で具体的な計算&出力方法が記載されています。このスクリプトはそのまま動かすと、
TAT1_custom_points_recalc.xlsx
TAT1_custom_points_recalc_recalc_matrix.csv
TAT1_custom_points_recalc_rates.csv
TAT1_custom_points_recalc_cal_rates.csv
の4ファイルが出力されてしまいますが、不要です。
あくまで、最終形態の「data\example\機械学習用データセット目標.xlsx」を目指してください。例えば、TAT1の「2-8」の計算結果を出力した場合は、「2-8」は、「TAT1_2-8」がこれにあたります。「TAT1_2-8_Rate」や、「TAT1_2-8_CalCurve」の列は不要です。

まずはこのような流れで、csv生データから、機械学習用のデータセットを自動で構築するスクリプトの作成を行います。
不明点は必ず私に確認して、実装前に方針レビューをお願いします。
