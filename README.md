# AmedasPlot
* [カラー高層実況天気図](https://tokiyui.github.io/Faxmap/) を表示しています。 [局地天気図](https://tokiyui.github.io/Faxmap/local.html) はここからご確認いただけます。
* ご要望がありましたら、Twitterアカウント @cho_tokisen までご連絡いただければ、可能な範囲で対応いたします。

## 注意！
* ここで表示している図は、00から20分頃：一つ前の正時、20分過ぎから59分：直近の正時の画像ですが、エラーなどによって更新が止まることがあります。
* 各種データは誤差や異常値を含む可能性がありますが、品質チェックをせずに作図しているため、不適切な情報が含まれる可能性があります。

## 今後の目標
* 過去の事例を解析できるよう、気象庁HPのJSON（10日分くらいしか残らない）以外からデータを取得したいと考えていますが、誰か代わりに改修してください。
* 海上の観測データを拡充したい

## データ取得元
* アメダスデータ、LIDEN、解析雨量：気象庁HPのJSON
* 船舶観測：[NOAA](https://www.ndbc.noaa.gov/ship_obs.php?uom=M&time=1)
* 高層観測：[Wyoming University](https://www.weather.uwyo.edu/upperair/sounding.shtml)
* 解析値：ECMWFの直近初期値(FT=12)を利用 (https://data.ecmwf.int/ecpds/home/opendata/)

## special thanks
* 本ページの作成プログラムの大部分は黒良さんのブログ（https://note.com/rkurora/）から流用させていただいています。
