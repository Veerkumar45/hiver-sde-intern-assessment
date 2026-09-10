@echo off
python -m src.pipeline --data data/raw/sample.csv --brand AppleSupport --limit 100
python -m src.evaluate --predictions outputs/predictions.csv
