@echo off
.venv\Scripts\python.exe train.py --csv "archive/HAM10000_metadata.csv" --img_dir "archive/Skin Cancer/Skin Cancer" --epochs 15 --batch_size 32 > training_log.txt 2>&1
