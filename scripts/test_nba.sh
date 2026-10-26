#!/usr/bin/env bash
set -euo pipefail
python test_NBA.py --dataset nba --data_path your/data/path/ --num_frame 10 --num_total_frame 40 --num_activities 9 --motion --multi_corr --model_path your/checkpoint/path/model.pth
