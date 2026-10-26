#!/usr/bin/env bash
set -euo pipefail
python train_nba_now.py --dataset nba --data_path your/data/path/ --num_frame 10 --num_total_frame 40 --num_activities 9 --motion --multi_corr --batch 2
