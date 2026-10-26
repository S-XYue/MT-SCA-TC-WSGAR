#!/usr/bin/env bash
set -euo pipefail
python train.py --dataset volleyball --data_path your/data/path/ --num_frame 5 --num_total_frame 10 --num_activities 8 --motion --enc_layers 2 --nheads 2 --nheads_agg 2 --weight_decay 1e-3 --test_batch 6 --batch 6
