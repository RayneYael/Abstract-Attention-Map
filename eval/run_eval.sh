#!/bin/bash

source /data/sfs/home/rensiyu/anaconda3/etc/profile.d/conda.sh
conda activate py311


OUTPUT_DIR="subplot_eval_results"

mkdir -p $OUTPUT_DIR


python screenSpot_pro.py \
    --save_path $OUTPUT_DIR \
    --data_path /data/sfs/home/rensiyu/Benchmark-Dataset/ScreenSpot-Pro


