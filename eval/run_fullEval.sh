#!/bin/bash

source /ssd/rensiyu/.conda/etc/profile.d/conda.sh
conda activate py311


OUTPUT_DIR="fullplot_eval_results"

mkdir -p $OUTPUT_DIR


python screenSpot_pro.py \
    --save_path $OUTPUT_DIR \
    --data_path /ssd/rensiyu/Benchmark-Dataset/ScreenSpot-Pro \
    --use_subplot




