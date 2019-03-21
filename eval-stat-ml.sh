#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python ./evaluate.py --dataset "math-AG" --modelname "stat-ML" \
    |& tee -a log_stat_ml_eval.txt
