#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset "stat-ML" \
    --model_load_only   0 \
    |& tee -a log_stat_ML_train.txt
