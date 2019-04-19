#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python ./train.py \
    --model_load_only   1 --dataset "stat-ML" --modelname "stat-ML"
