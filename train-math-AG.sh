#! /bin/bash

CUDA_VISIBLE_DEVICES=3 python ./train.py --dataset "math-AG" --modelname "math-AG" \
    --model_load_only   0 \
    |& tee -a log_math_AG_train.txt
