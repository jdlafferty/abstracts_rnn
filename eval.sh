#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./evaluate.py \
    --model_load_only   1 \
    |& tee -a log_evaluate.txt
