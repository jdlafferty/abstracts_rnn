#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_load_only   0 \
    |& tee -a log_charlm_layer3_hidden500_decaystep20000_train.txt
