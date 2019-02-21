#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python ./train.py \
    --model_load_only   1 \
    |& tee -a log_charlm_layer3_hidden500_decaystep20000_test.txt
