#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train.py --dataset "math-AG" --modelname "astro-ph" \
    --model_load_only   0 \
    |& tee -a log_astro_ph_train.txt
