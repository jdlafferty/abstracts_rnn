#! /bin/bash

CUDA_VISIBLE_DEVICES=4 python ./train.py --dataset "hep-th" --modelname "hep-th" \
    --model_load_only   0 \
    |& tee -a log_hep_th_train.txt
