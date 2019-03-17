#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python ./evaluate.py --dataset "math-AG" --modelname "hep-th" \
    |& tee -a log_hep_th_eval.txt
