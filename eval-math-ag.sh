#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python ./evaluate.py --dataset "hep-th" --modelname "math-AG" \
    |& tee -a log_math_ag_eval.txt
