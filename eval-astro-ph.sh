#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python ./evaluate.py --dataset "math-AG" --modelname "astro-ph" \
    |& tee -a log_astro_ph_eval.txt
