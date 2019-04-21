#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python ./train.py --dataset "astro-ph" --modelname "astro-ph" \
    --model_load_only   0 \
    |& tee -a log_astro_ph_train.txt

#from tensorflow.python.client import device_lib
#print device_lib.list_local_devices()
