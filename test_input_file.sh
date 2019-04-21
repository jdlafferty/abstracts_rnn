
CUDA_VISIBLE_DEVICES=2 python test_input_file.py --dataset "hep-th" --modelname "hep-th"
CUDA_VISIBLE_DEVICES=2 python test_input_file.py --dataset "astro-ph" --modelname "astro-ph"
CUDA_VISIBLE_DEVICES=2 python test_input_file.py --dataset "stat-ML" --modelname "stat-ML"
CUDA_VISIBLE_DEVICES=2 python test_input_file.py --dataset "math-AG" --modelname "math-AG"
tail -n +1 data/*log*

