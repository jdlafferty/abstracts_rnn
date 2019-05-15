import tensorflow as tf
import pdb
import numpy as np
from config import Config
from read_corpus import Reader
from rnns import RNNs
from copy import deepcopy
from util import *
np.random.seed(3)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_load_only", type=int, default=False)
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--modelname", type=str, default="")
args = parser.parse_args()




def main(_):
  config = Config(args.dataset, args.modelname,False)
  reader = Reader(config)
  with tf.Session() as sess:
    model = RNNs(sess, config, reader)
    model.load(config.checkpoint_dir)
    with open('input.pickle', 'rb') as myfile:
        fake_note = pickle.load(myfile)
        likelihoods = [777]*len(fake_note)
        for i in range(len(fake_note)):
            print(i)
            text = fake_note[i]
            likelihood = model.compute_likelihood(model.sess, text)
            likelihoods[i] = likelihood

    with open('data/%s_loglikelihood.pickle' % args.modelname, 'wb') as myfile:
        pickle.dump(likelihoods, myfile)


if __name__ == '__main__':
	tf.app.run()

#CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "hep-th" --modelname "hep-th"
#CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "astro-ph" --modelname "astro-ph"
#CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "stat-ML" --modelname "stat-ML"
#CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "math-AG" --modelname "math-AG"
