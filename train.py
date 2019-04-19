import tensorflow as tf
import numpy as np

from config import Config
from read_corpus import Reader
from rnns import RNNs
from copy import deepcopy
from util import *

np.random.seed(54)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_load_only", type=int, default=False)
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--modelname", type=str, default="")
args = parser.parse_args()

def main(_):
  config = Config(args.dataset, args.modelname)
  reader = Reader(config)

  with tf.Session() as sess:
    model = RNNs(sess, config, reader)

    if args.model_load_only:
      model.load(config.checkpoint_dir)
    else:
      model.train(config)


    #generate text...
    if config.generate_text:
      test_string0 = "We trained a neural network."
      test_string1 = "Fofo fofo fd ss dd ddfffsss."
      test_string2 = "The meadow is very beautiful"
      test_string3 = "We model the applicability "
      print(model.compute_likelihood(model.sess, test_string0))
      print(model.compute_likelihood(model.sess, test_string1))
      print(model.compute_likelihood(model.sess, test_string2))
      print(model.compute_likelihood(model.sess, test_string3))
      #model.generate_eq(model.sess)

if __name__ == '__main__':
	tf.app.run()
