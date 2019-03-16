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
args = parser.parse_args()

def main(_):
  config = Config(args.dataset)
  reader = Reader(config)

  with tf.Session() as sess:
    model = RNNs(sess, config, reader)

    if args.model_load_only:
      model.load(config.checkpoint_dir)
    else:
      model.train(config)


    #generate text...
    if config.generate_text:
      model.generate_eq(model.sess)


if __name__ == '__main__':
	tf.app.run()
