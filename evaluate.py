import tensorflow as tf
import numpy as np
import re

from config_models import Config
from read_corpus import Reader
from rnns import RNNs
from copy import deepcopy
from util import *

np.random.seed(54)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_load_only", type=int, default=True)
args = parser.parse_args()

def main(_):
  config = Config()
  reader = Reader(config)
  sess = tf.Session()
  model = RNNs(sess, config, reader)

  model.load(config.checkpoint_dir)
  test_pp = model.compute_perplexity(data_type="test", sess=model.sess)
  print("test perplexity: %f" % test_pp)


if __name__ == '__main__':
	tf.app.run()
