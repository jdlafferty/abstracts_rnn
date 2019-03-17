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
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--modelname", type=str, default="")
args = parser.parse_args()

def main(_):
  config = Config(args.dataset, args.modelname)
  reader = Reader(config)
  sess = tf.Session()
  model = RNNs(sess, config, reader)

  model.load(config.checkpoint_dir)
  test_pp = model.compute_perplexity(data_type="test", sess=model.sess)
  print("Perplexity of model %s on data %s: %f" % (args.modelname, args.dataset, test_pp))


if __name__ == '__main__':
	tf.app.run()
