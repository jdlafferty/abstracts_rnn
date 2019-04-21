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
    with open("data/input_file.txt", "r") as myfile:
        text = myfile.read()
        outfile = open('data/%s_loglikelihood.txt' % args.modelname, 'w')    
        #model.generate_eq(model.sess)
        likelihood = model.compute_likelihood(model.sess, text)
        print >>outfile, likelihood
        print(likelihood)

if __name__ == '__main__':
	tf.app.run()


for i in range(0,len(p_y_i[0])):
    print( 'i+1 %s: %f\n' % (self.reader.idx2word[i+1], p_y_i[0][i]))

