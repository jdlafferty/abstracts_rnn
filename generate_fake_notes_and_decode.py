
import tensorflow as tf
import numpy as np
import re
from config import Config
from read_corpus import Reader
from rnns import RNNs
from copy import deepcopy
from util import *
from nltk import tokenize
from hmmlearn import hmm
import viterbi_path

N = 100
types = ['astro-ph', 'hep-th', 'math-AG', 'stat-ML']
proportions = [ 0.1 , 0.5, 0.2, 0.2]
concat_splits = []
for t in types:
    config = Config(t,t, True)
    reader = Reader(config)
    concatenated = ''.join([reader.idx2word[i] for j in range(1,1000) for i in reader.X_valid[j]])
    concat_split = tokenize.sent_tokenize(concatenated)
    concat_splits.append(concat_split)

fake_note = [];
section_label = [];
for ti in range(4):
    section_length = int(round(proportions[ti] * N))
    fake_note += concat_splits[ti][0:section_length] 
    len(fake_note)
    section_label += [types[ti]]*section_length
    len(section_label)

with open('input.pickle', 'wb') as handle:
    pickle.dump(fake_note, handle)

emission_probabilities = np.zeros((4, len(fake_note)))
for ti in range(4):
    with open('data/%s_loglikelihood.pickle' % types[ti], 'rb') as handle:
        state_probs = pickle.load( handle)
        emission_probabilities[ti,] = state_probs

#np.exp(emission_probabilities)

transition_probability = np.array( [
    [0.925, 0.025,0.025,0.025],
    #[0.007, 0.979,0.007,0.007],
    [0.025, 0.925,0.025,0.025],
    [0.025, 0.025,0.925,0.025],
    [0.025, 0.025,0.025,0.925]])

start = np.array([0.25]*4)
reload(viterbi_path)
decoded = viterbi_path.viterbi_path( start, transition_probability, (emission_probabilities), scaled=False)

decoded
section_label

for i in range(len(decoded)):
    print "%s-%s" % (section_label[i],types[decoded[i]])

np.equal([ types[i] for i in decoded ] ,section_label)

sum(np.array(types)[decoded] == section_label)

np.equal(section_label,np.array(types)[decoded])

np.mean(section_label==[ types[i] for i in decoded ])
len(decoded)

types[]


CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "hep-th" --modelname "hep-th"
CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "astro-ph" --modelname "astro-ph"
CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "stat-ML" --modelname "stat-ML"
CUDA_VISIBLE_DEVICES=3 python compute_likelihood_pickle.py --dataset "math-AG" --modelname "math-AG"

#
## We have N observations, and we need the emission probabilities for each
## observation from each of the 4 states
#
#reader.X_valid[1]
#
#print(concat_split[:-1])
#print( 'i+1 %s: %f\n' % (self.reader.idx2word[i+1], p_y_i[0][i]))
#
#
#sess = tf.Session()
#model = RNNs(sess, config, reader)
#
#model.load(config.checkpoint_dir)
#test_pp = model.compute_perplexity(data_type="test", sess=model.sess)
#print("Perplexity of model %s on data %s: %f" % (args.modelname, args.dataset, test_pp))
#
#
