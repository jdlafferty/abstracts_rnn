import os, sys
import json, re, codecs
import itertools
import numpy as np
import nltk
import math
from collections import OrderedDict

import tensorflow as tf

from util import *
from collections import Counter

np.random.seed(54)

class Reader(object):
  def __init__(self, config):
    #self.vocab_size = config.vocab_size
    self.data_path = data_path = config.data_path
    if not os.path.exists(self.data_path): os.mkdir(self.data_path)
    self.raw_data_root = config.raw_data_root
    self.vocab_path = config.vocab_path

    #assume vocabulary is built in advance
    train_data_path = self.data_path+'/'+'X_train.pkl'
    if os.path.exists(train_data):
      print("loading the vocabulary..."); sys.stdout.flush()
      self.vocab, self.vocab_size = load_pkl(self.vocab_path)
      self._load()
    else:
      print("getting and saving the train, valid and test data...")
      sys.stdout.flush()
      train_objs, valid_objs, test_objs = self._read_and_get_objdata(self.raw_data_root)

      print("loading the vocabulary..."); sys.stdout.flush()
      self.vocab, self.vocab_size = load_pkl(self.vocab_path)
      print("preparing final run data..."); sys.stdout.flush()
      self.X_train, self.Y_train = self._obj_to_data(train_objs, data_type='train')
      self.X_valid, self.Y_valid = self._obj_to_data(valid_objs, data_type='valid')
      self.X_test, self.Y_test = self._obj_to_data(test_objs, data_type='test')

    self.n_train_docs = math.ceil((len(self.X_train) + 0.0))
    self.n_valid_docs = math.ceil((len(self.X_valid) + 0.0))
    self.n_test_docs = math.ceil((len(self.X_test) + 0.0))
    self.idx2word = {v: k for k,v in self.vocab.items()}
    print("vocabulary size: {}".format(self.vocab_size))
    print("number of training documents: {}".format(self.n_train_docs))
    print("number of validation documents: {}".format(self.n_valid_docs))
    print("number of testing documents: {}".format(self.n_test_docs))
    sys.stdout.flush()

  def _read_and_get_objdata(self, data_dir):
    obj = json.load(codecs.open(data_dir, "r", encoding="utf8"))
    data = []
    for arxiv_id in obj:
      text = obj[arxiv_id]
      data.append(text)
    #permutation = list(np.random.choice(range(len(data)), len(data), replace=False))
    #n_train = int(len(data) * 0.9)
    #train = list(np.array(data)[permutation])[:n_train]
    #valid = list(np.array(data)[permutation])[n_train:]
    #test = list(np.array(data)[permutation])[n_train:]
    n_train = int(len(data) * 0.9)
    train = data[:n_train]
    valid = data[n_train:]
    test = data[n_train:]
    return train, valid, test

  # def _clean_up_sent(self, sent): # sent: a list of words for one sentence
  #   ret_sent = []
  #   for word in sent:
  #     if (len(word) == 1) or (re.search("\d", word)): continue
  #     ret_sent.append(word.lower())
  #   return ret_sent


  def _build_vocab(self, train_objs, vocab_path):
    train = [] # just a list of all tokens in train
    for text in train_objs:
      text = [char for char in text.strip()]
      train += text
    word_freq = nltk.FreqDist(train)

    print("train length: ", len(train))
    vocab_size = self.vocab_size
    vocab = word_freq.most_common(vocab_size - 3)
    vocab_words = [x[0] for x in vocab]
    print("vocab tokens: ", vocab)
    print("vocab tokens length: ", len(vocab_words))

    vocab_words += ['<sos>', '<eos>', '<unk>']
    vocab = {x: i+1 for i,x in enumerate(vocab_words)}
    self.vocab = vocab
    print("official vocab length: ", len(vocab))
    self.vocab_size = len(vocab)
    save_pkl(vocab_path, [self.vocab, self.vocab_size])

  def _truncate_ver(self, seqs):
    max_len = 150
    step = max_len // 2
    new_seqs = []
    for seq in seqs:
      seq_len = len(seq)
      for i in range(0, seq_len -step, step):
        trunc_seq = seq[i: i+max_len]
        new_seqs.append(trunc_seq)
    return new_seqs

  def _obj_to_data(self, objs_data, data_type):
    seqs = []
    for text in objs_data:
      seq = [char for char in text.strip()]
      if seq != []:
        seqs.append(['<sos>']+ seq +['<eos>'])
    seqs_trunced = self._truncate_ver(seqs)  #truncated

    X = [seq[:-1] for seq in seqs_trunced] # for seq input
    Y = [seq[1:] for seq in seqs_trunced] # for seq target
    X = np.asarray([[self.vocab.get(tok, self.vocab_size) for tok in seq] for seq in X]) # token -> id
    # NOTE: vocab id starts from 1
    # NOTE: almost [n_examples, n_toks]; but be careful: this is still a np.array of lists
    Y = np.asarray([[self.vocab.get(tok, self.vocab_size) for tok in seq] for seq in Y])
    save_pkl(self.data_path+'/'+'X_'+data_type+'.pkl', X)
    save_pkl(self.data_path+'/'+'Y_'+data_type+'.pkl', Y)
    return X, Y

  def _load(self):
    print("loading the pickled data..."); sys.stdout.flush()
    self.X_train = load_pkl(self.data_path+'/'+'X_train'+'.pkl')
    self.Y_train = load_pkl(self.data_path+'/'+'Y_train'+'.pkl')
    self.X_valid = load_pkl(self.data_path+'/'+'X_valid'+'.pkl')
    self.Y_valid = load_pkl(self.data_path+'/'+'Y_valid'+'.pkl')
    self.X_test = load_pkl(self.data_path+'/'+'X_test'+'.pkl')
    self.Y_test = load_pkl(self.data_path+'/'+'Y_test'+'.pkl')

  def get_data_from_type(self, data_type):
    if data_type == "train":
      X = self.X_train
      Y = self.Y_train
    elif data_type == "valid":
      X = self.X_valid
      Y = self.Y_valid
    elif data_type == "test":
      X = self.X_test
      Y = self.Y_test
    else:
      raise Exception(" [!] Unkown data type %s: %s" % data_type)
    return X, Y

  def get_length(self, data):
  	"""
  	data is a document...a list of sentences (or seqs)
  	this returns the length of each sentence in data
  	"""
  	return [len(x) for x in data]

  def pad(self, data):
    """
    data is a document...a list of sentences (or seqs)
    this pads all the shorter sentences to
    match the length of the longest sequence
    """
    lengths = self.get_length(data)
    maxlen = max(lengths)
    n_rows = len(data)
    padded = np.zeros([n_rows, maxlen], dtype=np.int32)

    for i, length in enumerate(lengths):
    	padded[i, :length] = data[i]

    return padded


  def iterate_minibatches(self, data_type="train", batch_size=10, shuffle=False):
    X, Y = self.get_data_from_type(data_type)
    assert len(X) == len(Y)
    if shuffle:
      indices = np.arange(len(X))
      np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
      if shuffle:
        excerpt = indices[start_idx:start_idx + batch_size]
      else:
        excerpt = slice(start_idx, start_idx + batch_size)
      X_mini = X[excerpt] # X_mini: np.array of lists (each list = seq in tokens)
      Y_mini = Y[excerpt] # same as above; after padding => a complete np.array
      seq_len = self.get_length(X_mini)
      batch_size = len(X_mini) # need to get the actual batch_size (e.g., consider last minibatch)
      yield self.pad(X_mini), self.pad(Y_mini), seq_len, batch_size
