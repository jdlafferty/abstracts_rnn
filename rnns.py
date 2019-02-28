import tensorflow as tf
import numpy as np
import util
import time
import os, sys
import math
import datetime

from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.layers.core import Dense

try:
  linear = tf.nn.rnn_cell.linear
except:
  from tensorflow.contrib.rnn.python.ops import core_rnn_cell
  linear = core_rnn_cell._linear
  #from tensorflow.python.ops import rnn_cell_impl
  #linear = rnn_cell_impl._linear
  #from tensorflow.python.ops.rnn_cell import _linear as linear

class RNNs:
  def __init__(self, sess, config, reader):
    self.model_type = config.model_type

    self.sess = sess
    self.reader = reader

    self.vocab_size = reader.vocab_size

    # hyper parameters
    self.generator_embed_dim = config.generator_embed_dim
    self.n_hidden = config.n_hidden # rnn
    self.n_layers = config.n_layers # rnn
    self.init_lr = config.learning_rate
    self.pkeep_placeholder = tf.placeholder(tf.float32, name='keep_prob')
    self.train_pkeep = self.dropout = config.dropout
    self.max_grad_norm = config.max_grad_norm
    self.total_epoch = config.total_epoch
    self.init_scale = config.init_scale

    self.checkpoint_dir = config.checkpoint_dir
    self.cell_type = config.cell_type

    self.step = tf.Variable(0, dtype=tf.int32,
    			trainable=False, name="global_step")

    self.decay_step = config.decay_step
    self.lr = tf.train.exponential_decay(
        	      config.learning_rate, self.step, config.decay_step,
        	      config.decay_rate, staircase=True, name="lr")

    self._attrs = ["generator_embed_dim", "n_hidden", "n_layers", "init_lr", "decay_step", "dropout"]

    self.saver = tf.train.Saver()

    self.build_model()

  def get_model_dir(self):
    model_dir = ""
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "%s%s__" % (attr, getattr(self, attr))
    model_dir = model_dir.rstrip('_')
    return model_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess,
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    print checkpoint_dir

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Load SUCCESS")
      sys.stdout.flush()
      return True
    else:
      print(" [!] Load failed...")
      sys.stdout.flush()
      return False

  def build_model(self):
    self._X = tf.placeholder(tf.int32, [None, None], name="X") # one doc [batch_size, n_tokens]
    self._Y = tf.placeholder(tf.int32, [None, None], name="Y")
    #sequence lengths
    self._seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    #batch size
    self.batch_size = tf.placeholder(tf.int32, name="batch_size")

    #build the generator
    self.build_generator()
    #compute loss
    self.compute_loss()
    #compute cross entropy
    self.compute_cross_entropy()
    #optimizer and gradients
    trainable_vars = tf.trainable_variables()

    grads, _ = tf.clip_by_global_norm(
    			        tf.gradients(self.loss, trainable_vars),
    				        self.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.optim = optimizer.apply_gradients(zip(grads, trainable_vars),
    				      global_step=self.step)



  def build_generator(self):
    initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

    with tf.variable_scope("generator", initializer=initializer):
      if self.cell_type == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden)
      elif self.cell_type == 'lstm':
        rnn_cell = LSTMCell(self.n_hidden)
      elif self.cell_type == 'gru':
        rnn_cell = GRUCell(self.n_hidden)

      if self.train_pkeep < 1:
        rnn_cell = DropoutWrapper(rnn_cell, dtype=tf.float32,
                                  output_keep_prob=self.pkeep_placeholder)

      cell = MultiRNNCell([rnn_cell for i in range(self.n_layers)])
      self.rnn_cell = cell

      embedding = tf.get_variable("embedding",
                    [self.vocab_size, self.generator_embed_dim])
      self.embedding = embedding #to visualize nearest neighbors
      X_minus_1 = self._X - 1
      mask = tf.sign(tf.to_float(self._X))
      X_minus_1 = tf.cast(mask, tf.int32) * X_minus_1
      inputs = tf.nn.embedding_lookup(embedding, X_minus_1) # [batch_size, n_tokens, emb_dim]
      if self.train_pkeep < 1:
        inputs = tf.nn.dropout(inputs, self.pkeep_placeholder)

      input_layer = Dense(self.n_hidden, dtype=tf.float32, name='input_projection')
      inputs = input_layer(inputs)

      self.init_state_place = tf.placeholder(tf.float32, [self.n_layers, 2, None, self.n_hidden], "init_state_placeholder") # [n_layers, 2 (i.e., c or h), batch_size, n_hidden]
      tmp = tf.unstack(self.init_state_place, axis=0)
      rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(tmp[lay_i][0], tmp[lay_i][1]) for lay_i in range(self.n_layers)])

      outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                            initial_state=rnn_tuple_state,
                            sequence_length=self._seq_len,
                            dtype=tf.float32) # outputs: [batch_size, max_time, cell_state_size]  # state: LSTM tuple

      if self.train_pkeep < 1:
        outputs = tf.nn.dropout(outputs, self.pkeep_placeholder)

      output = tf.reshape(outputs, [-1, self.n_hidden]) # [batch_size x n_tokens, n_hidden]
      self.final_state = state #final state for each sequence

      self.V = tf.get_variable("V", [self.n_hidden, self.vocab_size])
      self.b = tf.get_variable("b", [self.vocab_size])
      logits = tf.matmul(output, self.V) + self.b # [batch_size x n_words, vocab_size] # right before softmax
      logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
      self.logits = logits # [batch_size, n_words, vocab_size]


      p_y_i = tf.nn.softmax(logits) # [batch_size, n_tokens, vocab_size]
      self.p_y_i = tf.reshape(p_y_i, [-1, self.vocab_size]) # [batch_size x n_tokens, vocab_size]


  def compute_loss(self):
    Y = tf.reshape(self._Y, [-1]) # [batch_size x n_tokens, ]
    mask = tf.sign(tf.to_float(Y))
    Y_minus_1 = Y - 1
    Y_minus_1 = tf.cast(mask, tf.int32) * Y_minus_1
    target_one_hots = tf.one_hot(Y_minus_1, self.vocab_size) # [batch_size x n_tokens, vocab_size] # NOTE: corrected
    # just get the prob of the correct word
    p_y = self.p_y_i * target_one_hots # [batch_size x n_tokens, vocab_size]
    p_y = tf.reduce_sum(p_y, reduction_indices=1) # [batch_size x n_tokens, ]
    p_y = mask * p_y
    log_p_y = tf.log(p_y + 1e-10) # [batch_size x n_tokens, ]
    log_p_y = tf.reshape(log_p_y, [self.batch_size, -1]) # [batch_size, n_tokens]
    sum_log_p_y = tf.reduce_sum(log_p_y, reduction_indices=1) # log likelihood summed across all tokens for each example # [batch_size, ]
    self.sum_log_p_y = sum_log_p_y


    self.loss = - tf.reduce_mean(sum_log_p_y) # mean across examples in batch


  def compute_cross_entropy(self):
    Y = tf.reshape(self._Y, [-1]) # [batch_size x n_tokens, ]
    mask = tf.sign(tf.to_float(Y))
    Y_minus_1 = Y - 1
    Y_minus_1 = tf.cast(mask, tf.int32) * Y_minus_1
    target_one_hots = tf.one_hot(Y_minus_1, self.vocab_size) #NOTE: corrected

    ce = - target_one_hots * tf.log(self.p_y_i + 1e-10) # [batch_size x n_tokens, vocab_size]
    ce = tf.reduce_sum(ce, reduction_indices=1) *mask # [batch_size x n_tokens,]
    self.ce = tf.reduce_sum(ce) / tf.reduce_sum(mask)




  def compute_perplexity(self, data_type, sess):
    """
    This is used for evaluation on validation and test
        "data_type" should be valid or test
    """
    costs = 0.0
    iters = 0

    for batch in self.reader.iterate_minibatches(data_type=data_type, batch_size=100, shuffle=False):
      X, Y, seq_len, batch_size = batch

      feed_dict = {self._X: X,
                   self._Y: Y,
                   self.init_state_place: np.zeros([self.n_layers, 2, batch_size, self.n_hidden]),
                   self.pkeep_placeholder: 1.0,
                   self._seq_len: seq_len,
                   self.batch_size: batch_size}
      ce = sess.run(self.ce, feed_dict=feed_dict) # update per doc?
      costs += ce
      iters += 1

    return np.exp(costs/iters)

  def train(self, config):
    # tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    self.load(self.checkpoint_dir)
    print "Starting training ..."

    start_time = time.time()

    train_batch_size = config.train_batch_size
    steps_per_epoch = math.ceil(self.reader.n_train_docs /train_batch_size)
    total_steps = steps_per_epoch * self.total_epoch

    start_step = self.step.eval()
    start_epoch = int(start_step / steps_per_epoch)

    step = start_step
    min_pp_val = 1000000
    for epoch in xrange(start_epoch, self.total_epoch):
      costs = 0.0
      steps_within = 0
      for (i, batch) in enumerate(self.reader.iterate_minibatches(data_type="train", batch_size=train_batch_size, shuffle=True)):
        if i == 1000:
          break
        X, Y, seq_len, batch_size = batch
        feed_dict = {self._X: X,
                     self._Y: Y,
                     self.init_state_place: np.zeros([self.n_layers, 2, batch_size, self.n_hidden]),
                     self.pkeep_placeholder: self.train_pkeep,
                     self._seq_len: seq_len,
                     self.batch_size: batch_size}
        _, loss, ce, lr = self.sess.run(
                [self.optim, self.loss, self.ce, self.lr],
                feed_dict=feed_dict)
        costs += ce
        steps_within += 1
        step += 1
        if step % 1000 == 0:
          print("Step: [%4d/%4d] time: %4.4f, lr: %.8f, loss: %.8f" \
          	% (step, total_steps, time.time() - start_time, lr, loss))
          sys.stdout.flush()

      # minibatch complete
      dt = datetime.datetime.now()
      print("%s" % dt.strftime("%B %d, %Y: %H:%M:%S"))
      print("Epoch: [%4d/%4d] (Step: [%4d/%4d]) time: %4.4f, perplexity: %.8f" \
        % (epoch+1, self.total_epoch, step, total_steps, time.time() - start_time, np.exp(costs/steps_within)))
      if epoch %5 ==0:
        self.save(self.checkpoint_dir, step)
      pp_val = self.compute_perplexity(data_type="valid", sess=self.sess)
      print("validation perplexity: %.8f" % (pp_val))
      if pp_val < min_pp_val:
        min_pp_val = pp_val
        pp_test = self.compute_perplexity(data_type="test", sess=self.sess)
        print("test perplexity: %.8f" % (pp_test))
      sys.stdout.flush()


  def generate_eq(self, sess):
    print("generating eqs...")

    seq_start_list = ["<sos>"]*100 # ["<sos> \\", "<sos> {", "<sos> P"]
    for seq_start in seq_start_list:
      batch_size = 1

      # initialize eq to generate
      generated_seq = [self.reader.vocab[word] for word in seq_start.split()]
      X = np.reshape(np.array([generated_seq[:-1]]), [1, -1])
      # assuming plainrnn
      feed_dict={self._X: X,
                 self.init_state_place: np.zeros([self.n_layers, 2, batch_size, self.n_hidden]),
                 self.pkeep_placeholder: 1.0,
                 self._seq_len: [len(generated_seq)],
                 self.batch_size: 1}

      state = sess.run(self.final_state, feed_dict=feed_dict)

      ### generate seq ###
      while True:
        X = np.reshape(np.array([generated_seq[-1:]]), [1, 1])
        state = np.stack([np.stack(list(layer)) for layer in state]) # layer: (c,h)
        feed_dict = {self._X: X,
                     self.init_state_place: state,
                     self.pkeep_placeholder: 1.0,
                     self._seq_len: [1],
                     self.batch_size: 1}

        state, p_y_i = sess.run([self.final_state, self.p_y_i], feed_dict=feed_dict)
        next_word_idx = np.random.choice(
                        np.arange(self.reader.vocab_size),
                        replace=True, p=p_y_i.reshape([-1])) + 1
        # next_word_idx = np.argmax(p_y_i.reshape([-1])) +1
        generated_seq.append(next_word_idx)
        if next_word_idx == self.reader.vocab_size -1 \
                                      or len(generated_seq) == 1000: # <eos>
          break
      output = [self.reader.idx2word[word_idx] for word_idx in generated_seq]
      print (("generated seq: %s" % "".join(output)).encode("utf8"))
      print
