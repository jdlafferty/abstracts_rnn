class Config:
  def __init__(self, dataset, modelname):
    self.dataset = dataset
    self.modelname = modelname
    #  dataset="astro-ph"
    self.vocab_path = "data/arxiv/vocab.pkl" # common vocabulary
    self.raw_data_root = "data/arxiv/%s.json" % self.dataset # location of raw data
    self.data_path = "./data/arxiv/%s" % self.dataset # location for storing pkl

    self.model_type = "rnn"
    self.checkpoint_dir="checkpoints/arxiv/%s/%s" % (self.modelname, self.model_type)

    self.decay_rate=0.95
    self.decay_step=20000
    self.generator_embed_dim=100
    self.n_hidden=500
    self.dropout=0.5 # this is keep-probability
    self.learning_rate=0.001
    # con_vocab_size=10000 #including <unk>
    # vocab_size=100 #including <sos>, <eos> and <unk>
    self.n_layers=2
    self.max_grad_norm = 1.0
    self.total_epoch=100
    self.train_batch_size=100
    self.init_scale=0.075
    self.print_topics=True
    self.generate_text=True
    self.cell_type = 'lstm'
