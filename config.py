class Config:
  dataset="abstracts"
  raw_data_root = "data/%s.json" % dataset # location of raw data

  data_path = "./data/%s" % dataset # location for storing pkl

  model_type = "rnn"
  checkpoint_dir="checkpoints/%s/%s" % (dataset, model_type)

  decay_rate=0.95
  decay_step=20000
  # n_topics=100
  # projector_embed_dim=500
  generator_embed_dim=100
  n_hidden=500
  dropout=0.5 # this is keep-probability
  learning_rate=0.001
  # con_vocab_size=10000 #including <unk>
  vocab_size=100 #including <sos>, <eos> and <unk>
  n_layers=2
  max_grad_norm = 1.0
  total_epoch=100
  train_batch_size=100
  init_scale=0.075
  print_topics=True
  generate_text=True
  cell_type = 'lstm'
