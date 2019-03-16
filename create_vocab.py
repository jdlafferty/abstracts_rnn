import json
import numpy as np
import codecs
import pickle
from collections import Counter


files = ["hep-th", "math-AG", "astro-ph", "stat-ML"]
charvoc = Counter()

for file in files:
   with open("/home/lafferty/abstracts_rnn/data/arxiv/%s.json" % file, 'r') as fp:
      abstract = json.load(fp)
      nabs = 0
      print("\n%s:" % file)
      for arxiv_id in abstract:
         if nabs < 10:
            print("%s: %s..." % (arxiv_id, abstract[arxiv_id][0:50]))
         nabs = nabs + 1
         charvoc.update([c for c in abstract[arxiv_id].strip()])
      print("processed %d abstracts" % nabs)


vocab_words = [c[0] for c in charvoc.most_common() if c[1] > 100]
print("\nvocab tokens: "),
print(vocab_words)
print("vocab length: %d" % len(vocab_words))

vocab_words += ['<sos>', '<eos>', '<unk>']
vocab = {x: i+1 for i,x in enumerate(vocab_words)}
vocab_size = len(vocab)

voc = [vocab, vocab_size]
with open('vocab.pkl', 'wb') as fp:
   pickle.dump(voc, fp)

with open('vocab.pkl', 'rb') as fp:
   rvoc = pickle.load(fp)

print("wrote/read file successfully: "),
print(rvoc == voc)

