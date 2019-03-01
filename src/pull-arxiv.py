
import arxiv
import time
import json
import re

def pull_abstracts(category, num_batches, results_per_batch):
    abstracts = {}
    print("pulling %d abstracts from %s" % (num_batches*results_per_batch, category))
    for batch_num in range(num_batches):
        batch  = arxiv.query(search_query='cat:' + category, start = batch_num*results_per_batch, max_results=results_per_batch)
        print("\t%d: retrieved %d articles from %s" % (batch_num, len(batch), category))
        for article in batch:
            abstracts[article['id']] = article['summary']
        time.sleep(3)
    filename = re.sub('\.','-',category)
    with open('%s.json' % filename, 'w') as outfile:
        json.dump(abstracts, outfile)
    print("\twrote abstracts to %s.json\n" % filename)


batches = 30
batchsize = 1000
pull_abstracts('stat.ML', batches, batchsize)
pull_abstracts('hep-th', batches, batchsize)
pull_abstracts('math.AG', batches, batchsize)
pull_abstracts('astro-ph', batches, batchsize)
