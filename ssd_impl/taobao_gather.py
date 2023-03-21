import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import os.path as osp
import os
import argparse
import scipy
import numpy as np
import json
import time

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from customNeighborSampler import MMAPNeighborSampler
from cache import FeatureCache
from lib.utils import *

import os, psutil
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

__file__ = os.path.abspath('')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Used in sample and gather c++
os.environ['GINEX_NUM_THREADS'] = str(128)

dataset_path = "../../taobao"

def get_mmap_dataset(path=dataset_path):
    indptr_path = os.path.join(path, 'indptr.dat')
    indices_path = os.path.join(path, 'indices.dat')
    features_path = os.path.join(path, 'features.dat')
    sampled_path = os.path.join(path, 'sampled.dat')
    conf_path = os.path.join(path, 'conf.json')

    conf = json.load(open(conf_path, 'r'))

    indptr = np.fromfile(indptr_path, dtype=conf['indptr_dtype']).reshape(tuple(conf['indptr_shape']))
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    indices = np.fromfile(indices_path, dtype=conf['indices_dtype']).reshape(tuple(conf['indices_shape']))
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    features_shape = conf['features_shape']
    features = np.memmap(features_path, mode='r+', shape=tuple(features_shape), dtype=conf['features_dtype'])
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    indptr = torch.from_numpy(indptr)
    indices = torch.from_numpy(indices)
    features = torch.from_numpy(features)
    sampled = torch.load(sampled_path)
    sampled = torch.flatten(sampled.t())

    num_nodes = conf['num_nodes']
    num_features = conf['features_shape'][1]

    return indptr, indices, features, sampled, num_features, num_nodes, features_path


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=1)
argparser.add_argument('--batch-size', type=int, default=2)
argparser.add_argument('--num-workers', type=int, default=30)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--sizes', type=str, default='10,10')
argparser.add_argument('--ginex-num-threads', type=int, default=128)
argparser.add_argument('--ramsize', type=int, default=2000)
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
args = argparser.parse_args()

indptr, indices, x, sampled, num_features, num_nodes, features_path  = get_mmap_dataset()

num_edges_sampled = 10000

sizes = [int(size) for size in args.sizes.split(',')]
train_loader = MMAPNeighborSampler(indptr, indices, node_idx=sampled[-2*num_edges_sampled:],
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)


# Determine feature cache size by amount of main memory left
# Assume we have 2GB RAM
curr_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
print(curr_used)

RAM_left = max((args.ramsize - curr_used - 100,0))
feature_size_mb = os.path.getsize(os.path.join(dataset_path, 'features.dat')) / 1024 ** 2
cache_ratio = RAM_left / feature_size_mb
feature_cache_size = min((int(num_nodes * cache_ratio), num_nodes))
print("cache size: ", RAM_left)
print("cache ratio: ", cache_ratio)
print("cache size: ", feature_cache_size, "/", num_nodes)
#features_cache_size = 1

cache = FeatureCache(feature_cache_size, num_nodes, x, num_features, False)

score_path = os.path.join(dataset_path, 'out_deg.pth')
score = torch.load(score_path)
sorted_indices = score
print("sorted len: ", len(sorted_indices))

# Fill cache with top out-degree
indices = []
for i in range(feature_cache_size):
    indices.append(sorted_indices[i])

indices = torch.tensor(indices, dtype=torch.long).cpu()

useLRU = True
cache.fill_cache(indices, useLRU)

print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

@torch.no_grad()
def train(epoch):

    pbar = tqdm(total=int(2*num_edges_sampled))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    total_hits = 0
    total_misses = 0

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        # Gather
        #print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        #batch_inputs, hits, misses = gather_ginex(features_path, ids, num_features, cache) #cache.lru_gather(ids)
        batch_inputs, hits, misses = cache.lru_gather(ids)
        total_hits += hits
        total_misses += misses

        torch.cuda.empty_cache()
        if not useLRU:
            #del(batch_inputs)
            tensor_free(batch_inputs)
        pbar.update(batch_size)

    pbar.close()
    print("Total Hits: ", total_hits, " perc: ", total_hits/(total_hits+total_misses))
    print("Total misses: ", total_misses, " perc: ", total_misses/(total_hits+total_misses))


for epoch in range(args.num_epochs):
    start = time.time()
    train(epoch)
    end = time.time()
    print('Epoch time: {:.4f} ms'.format((end - start) * 1000))
