from collections import OrderedDict
import os.path as osp
import os
import sys
import torch
from overflowDataset import OpenFlowDataset
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
import argparse
import scipy
import numpy as np


parser = argparse.ArgumentParser('Cache testing')
parser.add_argument('--subset', type=float, default=1, help='percentage of data to sample from')
parser.add_argument('--CPUCachePerc', type=int, default=20, help='CPU cache percentage of data')
parser.add_argument('--LRUOnly', default=True, action='store_true')
parser.add_argument('--no-LRUOnly', dest='LRUOnly', action='store_false')
parser.add_argument('--sizes', type=str, default='10,10')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

print("Using subset of: ", args.subset, "%, CPU cache of: ", args.CPUCachePerc, "% ", "LRU only: ", args.LRUOnly)

sizes = [int(size) for size in args.sizes.split(',')]

__file__ = os.path.abspath('')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OpenFlow')

print("Loading dataset...")
dataset = OpenFlowDataset(path)
data = dataset[0]
 
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.stats = []
 
    def get(self, key: int) -> int:
        if key not in self.cache:
            self.stats.append(0)
            return -1
        else:
            self.stats.append(1)
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)


subset = int(data.num_nodes / (100/args.subset))

# Assume CPU cache is 20% of data
CPUCacheNum = int(subset / (100/args.CPUCachePerc))

# Assume GPU cache is 0.25% of data
GPUCacheNum = int(subset / 200)
node_ids = torch.flatten(data.edge_index.t())


GPUCache = LRUCache(GPUCacheNum)

# We sample from end of data
temp_loader = NeighborSampler(data.edge_index, sizes=sizes, node_idx=node_ids[len(node_ids) - subset*2:], batch_size=2)

def runLRU():
  CPUCache  = LRUCache(CPUCacheNum)

  pbar = tqdm(total=subset*2)
  for batch_size, ids, adjs in temp_loader:
    for i in ids:
      i = int(i)
      val = CPUCache.get(i)
      if (val == -1):
        # Fetch from SSD
        CPUCache.put(i,i)
    pbar.update(batch_size)
  pbar.close()

  t = torch.tensor(CPUCache.stats)
  torch.save(t, "LRU_subset_" + str(args.subset) + "Ccache_" + str(args.CPUCachePerc) + '.pt')

def runStatic():
  CPUCache  = LRUCache(CPUCacheNum)

  coo = data.edge_index.numpy()
  v = np.ones_like(coo[0])
  coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(data.num_nodes, data.num_nodes))
  csc = coo.tocsc()
  csr = coo.tocsr()

  csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
  csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))
  out_num_neighbors = csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]
  in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1])

  sorted_vals_out, indices_out = torch.sort(out_num_neighbors, descending=True)

  # Populate cache with highest out degree nodes
  for i in range(CPUCacheNum):
    val = int(indices_out[i])
    CPUCache.put(val, val)
  
  pbar = tqdm(total=subset*2)
  for batch_size, ids, adjs in temp_loader:
    for i in ids:
      i = int(i)
      val = CPUCache.get(i)
    pbar.update(batch_size)
  pbar.close()

  t = torch.tensor(CPUCache.stats)
  torch.save(t, "static_subset_" + str(args.subset) + "Ccache_" + str(args.CPUCachePerc) + '.pt')

if args.LRUOnly:
   runLRU()
else:
   runStatic()