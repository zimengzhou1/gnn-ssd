import os.path as osp
import os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from overflowDataset import OpenFlowDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from neighbor_sampler import NeighborSampler
from functools import lru_cache
import random


def getCacheMiss(sampler, cacheSize):
  @lru_cache(maxsize=cacheSize)
  def get_value(key):
      return

  pbar = tqdm(total=sampleNum)
  for step, (batch_size, ids, adjs) in enumerate(sampler):
      for i in ids:
         get_value(int(i))
      pbar.update(1)
  pbar.close()
  info = get_value.cache_info()

  get_value.cache_clear()
  return info

def getHitRatio(info):
    return info.hits / (info.hits + info.misses)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    __file__ = os.path.abspath('')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OpenFlow')

    print("Loading dataset...")
    dataset = OpenFlowDataset(path)
    data = dataset[0]

    # 10% cache size
    cacheSize = 200000
    sampleNum = 1000000

    cacheSizes = [25000, 50000, 400000, 600000, 800000]

    for size in cacheSizes:
        # LRU
        print("Initializing sequential neighbor sampler for size ", size)
        train_loader = NeighborSampler(data.edge_index, sizes=[10,10], node_idx=data.edge_index[0][:sampleNum])
        cacheInfo = getCacheMiss(train_loader, size)
        print(cacheInfo)
        print("Cache hit ratio for sequential: ", getHitRatio(cacheInfo))


        print("Initializing random neighbor sampler")
        indices = torch.tensor(random.sample(range(len(data.edge_index[0])), sampleNum))
        sampled_values = data.edge_index[0][indices]
        random_loader = NeighborSampler(data.edge_index, sizes=[10,10], node_idx=sampled_values)
        cacheInfo = getCacheMiss(random_loader, size)
        print(cacheInfo)
        print("Cache hit ratio for sequential: ", getHitRatio(cacheInfo))



